"""FOCUS vocabulary replacement and embedding reinitialization for seq2seq ASR.

Ported from the LAPT framework (`src/tokenizer_utils.py`,
`src/model_utils.py:_initialize_focus_model`) and adapted to the two things
that differ here: the target corpus is ASR *transcripts* rather than raw web
text, and the base tokenizer belongs to Whisper rather than a text LM.

The pipeline is:

1. **Corpus** — write the normalized training transcripts to a JSONL file (for
   FOCUS's fastText step) and a matching plain-text file (for SentencePiece).
2. **Vocabulary** — train a fresh SentencePiece model over that text, then wrap
   it in a HuggingFace backend and *re-attach the base checkpoint's entire
   special-token block* (see `_append_base_special_tokens`).
3. **Embeddings** — hand the source model, source tokenizer, and new tokenizer
   to `deepfocus.FOCUS`, which copies embeddings for tokens the two vocabularies
   share and initializes the rest as sparsemax-weighted combinations of the
   shared ones, with weights from a fastText model trained on the target text.

Why this is safe for a model whose decoder is tied to its pretrained vocabulary
(the invariant in `.claude/CLAUDE.md` that says never to resize a seq2seq
vocabulary): that invariant exists because a bare `resize_token_embeddings()`
discards the pretrained decoder embeddings. FOCUS is the exception that earns
the resize — every row of the new matrix is either copied from or reconstructed
out of the pretrained rows, and the special-token block keeps its embeddings
verbatim because those tokens are carried over unchanged.
"""

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase


# Files copied verbatim from the base checkpoint into the new tokenizer
# directory. `tokenizer.json` is *not* among them -- that is the one file we
# rewrite -- and neither are `vocab.json` / `merges.txt`, whose presence would
# let a slow-tokenizer load silently resurrect the base vocabulary.
INHERITED_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "normalizer.json",
    "preprocessor_config.json",
)

FOCUS_EMBS_SUBDIR = "focus_embs"

# SentencePiece piece for out-of-vocabulary characters. Whisper's own tokenizer
# is byte-level and therefore has no unknown token at all (`unk_token` is an
# alias for `<|endoftext|>`), but the non-byte-level model trained here needs a
# real one, and both backend models index it by id.
UNK_PIECE = "<unk>"


# ---------------------------------------------------------------------------
# Paths and cache keys
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FocusPaths:
    """Filesystem locations for one FOCUS configuration.

    Attributes:
        tokenizer_dir: Directory holding the trained tokenizer, its SentencePiece
            model, and the cached embedding sidecars.
        corpus_jsonl: JSONL corpus (`{"text": ...}` per line) consumed by FOCUS's
            fastText step.
        corpus_txt: Plain-text corpus (one transcript per line) consumed by
            SentencePiece training.
    """

    tokenizer_dir: str
    corpus_jsonl: str
    corpus_txt: str


def is_enabled(args: DictConfig) -> bool:
    """Report whether FOCUS vocabulary replacement is switched on."""
    focus = args.get("focus")
    return bool(focus is not None and focus.get("enabled", False))


def tokenizer_id(args: DictConfig) -> str:
    """Name the tokenizer artifact for this configuration.

    Encodes only the fields that change the *vocabulary*; everything else is
    caught by `FocusTokenizerConfig.check_cached`. The FOCUS-embedding knobs are
    deliberately absent, so a change to them reuses the tokenizer and only
    recomputes the embedding sidecar.

    Returns:
        A directory-safe name such as `focus-v4k-unigram`.
    """
    from src.artifact_configs import format_number

    parts = [f"focus-v{format_number(args.focus.vocab_size)}"]

    algorithm = args.focus.get("tokenizer_algorithm")
    if algorithm is not None:
        parts.append(str(algorithm).lower())

    num_samples = args.focus.get("num_samples")
    if num_samples is not None:
        parts.append(f"s{format_number(num_samples)}")

    coverage = args.focus.get("character_coverage", 1.0)
    if float(coverage) != 1.0:
        parts.append(f"cc{str(coverage).replace('.', 'p')}")

    return "-".join(parts)


def resolve_paths(args: DictConfig, cache_dir: str) -> FocusPaths:
    """Derive every FOCUS path from config alone.

    Deriving rather than threading means `models.py` can find the corpus and the
    embedding cache without being handed state from `processors.py`.

    Args:
        args: Full Hydra config.
        cache_dir: The dataset cache directory (`args.dataset.cache_dir`).

    Returns:
        The `FocusPaths` for this run.
    """
    focus_root = os.path.join(cache_dir, "focus")
    corpus_stem = _corpus_stem(args)
    return FocusPaths(
        tokenizer_dir=os.path.join(focus_root, tokenizer_id(args)),
        corpus_jsonl=os.path.join(focus_root, f"{corpus_stem}.jsonl"),
        corpus_txt=os.path.join(focus_root, f"{corpus_stem}.txt"),
    )


def _corpus_stem(args: DictConfig) -> str:
    """Name the shared corpus file.

    The corpus depends only on the sampling parameters, not on the tokenizer, so
    it lives one level above the tokenizer directories and is reused across
    vocabulary sizes.
    """
    from src.artifact_configs import format_number

    num_samples = args.focus.get("num_samples")
    if num_samples is None:
        return f"corpus_all_seed{args.seed}"
    return f"corpus_s{format_number(num_samples)}_seed{args.seed}"


def embedding_hash(args: DictConfig) -> str:
    """Hash the inputs that determine the FOCUS embedding *values*.

    Independent of the vocabulary hyperparameters, which are already encoded in
    the tokenizer directory name that contains the sidecar. Changing a fastText
    knob therefore recomputes the embeddings without retraining the tokenizer.

    Returns:
        An 8-character hex digest.
    """
    from src.artifact_configs import DatasetConfig

    keys = {
        "data": DatasetConfig.from_args(args).to_dict(),
        "num_samples": args.focus.get("num_samples"),
        "seed": args.seed,
        "pretrained_name": args.model.pretrained_name,
        "fasttext_model_min_count": args.focus.get("fasttext_model_min_count", 1),
        "fasttext_model_epochs": args.focus.get("fasttext_model_epochs", 3),
        "fasttext_model_dim": args.focus.get("fasttext_model_dim", 100),
    }
    canonical = json.dumps(keys, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def _sidecar_paths(tokenizer_dir: str, digest: str) -> tuple[str, str, str]:
    """Return the (input, output, meta) sidecar paths for an embedding hash."""
    subdir = os.path.join(tokenizer_dir, FOCUS_EMBS_SUBDIR)
    return (
        os.path.join(subdir, f"{digest}.input.pt"),
        os.path.join(subdir, f"{digest}.output.pt"),
        os.path.join(subdir, f"{digest}.meta.yaml"),
    )


# ---------------------------------------------------------------------------
# Stage 1: corpus preparation
# ---------------------------------------------------------------------------

def prepare_corpus(
    texts: list[str],
    paths: FocusPaths,
    num_samples: Optional[int] = None,
    seed: int = 1,
) -> None:
    """Materialize the target-language corpus in the two formats FOCUS needs.

    Both files are written together and cached: if the JSONL already exists the
    whole step is skipped, so the corpus stays byte-identical across runs that
    share a sampling configuration.

    Args:
        texts: Normalized training transcripts.
        paths: Destination paths from `resolve_paths`.
        num_samples: Number of transcripts to sample, or None for all of them.
            ASR corpora are small enough that None is the sensible default.
        seed: Sampling seed; ignored when `num_samples` is None.

    Raises:
        ValueError: If `texts` is empty.
    """
    if os.path.exists(paths.corpus_jsonl) and os.path.exists(paths.corpus_txt):
        print(
            f"FOCUS corpus already present at {paths.corpus_jsonl}, reusing it",
            file=sys.stderr,
        )
        return

    if not texts:
        raise ValueError("Cannot prepare a FOCUS corpus from an empty transcript list.")

    selected = _sample_texts(texts, num_samples=num_samples, seed=seed)

    os.makedirs(os.path.dirname(paths.corpus_jsonl), exist_ok=True)
    written = 0
    with open(paths.corpus_jsonl, "w", encoding="utf-8") as jsonl_file, \
            open(paths.corpus_txt, "w", encoding="utf-8") as text_file:
        for text in selected:
            # Blank transcripts contribute nothing to either SentencePiece or
            # fastText and SentencePiece warns on every one of them.
            if not text.strip():
                continue
            # A transcript containing a newline would become two SentencePiece
            # sentences and two fastText lines, so flatten it first.
            flattened = " ".join(text.split())
            json.dump({"text": flattened}, jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")
            text_file.write(flattened + "\n")
            written += 1

    total_characters = sum(len(text) for text in selected)
    print(
        f"FOCUS corpus written to {paths.corpus_jsonl} "
        f"({written} transcripts, {total_characters:,} characters)",
        file=sys.stderr,
    )
    _warn_on_small_corpus(written, total_characters)


def _sample_texts(
    texts: list[str],
    num_samples: Optional[int],
    seed: int,
) -> list[str]:
    """Take a reproducible subsample of the transcripts, or all of them."""
    if num_samples is None or num_samples >= len(texts):
        if num_samples is not None and num_samples > len(texts):
            print(
                f"Warning: focus.num_samples={num_samples} exceeds the "
                f"{len(texts)} available transcripts; using all of them.",
                file=sys.stderr,
            )
        return list(texts)

    import random

    sampler = random.Random(seed)
    indices = sorted(sampler.sample(range(len(texts)), num_samples))
    return [texts[index] for index in indices]


def _warn_on_small_corpus(num_transcripts: int, total_characters: int) -> None:
    """Warn when the corpus is too thin for the fastText step to be meaningful.

    FOCUS was designed against monolingual web corpora of hundreds of millions
    of characters. ASR transcript sets are routinely four or five orders of
    magnitude smaller, and a fastText model fit on a few hundred thousand
    characters gives noisy similarities -- which shows up as new tokens being
    initialized from near-arbitrary neighbours rather than as an error.
    """
    if total_characters >= 1_000_000:
        return
    print(
        f"\n  WARNING: the FOCUS corpus holds only {total_characters:,} characters "
        f"across {num_transcripts} transcripts.\n"
        f"  FOCUS derives its similarities from a fastText model trained on this "
        f"text alone;\n"
        f"  below roughly a million characters those similarities are noisy and the "
        f"new-token\n"
        f"  embeddings degrade toward arbitrary convex combinations. Consider a "
        f"smaller\n"
        f"  focus.vocab_size, or supplying additional target-language text.\n",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Stage 2: vocabulary construction
# ---------------------------------------------------------------------------

def build_tokenizer(
    args: DictConfig,
    spec,
    paths: FocusPaths,
) -> PreTrainedTokenizerBase:
    """Train (or load) the FOCUS tokenizer for this configuration.

    Args:
        args: Full Hydra config.
        spec: The architecture's `ModelSpec`; its `tokenizer_class` is what the
            finished directory is reloaded as, so the result is a genuine
            `WhisperTokenizerFast` rather than a bare `PreTrainedTokenizerFast`.
        paths: Paths from `resolve_paths`; `corpus_txt` must already exist unless
            the tokenizer is cached.

    Returns:
        The trained tokenizer, reloaded from disk.
    """
    tokenizer_json = os.path.join(paths.tokenizer_dir, "tokenizer.json")
    if os.path.exists(tokenizer_json):
        print(
            f"Loading cached FOCUS tokenizer from {paths.tokenizer_dir}",
            file=sys.stderr,
        )
        return spec.tokenizer_class.from_pretrained(paths.tokenizer_dir)

    base_tokenizer = spec.tokenizer_class.from_pretrained(args.model.pretrained_name)
    algorithm = _resolve_algorithm(args, base_tokenizer)

    print(
        f"Training FOCUS tokenizer: algorithm={algorithm}, "
        f"vocab_size={args.focus.vocab_size}",
        file=sys.stderr,
    )

    os.makedirs(paths.tokenizer_dir, exist_ok=True)
    sp_model = _train_sentencepiece_model(
        text_file_path=paths.corpus_txt,
        output_dir=paths.tokenizer_dir,
        algorithm=algorithm,
        vocab_size=args.focus.vocab_size,
        character_coverage=args.focus.get("character_coverage", 1.0),
    )

    vocab_scores = [
        (sp_model.id_to_piece(index), sp_model.get_score(index))
        for index in range(sp_model.get_piece_size())
    ]

    if algorithm == "bpe":
        backend_tokenizer = _create_bpe_backend(
            spm_model_path=os.path.join(paths.tokenizer_dir, "spm.model"),
            vocab_scores=vocab_scores,
        )
    else:
        backend_tokenizer = _create_unigram_backend(vocab_scores)

    num_special = _append_base_special_tokens(backend_tokenizer, base_tokenizer)

    backend_tokenizer.save(os.path.join(paths.tokenizer_dir, "tokenizer.json"))
    _copy_inherited_files(args.model.pretrained_name, paths.tokenizer_dir)

    tokenizer = spec.tokenizer_class.from_pretrained(paths.tokenizer_dir)
    _validate_tokenizer(tokenizer, base_tokenizer, args.focus.vocab_size, num_special)

    print(
        f"FOCUS tokenizer saved to {paths.tokenizer_dir} "
        f"({args.focus.vocab_size} learned pieces + {num_special} inherited "
        f"special tokens = {len(tokenizer)} total)",
        file=sys.stderr,
    )
    return tokenizer


def _resolve_algorithm(args: DictConfig, base_tokenizer) -> str:
    """Pick the SentencePiece algorithm, defaulting to the base tokenizer's."""
    configured = args.focus.get("tokenizer_algorithm")
    if configured is not None:
        algorithm = str(configured).lower()
        if algorithm not in ("bpe", "unigram"):
            raise ValueError(
                f"focus.tokenizer_algorithm must be 'bpe' or 'unigram', "
                f"got {configured!r}"
            )
        return algorithm

    model_name = type(base_tokenizer.backend_tokenizer.model).__name__.lower()
    if "bpe" in model_name:
        return "bpe"
    if "unigram" in model_name:
        return "unigram"
    raise ValueError(
        f"Could not infer a SentencePiece algorithm from the base tokenizer's "
        f"{model_name!r} model. Set focus.tokenizer_algorithm explicitly."
    )


def _train_sentencepiece_model(
    text_file_path: str,
    output_dir: str,
    algorithm: str,
    vocab_size: int,
    character_coverage: float,
):
    """Train a SentencePiece model over the transcripts and load it back.

    Unlike LAPT's version this trains *no* special pieces beyond `<unk>`: the
    base checkpoint's special tokens are re-attached afterwards as added tokens
    so they keep their original relative order (see
    `_append_base_special_tokens`). Letting SentencePiece mint them instead
    would scatter them through the learned vocabulary and break the id
    arithmetic that Whisper's generation config depends on.

    Raises:
        ValueError: If SentencePiece cannot reach the requested vocabulary size,
            which on an ASR-sized corpus usually means `focus.vocab_size` is too
            large rather than that anything is misconfigured.
    """
    import sentencepiece as spm

    model_prefix = os.path.join(output_dir, "spm")
    train_args = {
        "input": text_file_path,
        "model_prefix": model_prefix,
        "model_type": algorithm,
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        # text normalization is already done by src/preprocessing.py; a second
        # pass here would desynchronize the vocabulary from the cached labels
        "normalization_rule_name": "identity",
        "hard_vocab_limit": True,
        "unk_id": 0,
        "unk_piece": UNK_PIECE,
        "bos_id": -1,
        "eos_id": -1,
        "pad_id": -1,
    }

    print(f"Training SentencePiece with args: {train_args}", file=sys.stderr)
    try:
        spm.SentencePieceTrainer.Train(**train_args)
    except RuntimeError as error:
        if "vocab_size" not in str(error):
            raise
        raise ValueError(
            f"SentencePiece could not build a {vocab_size}-piece vocabulary from "
            f"{text_file_path}.\n"
            f"ASR transcript corpora are small, and the reachable vocabulary size "
            f"is bounded by the number of distinct substrings in them. Lower "
            f"focus.vocab_size (1024-8192 is a reasonable range for a few hours "
            f"of speech) or supply more target-language text.\n"
            f"SentencePiece said: {error}"
        ) from error

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(f"{model_prefix}.model")

    actual_size = sp_model.get_piece_size()
    if actual_size != vocab_size:
        raise ValueError(
            f"Trained SentencePiece model has {actual_size} pieces but "
            f"{vocab_size} were requested."
        )
    return sp_model


def _apply_spm_pipeline(backend_tokenizer) -> None:
    """Install the normalizer, pre-tokenizer, and decoder SentencePiece implies.

    This deliberately *replaces* the base checkpoint's byte-level pipeline
    rather than reusing it: the learned pieces are SentencePiece pieces, written
    with `▁` for a leading space, and a `ByteLevel` pre-tokenizer would never
    produce a string that matches one. The two representations still line up
    where it matters, because `deepfocus.vocab_helper.canonicalize_vocab`
    decodes both vocabularies to text before comparing them (`Ġuku` and `▁uku`
    both canonicalize to `▁uku`), so overlap detection is unaffected.
    """
    from tokenizers import decoders, normalizers
    from tokenizers.pre_tokenizers import Metaspace

    backend_tokenizer.normalizer = normalizers.Sequence(normalizers=[])
    backend_tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    backend_tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")


def _create_unigram_backend(vocab_scores: list[tuple[str, float]]):
    """Build a `tokenizers.Tokenizer` around a Unigram model."""
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram

    # unk_id=0 matches the `unk_id` passed to SentencePiece training;
    # byte_fallback=False keeps the model consistent with a training run that
    # had byte fallback disabled.
    backend_tokenizer = Tokenizer(Unigram(vocab_scores, unk_id=0, byte_fallback=False))
    _apply_spm_pipeline(backend_tokenizer)
    return backend_tokenizer


def _create_bpe_backend(spm_model_path: str, vocab_scores: list[tuple[str, float]]):
    """Build a `tokenizers.Tokenizer` around a BPE model.

    A SentencePiece BPE model stores pieces and scores but no explicit merge
    list, so the merges are reconstructed from the scores exactly as
    HuggingFace's own `SpmConverter` does (higher score = earlier merge).
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers.convert_slow_tokenizer import SentencePieceExtractor

    _, merges = SentencePieceExtractor(spm_model_path).extract(vocab_scores)
    bpe_vocab = {piece: index for index, (piece, _score) in enumerate(vocab_scores)}

    backend_tokenizer = Tokenizer(
        BPE(
            bpe_vocab,
            merges,
            unk_token=UNK_PIECE,
            fuse_unk=True,
            byte_fallback=False,
            dropout=None,
        )
    )
    _apply_spm_pipeline(backend_tokenizer)
    return backend_tokenizer


def _append_base_special_tokens(backend_tokenizer, base_tokenizer) -> int:
    """Re-attach the base checkpoint's special-token block after the new pieces.

    This is the part that makes vocabulary replacement survivable for Whisper.
    Its added-token block is not decoration: it holds `<|startoftranscript|>`,
    the 99 language tokens, `<|transcribe|>`/`<|translate|>`,
    `<|notimestamps|>`, `<|endoftext|>`, and 1501 timestamp tokens -- and the
    generation config addresses all of them *by id*, including through the
    `lang_to_id` and `task_to_id` maps and through timestamp index arithmetic.

    Appending them in their original id order after the learned pieces means
    every one of those ids shifts by the same constant, which keeps the block
    internally consistent; `models.py:remap_generation_config` then rewrites the
    stored ids by looking each token back up by name.

    Args:
        backend_tokenizer: The freshly built `tokenizers.Tokenizer`, modified in
            place.
        base_tokenizer: The base checkpoint's tokenizer.

    Returns:
        The number of special tokens appended.

    Raises:
        ValueError: If a special token collides with a learned piece, which
            would make the appended block non-contiguous.
    """
    from tokenizers import AddedToken

    added_vocab = base_tokenizer.get_added_vocab()
    ordered_tokens = [token for token, _id in sorted(added_vocab.items(), key=lambda item: item[1])]

    learned_vocab = backend_tokenizer.get_vocab()
    collisions = [token for token in ordered_tokens if token in learned_vocab]
    if collisions:
        raise ValueError(
            f"{len(collisions)} of the base checkpoint's special tokens were also "
            f"learned as SentencePiece pieces (e.g. {collisions[:3]}), so they "
            f"cannot be appended as a contiguous block. This should be impossible "
            f"for transcripts that have been through src/preprocessing.py; check "
            f"that the corpus does not literally contain them."
        )

    vocab_size_before = backend_tokenizer.get_vocab_size(with_added_tokens=True)
    backend_tokenizer.add_special_tokens(
        [AddedToken(token, special=True, normalized=False) for token in ordered_tokens]
    )

    # Verify the block landed contiguously and in order; a silent reordering
    # here would corrupt every id in the generation config.
    for offset, token in enumerate(ordered_tokens):
        actual_id = backend_tokenizer.token_to_id(token)
        expected_id = vocab_size_before + offset
        if actual_id != expected_id:
            raise ValueError(
                f"Special token {token!r} was assigned id {actual_id}, expected "
                f"{expected_id}. The inherited special-token block must stay "
                f"contiguous and in its original order."
            )

    print(
        f"Appended {len(ordered_tokens)} inherited special tokens at ids "
        f"{vocab_size_before}-{vocab_size_before + len(ordered_tokens) - 1}",
        file=sys.stderr,
    )
    return len(ordered_tokens)


def _copy_inherited_files(pretrained_name: str, tokenizer_dir: str) -> None:
    """Copy the base checkpoint's auxiliary tokenizer files into the new dir.

    `tokenizer_config.json` carries the tokenizer class and its special-token
    declarations, `special_tokens_map.json` the role assignments,
    `normalizer.json` Whisper's English spelling table (loaded by
    `WhisperTokenizerFast` and used by its `normalize=True` decode path), and
    `preprocessor_config.json` the feature extractor settings that let the
    directory stand alone as a processor. Files the checkpoint does not have are
    skipped silently.

    `vocab.json` and `merges.txt` are pointedly *not* copied: they describe the
    old vocabulary, and leaving them beside the new `tokenizer.json` would let a
    slow-tokenizer load resurrect it.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for filename in INHERITED_TOKENIZER_FILES:
        destination = os.path.join(tokenizer_dir, filename)
        local_candidate = os.path.join(pretrained_name, filename)

        if os.path.isdir(pretrained_name):
            if os.path.exists(local_candidate):
                shutil.copyfile(local_candidate, destination)
            continue

        try:
            cached = hf_hub_download(pretrained_name, filename)
        except (EntryNotFoundError, OSError):
            continue
        shutil.copyfile(cached, destination)


def _validate_tokenizer(
    tokenizer,
    base_tokenizer,
    expected_learned: int,
    expected_special: int,
) -> None:
    """Check the reloaded tokenizer against what was written.

    Catches the two failure modes that would otherwise surface much later as an
    unexplained accuracy floor: a vocabulary that did not come out the expected
    size, and a special token whose string survived but whose role did not.
    """
    expected_total = expected_learned + expected_special
    if len(tokenizer) != expected_total:
        raise ValueError(
            f"FOCUS tokenizer has {len(tokenizer)} tokens, expected "
            f"{expected_total} ({expected_learned} learned + {expected_special} "
            f"inherited special)."
        )

    missing = [
        token
        for token in base_tokenizer.all_special_tokens
        if tokenizer.convert_tokens_to_ids(token) is None
    ]
    if missing:
        raise ValueError(
            f"FOCUS tokenizer is missing base special tokens: {missing}"
        )

    round_tripped = tokenizer.decode(
        tokenizer("a probe transcript", add_special_tokens=False).input_ids
    )
    print(
        f"FOCUS tokenizer validated (round-trip: {round_tripped!r})",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Stage 3: embedding initialization
# ---------------------------------------------------------------------------

def compute_embeddings(
    args: DictConfig,
    source_model,
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    paths: FocusPaths,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Produce the new embedding matrices, using a cached sidecar when possible.

    Args:
        args: Full Hydra config.
        source_model: The pretrained model, still at its original vocab size.
        source_tokenizer: The pretrained checkpoint's own tokenizer.
        target_tokenizer: The FOCUS tokenizer from `build_tokenizer`.
        paths: Paths from `resolve_paths`.

    Returns:
        `(input_embeddings, output_embeddings)`; the second is None whenever the
        model ties its input and output embeddings, as Whisper does.
    """
    digest = embedding_hash(args)
    input_pt, output_pt, meta_yaml = _sidecar_paths(paths.tokenizer_dir, digest)

    has_separate_output = not getattr(
        source_model.config, "tie_word_embeddings", True
    )

    cached = _load_cached_embeddings(input_pt, output_pt, has_separate_output)
    if cached is not None:
        print(f"Loaded cached FOCUS embeddings from {input_pt}", file=sys.stderr)
        return cached

    input_embeddings, output_embeddings = _run_focus(
        args=args,
        source_model=source_model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        corpus_jsonl=paths.corpus_jsonl,
        has_separate_output=has_separate_output,
    )

    os.makedirs(os.path.dirname(input_pt), exist_ok=True)
    torch.save(input_embeddings, input_pt)
    if output_embeddings is not None:
        torch.save(output_embeddings, output_pt)
    with open(meta_yaml, "w") as meta_file:
        yaml.dump(
            {
                "embedding_hash": digest,
                "tokenizer_id": tokenizer_id(args),
                "pretrained_name": args.model.pretrained_name,
                "target_vocab_size": len(target_tokenizer),
                "corpus": paths.corpus_jsonl,
                "seed": args.seed,
                "focus": OmegaConf.to_container(args.focus, resolve=True),
            },
            meta_file,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"Cached FOCUS embeddings to {input_pt}", file=sys.stderr)

    return input_embeddings, output_embeddings


def _load_cached_embeddings(
    input_pt: str,
    output_pt: str,
    has_separate_output: bool,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Load a matching embedding sidecar, or return None if there isn't one."""
    if not os.path.exists(input_pt):
        return None

    input_embeddings = torch.load(input_pt, weights_only=True)
    if not has_separate_output:
        return input_embeddings, None

    if not os.path.exists(output_pt):
        print(
            f"Warning: found cached input embeddings at {input_pt} but no "
            f"matching output embeddings; recomputing both.",
            file=sys.stderr,
        )
        return None
    return input_embeddings, torch.load(output_pt, weights_only=True)


def _run_focus(
    args: DictConfig,
    source_model,
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    corpus_jsonl: str,
    has_separate_output: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Call `deepfocus.FOCUS` for the input and, if untied, output embeddings."""
    try:
        from deepfocus import FOCUS
    except ImportError as error:
        raise ImportError(
            "focus.enabled=true requires the deepfocus package "
            "(`pip install deepfocus`); it is listed in environment.yml."
        ) from error

    if not os.path.exists(corpus_jsonl):
        raise FileNotFoundError(
            f"FOCUS corpus not found at {corpus_jsonl}. It is written during "
            f"tokenizer setup, so this means the pipeline ran out of order."
        )

    focus_kwargs = {
        "source_tokenizer": source_tokenizer,
        "target_tokenizer": target_tokenizer,
        "target_training_data_path": corpus_jsonl,
        "fasttext_model_epochs": args.focus.get("fasttext_model_epochs", 3),
        "fasttext_model_dim": args.focus.get("fasttext_model_dim", 100),
        "fasttext_model_min_count": args.focus.get("fasttext_model_min_count", 1),
        "seed": args.seed,
    }

    print(
        f"Running FOCUS: {len(source_tokenizer)} source tokens -> "
        f"{len(target_tokenizer)} target tokens",
        file=sys.stderr,
    )
    input_embeddings = FOCUS(
        source_embeddings=source_model.get_input_embeddings().weight,
        **focus_kwargs,
    )

    output_embeddings = None
    if has_separate_output:
        print(
            "Model does not tie its embeddings; running FOCUS on the output "
            "matrix as well",
            file=sys.stderr,
        )
        output_embeddings = FOCUS(
            source_embeddings=source_model.get_output_embeddings().weight,
            **focus_kwargs,
        )

    return input_embeddings, output_embeddings


# ---------------------------------------------------------------------------
# Stage 3b: installing the new embeddings on the model
# ---------------------------------------------------------------------------

def apply_to_model(
    args: DictConfig,
    model,
    tokenizer: PreTrainedTokenizerBase,
    spec,
    cache_dir: str,
) -> None:
    """Replace a loaded model's vocabulary with the FOCUS one, in place.

    Resizes the embedding matrix to the new vocabulary, overwrites every row
    with the FOCUS values, and rewrites the token ids stored in the model and
    generation configs so they address the new vocabulary rather than the old.

    Args:
        args: Full Hydra config.
        model: The freshly loaded pretrained model.
        tokenizer: The FOCUS tokenizer the model must now speak.
        spec: The architecture's `ModelSpec`.
        cache_dir: The dataset cache directory, for locating the FOCUS artifacts.
    """
    paths = resolve_paths(args, cache_dir)
    source_tokenizer = spec.tokenizer_class.from_pretrained(args.model.pretrained_name)

    input_embeddings, output_embeddings = compute_embeddings(
        args=args,
        source_model=model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=tokenizer,
        paths=paths,
    )

    _install_embeddings(model, tokenizer, input_embeddings, output_embeddings)
    remap_special_token_ids(model, source_tokenizer, tokenizer)

    print(
        f"FOCUS applied: vocabulary replaced with {len(tokenizer)} tokens "
        f"(was {len(source_tokenizer)})",
        file=sys.stderr,
    )


def _install_embeddings(
    model,
    tokenizer: PreTrainedTokenizerBase,
    input_embeddings: torch.Tensor,
    output_embeddings: Optional[torch.Tensor],
) -> None:
    """Resize the model's embedding modules and copy the FOCUS rows into them.

    The existing modules are resized and overwritten rather than replaced via
    `set_input_embeddings`, because some architectures subclass `nn.Embedding` to
    add forward-time behaviour (XGLM scales by sqrt(d_model)) that swapping the
    module would silently drop.
    """
    model.resize_token_embeddings(len(tokenizer))

    current_input = model.get_input_embeddings()
    with torch.no_grad():
        current_input.weight.data.copy_(
            input_embeddings.to(
                device=current_input.weight.device,
                dtype=current_input.weight.dtype,
            )
        )

    if output_embeddings is not None:
        current_output = model.get_output_embeddings()
        with torch.no_grad():
            current_output.weight.data.copy_(
                output_embeddings.to(
                    device=current_output.weight.device,
                    dtype=current_output.weight.dtype,
                )
            )
    else:
        # tied embeddings: re-point the output head at the resized input matrix
        model.tie_weights()


def remap_special_token_ids(
    model,
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Rewrite every stored token id so it addresses the new vocabulary.

    Both `model.config` and `model.generation_config` cache token ids resolved
    against the *old* vocabulary. Whisper's generation config is especially
    dense with them -- `decoder_start_token_id`, `no_timestamps_token_id`,
    `prev_sot_token_id`, and the whole `lang_to_id` / `task_to_id` maps -- and
    every one is now off by the size difference between the two vocabularies.
    Leaving them stale does not raise: `generate()` would simply prefix the
    decoder with whatever learned subword now sits at id 50258.

    Each id is recovered by name rather than by adding an offset, so the remap
    stays correct even if the special-token block ever changes shape.

    Args:
        model: The model whose configs should be rewritten, modified in place.
        source_tokenizer: The pretrained tokenizer the stored ids refer to.
        target_tokenizer: The FOCUS tokenizer the ids should refer to.
    """
    # Look the new id up in an explicit vocabulary dict rather than through
    # `convert_tokens_to_ids`, which maps an *absent* token to `unk_token_id`.
    # Whisper aliases its unknown token to `<|endoftext|>`, so a present-token
    # hit and an absent-token miss would be indistinguishable -- and the one
    # token that collision silently swallows is the end-of-sequence id, leaving
    # generation with no stop condition.
    target_vocab = target_tokenizer.get_vocab()

    def translate(old_id):
        """Map one old id to its new one, by token string.

        Returns None when the token has no counterpart in the new vocabulary,
        which is the normal outcome for an id that pointed into the replaced
        subword block rather than the inherited special-token block.
        """
        if old_id is None:
            return None
        token = source_tokenizer.convert_ids_to_tokens(int(old_id))
        if token is None:
            return None
        return target_vocab.get(token)

    scalar_fields = (
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "decoder_start_token_id",
        "no_timestamps_token_id",
        "prev_sot_token_id",
    )

    for holder in (model.config, getattr(model, "generation_config", None)):
        if holder is None:
            continue
        for field_name in scalar_fields:
            if not hasattr(holder, field_name):
                continue
            translated = translate(getattr(holder, field_name))
            if translated is not None:
                setattr(holder, field_name, translated)

        # `lang_to_id` and `task_to_id` are keyed by token string / task name,
        # so only the values need translating.
        for map_name in ("lang_to_id", "task_to_id"):
            id_map = getattr(holder, map_name, None)
            if not id_map:
                continue
            setattr(
                holder,
                map_name,
                {
                    key: translate(value)
                    for key, value in id_map.items()
                    if translate(value) is not None
                },
            )

        # The pretrained suppression lists index the *old* subword vocabulary
        # (Whisper suppresses tokens improbable in its pretraining languages).
        # Those ids have no meaning against a freshly learned vocabulary, so
        # clear them rather than translate a list that is mostly untranslatable.
        if getattr(holder, "suppress_tokens", None):
            holder.suppress_tokens = []
        begin_suppress = getattr(holder, "begin_suppress_tokens", None)
        if begin_suppress:
            translated_begin = [
                translated
                for translated in (translate(old) for old in begin_suppress)
                if translated is not None
            ]
            holder.begin_suppress_tokens = translated_begin

        # forced_decoder_ids is the superseded (pre-4.34) prefix mechanism and
        # holds (position, id) pairs against the old vocabulary.
        if getattr(holder, "forced_decoder_ids", None):
            holder.forced_decoder_ids = None

    _verify_remapped_ids(model, target_tokenizer)

    print(
        "Remapped model and generation config token ids onto the FOCUS "
        "vocabulary (pretrained suppress_tokens cleared)",
        file=sys.stderr,
    )


def _verify_remapped_ids(model, target_tokenizer: PreTrainedTokenizerBase) -> None:
    """Assert that no stored token id still points outside the new vocabulary.

    A stale id is not a crash, it is a silently broken run -- an out-of-range
    `eos_token_id` means generation never stops, and an out-of-range
    `decoder_start_token_id` means the decoder is prompted with a token the
    embedding matrix does not contain. Both are worth failing loudly at startup.
    """
    vocab_size = len(target_tokenizer)
    checked_fields = (
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "decoder_start_token_id",
        "no_timestamps_token_id",
        "prev_sot_token_id",
    )

    stale = []
    for holder_name, holder in (
        ("config", model.config),
        ("generation_config", getattr(model, "generation_config", None)),
    ):
        if holder is None:
            continue
        for field_name in checked_fields:
            value = getattr(holder, field_name, None)
            if value is not None and int(value) >= vocab_size:
                stale.append(f"{holder_name}.{field_name}={value}")
        for map_name in ("lang_to_id", "task_to_id"):
            for key, value in (getattr(holder, map_name, None) or {}).items():
                if value is not None and int(value) >= vocab_size:
                    stale.append(f"{holder_name}.{map_name}[{key}]={value}")

    if stale:
        raise ValueError(
            f"FOCUS remapping left token ids pointing outside the new "
            f"{vocab_size}-token vocabulary: {stale}. This means a special "
            f"token was not carried over from the base checkpoint."
        )
