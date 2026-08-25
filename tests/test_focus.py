"""Tests for src/focus.py (FOCUS vocabulary replacement)."""

import json
import os

import pytest
from omegaconf import OmegaConf

from src import focus
from src.artifact_configs import (
    FocusTokenizerConfig,
    format_number,
    processed_cache_dirname,
)


sentencepiece = pytest.importorskip("sentencepiece", reason="FOCUS needs sentencepiece")


@pytest.fixture
def whisper_config(base_config):
    """A Whisper config with FOCUS switched on."""
    config = base_config.copy()
    config.model = OmegaConf.create({
        "type": "whisper",
        "pretrained_name": "openai/whisper-small",
        "short_name": "whisper_small",
        "language": "sw",
        "task": "transcribe",
    })
    config.dataset.max_label_length = 448
    config.focus = OmegaConf.create({
        "enabled": True,
        "vocab_size": 512,
        "tokenizer_algorithm": "unigram",
        "character_coverage": 1.0,
        "num_samples": None,
        "fasttext_model_min_count": 1,
        "fasttext_model_epochs": 3,
        "fasttext_model_dim": 100,
    })
    return config


@pytest.fixture
def zulu_texts():
    """Synthetic Bantu-shaped text, varied enough to train a 512-piece vocab.

    SentencePiece can only reach a vocabulary size the corpus actually supports,
    so the fixture builds a few thousand distinct word forms from a prefix /
    stem / suffix product rather than repeating a handful of sentences.
    """
    import random

    prefixes = ["um", "aba", "isi", "izi", "uku", "ama", "ubu", "ulu"]
    stems = [
        "ntu", "khathi", "hamba", "bonga", "khulu", "fund", "sebenz",
        "thand", "buk", "phum", "ngen", "cul",
    ]
    suffixes = ["a", "ile", "isa", "ana", "eni", "wami", "wakho", "wethu"]

    words = [
        prefix + stem + suffix
        for prefix in prefixes
        for stem in stems
        for suffix in suffixes
    ]
    sampler = random.Random(1)
    return [" ".join(sampler.sample(words, 6)) for _ in range(3000)]


class TestEnablement:
    def test_disabled_by_default(self, base_config):
        assert focus.is_enabled(base_config) is False

    def test_disabled_when_flag_false(self, base_config):
        base_config.focus = OmegaConf.create({"enabled": False})
        assert focus.is_enabled(base_config) is False

    def test_enabled(self, whisper_config):
        assert focus.is_enabled(whisper_config) is True


class TestNamingAndPaths:
    def test_format_number(self):
        assert format_number(512) == "512"
        assert format_number(4096) == "4k"
        assert format_number(2_000_000) == "2m"

    def test_tokenizer_id_encodes_vocab_and_algorithm(self, whisper_config):
        assert focus.tokenizer_id(whisper_config) == "focus-v512-unigram"

    def test_tokenizer_id_omits_inherited_algorithm(self, whisper_config):
        whisper_config.focus.tokenizer_algorithm = None
        assert focus.tokenizer_id(whisper_config) == "focus-v512"

    def test_tokenizer_id_tracks_vocab_size(self, whisper_config):
        whisper_config.focus.vocab_size = 4096
        assert "v4k" in focus.tokenizer_id(whisper_config)

    def test_paths_rooted_in_cache_dir(self, whisper_config, tmp_dir):
        paths = focus.resolve_paths(whisper_config, tmp_dir)
        assert paths.tokenizer_dir.startswith(os.path.join(tmp_dir, "focus"))
        assert paths.corpus_jsonl.endswith(".jsonl")
        assert paths.corpus_txt.endswith(".txt")

    def test_corpus_shared_across_vocab_sizes(self, whisper_config, tmp_dir):
        small = focus.resolve_paths(whisper_config, tmp_dir)
        whisper_config.focus.vocab_size = 4096
        large = focus.resolve_paths(whisper_config, tmp_dir)
        assert small.corpus_jsonl == large.corpus_jsonl
        assert small.tokenizer_dir != large.tokenizer_dir

    def test_processed_cache_name_includes_focus_id(self, whisper_config):
        name = processed_cache_dirname(whisper_config)
        assert "focus-v512-unigram" in name

    def test_processed_cache_name_excludes_focus_when_off(self, base_config):
        assert "focus" not in processed_cache_dirname(base_config)


class TestEmbeddingHash:
    def test_stable(self, whisper_config):
        assert focus.embedding_hash(whisper_config) == focus.embedding_hash(whisper_config)

    def test_changes_with_fasttext_knob(self, whisper_config):
        before = focus.embedding_hash(whisper_config)
        whisper_config.focus.fasttext_model_min_count = 5
        assert focus.embedding_hash(whisper_config) != before

    def test_independent_of_vocab_size(self, whisper_config):
        """The tokenizer directory already separates vocabulary sizes, so the
        embedding key must not duplicate that distinction."""
        before = focus.embedding_hash(whisper_config)
        whisper_config.focus.vocab_size = 4096
        assert focus.embedding_hash(whisper_config) == before


class TestCorpusPreparation:
    def test_writes_both_formats(self, whisper_config, tmp_dir, zulu_texts):
        paths = focus.resolve_paths(whisper_config, tmp_dir)
        focus.prepare_corpus(zulu_texts, paths)

        assert os.path.exists(paths.corpus_jsonl)
        assert os.path.exists(paths.corpus_txt)

        with open(paths.corpus_jsonl) as jsonl_file:
            records = [json.loads(line) for line in jsonl_file]
        with open(paths.corpus_txt) as text_file:
            lines = [line.rstrip("\n") for line in text_file]

        assert len(records) == len(lines) == len(zulu_texts)
        assert [record["text"] for record in records] == lines

    def test_skips_blank_transcripts(self, whisper_config, tmp_dir):
        paths = focus.resolve_paths(whisper_config, tmp_dir)
        focus.prepare_corpus(["real text", "   ", ""], paths)
        with open(paths.corpus_txt) as text_file:
            assert text_file.read().splitlines() == ["real text"]

    def test_flattens_embedded_newlines(self, whisper_config, tmp_dir):
        """A newline would split one transcript into two training sentences."""
        paths = focus.resolve_paths(whisper_config, tmp_dir)
        focus.prepare_corpus(["first line\nsecond line"], paths)
        with open(paths.corpus_txt) as text_file:
            assert text_file.read().splitlines() == ["first line second line"]

    def test_empty_texts_raises(self, whisper_config, tmp_dir):
        paths = focus.resolve_paths(whisper_config, tmp_dir)
        with pytest.raises(ValueError, match="empty transcript list"):
            focus.prepare_corpus([], paths)

    def test_subsampling_is_reproducible(self, whisper_config, tmp_dir, zulu_texts):
        first = focus.resolve_paths(whisper_config, tmp_dir)
        focus.prepare_corpus(zulu_texts, first, num_samples=20, seed=1)
        with open(first.corpus_txt) as text_file:
            first_lines = text_file.read().splitlines()

        second_dir = os.path.join(tmp_dir, "again")
        second = focus.resolve_paths(whisper_config, second_dir)
        focus.prepare_corpus(zulu_texts, second, num_samples=20, seed=1)
        with open(second.corpus_txt) as text_file:
            second_lines = text_file.read().splitlines()

        assert first_lines == second_lines
        assert len(first_lines) == 20

    def test_cached_corpus_is_not_rewritten(self, whisper_config, tmp_dir, zulu_texts):
        paths = focus.resolve_paths(whisper_config, tmp_dir)
        focus.prepare_corpus(zulu_texts, paths)
        focus.prepare_corpus(["completely different"], paths)
        with open(paths.corpus_txt) as text_file:
            assert text_file.read().splitlines()[0] != "completely different"


class TestTokenizerConfigTracking:
    def test_roundtrips(self, whisper_config, tmp_dir):
        config = FocusTokenizerConfig.from_args(whisper_config)
        path = os.path.join(tmp_dir, "focus_config.yaml")
        config.save(path)
        assert config.check_cached(path) is True

    def test_detects_vocab_size_change(self, whisper_config, tmp_dir):
        path = os.path.join(tmp_dir, "focus_config.yaml")
        FocusTokenizerConfig.from_args(whisper_config).save(path)
        whisper_config.focus.vocab_size = 1024
        with pytest.raises(ValueError, match="CONFIG MISMATCH"):
            FocusTokenizerConfig.from_args(whisper_config).check_cached(path)

    def test_ignores_fasttext_knobs(self, whisper_config, tmp_dir):
        """fastText settings change embeddings, not the vocabulary, so they
        must not invalidate a cached tokenizer."""
        path = os.path.join(tmp_dir, "focus_config.yaml")
        FocusTokenizerConfig.from_args(whisper_config).save(path)
        whisper_config.focus.fasttext_model_min_count = 7
        assert FocusTokenizerConfig.from_args(whisper_config).check_cached(path) is True


class TestCtcRejection:
    def test_ctc_model_rejects_focus(self, base_config, tmp_dir, sample_texts):
        """A CTC model already derives its vocabulary from the transcripts."""
        from src.processors import setup_tokenizer

        base_config.focus = OmegaConf.create({"enabled": True, "vocab_size": 512})
        with pytest.raises(ValueError, match="not supported for model type"):
            setup_tokenizer(
                base_config,
                train_texts=sample_texts,
                vocab_dir=tmp_dir,
                cache_dir=tmp_dir,
            )


@pytest.mark.network
class TestTokenizerConstruction:
    """Exercises the real build against openai/whisper-small.

    These are the cases where the port could go quietly wrong: the inherited
    special-token block landing out of order, the language/task prefix not being
    rebuilt against the new ids, or the stored generation-config ids still
    addressing the replaced vocabulary.
    """

    @pytest.fixture
    def built(self, whisper_config, tmp_dir, zulu_texts):
        from src.processors import get_model_spec

        spec = get_model_spec("whisper")
        paths = focus.resolve_paths(whisper_config, tmp_dir)
        focus.prepare_corpus(zulu_texts, paths)
        tokenizer = focus.build_tokenizer(whisper_config, spec, paths)
        return tokenizer, spec, paths

    def test_is_a_whisper_tokenizer(self, built):
        from transformers import WhisperTokenizerFast

        tokenizer, _spec, _paths = built
        assert isinstance(tokenizer, WhisperTokenizerFast)

    def test_size_is_learned_plus_inherited(self, built):
        from transformers import WhisperTokenizerFast

        tokenizer, _spec, _paths = built
        base = WhisperTokenizerFast.from_pretrained("openai/whisper-small")
        assert len(tokenizer) == 512 + len(base.get_added_vocab())

    def test_special_block_is_contiguous_and_ordered(self, built):
        from transformers import WhisperTokenizerFast

        tokenizer, _spec, _paths = built
        base = WhisperTokenizerFast.from_pretrained("openai/whisper-small")
        base_order = [
            token
            for token, _id in sorted(base.get_added_vocab().items(), key=lambda i: i[1])
        ]
        new_ids = [tokenizer.convert_tokens_to_ids(token) for token in base_order]
        assert new_ids == list(range(512, 512 + len(base_order)))

    def test_prefix_tokens_use_new_ids(self, built, whisper_config):
        from src.processors import _configure_prefix_tokens

        tokenizer, spec, _paths = built
        _configure_prefix_tokens(whisper_config, spec, tokenizer)
        prefix = tokenizer("umuntu wami").input_ids[: len(tokenizer.prefix_tokens)]
        assert tokenizer.convert_ids_to_tokens(prefix) == [
            "<|startoftranscript|>",
            "<|sw|>",
            "<|transcribe|>",
            "<|notimestamps|>",
        ]

    def test_round_trips_target_language_text(self, built):
        tokenizer, _spec, _paths = built
        text = "umuntuwami abahambaile isikhathiana"
        decoded = tokenizer.decode(
            tokenizer(text).input_ids, skip_special_tokens=True
        ).strip()
        assert decoded == text

    def test_unseen_character_becomes_unk(self, built):
        """The learned vocabulary is not byte-level, unlike Whisper's own.

        A character absent from the training transcripts has no piece and no
        byte fallback, so it encodes to `<unk>` rather than round-tripping. This
        is why `focus.character_coverage` should stay at 1.0 and why the
        transcripts must be normalized the same way at train and eval time.
        """
        tokenizer, _spec, _paths = built
        decoded = tokenizer.decode(
            tokenizer("umuntu \u4e2d").input_ids, skip_special_tokens=True
        )
        assert "<unk>" in decoded

    def test_vocabulary_is_smaller_than_the_base(self, built):
        from transformers import WhisperTokenizerFast

        tokenizer, _spec, _paths = built
        base = WhisperTokenizerFast.from_pretrained("openai/whisper-small")
        assert len(tokenizer) < len(base)

    def test_reload_from_cache_is_identical(self, built, whisper_config):
        tokenizer, spec, paths = built
        reloaded = focus.build_tokenizer(whisper_config, spec, paths)
        assert reloaded.get_vocab() == tokenizer.get_vocab()

    def test_vocab_size_beyond_corpus_errors_helpfully(
        self, whisper_config, tmp_dir, zulu_texts
    ):
        from src.processors import get_model_spec

        whisper_config.focus.vocab_size = 200_000
        paths = focus.resolve_paths(whisper_config, tmp_dir)
        focus.prepare_corpus(zulu_texts, paths)
        with pytest.raises(ValueError, match="Lower focus.vocab_size"):
            focus.build_tokenizer(whisper_config, get_model_spec("whisper"), paths)


@pytest.mark.network
class TestGenerationConfigRemap:
    def test_every_stored_id_moves_into_range(self, whisper_config, tmp_dir, zulu_texts):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperTokenizerFast

        from src.processors import get_model_spec

        spec = get_model_spec("whisper")
        paths = focus.resolve_paths(whisper_config, tmp_dir)
        focus.prepare_corpus(zulu_texts, paths)
        target = focus.build_tokenizer(whisper_config, spec, paths)
        source = WhisperTokenizerFast.from_pretrained("openai/whisper-small")

        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        placeholder = torch.zeros(len(target), model.config.d_model)
        focus._install_embeddings(model, target, placeholder, None)
        focus.remap_special_token_ids(model, source, target)

        generation_config = model.generation_config
        # The end-of-sequence id is the one that a naive `convert_tokens_to_ids`
        # remap silently drops, because Whisper aliases its unknown token to it.
        assert generation_config.eos_token_id == target.convert_tokens_to_ids(
            "<|endoftext|>"
        )
        assert generation_config.decoder_start_token_id == target.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
        assert generation_config.lang_to_id["<|sw|>"] == target.convert_tokens_to_ids(
            "<|sw|>"
        )
        assert generation_config.task_to_id["transcribe"] == (
            target.convert_tokens_to_ids("<|transcribe|>")
        )
        # The pretrained suppression list indexes the replaced subword block.
        assert generation_config.suppress_tokens == []
        assert model.config.vocab_size == len(target)
