"""Character vocabulary generation and CTC tokenizer construction.

Extracts unique characters from normalized training transcriptions, assigns
integer IDs (with [PAD] at index 0 as required by CTC blank), and builds a
Wav2Vec2CTCTokenizer.
"""

import json
import os
import sys
from collections import Counter

from transformers import Wav2Vec2CTCTokenizer


def generate_vocab(
    texts: list[str],
    min_frequency: int = 1,
) -> dict[str, int]:
    """Extract character vocabulary from transcription texts.

    Args:
        texts: List of normalized transcription strings.
        min_frequency: Minimum character occurrence count to include.

    Returns:
        Dictionary mapping characters/special tokens to integer IDs.
        [PAD] is always at index 0, [UNK] follows, then sorted characters.
    """
    char_counts = Counter()
    for text in texts:
        char_counts.update(text)

    # filter by minimum frequency
    chars = sorted(
        char for char, count in char_counts.items()
        if count >= min_frequency
    )

    # build vocab: [PAD]=0 (CTC blank), [UNK]=1, then characters
    vocab: dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
    # space gets a special pipe representation for CTC convention
    for char in chars:
        if char == " ":
            vocab["|"] = len(vocab)
        else:
            vocab[char] = len(vocab)

    return vocab


def save_vocab(vocab: dict[str, int], path: str) -> None:
    """Write vocab dictionary to a JSON file.

    Args:
        vocab: Character-to-ID mapping.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocab ({len(vocab)} tokens) to {path}", file=sys.stderr)


def load_vocab(path: str) -> dict[str, int]:
    """Load vocab dictionary from a JSON file.

    Args:
        path: Path to vocab JSON file.

    Returns:
        Character-to-ID mapping.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_ctc_tokenizer(vocab_path: str) -> Wav2Vec2CTCTokenizer:
    """Build a CTC tokenizer from a vocab JSON file.

    Args:
        vocab_path: Path to the vocab.json file.

    Returns:
        Configured Wav2Vec2CTCTokenizer.
    """
    # bos_token and eos_token are suppressed: Wav2Vec2CTCTokenizer would
    # otherwise add <s> and </s> on top of the vocab file, inflating
    # vocab_size and adding unused neurons to the model's output head.
    # CTC only needs a blank token ([PAD] at index 0); BOS/EOS are seq2seq
    # concepts that don't apply here.
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token=None,
        eos_token=None,
    )
    return tokenizer
