"""Tests for src/vocab.py."""

import json
import os

from src.vocab import create_ctc_tokenizer, generate_vocab, load_vocab, save_vocab


class TestGenerateVocab:
    def test_pad_at_index_zero(self, sample_texts):
        vocab = generate_vocab(sample_texts)
        assert vocab["[PAD]"] == 0

    def test_unk_at_index_one(self, sample_texts):
        vocab = generate_vocab(sample_texts)
        assert vocab["[UNK]"] == 1

    def test_space_mapped_to_pipe(self, sample_texts):
        vocab = generate_vocab(sample_texts)
        assert "|" in vocab
        assert " " not in vocab

    def test_all_chars_present(self, sample_texts):
        vocab = generate_vocab(sample_texts)
        all_chars = set("".join(sample_texts)) - {" "}
        for char in all_chars:
            assert char in vocab, f"Missing character: {char}"

    def test_min_frequency_filter(self):
        texts = ["aaa bbb", "aaa ccc"]
        vocab = generate_vocab(texts, min_frequency=2)
        assert "a" in vocab
        # 'b' appears only in one text (3 times), 'c' also (3 times)
        # both appear 3 times which is >= 2
        assert "b" in vocab

    def test_min_frequency_excludes_rare(self):
        texts = ["aaaa", "aaab"]
        vocab = generate_vocab(texts, min_frequency=5)
        # 'a' appears 7 times, 'b' appears 1 time
        assert "a" in vocab
        assert "b" not in vocab

    def test_unique_ids(self, sample_texts):
        vocab = generate_vocab(sample_texts)
        ids = list(vocab.values())
        assert len(ids) == len(set(ids))

    def test_contiguous_ids(self, sample_texts):
        vocab = generate_vocab(sample_texts)
        ids = sorted(vocab.values())
        assert ids == list(range(len(ids)))


class TestSaveLoadVocab:
    def test_round_trip(self, tmp_dir, sample_texts):
        vocab = generate_vocab(sample_texts)
        path = os.path.join(tmp_dir, "vocab.json")
        save_vocab(vocab, path)
        loaded = load_vocab(path)
        assert loaded == vocab

    def test_creates_parent_dirs(self, tmp_dir):
        path = os.path.join(tmp_dir, "nested", "dir", "vocab.json")
        save_vocab({"[PAD]": 0}, path)
        assert os.path.exists(path)


class TestCreateCtcTokenizer:
    def test_creates_tokenizer(self, vocab_path):
        tokenizer = create_ctc_tokenizer(vocab_path)
        assert tokenizer.pad_token == "[PAD]"
        assert tokenizer.unk_token == "[UNK]"
        assert tokenizer.word_delimiter_token == "|"

    def test_vocab_size(self, vocab_path, sample_vocab):
        tokenizer = create_ctc_tokenizer(vocab_path)
        assert len(tokenizer) == len(sample_vocab)

    def test_encode_decode(self, vocab_path):
        tokenizer = create_ctc_tokenizer(vocab_path)
        encoded = tokenizer("hello").input_ids
        assert isinstance(encoded, list)
        assert all(isinstance(i, int) for i in encoded)
