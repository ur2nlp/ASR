"""LM-boosted CTC decoding via pyctcdecode + KenLM.

Provides utilities to build a beam search decoder with an optional KenLM
language model, wrap it in a Wav2Vec2ProcessorWithLM, and decode logits.
"""

import sys

import numpy as np
from omegaconf import DictConfig
from transformers import ProcessorMixin


def build_lm_decoder(
    processor: ProcessorMixin,
    arpa_path: str | None = None,
    alpha: float = 0.5,
    beta: float = 1.5,
):
    """Build a pyctcdecode BeamSearchDecoderCTC.

    Args:
        processor: The processor whose tokenizer defines the label set.
        arpa_path: Path to a KenLM ARPA file. If None, uses no LM (pure
            prefix beam search).
        alpha: LM weight.
        beta: Word insertion bonus.

    Returns:
        A pyctcdecode BeamSearchDecoderCTC instance.
    """
    from pyctcdecode import build_ctcdecoder

    # get labels in model output order (index 0 = CTC blank = [PAD])
    vocab = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    labels = [token for token, _idx in sorted_vocab]

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=arpa_path,
        alpha=alpha,
        beta=beta,
    )

    return decoder


def build_processor_with_lm(
    processor: ProcessorMixin,
    arpa_path: str | None = None,
    alpha: float = 0.5,
    beta: float = 1.5,
    beam_width: int = 100,
):
    """Wrap a processor with an LM decoder for integrated decoding.

    Args:
        processor: Base processor (feature extractor + tokenizer).
        arpa_path: Path to KenLM ARPA file.
        alpha: LM weight.
        beta: Word insertion bonus.
        beam_width: Beam width.

    Returns:
        A Wav2Vec2ProcessorWithLM instance.
    """
    from transformers import Wav2Vec2ProcessorWithLM

    decoder = build_lm_decoder(
        processor,
        arpa_path=arpa_path,
        alpha=alpha,
        beta=beta,
    )

    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder,
    )

    print(
        f"Built processor with LM decoder "
        f"(arpa={arpa_path}, alpha={alpha}, beta={beta}, beam={beam_width})",
        file=sys.stderr,
    )
    return processor_with_lm


def decode_logits(
    logits: np.ndarray,
    processor_with_lm,
    beam_width: int = 100,
) -> list[str]:
    """Decode model logits using beam search with optional LM.

    Args:
        logits: Model output logits, shape (batch_size, seq_len, vocab_size).
        processor_with_lm: A Wav2Vec2ProcessorWithLM instance.
        beam_width: Beam width for decoding.

    Returns:
        List of decoded transcription strings.
    """
    results = processor_with_lm.batch_decode(
        logits,
        beam_width=beam_width,
    )
    return results.text


def setup_lm_decoding(
    processor: ProcessorMixin,
    lm_config: DictConfig,
):
    """Conditionally build LM decoding from config.

    Args:
        processor: Base processor.
        lm_config: The `lm` section of the Hydra config.

    Returns:
        A Wav2Vec2ProcessorWithLM if lm.enabled, else None.
    """
    if not lm_config.get("enabled", False):
        return None

    arpa_path = lm_config.get("arpa_path")
    if arpa_path is None:
        print("LM decoding enabled but no arpa_path provided, skipping", file=sys.stderr)
        return None

    return build_processor_with_lm(
        processor,
        arpa_path=arpa_path,
        alpha=lm_config.get("alpha", 0.5),
        beta=lm_config.get("beta", 1.5),
        beam_width=lm_config.get("beam_width", 100),
    )
