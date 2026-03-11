"""ModelSpec registry and processor construction.

Handles architecture divergence between wav2vec2/HuBERT (input_values +
Wav2Vec2FeatureExtractor) and W2V-BERT (input_features +
SeamlessM4TFeatureExtractor). The ModelSpec dataclass captures these
differences and serves as the extensibility point for future model types.
"""

import sys
from dataclasses import dataclass

from omegaconf import DictConfig
from transformers import (
    HubertForCTC,
    ProcessorMixin,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


@dataclass
class ModelSpec:
    """Captures architecture-specific types and configuration.

    Attributes:
        model_class: The HuggingFace model class for CTC fine-tuning.
        feature_extractor_class: The feature extractor class to use.
        processor_class: The processor class that combines feature extractor
            and tokenizer.
        input_column: Name of the model input key ("input_values" or
            "input_features").
    """

    model_class: type
    feature_extractor_class: type
    processor_class: type
    input_column: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "wav2vec2": ModelSpec(
        model_class=Wav2Vec2ForCTC,
        feature_extractor_class=Wav2Vec2FeatureExtractor,
        processor_class=Wav2Vec2Processor,
        input_column="input_values",
    ),
    "hubert": ModelSpec(
        model_class=HubertForCTC,
        feature_extractor_class=Wav2Vec2FeatureExtractor,
        processor_class=Wav2Vec2Processor,
        input_column="input_values",
    ),
    "w2v_bert": ModelSpec(
        model_class=Wav2Vec2BertForCTC,
        feature_extractor_class=SeamlessM4TFeatureExtractor,
        processor_class=Wav2Vec2Processor,
        input_column="input_features",
    ),
}


def get_model_spec(model_type: str) -> ModelSpec:
    """Look up the ModelSpec for a given model type string.

    Args:
        model_type: One of the keys in MODEL_SPECS.

    Returns:
        The corresponding ModelSpec.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_SPECS.keys())}"
        )
    return MODEL_SPECS[model_type]


def setup_processor(
    args: DictConfig,
    tokenizer: Wav2Vec2CTCTokenizer,
) -> ProcessorMixin:
    """Construct a processor by pairing a tokenizer with the appropriate
    feature extractor for the configured model architecture.

    Args:
        args: Full Hydra config (needs args.model.type and args.audio).
        tokenizer: A CTC tokenizer built from the training vocab.

    Returns:
        A processor combining the feature extractor and tokenizer.
    """
    spec = get_model_spec(args.model.type)

    feature_extractor = spec.feature_extractor_class(
        feature_size=args.audio.feature_size,
        sampling_rate=args.audio.sampling_rate,
        padding_value=0.0,
        do_normalize=args.audio.do_normalize,
        return_attention_mask=args.audio.return_attention_mask,
    )

    processor = spec.processor_class(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    print(
        f"Built processor: {spec.processor_class.__name__} "
        f"(input_column={spec.input_column})",
        file=sys.stderr,
    )
    return processor
