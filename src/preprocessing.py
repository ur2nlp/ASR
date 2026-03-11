"""Text normalization for ASR transcriptions.

Provides pluggable, config-driven text normalization applied during dataset
loading (before caching). Normalization runs before vocab generation so the
character set is clean and consistent.
"""

import re
import unicodedata
from collections.abc import Callable

from omegaconf import DictConfig


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    unicode_normalize: str | None = "NFC",
    custom_replacements: dict[str, str] | None = None,
) -> str:
    """Apply normalization steps to a transcription string.

    Args:
        text: Raw transcription text.
        lowercase: Whether to lowercase the text.
        remove_punctuation: Whether to strip punctuation (keeps letters,
            digits, spaces, and apostrophes).
        unicode_normalize: Unicode normalization form (NFC, NFKC, NFD, NFKD)
            or None to skip.
        custom_replacements: Mapping of strings to replace before other steps.

    Returns:
        Normalized text string.
    """
    if custom_replacements:
        for old, new in custom_replacements.items():
            text = text.replace(old, new)

    if unicode_normalize:
        text = unicodedata.normalize(unicode_normalize, text)

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        # keep letters (any script), digits, spaces, and apostrophes
        text = re.sub(r"[^\w\s']", "", text, flags=re.UNICODE)
        # collapse whitespace that may result from removed chars
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_text_normalizer(config: DictConfig) -> Callable[[str], str]:
    """Build a normalization closure from a preprocessing config section.

    Args:
        config: The `preprocessing` section of the Hydra config.

    Returns:
        A callable that normalizes a single transcription string.
    """
    lowercase = config.get("lowercase", True)
    remove_punctuation = config.get("remove_punctuation", True)
    unicode_norm = config.get("unicode_normalize", "NFC")
    custom_replacements = dict(config.get("custom_replacements", {})) or None

    def _normalize(text: str) -> str:
        return normalize_text(
            text,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
            unicode_normalize=unicode_norm,
            custom_replacements=custom_replacements,
        )

    return _normalize
