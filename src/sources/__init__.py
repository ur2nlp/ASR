"""Audio dataset sources, one module per `type` a configuration can name.

Importing this package registers every type. The imports below are what makes
`SOURCE_TYPES` complete, so a module missing from this list is invisible to
config dispatch even though its own tests pass -- see
`tests/test_sources_registration.py`.
"""

from src.sources import audiofolder, concat, huggingface, paired  # noqa: F401
from src.sources.base import SOURCE_TYPES, AudioSourceArtifact
from src.sources.factory import make_source, normalize_sources, source_config_record

__all__ = [
    "AudioSourceArtifact",
    "SOURCE_TYPES",
    "make_source",
    "normalize_sources",
    "source_config_record",
]
