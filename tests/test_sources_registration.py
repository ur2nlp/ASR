"""Tests that every source type is reachable the way a config reaches it."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

IMPORT_THE_PACKAGE = """
import json
from src.sources import SOURCE_TYPES
print(json.dumps(SOURCE_TYPES.known_types()))
"""

IMPORT_EVERY_MODULE = """
import importlib, json, pkgutil
import src.sources
from src.sources import SOURCE_TYPES
for module in pkgutil.iter_modules(src.sources.__path__):
    importlib.import_module(f'src.sources.{module.name}')
print(json.dumps(SOURCE_TYPES.known_types()))
"""


def _registered_types(script: str) -> list[str]:
    """Run `script` in a clean interpreter and return the types it registered.

    A subprocess is not incidental. `SOURCE_TYPES` is process-global mutable
    state and pytest runs everything in one process, so a sibling test
    importing `src.sources.paired` directly would register that type as a side
    effect -- masking exactly the omission these tests exist to catch. Only a
    fresh interpreter answers the question honestly.
    """
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=REPO_ROOT,
        env={**os.environ, 'PYTHONPATH': str(REPO_ROOT), 'KMP_DUPLICATE_LIB_OK': 'TRUE'},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


class TestEveryTypeModuleIsWiredIn:
    """Registration is an import side effect, so a module absent from
    `src/sources/__init__.py` is invisible to config dispatch -- while its own
    test file still passes, because importing the module registers it.
    """

    def test_importing_the_package_registers_every_type_module(self):
        from_package = _registered_types(IMPORT_THE_PACKAGE)
        from_every_module = _registered_types(IMPORT_EVERY_MODULE)
        assert sorted(from_package) == sorted(from_every_module)

    def test_the_registry_is_not_empty(self):
        assert _registered_types(IMPORT_THE_PACKAGE)

    def test_the_types_configs_reference_are_present(self):
        registered = set(_registered_types(IMPORT_THE_PACKAGE))
        assert {"paired", "audiofolder", "huggingface", "concat"} <= registered


class TestConcatPreservesSplits:
    """The behaviour ASR needs from `lapt_core`'s composite, guarded here.

    Speech corpora routinely arrive pre-split, so a concat that keeps only
    `train` would silently discard held-out data. `AudioConcatDataset` used to
    override `build` to get this; as of lapt-core 0.1.1 it is the shared
    behaviour and the override is gone. This test is what would notice if a
    future re-pin moved back to a version that drops splits.

    Stub children rather than real sources: the point is the composite's
    split handling, and building real `paired` corpora would drag audio
    decoding into a test that is not about audio.
    """

    def _resolve(self, tmp_dir, split_sets):
        from datasets import Dataset, DatasetDict
        from lapt_core.dataset_artifacts import DatasetArtifact

        from src.sources.concat import AudioConcatDataset

        class StubSource(DatasetArtifact):
            def __init__(self, cache_dir, splits):
                super().__init__(cache_dir)
                self.splits = splits

            def config(self):
                return {"type": "stub", "splits": sorted(self.splits)}

            def build(self, deps):
                return DatasetDict({
                    name: Dataset.from_dict({"transcription": values})
                    for name, values in self.splits.items()
                })

        sources = [
            {"id": f"s{index}", "splits": splits}
            for index, splits in enumerate(split_sets)
        ]
        composite = AudioConcatDataset(
            os.path.join(tmp_dir, "concat"),
            sources,
            child_factory=lambda cache_dir, cfg, seed=1: StubSource(
                cache_dir, cfg["splits"]
            ),
        )
        return composite.resolve()

    def test_every_split_survives_concatenation(self, tmp_dir):
        result = self._resolve(tmp_dir, [
            {"train": ["a1"], "test": ["a2"]},
            {"train": ["b1"], "validation": ["b2"]},
        ])
        assert set(result) == {"train", "test", "validation"}
        assert result["train"]["transcription"] == ["a1", "b1"]
        assert result["test"]["transcription"] == ["a2"]
        assert result["validation"]["transcription"] == ["b2"]

    def test_a_source_without_train_is_not_an_error(self, tmp_dir):
        result = self._resolve(tmp_dir, [{"test": ["only"]}])
        assert set(result) == {"test"}
