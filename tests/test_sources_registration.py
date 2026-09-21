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
