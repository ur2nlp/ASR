"""Tests for tools/registry.py — extract_params, upsert_entry, diff_runs."""

import pytest

from tools.registry import (
    categorize_debt,
    diff_runs,
    extract_params,
    format_param_value,
    load_registry,
    save_registry,
    upsert_entry,
)


@pytest.fixture
def sample_config():
    """Minimal training_config.yaml structure matching v81."""
    return {
        "experiment_id": "v81",
        "dataset": {
            "alpha": 0.5,
            "total_samples": 9_000_000,
        },
        "training": {
            "learning_rate": 2.0e-05,
            "train_batch_size": 18,
            "gradient_accumulation_steps": 10,
            "dropout": 0.25,
            "weight_decay": 0.1,
            "max_steps": 200_000,
        },
        "focus": {
            "enabled": True,
            "vocab_size": 32768,
            "seed_lambda": 0.99,
            "seed_vocab_multiplier": 5.0,
        },
    }


@pytest.fixture
def sample_config_v82():
    """A second config with different lr and dropout."""
    return {
        "experiment_id": "v82",
        "dataset": {
            "alpha": 0.5,
            "total_samples": 9_000_000,
        },
        "training": {
            "learning_rate": 2.2e-05,
            "train_batch_size": 18,
            "gradient_accumulation_steps": 10,
            "dropout": 0.3,
            "weight_decay": 0.1,
            "max_steps": 200_000,
        },
        "focus": {
            "enabled": True,
            "vocab_size": 32768,
            "seed_lambda": 0.99,
            "seed_vocab_multiplier": 5.0,
        },
    }


class TestCategorizeDebt:
    @pytest.fixture
    def registry(self):
        return {
            "v01": {"params": {}, "note": "annotated", "observation": "seen"},
            "v02": {"params": {}, "note": "", "observation": "seen"},
            "v03": {"params": {}, "note": "   ", "observation": None},
            "v10": {"params": {}, "note": "ok"},  # observation key absent
        }

    def test_unregistered_runs(self, registry):
        # v04/v05 have files but no registry row
        debt = categorize_debt(registry, {"v01", "v04"}, {"v01", "v05"})
        assert debt["unregistered"] == ["v04", "v05"]

    def test_orphan_rows(self, registry):
        # registry rows with no backing files
        debt = categorize_debt(registry, {"v01"}, {"v02"})
        assert debt["orphan"] == ["v03", "v10"]

    def test_empty_note_includes_blank_and_whitespace(self, registry):
        debt = categorize_debt(registry, set(), set())
        # v02 (empty) and v03 (whitespace) are blank; v01/v10 are not
        assert debt["empty_note"] == ["v02", "v03"]

    def test_empty_observation_includes_absent_and_none(self, registry):
        debt = categorize_debt(registry, set(), set())
        # v03 (None) and v10 (key absent) count as empty
        assert debt["empty_observation"] == ["v03", "v10"]

    def test_natural_sort_order(self, registry):
        # v2 / v100 are not registry keys, so both are unregistered
        debt = categorize_debt(registry, {"v2", "v100"}, set())
        # numeric-aware sort: v2 before v100 (lexical would give v100 first)
        assert debt["unregistered"] == ["v2", "v100"]


class TestExtractParams:
    def test_basic_extraction(self, sample_config):
        exp_id, params = extract_params(sample_config)
        assert exp_id == "v81"
        assert params["lr"] == 2.0e-05
        assert params["effective_batch"] == 180
        assert params["dropout"] == 0.25
        assert params["weight_decay"] == 0.1
        assert params["alpha"] == 0.5
        assert params["total_samples"] == 9_000_000
        assert params["max_steps"] == 200_000
        assert params["focus_enabled"] is True
        assert params["vocab_size"] == 32768
        assert params["seed_lambda"] == 0.99
        assert params["seed_vocab_multiplier"] == 5.0

    def test_missing_experiment_id(self):
        config = {"training": {"learning_rate": 1e-4}}
        with pytest.raises(ValueError, match="experiment_id"):
            extract_params(config)

    def test_missing_sections_graceful(self):
        """Configs with missing sections should still extract what's available."""
        config = {
            "experiment_id": "v_minimal",
            "training": {"learning_rate": 1e-4, "max_steps": 1000},
        }
        exp_id, params = extract_params(config)
        assert exp_id == "v_minimal"
        assert params["lr"] == 1e-4
        assert params["max_steps"] == 1000
        # missing fields should not be in params
        assert "alpha" not in params
        assert "vocab_size" not in params

    def test_effective_batch_defaults(self):
        """If batch or grad_accum missing, defaults to 1."""
        config = {
            "experiment_id": "v_nograd",
            "training": {"train_batch_size": 32},
        }
        _, params = extract_params(config)
        assert params["effective_batch"] == 32

    def test_focus_disabled(self):
        config = {
            "experiment_id": "v_nofocus",
            "focus": {"enabled": False},
        }
        _, params = extract_params(config)
        assert params["focus_enabled"] is False

    def test_extracts_all_scalars(self):
        """All scalar params from training/dataset/focus should be extracted."""
        config = {
            "experiment_id": "v_full",
            "training": {
                "learning_rate": 2e-5,
                "max_grad_norm": 4.0,
                "bf16": True,
                "train_batch_size": 18,
                "gradient_accumulation_steps": 10,
            },
            "focus": {
                "enabled": True,
                "character_coverage": 0.999,
            },
        }
        _, params = extract_params(config)
        assert params["max_grad_norm"] == 4.0
        assert params["bf16"] is True
        assert params["character_coverage"] == 0.999
        # train_batch_size should be extracted as-is, plus effective_batch computed
        assert params["train_batch_size"] == 18
        assert params["effective_batch"] == 180

    def test_skips_hydra_interpolations(self):
        """Hydra interpolation strings like ${divide:...} should be skipped."""
        config = {
            "experiment_id": "v_hydra",
            "training": {
                "logging_steps": "${divide:${training.max_steps},400}",
                "max_steps": 100000,
            },
        }
        _, params = extract_params(config)
        assert "logging_steps" not in params
        assert params["max_steps"] == 100000


class TestUpsertEntry:
    def test_new_entry(self):
        registry = {}
        upsert_entry(registry, "v81", {"lr": 2e-5, "dropout": 0.25})
        assert "v81" in registry
        assert registry["v81"]["params"]["lr"] == 2e-5
        assert registry["v81"]["era"] == ""
        assert registry["v81"]["group"] == ""
        assert registry["v81"]["note"] == ""
        assert registry["v81"]["observation"] == ""

    def test_preserves_human_fields(self):
        """Re-extracting params must not overwrite era/group/note/observation."""
        registry = {
            "v81": {
                "params": {"lr": 1e-5},
                "era": "5M-eng-seeded",
                "group": "dropout-sweep",
                "note": "Testing dropout",
                "observation": "Looks good",
            }
        }
        upsert_entry(registry, "v81", {"lr": 2e-5, "dropout": 0.25})
        assert registry["v81"]["params"]["lr"] == 2e-5
        assert registry["v81"]["params"]["dropout"] == 0.25
        assert registry["v81"]["era"] == "5M-eng-seeded"
        assert registry["v81"]["group"] == "dropout-sweep"
        assert registry["v81"]["note"] == "Testing dropout"
        assert registry["v81"]["observation"] == "Looks good"


class TestDiffRuns:
    def test_varying_params(self, sample_config, sample_config_v82):
        registry = {}
        eid1, p1 = extract_params(sample_config)
        eid2, p2 = extract_params(sample_config_v82)
        upsert_entry(registry, eid1, p1)
        upsert_entry(registry, eid2, p2)

        varying, constant = diff_runs(registry, ["v81", "v82"])
        assert "lr" in varying
        assert "dropout" in varying
        # these should be constant
        assert "weight_decay" in constant
        assert "max_steps" in constant
        assert "alpha" in constant

    def test_identical_runs(self, sample_config):
        registry = {}
        eid, params = extract_params(sample_config)
        upsert_entry(registry, eid, params)
        # duplicate under different name
        registry["v81_copy"] = {"params": dict(params)}

        varying, constant = diff_runs(registry, ["v81", "v81_copy"])
        assert len(varying) == 0
        assert len(constant) > 0

    def test_missing_run(self):
        """Missing runs should be handled gracefully (empty params)."""
        registry = {"v81": {"params": {"lr": 2e-5}}}
        varying, constant = diff_runs(registry, ["v81", "v_missing"])
        # lr should vary (present vs None)
        assert "lr" in varying

    def test_detects_non_key_param_variation(self):
        """diff should detect variation in ANY param, not just a curated list."""
        registry = {
            "v5L": {"params": {"lr": 1e-5, "max_grad_norm": 4.0}},
            "v4L": {"params": {"lr": 1e-5, "max_grad_norm": 1.0}},
        }
        varying, constant = diff_runs(registry, ["v5L", "v4L"])
        assert "max_grad_norm" in varying
        assert "lr" in constant


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        registry_path = tmp_path / "registry.yaml"
        registry = {
            "v81": {
                "params": {"lr": 2e-5, "max_steps": 200000},
                "era": "test",
                "group": "",
                "note": "round-trip test",
                "observation": "",
            }
        }
        save_registry(registry, registry_path)
        loaded = load_registry(registry_path)
        assert loaded["v81"]["params"]["lr"] == 2e-5
        assert loaded["v81"]["era"] == "test"
        assert loaded["v81"]["note"] == "round-trip test"

    def test_load_missing(self, tmp_path):
        result = load_registry(tmp_path / "nonexistent.yaml")
        assert result == {}


class TestFormatParamValue:
    def test_large_int(self):
        assert format_param_value(9_000_000) == "9M"
        assert format_param_value(200_000) == "200k"
        # non-round values stay as-is
        assert format_param_value(32768) == "32768"

    def test_small_float(self):
        assert format_param_value(2e-5) == "2.0e-05"

    def test_regular_float(self):
        assert format_param_value(0.25) == "0.25"

    def test_bool(self):
        assert format_param_value(True) == "true"
        assert format_param_value(False) == "false"
