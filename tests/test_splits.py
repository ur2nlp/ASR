"""Tests for src/splits.py (metadata-driven split assignment)."""

import csv
import os

import pytest
from omegaconf import OmegaConf

from src.splits import (
    SplitSpec,
    assign_splits,
    keys_from_values,
    load_metadata,
    matches,
    save_split_record,
)

METADATA_ROWS = [
    {"recording": "r1", "transcriber": "A", "speakers": "S1;S2", "language": "zulu"},
    {"recording": "r2", "transcriber": "A", "speakers": "S3;S4", "language": "zulu"},
    {"recording": "r3", "transcriber": "B", "speakers": "S1;S3", "language": "zulu"},
    {"recording": "r4", "transcriber": "C", "speakers": "S2;S4", "language": "zulu"},
    {"recording": "r5", "transcriber": "C", "speakers": "", "language": "english"},
]

# two segments per recording, named `{recording}_{index}.wav` as segmenters commonly do
AUDIO_PATHS = [f"{recording}_{index:03d}.wav" for recording in ("r1", "r2", "r3", "r4", "r5")
               for index in range(2)]

SEGMENT_PATTERN = r"^(.+)_\d+\.wav$"


def _write_metadata(directory, rows=METADATA_ROWS) -> str:
    path = os.path.join(directory, "metadata.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _spec(tmp_dir, rows=METADATA_ROWS, **overrides) -> SplitSpec:
    config = {
        "enabled": True,
        "name": "test",
        "metadata": _write_metadata(tmp_dir, rows),
        "key_pattern": SEGMENT_PATTERN,
        "multi_valued": ["speakers"],
        "dev_size": 0,
    }
    config.update(overrides)
    return SplitSpec.from_config(OmegaConf.create(config))


def _recordings(indices) -> set[str]:
    return {AUDIO_PATHS[index].split("_")[0] for index in indices}


class TestSelectors:
    def test_and_across_columns_or_within(self, tmp_dir):
        spec = _spec(tmp_dir, test={"transcriber": ["A", "C"], "language": ["zulu"]})
        result = assign_splits(AUDIO_PATHS, spec)
        assert _recordings(result["test"]) == {"r1", "r2", "r4"}

    def test_list_of_clauses_is_or(self, tmp_dir):
        spec = _spec(tmp_dir, test=[{"transcriber": ["B"]}, {"language": ["english"]}])
        result = assign_splits(AUDIO_PATHS, spec)
        assert _recordings(result["test"]) == {"r3", "r5"}

    def test_multi_valued_matches_any_element(self, tmp_dir):
        spec = _spec(tmp_dir, test={"speakers": ["S2"]})
        result = assign_splits(AUDIO_PATHS, spec)
        assert _recordings(result["test"]) == {"r1", "r4"}

    def test_unknown_column_is_an_error(self, tmp_dir):
        spec = _spec(tmp_dir, test={"accent": ["x"]})
        with pytest.raises(KeyError, match="accent"):
            assign_splits(AUDIO_PATHS, spec)

    def test_scalar_value_is_accepted(self, tmp_dir):
        spec = _spec(tmp_dir, test={"transcriber": "B"})
        assert _recordings(assign_splits(AUDIO_PATHS, spec)["test"]) == {"r3"}

    def test_matches_directly(self):
        row = {"transcriber": ("A",), "speakers": ("S1", "S2")}
        assert matches(row, ((("speakers", ("S2",)),),))
        assert not matches(row, ((("transcriber", ("B",)),),))


class TestAssignment:
    def test_precedence_exclude_test_dev_train(self, tmp_dir):
        spec = _spec(
            tmp_dir,
            exclude={"language": ["english"]},
            test={"transcriber": ["A"]},
            dev={"transcriber": ["B"]},
        )
        result = assign_splits(AUDIO_PATHS, spec)
        assert _recordings(result["test"]) == {"r1", "r2"}
        assert _recordings(result["dev"]) == {"r3"}
        assert _recordings(result["train"]) == {"r4"}

    def test_splits_are_disjoint_and_cover_the_rest(self, tmp_dir):
        spec = _spec(tmp_dir, test={"transcriber": ["A"]}, dev_size=0.25)
        result = assign_splits(AUDIO_PATHS, spec)
        assigned = [index for indices in result.values() for index in indices]
        assert sorted(assigned) == list(range(len(AUDIO_PATHS)))

    def test_sampled_dev_keeps_recordings_whole(self, tmp_dir):
        """Segments of one recording must not straddle train and dev."""
        spec = _spec(tmp_dir, dev_size=0.3)
        result = assign_splits(AUDIO_PATHS, spec)
        assert _recordings(result["dev"]).isdisjoint(_recordings(result["train"]))
        assert len(result["dev"]) >= 3

    def test_sampled_dev_groups_by_column(self, tmp_dir):
        spec = _spec(tmp_dir, dev_size=1, dev_group_by="transcriber")
        result = assign_splits(AUDIO_PATHS, spec)
        transcriber = {row["recording"]: row["transcriber"] for row in METADATA_ROWS}
        dev_transcribers = {transcriber[r] for r in _recordings(result["dev"])}
        train_transcribers = {transcriber[r] for r in _recordings(result["train"])}
        assert len(dev_transcribers) == 1
        assert dev_transcribers.isdisjoint(train_transcribers)

    def test_sampled_dev_is_seeded(self, tmp_dir):
        spec = _spec(tmp_dir, dev_size=0.3)
        assert assign_splits(AUDIO_PATHS, spec, seed=1) == assign_splits(AUDIO_PATHS, spec, seed=1)

    def test_train_selector_restricts_training(self, tmp_dir):
        spec = _spec(tmp_dir, test={"transcriber": ["A"]}, train={"transcriber": ["B"]})
        result = assign_splits(AUDIO_PATHS, spec)
        assert _recordings(result["train"]) == {"r3"}

    def test_empty_training_split_is_an_error(self, tmp_dir):
        spec = _spec(tmp_dir, test={"language": ["zulu", "english"]})
        with pytest.raises(ValueError, match="no training examples"):
            assign_splits(AUDIO_PATHS, spec)

    def test_unmatched_examples_error_by_default(self, tmp_dir):
        spec = _spec(tmp_dir)
        with pytest.raises(ValueError, match="no row"):
            assign_splits(AUDIO_PATHS + ["r9_000.wav"], spec)

    def test_unmatched_examples_can_be_dropped(self, tmp_dir):
        spec = _spec(tmp_dir, unmatched="drop")
        result = assign_splits(AUDIO_PATHS + ["r9_000.wav"], spec)
        assert len(AUDIO_PATHS) not in result["train"]


class TestKeys:
    def test_pattern_group_is_the_key(self):
        assert keys_from_values(["a/b/rec_001.wav"], SEGMENT_PATTERN) == ["rec"]

    def test_unmatched_pattern_gives_none(self):
        assert keys_from_values(["nothing.flac", None], SEGMENT_PATTERN) == [None, None]

    def test_duplicate_basenames_are_an_error(self):
        """Caching keeps only basenames, so these two are indistinguishable."""
        with pytest.raises(ValueError, match="key_from"):
            keys_from_values(["spk1/utt_001.wav", "spk2/utt_001.wav"], SEGMENT_PATTERN)

    def test_column_values_may_repeat(self):
        keys = keys_from_values(["r1", "r1"], r"^(.+)$", from_paths=False)
        assert keys == ["r1", "r1"]

    def test_key_from_column(self, tmp_dir):
        spec = _spec(tmp_dir, key_from="recording_id", key_pattern=r"^(.+)$",
                     test={"transcriber": ["B"]})
        ids = ["r1", "r1", "r3", "r4"]
        assert assign_splits(ids, spec)["test"] == [2]


class TestSplitId:
    def test_changes_with_selectors(self, tmp_dir):
        first = _spec(tmp_dir, test={"transcriber": ["A"]}).split_id(seed=1)
        second = _spec(tmp_dir, test={"transcriber": ["B"]}).split_id(seed=1)
        assert first != second
        assert first.startswith("test-")

    def test_changes_with_seed(self, tmp_dir):
        spec = _spec(tmp_dir, dev_size=0.2)
        assert spec.split_id(seed=1) != spec.split_id(seed=2)

    def test_changes_with_metadata_contents(self, tmp_dir):
        before = _spec(tmp_dir).split_id(seed=1)
        edited = [dict(row) for row in METADATA_ROWS]
        edited[0]["transcriber"] = "Z"
        assert _spec(tmp_dir, rows=edited).split_id(seed=1) != before

    def test_ignores_metadata_location(self, tmp_dir):
        first = _spec(tmp_dir)
        moved_dir = os.path.join(tmp_dir, "moved")
        os.makedirs(moved_dir)
        moved = _spec(moved_dir)
        assert first.split_id(seed=1) == moved.split_id(seed=1)


class TestLoading:
    def test_missing_metadata_file(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            SplitSpec.from_config(OmegaConf.create({
                "enabled": True, "metadata": os.path.join(tmp_dir, "absent.csv"),
            }))

    def test_duplicate_keys_rejected(self, tmp_dir):
        path = _write_metadata(tmp_dir, METADATA_ROWS + [METADATA_ROWS[0]])
        spec = SplitSpec.from_config(OmegaConf.create({"enabled": True, "metadata": path}))
        with pytest.raises(ValueError, match="Duplicate"):
            load_metadata(spec)

    def test_multi_valued_cells_split(self, tmp_dir):
        metadata = load_metadata(_spec(tmp_dir))
        assert metadata["r1"]["speakers"] == ("S1", "S2")
        assert metadata["r5"]["speakers"] == ()
        assert metadata["r1"]["transcriber"] == ("A",)


class TestSplitRecord:
    def test_writes_record_and_assignment(self, tmp_dir):
        spec = _spec(tmp_dir, test={"transcriber": ["B"]})
        assignment = assign_splits(AUDIO_PATHS, spec)
        root = os.path.join(tmp_dir, "splits", spec.split_id(seed=1))
        save_split_record(root, spec, 1, assignment, AUDIO_PATHS)

        with open(os.path.join(root, "assignment.csv")) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(AUDIO_PATHS)
        test_rows = {row["example"] for row in rows if row["split"] == "test"}
        assert test_rows == {"r3_000.wav", "r3_001.wav"}
        assert os.path.exists(os.path.join(root, "split.yaml"))
