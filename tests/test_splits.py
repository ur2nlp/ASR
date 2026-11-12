"""Tests for src/splits.py (metadata-driven split assignment)."""

import csv
import os

import pytest
import yaml
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


class TestMatchedHoldout:
    """Partial test holdout, remainder routing and training budgets."""

    # 3 transcribers x 4 sessions x 3 recordings, 2 segments each; T = "A"
    ROWS = [
        {
            "recording": f"{transcriber}{session}r{recording}",
            "transcriber": transcriber,
            "session": f"{transcriber}{session}",
        }
        for transcriber in ("A", "B", "C")
        for session in range(4)
        for recording in range(3)
    ]
    PATHS = [f"{row['recording']}_{index:03d}.wav" for row in ROWS for index in range(2)]
    # vary durations so hour and example budgets differ
    DURATIONS = [float(3 + (position % 5)) for position in range(len(PATHS))]

    def _spec(self, tmp_dir, **overrides) -> SplitSpec:
        config = {
            "enabled": True,
            "name": "matched",
            "metadata": _write_metadata(tmp_dir, self.ROWS),
            "key_pattern": SEGMENT_PATTERN,
            "test": {"transcriber": ["A"]},
            "test_size": 0.5,
            "dev_size": 0.2,
            "dev_group_by": "session",
            "train_budget": "matched",
        }
        config.update(overrides)
        return SplitSpec.from_config(OmegaConf.create(config))

    def _assign(self, spec):
        return assign_splits(self.PATHS, spec, seed=1, durations=self.DURATIONS)

    def _hours(self, indices) -> float:
        return sum(self.DURATIONS[index] for index in indices) / 3600.0

    def _sessions(self, indices) -> set[str]:
        return {self.PATHS[index][:2] for index in indices}

    def _transcribers(self, indices) -> set[str]:
        return {self.PATHS[index][0] for index in indices}

    def test_partial_test_keeps_groups_whole(self, tmp_dir):
        result = self._assign(self._spec(tmp_dir))
        test_sessions = self._sessions(result["test"])
        assert self._transcribers(result["test"]) == {"A"}
        assert 0 < len(test_sessions) < 4

    def test_pair_shares_test_and_dev(self, tmp_dir):
        without = self._assign(self._spec(tmp_dir, test_remainder="exclude"))
        with_t = self._assign(self._spec(tmp_dir, test_remainder="train"))
        assert without["test"] == with_t["test"]
        assert without["dev"] == with_t["dev"]

    def test_dev_never_holds_the_held_out_transcriber(self, tmp_dir):
        result = self._assign(self._spec(tmp_dir, test_remainder="train"))
        assert "A" not in self._transcribers(result["dev"])

    def test_without_variant_trains_on_no_t(self, tmp_dir):
        result = self._assign(self._spec(tmp_dir, test_remainder="exclude"))
        assert "A" not in self._transcribers(result["train"])

    def test_with_variant_trains_on_the_whole_remainder(self, tmp_dir):
        result = self._assign(self._spec(tmp_dir, test_remainder="train"))
        a_train = self._sessions([i for i in result["train"] if self.PATHS[i][0] == "A"])
        a_test = self._sessions(result["test"])
        assert a_train == {f"A{session}" for session in range(4)} - a_test

    def test_matched_budget_equalizes_hours(self, tmp_dir):
        without = self._assign(self._spec(tmp_dir, test_remainder="exclude"))
        with_t = self._assign(self._spec(tmp_dir, test_remainder="train"))
        longest = max(self.DURATIONS) / 3600.0
        assert abs(self._hours(with_t["train"]) - self._hours(without["train"])) <= longest

    def test_budget_ignores_examples_the_length_filter_drops(self, tmp_dir):
        """Matching must hold *after* filtering, when T's segments are long."""
        durations = [
            40.0 if path.startswith("A") and index % 2 == 0 else duration
            for index, (path, duration) in enumerate(zip(self.PATHS, self.DURATIONS))
        ]

        def surviving_hours(result):
            return sum(durations[i] for i in result["train"] if durations[i] <= 30) / 3600

        results = {
            remainder: assign_splits(
                self.PATHS,
                self._spec(tmp_dir, test_remainder=remainder),
                seed=1,
                durations=durations,
                max_duration=30.0,
            )
            for remainder in ("exclude", "train")
        }
        difference = surviving_hours(results["train"]) - surviving_hours(results["exclude"])
        assert abs(difference) <= max(self.DURATIONS) / 3600.0

    def test_matched_budget_in_examples(self, tmp_dir):
        overrides = {"train_budget_unit": "examples"}
        without = self._assign(self._spec(tmp_dir, test_remainder="exclude", **overrides))
        with_t = self._assign(self._spec(tmp_dir, test_remainder="train", **overrides))
        assert len(with_t["train"]) == len(without["train"])

    def test_matched_is_a_no_op_without_remainder(self, tmp_dir):
        capped = self._assign(self._spec(tmp_dir))
        uncapped = self._assign(self._spec(tmp_dir, train_budget=None))
        assert capped["train"] == uncapped["train"]

    def test_numeric_budget_caps_training(self, tmp_dir):
        spec = self._spec(tmp_dir, train_budget=10, train_budget_unit="examples")
        assert len(self._assign(spec)["train"]) == 10

    def test_remainder_larger_than_budget_is_an_error(self, tmp_dir):
        spec = self._spec(
            tmp_dir, test_remainder="train", train_budget=1, train_budget_unit="examples"
        )
        with pytest.raises(ValueError, match="exceeds"):
            self._assign(spec)

    def test_hour_budget_needs_durations(self, tmp_dir):
        with pytest.raises(ValueError, match="durations"):
            assign_splits(self.PATHS, self._spec(tmp_dir), seed=1)

    def test_remainder_train_needs_test_size(self, tmp_dir):
        with pytest.raises(ValueError, match="test_size"):
            self._spec(tmp_dir, test_remainder="train", test_size=None)

    def test_unknown_budget_word_rejected(self, tmp_dir):
        with pytest.raises(ValueError, match="train_budget"):
            self._spec(tmp_dir, train_budget="half")

    def test_variants_get_different_ids(self, tmp_dir):
        without = self._spec(tmp_dir, test_remainder="exclude").split_id(seed=1)
        with_t = self._spec(tmp_dir, test_remainder="train").split_id(seed=1)
        assert without != with_t

    def _hour_spec(self, tmp_dir, test_hours, **overrides):
        return self._spec(tmp_dir, test_size=test_hours, test_size_unit="hours", **overrides)

    def test_test_size_in_hours_reaches_target_within_one_group(self, tmp_dir):
        # each A session is 6 segments; one session's hours set the overshoot bound
        session_hours = {
            session: self._hours(
                [i for i, path in enumerate(self.PATHS) if path.startswith(session)]
            )
            for session in (f"A{index}" for index in range(4))
        }
        target = 1.5 * min(session_hours.values())
        result = self._assign(self._hour_spec(tmp_dir, target))
        held_out = self._hours(result["test"])
        assert target <= held_out <= target + max(session_hours.values())
        assert self._transcribers(result["test"]) == {"A"}

    def test_test_hours_count_only_usable_audio(self, tmp_dir):
        """Segments the length filter will drop must not count toward the test target."""
        durations = [
            40.0 if path.startswith("A") and index % 2 == 0 else duration
            for index, (path, duration) in enumerate(zip(self.PATHS, self.DURATIONS))
        ]
        usable_per_session = max(
            sum(
                durations[i]
                for i, path in enumerate(self.PATHS)
                if path.startswith(f"A{session}") and durations[i] <= 30
            )
            for session in range(4)
        ) / 3600.0
        target = 2.5 * usable_per_session
        result = assign_splits(
            self.PATHS,
            self._hour_spec(tmp_dir, target),
            seed=1,
            durations=durations,
            max_duration=30.0,
        )
        usable = sum(durations[i] for i in result["test"] if durations[i] <= 30) / 3600.0
        assert usable >= target
        # had the 40 s segments counted, two sessions would already have met the target
        assert len(self._sessions(result["test"])) >= 3

    def test_hour_test_pair_shares_test_and_dev(self, tmp_dir):
        target = 0.01
        without = self._assign(self._hour_spec(tmp_dir, target, test_remainder="exclude"))
        with_t = self._assign(self._hour_spec(tmp_dir, target, test_remainder="train"))
        assert without["test"] == with_t["test"]
        assert without["dev"] == with_t["dev"]

    def test_test_hours_need_durations(self, tmp_dir):
        spec = self._hour_spec(tmp_dir, 0.01, train_budget=None)
        with pytest.raises(ValueError, match="test_size in hours"):
            assign_splits(self.PATHS, spec, seed=1)

    def test_unknown_test_size_unit_rejected(self, tmp_dir):
        with pytest.raises(ValueError, match="test_size_unit"):
            self._spec(tmp_dir, test_size_unit="minutes")

    def test_test_size_unit_changes_the_id(self, tmp_dir):
        examples = self._spec(tmp_dir, test_size=2).split_id(seed=1)
        hours = self._spec(tmp_dir, test_size=2, test_size_unit="hours").split_id(seed=1)
        assert examples != hours


class TestIdStability:
    def test_new_options_absent_from_record_when_unset(self, tmp_dir):
        """Adding the options must not change the id of a split that ignores them.

        Run directories are named by split id, so a changed id would orphan
        every fold already trained.
        """
        record = _spec(tmp_dir, test={"transcriber": ["A"]}).record(seed=1)
        for option in (
            "test_size", "test_size_unit", "test_group_by", "test_remainder",
            "train_budget", "train_budget_unit",
        ):
            assert option not in record

    def test_default_test_size_unit_absent_from_record(self, tmp_dir):
        """A split using test_size in examples keeps the id it had before the unit existed."""
        spec = _spec(tmp_dir, test={"transcriber": ["A"]}, test_size=0.5)
        assert "test_size" in spec.record(seed=1)
        assert "test_size_unit" not in spec.record(seed=1)

    def test_split_record_has_hours_when_given_durations(self, tmp_dir):
        spec = _spec(tmp_dir, test={"transcriber": ["B"]})
        durations = [3600.0] * len(AUDIO_PATHS)
        assignment = assign_splits(AUDIO_PATHS, spec, durations=durations)
        root = os.path.join(tmp_dir, "record")
        save_split_record(root, spec, 1, assignment, AUDIO_PATHS, durations=durations)
        with open(os.path.join(root, "split.yaml")) as handle:
            record = yaml.safe_load(handle)
        assert record["hours"]["test"] == 2.0
