"""Metadata-driven train/dev/test assignment.

A *metadata table* is a CSV with one row per unit of the corpus -- a recording, a
speaker session, whatever the files can be mapped to -- and any number of
attribute columns (transcriber, speaker, date, ...). Each example is mapped to
its row by a *key*. By default the key is read from the audio file's basename
with `key_pattern`, so segments cut from one recording (`{stem}_{index}.wav`)
all inherit that recording's attributes. Only the basename is available: when
`datasets` embeds audio in a saved dataset it keeps `os.path.basename(path)`
and drops the directory. A source whose basenames are not unique, or that has
no file paths at all (many Hub datasets), should set `key_from` to a dataset
column holding an id instead.

A *selector* picks rows by attribute. It is a mapping from column to the values
accepted in that column; a row matches when every listed column holds one of the
listed values (AND across columns, OR within one). A list of mappings is the OR
of its members. Columns named in `multi_valued` hold several values separated by
`multi_value_separator`, and match when any one of them is accepted:

    test: {speaker: [S01]}
    test: {speaker: [S01, S02], session: [morning]}
    exclude: [{language: [english]}, {speakers: [S09]}]

For a *matched* comparison -- does having transcriber T in training help on
T's held-out data, at equal training size? -- hold out only part of T
(`test_size`, by `test_group_by` group), decide what happens to the rest with
`test_remainder` (`exclude` or `train`), and cap the training set with
`train_budget: matched`, which sets it to the size the `exclude` variant would
have. The two variants then share their test and dev sets exactly, and differ
only in whether T's remaining data replaces an equal amount of everyone else's.

The assignment is pure Python over example indices. It reads nothing but the
metadata file and the audio basenames, so the expensive cached artifacts --
audio and extracted features -- are shared by every split configuration, and
only what genuinely depends on the training rows (the vocabulary, a FOCUS
tokenizer) is rebuilt per split.
"""

import csv
import hashlib
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

import yaml
from lapt_core.artifacts import config_digest
from omegaconf import DictConfig, OmegaConf

SPLIT_NAMES = ("train", "dev", "test")


@dataclass(frozen=True)
class SplitSpec:
    """A resolved `split` config section.

    Attributes:
        name: Readable prefix of the split id, e.g. `loto`.
        metadata: Path to the metadata CSV.
        key_column: Metadata column holding each row's key.
        key_from: `path` to read keys from audio basenames, or the name of a
            dataset column holding them.
        key_pattern: Regex applied to the key source; group 1 is the key.
        multi_valued: Columns holding several separated values.
        multi_value_separator: Separator within a multi-valued cell.
        exclude: Selector for rows dropped from every split.
        test: Selector for the test split, or None for no test split.
        dev: Selector for the dev split, or None to sample dev from the rest.
        train: Selector restricting the training split, or None for no restriction.
        dev_size: Fraction (< 1) or number of *examples* to sample for dev,
            used only when `dev` is None.
        dev_group_by: Column whose values are kept whole when sampling dev, so
            that no group straddles train and dev. Defaults to `key_column`.
        unmatched: `error` or `drop` -- what to do with an example whose key has
            no metadata row.
        test_size: Fraction (< 1) or number of the `test`-selected examples to
            hold out, as whole `test_group_by` groups; None holds out all of them.
        test_group_by: Column kept whole when sampling test. Defaults to
            `dev_group_by`, then `key_column`.
        test_remainder: `exclude` or `train` -- where the `test`-selected
            examples not drawn into test go. Never into dev, so that both
            choices share one dev set.
        train_budget: Cap on the training split: a number (in
            `train_budget_unit`), `matched` for the size the training split has
            with `test_remainder: exclude`, or None for no cap. The test
            remainder, when trained on, is always kept; the rest is filled up
            to the budget by whole `dev_group_by` groups in seeded random order.
        train_budget_unit: `hours` (audio duration) or `examples`.
    """

    name: str
    metadata: str
    key_column: str = "recording"
    key_from: str = "path"
    key_pattern: str = r"^(.+?)\.[^.]+$"
    multi_valued: tuple[str, ...] = ()
    multi_value_separator: str = ";"
    exclude: tuple = ()
    test: tuple | None = None
    dev: tuple | None = None
    train: tuple | None = None
    dev_size: float = 0.1
    dev_group_by: str | None = None
    unmatched: str = "error"
    test_size: float | None = None
    test_group_by: str | None = None
    test_remainder: str = "exclude"
    train_budget: float | str | None = None
    train_budget_unit: str = "hours"
    metadata_digest: str = field(default="", compare=False)

    @classmethod
    def from_config(cls, split_config: DictConfig) -> "SplitSpec":
        """Build a spec from the `split` config section.

        Args:
            split_config: The `split` section; must have `enabled: true`.

        Returns:
            The resolved spec, with the metadata file's content digest attached.

        Raises:
            FileNotFoundError: If the metadata file does not exist.
            ValueError: If an enumerated option has an unknown value, or the
                options contradict each other.
        """
        plain = OmegaConf.to_container(split_config, resolve=True)
        metadata = plain["metadata"]
        if not metadata or not os.path.isfile(metadata):
            raise FileNotFoundError(f"split.metadata does not exist: {metadata}")

        unmatched = plain.get("unmatched", "error")
        if unmatched not in ("error", "drop"):
            raise ValueError(f"split.unmatched must be 'error' or 'drop', got {unmatched!r}")

        test_size = plain.get("test_size")
        test_remainder = plain.get("test_remainder") or "exclude"
        if test_remainder not in ("exclude", "train"):
            raise ValueError(
                f"split.test_remainder must be 'exclude' or 'train', got {test_remainder!r}"
            )
        if test_remainder == "train" and test_size is None:
            raise ValueError(
                "split.test_remainder=train needs split.test_size: without it every "
                "test-selected example is held out and there is no remainder to train on."
            )

        train_budget = plain.get("train_budget")
        if isinstance(train_budget, str) and train_budget != "matched":
            raise ValueError(
                f"split.train_budget must be a number, 'matched' or null, got {train_budget!r}"
            )
        train_budget_unit = plain.get("train_budget_unit") or "hours"
        if train_budget_unit not in ("hours", "examples"):
            raise ValueError(
                f"split.train_budget_unit must be 'hours' or 'examples', "
                f"got {train_budget_unit!r}"
            )

        return cls(
            name=plain.get("name") or "split",
            metadata=metadata,
            key_column=plain.get("key_column", "recording"),
            key_from=plain.get("key_from") or "path",
            key_pattern=plain.get("key_pattern", cls.key_pattern),
            multi_valued=tuple(plain.get("multi_valued") or ()),
            multi_value_separator=plain.get("multi_value_separator", ";"),
            exclude=_normalize_selector(plain.get("exclude")) or (),
            test=_normalize_selector(plain.get("test")),
            dev=_normalize_selector(plain.get("dev")),
            train=_normalize_selector(plain.get("train")),
            dev_size=plain.get("dev_size", 0.1),
            dev_group_by=plain.get("dev_group_by"),
            unmatched=unmatched,
            test_size=test_size,
            test_group_by=plain.get("test_group_by"),
            test_remainder=test_remainder,
            train_budget=train_budget,
            train_budget_unit=train_budget_unit,
            metadata_digest=_file_digest(metadata),
        )

    def record(self, seed: int) -> dict:
        """Return everything that determines the assignment, for hashing and saving."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "metadata_sha256": self.metadata_digest,
            "key_column": self.key_column,
            "key_from": self.key_from,
            "key_pattern": self.key_pattern,
            "multi_valued": list(self.multi_valued),
            "multi_value_separator": self.multi_value_separator,
            "exclude": _selector_record(self.exclude),
            "test": _selector_record(self.test),
            "dev": _selector_record(self.dev),
            "train": _selector_record(self.train),
            "dev_size": self.dev_size,
            "dev_group_by": self.dev_group_by,
            "unmatched": self.unmatched,
            "seed": seed,
            # Options added after the first split configs had been run. They are
            # recorded only when set, so that adding them did not change the id
            # -- and so the run directory -- of a split that does not use them.
            **{
                option: value
                for option, value in (
                    ("test_size", self.test_size),
                    ("test_group_by", self.test_group_by),
                    ("test_remainder", None if self.test_remainder == "exclude"
                     else self.test_remainder),
                    ("train_budget", self.train_budget),
                    ("train_budget_unit", None if self.train_budget is None
                     else self.train_budget_unit),
                )
                if value is not None
            },
        }

    def split_id(self, seed: int) -> str:
        """Name this split configuration: `{name}-{digest}`.

        The digest covers the selectors, the dev parameters, the seed and the
        metadata file's *contents*, so editing the metadata yields a new id
        rather than silently reusing a vocabulary built under the old one. The
        path of the metadata file is excluded: moving it changes nothing.
        """
        record = self.record(seed)
        del record["metadata"]
        return f"{_slugify(self.name)}-{config_digest(record)}"


def is_enabled(args: DictConfig) -> bool:
    """Report whether a metadata split is configured."""
    split = args.get("split")
    return bool(split is not None and split.get("enabled", False))


def split_id(args: DictConfig) -> str | None:
    """Return the configured split's id, or None when no split is enabled."""
    if not is_enabled(args):
        return None
    return SplitSpec.from_config(args.split).split_id(args.seed)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_metadata(spec: SplitSpec) -> dict[str, dict[str, tuple[str, ...]]]:
    """Read the metadata CSV into `{key: {column: values}}`.

    Every cell becomes a tuple of values -- one value for an ordinary column,
    the separated values for a multi-valued one -- so matching treats both the
    same way.

    Raises:
        ValueError: If the key column is missing or a key is duplicated.
    """
    with open(spec.metadata, newline="", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if reader.fieldnames is None or spec.key_column not in reader.fieldnames:
            raise ValueError(
                f"Metadata {spec.metadata} has no key column {spec.key_column!r} "
                f"(columns: {reader.fieldnames})"
            )

        metadata: dict[str, dict[str, tuple[str, ...]]] = {}
        for row in reader:
            key = row[spec.key_column]
            if key in metadata:
                raise ValueError(f"Duplicate key {key!r} in {spec.metadata}")
            attributes = {}
            for column, value in row.items():
                value = value or ""
                if column in spec.multi_valued:
                    attributes[column] = tuple(
                        part for part in value.split(spec.multi_value_separator) if part
                    )
                else:
                    attributes[column] = (value,)
            metadata[key] = attributes
    return metadata


def keys_from_values(
    values: list[str | None],
    key_pattern: str,
    from_paths: bool = True,
) -> list[str | None]:
    """Map each example's key source to its metadata key via `key_pattern`.

    Args:
        values: Audio paths (reduced to basenames) or dataset-column values.
        key_pattern: Regex whose group 1 is the key.
        from_paths: Whether `values` are file paths. Two examples sharing a
            basename are then indistinguishable -- the directory that told them
            apart did not survive caching -- so that raises.

    Returns:
        One key per example; None where the value is missing or the pattern
        does not match. The caller decides whether that is fatal.

    Raises:
        ValueError: If `from_paths` and two examples share a basename.
    """
    if from_paths:
        basenames = [os.path.basename(value) if value else value for value in values]
        duplicates = [name for name, count in Counter(basenames).items() if name and count > 1]
        if duplicates:
            raise ValueError(
                f"{len(duplicates)} audio basename(s) occur more than once (e.g. "
                f"{duplicates[0]!r}), so they cannot be told apart by filename -- "
                f"cached audio keeps only the basename. Set split.key_from to a "
                f"dataset column that identifies each example."
            )
        values = basenames

    pattern = re.compile(key_pattern)
    keys: list[str | None] = []
    for value in values:
        if value is None or value == "":
            keys.append(None)
            continue
        match = pattern.search(str(value))
        keys.append(match.group(1) if match else None)
    return keys


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

def matches(attributes: dict[str, tuple[str, ...]], selector: tuple) -> bool:
    """Report whether a metadata row is picked by a selector.

    Args:
        attributes: One row from `load_metadata`.
        selector: A normalized selector: a tuple of clauses, each a tuple of
            `(column, accepted_values)` pairs.

    Returns:
        True when any clause matches; a clause matches when every one of its
        columns holds at least one accepted value.

    Raises:
        KeyError: If a selector names a column the metadata does not have.
    """
    for clause in selector:
        clause_matches = True
        for column, accepted in clause:
            if column not in attributes:
                raise KeyError(
                    f"Selector column {column!r} is not in the metadata "
                    f"(columns: {sorted(attributes)})"
                )
            if not set(attributes[column]) & set(accepted):
                clause_matches = False
                break
        if clause_matches:
            return True
    return False


def _normalize_selector(raw) -> tuple | None:
    """Turn a selector from config into a hashable tuple of clauses.

    Accepts a mapping (one clause) or a list of mappings (OR of clauses). A
    scalar value is treated as a one-element list. Values are compared as
    strings, since metadata cells are strings.
    """
    if raw is None:
        return None
    clauses = raw if isinstance(raw, list) else [raw]
    normalized = []
    for clause in clauses:
        if not isinstance(clause, dict) or not clause:
            raise ValueError(f"A selector clause must be a non-empty mapping, got {clause!r}")
        pairs = []
        for column, values in sorted(clause.items()):
            if not isinstance(values, list):
                values = [values]
            pairs.append((column, tuple(str(value) for value in values)))
        normalized.append(tuple(pairs))
    return tuple(normalized)


def _selector_record(selector: tuple | None) -> list | None:
    """Render a normalized selector back as plain lists of mappings."""
    if selector is None:
        return None
    return [{column: list(values) for column, values in clause} for clause in selector]


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def assign_splits(
    key_sources: list[str | None],
    spec: SplitSpec,
    seed: int = 1,
    durations: list[float] | None = None,
    max_duration: float | None = None,
) -> dict[str, list[int]]:
    """Assign example indices to train/dev/test from metadata.

    Order of precedence: `exclude`; then `test` (all selected examples, or a
    `test_size` sample of their groups, leaving a remainder); then `dev` (by
    selector, or sampled by group from the examples *not* selected for test);
    then `train` (what remains, restricted by the `train` selector, plus the
    test remainder when `test_remainder` is `train`, capped by `train_budget`).
    An example belongs to at most one split.

    Test and dev are drawn before `test_remainder` is consulted, so two specs
    differing only in it hold out exactly the same test and dev examples.

    Args:
        key_sources: Each example's key source in dataset order: its audio
            path when `spec.key_from` is `path`, else its value in that column.
        spec: The split specification.
        seed: Seed for the grouped test, dev and budget samples.
        durations: Each example's audio duration in seconds. Required when
            `train_budget` is measured in hours; otherwise used only for the
            log.
        max_duration: The audio-length filter applied after assignment, in
            seconds. Longer examples are still assigned, but count nothing
            toward `train_budget`: they will be dropped, and counting them
            would let a variant whose extra data is mostly long segments come
            out short once filtered. Ignored when there is no budget, so it
            never changes an uncapped assignment.

    Returns:
        `{split: sorted example indices}` for each non-empty split.

    Raises:
        ValueError: If an example has no metadata row and `unmatched` is
            `error`, if the training split comes out empty, if an hour budget
            is set without durations, or if the trained-on test remainder alone
            exceeds the budget.
    """
    metadata = load_metadata(spec)
    keys = keys_from_values(key_sources, spec.key_pattern, from_paths=spec.key_from == "path")

    unmatched = [index for index, key in enumerate(keys) if key not in metadata]
    if unmatched and spec.unmatched == "error":
        preview = ", ".join(str(key_sources[index]) for index in unmatched[:3])
        raise ValueError(
            f"{len(unmatched)} example(s) have no row in {spec.metadata} "
            f"(e.g. {preview}). Check split.key_pattern, or set "
            f"split.unmatched=drop to leave them out."
        )
    if unmatched:
        print(
            f"Split: dropping {len(unmatched)} example(s) with no metadata row",
            file=sys.stderr,
        )

    if spec.train_budget is not None and spec.train_budget_unit == "hours" and durations is None:
        raise ValueError("split.train_budget in hours needs example durations.")

    def column_of(index: int, column: str) -> tuple[str, ...]:
        return metadata[keys[index]][column]

    dev_group_column = spec.dev_group_by or spec.key_column
    test_group_column = spec.test_group_by or dev_group_column

    candidates: list[int] = []
    excluded = 0
    for index, key in enumerate(keys):
        if key not in metadata:
            continue
        if spec.exclude and matches(metadata[key], spec.exclude):
            excluded += 1
            continue
        candidates.append(index)

    assignment: dict[str, list[int]] = {"train": [], "dev": [], "test": []}

    # test, and the part of the test selection that is not held out
    selected: list[int] = []
    remaining: list[int] = []
    for index in candidates:
        if spec.test is not None and matches(metadata[keys[index]], spec.test):
            selected.append(index)
        else:
            remaining.append(index)

    test_remainder: list[int] = []
    if spec.test_size is None:
        assignment["test"] = selected
    else:
        groups = [column_of(index, test_group_column) for index in selected]
        test_mask = _sample_groups(groups, spec.test_size, seed)
        for index, in_test in zip(selected, test_mask):
            if in_test:
                assignment["test"].append(index)
            else:
                test_remainder.append(index)

    # dev, from the examples not selected for test
    if spec.dev is not None:
        rest: list[int] = []
        for index in remaining:
            if matches(metadata[keys[index]], spec.dev):
                assignment["dev"].append(index)
            else:
                rest.append(index)
        remaining = rest
    else:
        groups = [column_of(index, dev_group_column) for index in remaining]
        dev_mask = _sample_groups(groups, spec.dev_size, seed)
        rest = []
        for index, in_dev in zip(remaining, dev_mask):
            if in_dev:
                assignment["dev"].append(index)
            else:
                rest.append(index)
        remaining = rest

    # train
    pool = [
        index for index in remaining
        if spec.train is None or matches(metadata[keys[index]], spec.train)
    ]
    kept_remainder: list[int] = []
    if spec.test_remainder == "train":
        kept_remainder = [
            index for index in test_remainder
            if spec.train is None or matches(metadata[keys[index]], spec.train)
        ]

    def budget_size(index: int) -> float:
        """An example's contribution to the budget, zero if it will be filtered."""
        if (
            max_duration is not None
            and durations is not None
            and durations[index] > max_duration
        ):
            return 0.0
        if spec.train_budget_unit == "examples":
            return 1.0
        return durations[index] / 3600.0

    def size_of(indices: list[int]) -> float:
        return sum(budget_size(index) for index in indices)

    if spec.train_budget is None:
        filled = pool
    else:
        if spec.train_budget == "matched":
            budget = size_of(pool)
        else:
            budget = float(spec.train_budget)
        mandatory = size_of(kept_remainder)
        if mandatory > budget:
            raise ValueError(
                f"The trained-on test remainder alone ({mandatory:.2f} "
                f"{spec.train_budget_unit}) exceeds split.train_budget "
                f"({budget:.2f}). Raise the budget or test_size."
            )
        groups = [column_of(index, dev_group_column) for index in pool]
        sizes = [budget_size(index) for index in pool]
        fill_mask = _fill_to_budget(groups, budget - mandatory, seed, sizes)
        filled = [index for index, keep in zip(pool, fill_mask) if keep]
    assignment["train"] = sorted(filled + kept_remainder)

    if not assignment["train"]:
        raise ValueError("The split configuration leaves no training examples.")

    counts = ", ".join(f"{name}={len(indices)}" for name, indices in assignment.items())
    if durations is not None:
        hours = ", ".join(
            f"{name}={sum(durations[index] for index in indices) / 3600.0:.2f}h"
            for name, indices in assignment.items()
        )
        counts = f"{counts}; {hours}"
    # held-out remainder not trained on, plus pool examples cut by the budget
    unused = len(test_remainder) - len(kept_remainder) + len(pool) - len(filled)
    print(
        f"Split {spec.split_id(seed)}: {counts} (excluded={excluded}, unused={unused})",
        file=sys.stderr,
    )
    return {name: indices for name, indices in assignment.items() if indices}


def _fill_to_budget(
    groups: list[tuple[str, ...]],
    budget: float,
    seed: int,
    sizes: list[float] | None = None,
) -> list[bool]:
    """Pick examples, whole groups at a time in seeded random order, up to a budget.

    Groups are shuffled and their examples taken in order until the running
    total reaches `budget`, so the result overshoots by at most one example.
    The last group taken may be partial. That is harmless here: what is left
    out is not used anywhere, so a split group cannot leak between splits.

    Args:
        groups: Each example's group value.
        budget: Target total size.
        seed: Shuffle seed.
        sizes: Each example's size; None counts every example as 1.

    Returns:
        A per-example mask, True for kept.
    """
    if sizes is None:
        sizes = [1.0] * len(groups)

    members: dict[tuple[str, ...], list[int]] = {}
    for position, group in enumerate(groups):
        members.setdefault(group, []).append(position)
    ordered = sorted(members, key=str)
    random.Random(seed).shuffle(ordered)

    keep = [False] * len(groups)
    total = 0.0
    for group in ordered:
        for position in members[group]:
            if total >= budget:
                return keep
            keep[position] = True
            total += sizes[position]
    return keep


def _sample_groups(
    groups: list[tuple[str, ...]],
    dev_size: float | int,
    seed: int,
) -> list[bool]:
    """Pick whole groups for dev until the target number of examples is reached.

    Groups are shuffled with a seeded generator and taken in order until the
    dev split holds at least `dev_size` examples (a fraction of all examples
    when below 1). Keeping groups whole is the point: segments cut from one
    recording must not straddle train and dev, or dev error is optimistic.

    Args:
        groups: Each example's group value (a tuple, as metadata cells are).
        dev_size: Fraction (< 1) or number of examples.
        seed: Shuffle seed.

    Returns:
        A per-example mask, True for dev.
    """
    if not groups or not dev_size:
        return [False] * len(groups)

    if dev_size < 1:
        target = int(round(dev_size * len(groups)))
    else:
        target = int(dev_size)

    sizes = Counter(groups)
    ordered = sorted(sizes, key=str)
    random.Random(seed).shuffle(ordered)

    chosen: set = set()
    taken = 0
    for group in ordered:
        if taken >= target:
            break
        chosen.add(group)
        taken += sizes[group]

    return [group in chosen for group in groups]


def save_split_record(
    split_root: str,
    spec: SplitSpec,
    seed: int,
    assignment: dict[str, list[int]],
    key_sources: list[str | None],
    durations: list[float] | None = None,
) -> None:
    """Write what produced this split, and which example went where.

    `split.yaml` records the resolved spec and the split sizes;
    `assignment.csv` lists every assigned example's key source (audio basename
    or id) and split, for per-recording error analysis after training. Both are
    deterministic, so rewriting them on every run is harmless.

    Args:
        split_root: The split's directory, `<cache_dir>/splits/<split_id>`.
        spec: The split specification.
        seed: Global seed.
        assignment: `{split: example indices}`.
        key_sources: Each example's key source, as given to `assign_splits`.
        durations: Each example's audio duration in seconds, to record each
            split's hours; omitted from the record when None.
    """
    os.makedirs(split_root, exist_ok=True)

    record = spec.record(seed)
    record["split_id"] = spec.split_id(seed)
    record["counts"] = {name: len(indices) for name, indices in assignment.items()}
    if durations is not None:
        record["hours"] = {
            name: round(sum(durations[index] for index in indices) / 3600.0, 3)
            for name, indices in assignment.items()
        }
    with open(os.path.join(split_root, "split.yaml"), "w", encoding="utf-8") as record_file:
        yaml.safe_dump(record, record_file, sort_keys=False)

    with open(
        os.path.join(split_root, "assignment.csv"), "w", newline="", encoding="utf-8"
    ) as assignment_file:
        writer = csv.writer(assignment_file)
        writer.writerow(["example", "split"])
        for name in SPLIT_NAMES:
            for index in assignment.get(name, []):
                writer.writerow([key_sources[index], name])
    print(f"Saved split record to {split_root}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_digest(path: str) -> str:
    """Return the SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _slugify(value: str) -> str:
    """Reduce a name to a filesystem-safe token."""
    safe = [character if character.isalnum() else "-" for character in str(value)]
    return "".join(safe).strip("-").lower()
