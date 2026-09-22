"""Move untokenized cache records inside the directory they describe.

Before the dataset stage became an artifact, a run recorded its dataset
parameters in `<cache_dir>/dataset_config.yaml` -- beside the cached data
rather than inside it. The artifact layer looks for `config.yaml` *within*
`<cache_dir>/untokenized/`, and refuses a cache it cannot verify, so an
existing cache needs its record moved before it can be reused.

Two things beyond a move are needed, which is why this is a script rather than
an `mv`:

  * The record's *shape* changed. The old one carried `preprocessing` and
    `seed`, neither of which can affect a cache written before normalization
    and splitting run, and it lacked the column names, which can. The target
    record is built by handing the cached parameters to the source class
    itself, so this cannot drift from the code it migrates toward.
  * A `concat` cache's children never had a record at all -- only the
    top-level one was ever written -- so their records are reconstructed from
    the `sources` list in the parent's legacy record.

The legacy record is *copied*, not deleted, so this is reversible: remove the
written `config.yaml` files and nothing has changed.

Where the old record cannot answer a question, the plan says so explicitly
rather than guessing silently: `audio_column` and `text_column` were never
tracked, so they are filled from the source class's defaults and flagged as
assumed. Check those lines before applying if your config renames columns.

Prints a plan and changes nothing unless `--apply` is passed.
"""

import argparse
import os
import sys

import yaml

# Import `src` without an editable install, for running straight from a checkout.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lapt_core.artifacts import CONFIG_FILENAME

from src.sources import SOURCE_TYPES, make_source

# The record name the dataset stage used before it became an artifact, and the
# subdirectory the data itself has always lived in.
LEGACY_CONFIG_FILENAME = "dataset_config.yaml"
ARTIFACT_SUBDIR = "untokenized"

# Fields the old record carried that the new one deliberately drops, and the
# reason each is gone. Reported so a migration explains itself.
DROPPED_FIELDS = {
    "preprocessing": "normalization runs after this cache is written",
    "seed": "the dev split is carved out after this cache is written",
    "id": "the id names the cache directory, it does not key the contents",
}

# Fields the new record needs that the old one never tracked.
ASSUMED_FIELDS = ("audio_column", "text_column")


def find_legacy_records(root: str) -> list[str]:
    """Find every cache directory holding a legacy record.

    Args:
        root: Directory to walk.

    Returns:
        Sorted paths of the directories, not the records themselves.
    """
    found = []
    for dirpath, _, filenames in os.walk(root):
        if LEGACY_CONFIG_FILENAME in filenames:
            found.append(dirpath)
    return sorted(found)


def read_record(path: str) -> dict | None:
    """Read a YAML record, returning None if absent or empty.

    Args:
        path: Full path of the record file.

    Returns:
        The parsed record, or None.
    """
    if not os.path.exists(path):
        return None
    with open(path) as record_file:
        return yaml.safe_load(record_file)


def target_record(source_config: dict, seed: int) -> dict:
    """Build the record the current code would write for a source config.

    Args:
        source_config: The source parameters, from a legacy record or a
            parent's `sources` entry.
        seed: Seed to pass to sources that sample.

    Returns:
        The artifact's `config()` dict.

    Raises:
        ValueError: If the config names a type the registry does not know.
    """
    return make_source("", source_config, seed).config()


def plan_for(cache_dir: str, source_config: dict, seed: int) -> dict:
    """Decide what one cache directory needs.

    Args:
        cache_dir: Directory whose `untokenized/` subdirectory holds the data.
        source_config: Parameters describing that data.
        seed: Seed to pass to sources that sample.

    Returns:
        A plan dict with `status`, `record`, `dropped`, `assumed`, `detail`
        and `config_path`. Status is one of `write`, `already-migrated`,
        `differs`, `no-data`, `unknown-type`.
    """
    artifact_dir = os.path.join(cache_dir, ARTIFACT_SUBDIR)
    config_path = os.path.join(artifact_dir, CONFIG_FILENAME)
    blank = {"record": None, "dropped": [], "assumed": [], "config_path": config_path}

    try:
        record = target_record(source_config, seed)
    except ValueError as lookup_error:
        return {"status": "unknown-type", "detail": str(lookup_error), **blank}

    if not os.path.isdir(artifact_dir):
        return {"status": "no-data", "detail": "", **blank}

    dropped = [key for key in DROPPED_FIELDS if key in source_config]
    # Only fields the target record actually carries can be "assumed": a source
    # type that does not key on columns at all (`paired`) leaves them out of
    # `config()` entirely, and reporting them as assumed-None would be noise.
    assumed = [
        key for key in ASSUMED_FIELDS
        if key not in source_config and key in record
    ]

    existing = read_record(config_path)
    if existing is not None:
        status = "already-migrated" if existing == record else "differs"
    else:
        status = "write"

    return {
        "status": status,
        "detail": "",
        "record": record,
        "dropped": dropped,
        "assumed": assumed,
        "config_path": config_path,
    }


def plans_for_tree(cache_dir: str, legacy: dict, seed: int) -> list[tuple[str, dict]]:
    """Build plans for a cache directory and, for a concat, its children.

    A concat's children were cached under `<cache_dir>/<source_id>/` and never
    had a record of their own, so theirs are reconstructed from the parent's
    `sources` list.

    Args:
        cache_dir: Directory holding the legacy record.
        legacy: The parsed legacy record.
        seed: Seed to pass to sources that sample.

    Returns:
        `(cache_dir, plan)` pairs, parent first.
    """
    plans = [(cache_dir, plan_for(cache_dir, legacy, seed))]

    if legacy.get("type") == "concat":
        for index, child in enumerate(legacy.get("sources") or []):
            child_id = child.get("id", f"source_{index}")
            child_dir = os.path.join(cache_dir, child_id)
            plans.append((child_dir, plan_for(child_dir, child, seed)))

    return plans


def describe(cache_dir: str, plan: dict, root: str) -> None:
    """Print one directory's plan to stdout.

    Args:
        cache_dir: The directory the plan is for.
        plan: The plan from `plan_for`.
        root: Root the path is reported relative to.
    """
    print(f"  [{plan['status']}] {os.path.relpath(cache_dir, root)}")
    if plan["status"] in ("already-migrated", "no-data", "unknown-type"):
        if plan["detail"]:
            print(f"      ! {plan['detail']}")
        return
    for key in plan["dropped"]:
        print(f"      - {key}: dropped, {DROPPED_FIELDS[key]}")
    for key in plan["assumed"]:
        print(f"      ? {key}: assumed {plan['record'].get(key)!r}, never tracked before")


def write_record(plan: dict) -> None:
    """Write a plan's record into the artifact directory.

    Args:
        plan: A plan whose status is `write` or `differs`.
    """
    with open(plan["config_path"], "w") as config_file:
        yaml.dump(plan["record"], config_file, default_flow_style=False, sort_keys=False)


def main() -> int:
    """Plan, and optionally apply, the record migration.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root",
        nargs="?",
        default="data",
        help="Directory to walk for cache directories (default: data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed to record for sources that sample (default: 1)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the records. Without this, only the plan is printed.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"Not a directory: {args.root}", file=sys.stderr)
        return 1

    legacy_dirs = find_legacy_records(args.root)
    if not legacy_dirs:
        print(f"No {LEGACY_CONFIG_FILENAME} found under {args.root}.")
        print(f"Known dataset types: {', '.join(SOURCE_TYPES.known_types())}")
        return 0

    print(f"Found {len(legacy_dirs)} cache(s) with a legacy record under {args.root}:\n")

    to_write = []
    for cache_dir in legacy_dirs:
        legacy = read_record(os.path.join(cache_dir, LEGACY_CONFIG_FILENAME)) or {}
        for target_dir, plan in plans_for_tree(cache_dir, legacy, args.seed):
            describe(target_dir, plan, args.root)
            if plan["status"] in ("write", "differs"):
                to_write.append(plan)

    if not to_write:
        print("\nNothing to do.")
        return 0

    if not args.apply:
        print(f"\n{len(to_write)} record(s) would be written. Re-run with --apply.")
        return 0

    for plan in to_write:
        write_record(plan)
        print(f"Wrote {plan['config_path']}")
    print(f"\nWrote {len(to_write)} record(s). The legacy files are left in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
