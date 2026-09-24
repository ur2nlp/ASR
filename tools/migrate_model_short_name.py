"""Rename the caches that a change to `model.short_name` would orphan.

`short_name` is read in three places, and renaming it moves two cache
directories out from under the code:

    data/{id}/processed_{slug}_.../     feature-extracted + label-encoded data
    data/{id}/focus/{vocab}-{slug}/     the FOCUS tokenizer

Renaming the directories is most of the job, but not all of it. The processed
cache's own record stores `focus_tokenizer_id`, which *ends* with the model
slug (`ProcessedDatasetConfig`, via `tokenizer_id`), so moving the directory
leaves a record that still names the old model and the cache is refused with a
ConfigMismatchError. This rewrites that field as well.

The FOCUS tokenizer's record needs no such fix: `FocusTokenizerConfig` tracks
`pretrained_name`, not `short_name`, so only its directory name is stale. That
is the difference between renaming a directory and retraining SentencePiece and
fastText.

Run directories under `models/` are deliberately left alone. They hold
checkpoints rather than caches, nothing keys on their names, and the
`training_config.yaml` inside each one records the `short_name` the run actually
used -- which is history, and which `registry.py` reads.

Reversible: run it again with OLD and NEW swapped.

Prints a plan and changes nothing unless `--apply` is passed.
"""

import argparse
import os
import sys

import yaml

# Import `src` without an editable install, for running straight from a checkout.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.artifact_configs import _slugify

PROCESSED_PREFIX = "processed_"
FOCUS_SUBDIR = "focus"
CONFIG_FILENAME = "config.yaml"


def processed_dirs(dataset_dir: str, old_slug: str) -> list[str]:
    """Find processed subcaches built under the old model name.

    Matched on a component boundary so that `whisper-small` does not also claim
    `processed_whisper-small-v2_...`.

    Args:
        dataset_dir: A `data/{id}` directory.
        old_slug: The slugified old `short_name`.

    Returns:
        Sorted directory paths.
    """
    exact = PROCESSED_PREFIX + old_slug
    found = [
        os.path.join(dataset_dir, name)
        for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))
        and (name == exact or name.startswith(exact + "_"))
    ]
    return sorted(found)


def focus_dirs(dataset_dir: str, old_slug: str) -> list[str]:
    """Find FOCUS tokenizer directories built under the old model name.

    Args:
        dataset_dir: A `data/{id}` directory.
        old_slug: The slugified old `short_name`.

    Returns:
        Sorted directory paths.
    """
    focus_root = os.path.join(dataset_dir, FOCUS_SUBDIR)
    if not os.path.isdir(focus_root):
        return []
    suffix = "-" + old_slug
    found = [
        os.path.join(focus_root, name)
        for name in os.listdir(focus_root)
        if os.path.isdir(os.path.join(focus_root, name)) and name.endswith(suffix)
    ]
    return sorted(found)


def renamed(path: str, old_slug: str, new_slug: str) -> str:
    """Return the path with the trailing or embedded old slug swapped.

    Args:
        path: The directory being renamed.
        old_slug: Slugified old `short_name`.
        new_slug: Slugified new `short_name`.

    Returns:
        The new full path.
    """
    parent, name = os.path.split(path)
    if name.startswith(PROCESSED_PREFIX):
        rest = name[len(PROCESSED_PREFIX):]
        name = PROCESSED_PREFIX + new_slug + rest[len(old_slug):]
    elif name.endswith("-" + old_slug):
        name = name[: -len(old_slug)] + new_slug
    return os.path.join(parent, name)


def record_needs_rewrite(config_path: str, old_slug: str) -> str | None:
    """Return the stale `focus_tokenizer_id` in a processed record, if any.

    Args:
        config_path: Path to a processed subcache's config.yaml.
        old_slug: Slugified old `short_name`.

    Returns:
        The stale value, or None if absent, unreadable or already correct.
    """
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path) as config_file:
            data = yaml.safe_load(config_file) or {}
    except yaml.YAMLError:
        return None
    value = data.get("focus_tokenizer_id")
    if isinstance(value, str) and value.endswith("-" + old_slug):
        return value
    return None


def rewrite_record(config_path: str, old_slug: str, new_slug: str) -> None:
    """Swap the model slug in a processed record's `focus_tokenizer_id`.

    Args:
        config_path: Path to a processed subcache's config.yaml.
        old_slug: Slugified old `short_name`.
        new_slug: Slugified new `short_name`.
    """
    with open(config_path) as config_file:
        data = yaml.safe_load(config_file) or {}
    value = data["focus_tokenizer_id"]
    data["focus_tokenizer_id"] = value[: -len(old_slug)] + new_slug
    with open(config_path, "w") as config_file:
        yaml.dump(data, config_file, default_flow_style=False, sort_keys=False)


def main() -> int:
    """Plan, and optionally apply, the rename.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old", help="The current model.short_name")
    parser.add_argument("new", help="The model.short_name you are moving to")
    parser.add_argument(
        "root", nargs="?", default="data",
        help="Directory holding the per-dataset caches (default: data)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Perform the renames. Without this, only the plan is printed.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"Not a directory: {args.root}", file=sys.stderr)
        return 1

    old_slug, new_slug = _slugify(args.old), _slugify(args.new)
    if old_slug == new_slug:
        print(f"'{args.old}' and '{args.new}' slugify to the same token "
              f"({old_slug!r}); nothing on disk would change.")
        return 0

    print(f"{args.old} -> {args.new}   (slug {old_slug} -> {new_slug})\n")

    renames: list[tuple[str, str]] = []
    rewrites: list[tuple[str, str]] = []
    collisions: list[str] = []

    for entry in sorted(os.listdir(args.root)):
        dataset_dir = os.path.join(args.root, entry)
        if not os.path.isdir(dataset_dir):
            continue

        for source in processed_dirs(dataset_dir, old_slug) + focus_dirs(dataset_dir, old_slug):
            target = renamed(source, old_slug, new_slug)
            rel_source = os.path.relpath(source, args.root)
            rel_target = os.path.relpath(target, args.root)
            if os.path.exists(target):
                print(f"  [COLLISION] {rel_source}\n              -> {rel_target} already exists, skipped")
                collisions.append(rel_source)
                continue
            print(f"  [rename] {rel_source}\n           -> {rel_target}")
            renames.append((source, target))

            stale = record_needs_rewrite(os.path.join(source, CONFIG_FILENAME), old_slug)
            if stale is not None:
                fixed = stale[: -len(old_slug)] + new_slug
                print(f"           record: focus_tokenizer_id {stale} -> {fixed}")
                rewrites.append((target, stale))

    print(f"\n{len(renames)} rename(s), {len(rewrites)} record rewrite(s), "
          f"{len(collisions)} collision(s).")

    if not renames:
        print("Nothing found under the old name. Check --help for the expected layout.")
        return 0

    if not args.apply:
        print("\nRe-run with --apply to perform them.")
        return 0

    for source, target in renames:
        os.rename(source, target)
    for target, _stale in rewrites:
        rewrite_record(os.path.join(target, CONFIG_FILENAME), old_slug, new_slug)
    print(f"\nRenamed {len(renames)} director(ies) and rewrote {len(rewrites)} record(s).")
    print("Reversible: run again with the names swapped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
