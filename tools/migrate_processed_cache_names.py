"""Drop the redundant model slug from processed-subcache names.

`processed_cache_dirname` used to append the whole `tokenizer_id` to a name that
already opened with the model slug, so the model appeared twice:

    processed_whisper-medium-zu_sr16k_a30_l448_sw-transcribe_focus-v4k-whisper-medium-zu
              ^^^^^^^^^^^^^^^^^                                       ^^^^^^^^^^^^^^^^^

It now appends `vocabulary_id`, which omits the model. This renames existing
subcaches to match, so they are found rather than rebuilt -- and a processed
subcache here runs to tens of gigabytes.

A pure directory rename: the record inside is untouched. `focus_tokenizer_id`
stores the full `tokenizer_id` and always did, because that is the actual cache
key -- two base models must not share a tokenizer. The directory name is the
readable label, and it is only the label that was repeating itself.

Caches predating per-model tokenizer keying already read `..._focus-v4k` with no
model slug, and are left alone.

Reversible only by renaming back by hand; nothing is deleted.

Prints a plan and changes nothing unless `--apply` is passed.
"""

import argparse
import os
import sys

PROCESSED_PREFIX = "processed_"
FOCUS_PREFIX = "focus-"


def redundant_suffix(name: str) -> tuple[str, str] | None:
    """Return (new name, the slug being dropped) if this name repeats its model.

    The name is `processed_{model}_{settings...}_{focus segment}`. Components are
    joined with `_`, and slugs contain `-` rather than `_`, so the model is the
    first component and the FOCUS segment is the last.

    Args:
        name: A processed subcache directory name.

    Returns:
        `(new_name, slug)`, or None if there is nothing to drop.
    """
    if not name.startswith(PROCESSED_PREFIX):
        return None
    components = name[len(PROCESSED_PREFIX):].split("_")
    if len(components) < 2:
        return None

    model_slug, last = components[0], components[-1]
    if not last.startswith(FOCUS_PREFIX):
        return None
    if not last.endswith("-" + model_slug):
        return None

    components[-1] = last[: -len("-" + model_slug)]
    return PROCESSED_PREFIX + "_".join(components), model_slug


def main() -> int:
    """Plan, and optionally apply, the renames.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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

    renames: list[tuple[str, str]] = []
    collisions = 0

    for entry in sorted(os.listdir(args.root)):
        dataset_dir = os.path.join(args.root, entry)
        if not os.path.isdir(dataset_dir):
            continue
        for name in sorted(os.listdir(dataset_dir)):
            source = os.path.join(dataset_dir, name)
            if not os.path.isdir(source):
                continue
            result = redundant_suffix(name)
            if result is None:
                continue
            new_name, slug = result
            target = os.path.join(dataset_dir, new_name)
            if os.path.exists(target):
                print(f"  [COLLISION] {entry}/{name}\n              -> {new_name} exists, skipped")
                collisions += 1
                continue
            print(f"  [rename] {entry}/{name}\n           -> {new_name}   (dropped '-{slug}')")
            renames.append((source, target))

    print(f"\n{len(renames)} rename(s), {collisions} collision(s).")

    if not renames:
        print("Nothing to do: no subcache repeats its model slug.")
        return 0

    if not args.apply:
        print("\nRe-run with --apply to perform them.")
        return 0

    for source, target in renames:
        os.rename(source, target)
    print(f"\nRenamed {len(renames)} subcache(s). Records untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
