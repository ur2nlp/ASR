"""Re-key a fetched-run mirror by experiment_id and flatten it by artifact kind.

Runs used to be mirrored into `outputs/{run_id}/` with `run_id` derived from the
remote directory path. That id depended on which `-b` root the inventory was run
against, so the same run fetched from `models/` and from `models/zulu/` landed in
two directories -- and nothing could tell they were the same run. The fetcher now
keys runs by the `experiment_id` in each `training_config.yaml`, and stores them
the way LAPT does:

    outputs/{run_id}/trainer_state.json     ->  outputs/trainer_states/{exp_id}.json
    outputs/{run_id}/training_config.yaml   ->  outputs/configs/{exp_id}.yaml

Files are *copied*, not moved, so this is reversible: delete the two new
directories and nothing has changed. The old per-run directories are left for
you to remove once you are satisfied; `--list-redundant` prints them.

Where two source directories map to the same experiment_id, their contents are
compared. Identical files are deduplicated silently; differing ones are reported
and neither is written, because picking a winner is not this script's call.

Prints a plan and changes nothing unless `--apply` is passed.
"""

import argparse
import filecmp
import json
import os
import shutil
import sys

import yaml

TRAINER_STATE = "trainer_state.json"
TRAINING_CONFIG = "training_config.yaml"
SKIP_DIRS = {"hydra", "trainer_states", "configs", "fetched"}


def find_run_dirs(root: str) -> list[str]:
    """Find mirrored run directories directly under `root`.

    Args:
        root: The outputs directory to scan.

    Returns:
        Sorted paths of directories holding a trainer_state.json.
    """
    found = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or name in SKIP_DIRS:
            continue
        if os.path.isfile(os.path.join(path, TRAINER_STATE)):
            found.append(path)
    return found


def read_experiment_id(run_dir: str) -> str | None:
    """Return the experiment_id a run's config declares, or None.

    Args:
        run_dir: Directory holding training_config.yaml.

    Returns:
        The experiment id, or None if absent or unreadable.
    """
    config_path = os.path.join(run_dir, TRAINING_CONFIG)
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path) as config_file:
            data = yaml.safe_load(config_file) or {}
    except yaml.YAMLError:
        return None
    exp_id = data.get("experiment_id")
    return str(exp_id) if exp_id else None


def group_by_experiment(run_dirs: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Group run directories by the experiment_id they declare.

    Args:
        run_dirs: Directories to group.

    Returns:
        A `(groups, unidentified)` pair.
    """
    groups: dict[str, list[str]] = {}
    unidentified: list[str] = []
    for run_dir in run_dirs:
        exp_id = read_experiment_id(run_dir)
        if exp_id is None:
            unidentified.append(run_dir)
            continue
        groups.setdefault(exp_id, []).append(run_dir)
    return groups, unidentified


def agree(run_dirs: list[str], filename: str) -> bool:
    """Return whether every directory holds a byte-identical copy of `filename`.

    Args:
        run_dirs: Directories to compare.
        filename: The file to compare across them.

    Returns:
        True if all present copies are identical.
    """
    paths = [
        os.path.join(d, filename) for d in run_dirs
        if os.path.isfile(os.path.join(d, filename))
    ]
    if not paths:
        return False
    return all(filecmp.cmp(paths[0], other, shallow=False) for other in paths[1:])


def progress(run_dir: str) -> tuple[int, int]:
    """Rank a copy of a run by how far through training it got.

    Two mirrors of the same run differ only because one was fetched later, so
    the further-along copy is the newer one. A finished run outranks an
    unfinished one at the same step.

    Args:
        run_dir: Directory holding trainer_state.json.

    Returns:
        A `(finished, global_step)` sort key; `(-1, -1)` if unreadable.
    """
    try:
        with open(os.path.join(run_dir, TRAINER_STATE)) as state_file:
            data = json.load(state_file)
    except (OSError, json.JSONDecodeError):
        return (-1, -1)
    log_history = data.get("log_history") or []
    finished = "train_runtime" in (log_history[-1] if log_history else {})
    return (int(finished), int(data.get("global_step", 0)))


def main() -> int:
    """Plan, and optionally apply, the mirror migration.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root", nargs="?", default="outputs",
        help="Outputs directory to migrate (default: outputs)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the new layout. Without this, only the plan is printed.",
    )
    parser.add_argument(
        "--on-conflict", choices=("skip", "latest"), default="skip",
        help=(
            "What to do when two directories map to one experiment_id with "
            "differing contents: 'skip' leaves both alone (default), 'latest' "
            "keeps the copy that got further through training."
        ),
    )
    parser.add_argument(
        "--list-redundant", action="store_true",
        help="After a successful plan, list the old directories now superseded.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"Not a directory: {args.root}", file=sys.stderr)
        return 1

    run_dirs = find_run_dirs(args.root)
    if not run_dirs:
        print(f"No mirrored run directories found under {args.root}.")
        return 0

    groups, unidentified = group_by_experiment(run_dirs)
    states_dir = os.path.join(args.root, "trainer_states")
    configs_dir = os.path.join(args.root, "configs")

    print(f"{len(run_dirs)} run director(ies) -> {len(groups)} experiment(s)\n")

    planned: list[tuple[str, str, str]] = []
    conflicts: list[str] = []
    for exp_id, dirs in sorted(groups.items()):
        names = [os.path.relpath(d, args.root) for d in dirs]
        if len(dirs) > 1:
            if agree(dirs, TRAINER_STATE):
                print(f"  [dedupe] {exp_id}: {len(dirs)} identical copies ({', '.join(names)})")
            elif args.on_conflict == "latest":
                dirs = sorted(dirs, key=progress, reverse=True)
                winner, steps = os.path.relpath(dirs[0], args.root), progress(dirs[0])[1]
                print(f"  [resolve] {exp_id}: copies differ, keeping {winner} "
                      f"(step {steps}, furthest along)")
            else:
                print(f"  [CONFLICT] {exp_id}: copies differ ({', '.join(names)}) -- skipped")
                print("             re-run with --on-conflict latest to keep the "
                      "further-along copy")
                conflicts.append(exp_id)
                continue
        else:
            print(f"  [move]   {exp_id}: {names[0]}")
        source = dirs[0]
        planned.append((exp_id, os.path.join(source, TRAINER_STATE),
                        os.path.join(source, TRAINING_CONFIG)))

    for run_dir in unidentified:
        print(f"  [skip]   {os.path.relpath(run_dir, args.root)}: no experiment_id in config")

    print(f"\n{len(planned)} experiment(s) to write, {len(conflicts)} conflict(s), "
          f"{len(unidentified)} unidentified.")

    if args.list_redundant:
        print("\nSuperseded directories (remove once satisfied):")
        for exp_id, dirs in sorted(groups.items()):
            if exp_id in conflicts:
                continue
            for run_dir in dirs:
                print(f"  {run_dir}")

    if not args.apply:
        print("\nRe-run with --apply to write them.")
        return 0

    os.makedirs(states_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)
    for exp_id, state_path, config_path in planned:
        shutil.copy2(state_path, os.path.join(states_dir, f"{exp_id}.json"))
        if os.path.isfile(config_path):
            shutil.copy2(config_path, os.path.join(configs_dir, f"{exp_id}.yaml"))
    print(f"\nWrote {len(planned)} experiment(s). The old directories are left in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
