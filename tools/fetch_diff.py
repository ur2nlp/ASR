"""Compare the remote run inventory against the local mirror and emit scp commands.

Reads tab-separated lines from stdin (produced by tools/remote_inventory.sh):
    run_id \t run_dir \t trainer_state_path \t config_path

Mirrors each run into ``outputs/fetched/{run_id}/`` (trainer_state.json +
training_config.yaml) so that tools/training_plot.py can read them locally. Runs
that already finished locally (complete / early_stopped) are never re-fetched;
runs still in progress have only their trainer_state refreshed.

Usage:
    ssh circ 'bash -s' < tools/remote_inventory.sh | python tools/fetch_diff.py --dry-run
    ssh circ 'bash -s' < tools/remote_inventory.sh | python tools/fetch_diff.py | bash
"""

import argparse
import json
import sys
from pathlib import Path


FETCHED_DIR = Path("outputs/fetched")


def get_local_status(trainer_state_path: Path) -> str:
    """Determine a run's status from a local trainer_state.json file."""
    with open(trainer_state_path) as state_file:
        data = json.load(state_file)
    log_history = data.get("log_history") or []
    last = log_history[-1] if log_history else {}
    has_runtime = "train_runtime" in last
    global_step = data.get("global_step", 0)
    max_steps = data.get("max_steps", 0)
    if has_runtime and max_steps and global_step >= max_steps:
        return "complete"
    if has_runtime:
        return "early_stopped"
    return "training"


def main():
    parser = argparse.ArgumentParser(
        description="Diff remote runs against the local mirror and emit scp commands",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without emitting scp commands",
    )
    parser.add_argument(
        "--host",
        default="circ",
        help="SSH host to scp from (default: circ)",
    )
    args = parser.parse_args()

    to_fetch: list[tuple[str, str, str | None]] = []
    skipped = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            print(f"Warning: malformed line: {line}", file=sys.stderr)
            continue
        run_id, _run_dir, remote_trainer_state, remote_config = parts[:4]

        local_trainer_state = FETCHED_DIR / run_id / "trainer_state.json"
        if local_trainer_state.exists():
            try:
                status = get_local_status(local_trainer_state)
            except (json.JSONDecodeError, KeyError):
                status = "unknown"
            if status in ("complete", "early_stopped"):
                skipped += 1
                continue

        local_config = FETCHED_DIR / run_id / "training_config.yaml"
        need_config = remote_config if not local_config.exists() else None
        to_fetch.append((run_id, remote_trainer_state, need_config))

    print(f"# {skipped} skipped (finished locally)", file=sys.stderr)
    print(f"# {len(to_fetch)} to fetch", file=sys.stderr)

    if not to_fetch:
        print("# Nothing to fetch", file=sys.stderr)
        return

    for run_id, remote_trainer_state, remote_config in to_fetch:
        dest_dir = FETCHED_DIR / run_id
        if args.dry_run:
            note = " +config" if remote_config else ""
            print(f"  {run_id}{note}", file=sys.stderr)
            continue
        print(f'mkdir -p "{dest_dir}"')
        print(f'scp "{args.host}:{remote_trainer_state}" "{dest_dir}/trainer_state.json"')
        if remote_config:
            print(f'scp "{args.host}:{remote_config}" "{dest_dir}/training_config.yaml"')


if __name__ == "__main__":
    main()
