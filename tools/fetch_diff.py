"""Compare remote run inventory against local files to decide what to fetch.

Reads tab-separated lines from stdin:
    experiment_id \t dir_name \t trainer_state_path \t config_path

Checks local trainer_states/ and configs/ directories to determine what
already exists. Prints scp commands for runs that need fetching.

The remote host is an ssh destination (a hostname, or an alias from your
ssh config) given by --remote or the ASR_REMOTE environment variable. Nothing
is hardcoded: the host and the remote paths belong to whoever runs this.

Runs are keyed by the `experiment_id` their training_config.yaml declares, not
by their directory path, so the same run inventoried from different -b roots
lands in one place rather than two. Ids are taken from the config verbatim and
are never normalized, so whatever a run declared is what it is filed under.

Usage:
    INV="ssh $ASR_REMOTE 'bash -s' < tools/remote_inventory.sh -- -b /path/to/models"
    eval "$INV" | python tools/fetch_diff.py --dry-run
    eval "$INV" | python tools/fetch_diff.py | bash
"""

from lapt_core.fetch_diff import main


if __name__ == "__main__":
    # No id normalization: ASR ids come from each run's config verbatim and have
    # no hand-typed variants, so padding would rewrite ids the records never use.
    main(env_prefix="ASR", epilog=__doc__)
