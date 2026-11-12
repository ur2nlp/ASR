"""Find the terms that distinguish each Zulu transcriber's raw transcripts.

Transcribers differ in conventions (``[uhm]``, ``(re)``, ``...``, spelling, code-switch
handling) as well as in what their recordings are about. This tool scores every term per
transcriber two ways:

- **TF-IDF**, with each transcriber's whole output as one document:
  ``tf = count / tokens``, ``idf = log(D / df)`` over the D transcribers. With only a
  handful of documents, a term every transcriber uses scores 0 however unevenly they use
  it, so TF-IDF finds terms *peculiar* to a few transcribers.
- **Weighted log-odds with an informative Dirichlet prior** (Monroe, Colaresi & Quinn
  2008, "Fightin' Words", Political Analysis 16(4)), each transcriber against all the
  others, reported as a z-score. It finds terms used at different *rates*, shrinking
  rare terms toward the corpus rate, so it also catches shared conventions used much more
  by one transcriber.

Tokens come from the *raw* transcript lines (not ``clean_text`` output, which strips the
annotations): bracketed annotations (``[...]``, ``(...)``, ``<...>``) are single tokens,
words are runs of letters/digits/apostrophes (lowercased unless ``--keep-case``), and
other punctuation runs (``...``, ``?``) are tokens too. Timestamp lines are skipped.

Because small transcribers mostly own whole sessions, topic and name words would
dominate; a term only counts for a transcriber if it occurs in at least
``--min-sessions`` of that transcriber's sessions.

Outputs a CSV (one row per transcriber x term with both scores and rates) and prints
each transcriber's top terms. Standard library only, so it is fine on circ's login node.

Usage:
    python tools/transcriber_terms.py --raw-dir data/zulu/raw \\
        --metadata data/zulu/metadata.csv --output outputs/transcriber_terms.csv
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter, defaultdict

from tools.prepare_zulu import TIMESTAMP_LINE

TOKEN = re.compile(
    r"\[[^\]]*\]"
    r"|\([^)]*\)"
    r"|<[^>]*>"
    r"|[\w'’]+"
    r"|[^\w\s]+"
)
ANNOTATION_START = "[(<"


def tokenize(line: str, keep_case: bool = False) -> list[str]:
    """Split a raw transcript line into annotation, word and punctuation tokens.

    Annotations are kept verbatim (lowercased unless `keep_case`) so that ``[uhm]`` and
    ``[Uhm]`` stay comparable across transcribers by default.
    """
    tokens = TOKEN.findall(line)
    if keep_case:
        return tokens
    return [token.lower() for token in tokens]


def read_transcript_tokens(path: str, keep_case: bool) -> list[str]:
    """Tokens of every non-timestamp line of a transcript."""
    tokens: list[str] = []
    with open(path, encoding="utf-8") as transcript_file:
        for line in transcript_file:
            stripped = line.strip()
            if not stripped or TIMESTAMP_LINE.match(stripped):
                continue
            tokens.extend(tokenize(stripped, keep_case))
    return tokens


def collect_counts(
    raw_dir: str,
    metadata_path: str,
    keep_case: bool,
) -> tuple[dict[str, Counter], dict[str, dict[str, set]]]:
    """Count tokens per transcriber and record which sessions each term occurs in.

    Reads only ``{recording}.txt`` files named in the anonymized metadata.

    Returns:
        ``(counts, sessions)``: ``counts[transcriber][term]`` and
        ``sessions[transcriber][term]`` = set of sessions it occurs in.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    sessions: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    with open(metadata_path, encoding="utf-8", newline="") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    for row in rows:
        if row["language"] == "english":
            continue
        path = os.path.join(raw_dir, f"{row['recording']}.txt")
        if not os.path.isfile(path):
            continue
        transcriber = row["transcriber"]
        for token in read_transcript_tokens(path, keep_case):
            counts[transcriber][token] += 1
            sessions[transcriber][token].add(row["session"])
    return counts, sessions


def tfidf_scores(counts: dict[str, Counter], eligible: dict[str, set]) -> dict[str, dict]:
    """TF-IDF of each eligible term per transcriber, one document per transcriber."""
    num_documents = len(counts)
    document_frequency: Counter = Counter()
    for transcriber_counts in counts.values():
        document_frequency.update(transcriber_counts.keys())
    scores: dict[str, dict] = {}
    for transcriber, transcriber_counts in counts.items():
        total = sum(transcriber_counts.values())
        scores[transcriber] = {
            term: (transcriber_counts[term] / total)
            * math.log(num_documents / document_frequency[term])
            for term in eligible[transcriber]
        }
    return scores


def log_odds_scores(
    counts: dict[str, Counter],
    eligible: dict[str, set],
    prior_strength: float,
) -> dict[str, dict]:
    """Weighted log-odds z-scores of each eligible term, transcriber vs. all others.

    The prior is the corpus-wide term distribution scaled to `prior_strength` pseudo-
    tokens (Monroe et al. 2008, eq. 16-22).
    """
    corpus: Counter = Counter()
    for transcriber_counts in counts.values():
        corpus.update(transcriber_counts)
    corpus_total = sum(corpus.values())
    prior = {term: prior_strength * count / corpus_total for term, count in corpus.items()}

    scores: dict[str, dict] = {}
    for transcriber, transcriber_counts in counts.items():
        own_total = sum(transcriber_counts.values())
        rest_total = corpus_total - own_total
        transcriber_scores = {}
        for term in eligible[transcriber]:
            own = transcriber_counts[term]
            rest = corpus[term] - own
            alpha = prior[term]
            own_log_odds = math.log((own + alpha) / (own_total + prior_strength - own - alpha))
            rest_log_odds = math.log((rest + alpha) / (rest_total + prior_strength - rest - alpha))
            variance = 1.0 / (own + alpha) + 1.0 / (rest + alpha)
            transcriber_scores[term] = (own_log_odds - rest_log_odds) / math.sqrt(variance)
        scores[transcriber] = transcriber_scores
    return scores


def is_annotation(term: str) -> bool:
    return term[:1] in ANNOTATION_START


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", default="data/zulu/raw")
    parser.add_argument("--metadata", default="data/zulu/metadata.csv")
    parser.add_argument("--output", default="outputs/transcriber_terms.csv")
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=3,
        help="a term counts for a transcriber only if it occurs in this many of its sessions",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=10000.0,
        help="pseudo-tokens in the log-odds Dirichlet prior (larger shrinks rare terms more)",
    )
    parser.add_argument("--top", type=int, default=20, help="terms to print per transcriber")
    parser.add_argument("--keep-case", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts, sessions = collect_counts(args.raw_dir, args.metadata, args.keep_case)
    if not counts:
        sys.exit("no transcripts found")
    eligible = {
        transcriber: {
            term
            for term, term_sessions in sessions[transcriber].items()
            if len(term_sessions) >= args.min_sessions
        }
        for transcriber in counts
    }
    tfidf = tfidf_scores(counts, eligible)
    log_odds = log_odds_scores(counts, eligible, args.prior_strength)

    corpus: Counter = Counter()
    for transcriber_counts in counts.values():
        corpus.update(transcriber_counts)
    corpus_total = sum(corpus.values())

    rows = []
    for transcriber in sorted(counts):
        own_total = sum(counts[transcriber].values())
        rest_total = corpus_total - own_total
        for term in eligible[transcriber]:
            own = counts[transcriber][term]
            rest = corpus[term] - own
            rows.append(
                {
                    "transcriber": transcriber,
                    "term": term,
                    "annotation": is_annotation(term),
                    "count": own,
                    "sessions": len(sessions[transcriber][term]),
                    "per_10k": round(10000 * own / own_total, 3),
                    "others_per_10k": round(10000 * rest / rest_total, 3) if rest_total else None,
                    "transcribers_using": sum(term in other for other in counts.values()),
                    "tfidf": tfidf[transcriber][term],
                    "log_odds_z": round(log_odds[transcriber][term], 3),
                }
            )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for transcriber in sorted(counts):
        own_rows = [row for row in rows if row["transcriber"] == transcriber]
        tokens = sum(counts[transcriber].values())
        print(f"\n== {transcriber} ({tokens:,} tokens)")
        for label, key in (("TF-IDF", "tfidf"), ("log-odds z", "log_odds_z")):
            top = sorted(own_rows, key=lambda row: row[key], reverse=True)[: args.top]
            terms = ", ".join(
                f"{row['term']} ({row['per_10k']:g} vs {row['others_per_10k']:g})" for row in top
            )
            print(f"  {label}: {terms}")
    print(f"\nwrote {len(rows)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
