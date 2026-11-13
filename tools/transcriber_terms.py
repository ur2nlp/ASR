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

With ``--text label`` the tokens are instead the training labels: each line goes through
``prepare_zulu.clean_text`` (annotations removed) and ``src.preprocessing.normalize_text``
with its defaults, which match ``configs/main.yaml`` (lowercase, punctuation removed except
``'``, NFC), and is split on whitespace. Only differences that reach the model remain:
spelling variants, how fillers are written, elisions. This mode imports ``src`` (and so
``omegaconf``); run it from the project environment.

Because small transcribers mostly own whole sessions, topic and name words would
dominate; a term only counts for a transcriber if it occurs in at least
``--min-sessions`` of that transcriber's sessions.

With ``--char-ngrams N`` the terms are instead within-word character N-grams of the word
tokens, with ``_`` marking the word edges (``_ngi``, ``ampe``, ``mpe_``; never spanning two
words). Overlapping N-grams of one distinctive word would otherwise just repeat the
word-level result, so each N-gram is also scored by how many *different words* its excess
comes from. For transcriber T and each word type w containing the N-gram, T's excess is
``k * (count_T(w) - count_others(w) * tokens_T / tokens_others)`` (k = occurrences of the
N-gram in w); over the words with positive excess, ``effective_words = 1 / sum(p_w ** 2)``
with p_w each word's share of the total. About 1 means a fragment of one word; much more
means a pattern spread over many words that are each too rare to stand out (e.g. one
morpheme spelled differently). The session filter applies to the N-gram, not its words,
so rare words still count. The top N-grams per transcriber are then grouped: each joins
the first earlier (higher-scoring) group whose supporting words -- the fewest words making
up ``--support-share`` of its excess -- overlap its own with Jaccard >= ``--group-jaccard``,
so ``hlam``, ``lamp``, ``ampe`` from one word collapse into one group.

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

from tools.prepare_zulu import TIMESTAMP_LINE, clean_text

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


def label_tokens(line: str) -> list[str]:
    """Words of a raw transcript line as they appear in the training labels."""
    from src.preprocessing import normalize_text

    return normalize_text(clean_text(line)).split()


def read_transcript_tokens(path: str, keep_case: bool, text: str = "raw") -> list[str]:
    """Tokens of every non-timestamp line of a transcript.

    Args:
        path: Transcript path.
        keep_case: Keep case in raw mode.
        text: `raw` (words, annotations, punctuation) or `label` (training labels).
    """
    tokens: list[str] = []
    with open(path, encoding="utf-8") as transcript_file:
        for line in transcript_file:
            stripped = line.strip()
            if not stripped or TIMESTAMP_LINE.match(stripped):
                continue
            if text == "label":
                tokens.extend(label_tokens(stripped))
            else:
                tokens.extend(tokenize(stripped, keep_case))
    return tokens


def collect_counts(
    raw_dir: str,
    metadata_path: str,
    keep_case: bool,
    text: str = "raw",
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
        for token in read_transcript_tokens(path, keep_case, text):
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


WORD = re.compile(r"^[\w'’]+$")


def word_ngrams(word: str, size: int) -> list[str]:
    """Within-word character n-grams of ``_word_`` (``_`` marks the word edges)."""
    padded = f"_{word}_"
    return [padded[start : start + size] for start in range(len(padded) - size + 1)]


def excess_by_word(
    ngram: str,
    words: dict[str, int],
    own_counts: Counter,
    other_counts: Counter,
    own_total: int,
    other_total: int,
) -> dict[str, float]:
    """Each word's positive contribution to a transcriber's excess use of an n-gram.

    Args:
        ngram: The n-gram.
        words: Word types containing it, with the number of occurrences in each.
        own_counts: The transcriber's word counts.
        other_counts: Everyone else's word counts.
        own_total: The transcriber's word tokens.
        other_total: Everyone else's word tokens.

    Returns:
        ``{word: excess}`` for the words whose excess is positive.
    """
    excess = {}
    scale = own_total / other_total if other_total else 0.0
    for word, occurrences in words.items():
        value = occurrences * (own_counts[word] - other_counts[word] * scale)
        if value > 0:
            excess[word] = value
    return excess


def dispersion(excess: dict[str, float]) -> tuple[float, str | None, float]:
    """Effective number of contributing words, the top word, and its share of the excess."""
    total = sum(excess.values())
    if total <= 0:
        return 0.0, None, 0.0
    shares = {word: value / total for word, value in excess.items()}
    top_word = max(shares, key=shares.get)
    effective = 1.0 / sum(share * share for share in shares.values())
    return effective, top_word, shares[top_word]


def supporting_words(excess: dict[str, float], share: float, limit: int = 50) -> set[str]:
    """The fewest top-excess words that together make up `share` of the total excess."""
    total = sum(excess.values())
    chosen: set[str] = set()
    covered = 0.0
    for word, value in sorted(excess.items(), key=lambda item: item[1], reverse=True):
        if covered >= share * total or len(chosen) >= limit:
            break
        chosen.add(word)
        covered += value
    return chosen


def group_by_support(
    ranked: list[tuple[str, set[str]]],
    min_jaccard: float,
) -> list[int]:
    """Assign each n-gram (in rank order) to the first group whose support overlaps enough.

    Args:
        ranked: ``(ngram, supporting words)``, best first.
        min_jaccard: Jaccard overlap with a group's first member needed to join it.

    Returns:
        A group index per n-gram; groups are numbered in order of their best member.
    """
    leaders: list[set[str]] = []
    groups = []
    for _, support in ranked:
        assigned = None
        for index, leader in enumerate(leaders):
            union = support | leader
            if union and len(support & leader) / len(union) >= min_jaccard:
                assigned = index
                break
        if assigned is None:
            leaders.append(support)
            assigned = len(leaders) - 1
        groups.append(assigned)
    return groups


def run_ngrams(
    args: argparse.Namespace,
    counts: dict[str, Counter],
    sessions: dict[str, dict[str, set]],
) -> None:
    """Score within-word character n-grams with dispersion over words, and group them."""
    word_counts = {
        transcriber: Counter(
            {term: count for term, count in transcriber_counts.items() if WORD.match(term)}
        )
        for transcriber, transcriber_counts in counts.items()
    }
    corpus_words: Counter = Counter()
    for transcriber_words in word_counts.values():
        corpus_words.update(transcriber_words)
    corpus_word_total = sum(corpus_words.values())

    # n-gram -> {word type: occurrences of the n-gram in it}
    words_of: dict[str, dict[str, int]] = defaultdict(dict)
    for word in corpus_words:
        for ngram, occurrences in Counter(word_ngrams(word, args.char_ngrams)).items():
            words_of[ngram][word] = occurrences

    ngram_counts: dict[str, Counter] = {}
    ngram_sessions: dict[str, dict[str, set]] = {}
    for transcriber, transcriber_words in word_counts.items():
        transcriber_ngrams: Counter = Counter()
        transcriber_ngram_sessions: dict[str, set] = defaultdict(set)
        for word, count in transcriber_words.items():
            word_sessions = sessions[transcriber][word]
            for ngram, occurrences in Counter(word_ngrams(word, args.char_ngrams)).items():
                transcriber_ngrams[ngram] += occurrences * count
                transcriber_ngram_sessions[ngram] |= word_sessions
        ngram_counts[transcriber] = transcriber_ngrams
        ngram_sessions[transcriber] = transcriber_ngram_sessions

    eligible = {
        transcriber: {
            ngram
            for ngram, ngram_session_set in ngram_sessions[transcriber].items()
            if len(ngram_session_set) >= args.min_sessions
        }
        for transcriber in ngram_counts
    }
    log_odds = log_odds_scores(ngram_counts, eligible, args.prior_strength)

    corpus_ngrams: Counter = Counter()
    for transcriber_ngrams in ngram_counts.values():
        corpus_ngrams.update(transcriber_ngrams)
    corpus_ngram_total = sum(corpus_ngrams.values())

    rows = []
    for transcriber in sorted(ngram_counts):
        own_words = word_counts[transcriber]
        other_words = corpus_words - own_words
        own_word_total = sum(own_words.values())
        other_word_total = corpus_word_total - own_word_total
        own_ngram_total = sum(ngram_counts[transcriber].values())
        other_ngram_total = corpus_ngram_total - own_ngram_total

        ranked = sorted(
            eligible[transcriber], key=lambda ngram: log_odds[transcriber][ngram], reverse=True
        )[: args.top_k]
        supports = []
        details = []
        for ngram in ranked:
            excess = excess_by_word(
                ngram,
                words_of[ngram],
                own_words,
                other_words,
                own_word_total,
                other_word_total,
            )
            effective, top_word, top_share = dispersion(excess)
            supports.append((ngram, supporting_words(excess, args.support_share)))
            details.append((effective, top_word, top_share, len(excess)))
        groups = group_by_support(supports, args.group_jaccard)

        for rank, (ngram, (effective, top_word, top_share, contributing), group) in enumerate(
            zip(ranked, details, groups)
        ):
            own = ngram_counts[transcriber][ngram]
            rest = corpus_ngrams[ngram] - own
            rows.append(
                {
                    "transcriber": transcriber,
                    "rank": rank + 1,
                    "ngram": ngram,
                    "log_odds_z": round(log_odds[transcriber][ngram], 3),
                    "count": own,
                    "sessions": len(ngram_sessions[transcriber][ngram]),
                    "per_10k": round(10000 * own / own_ngram_total, 3),
                    "others_per_10k": round(10000 * rest / other_ngram_total, 3),
                    "word_types": len(words_of[ngram]),
                    "contributing_words": contributing,
                    "effective_words": round(effective, 2),
                    "top_word": top_word,
                    "top_word_share": round(top_share, 3),
                    "group": group + 1,
                    "support": " ".join(sorted(supports[rank][1])[:12]),
                }
            )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for transcriber in sorted(ngram_counts):
        own_rows = [row for row in rows if row["transcriber"] == transcriber]
        print(f"\n== {transcriber}")
        shown = 0
        for group in sorted({row["group"] for row in own_rows}):
            members = [row for row in own_rows if row["group"] == group]
            leader = members[0]
            names = " ".join(row["ngram"] for row in members[:6])
            print(
                f"  [{leader['log_odds_z']:6.1f}] eff={leader['effective_words']:5.1f} "
                f"top={leader['top_word']} ({leader['top_word_share']:.0%})  {names}"
            )
            shown += 1
            if shown >= args.top:
                break
    print(f"\nwrote {len(rows)} rows to {args.output}", file=sys.stderr)


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
    parser.add_argument("--keep-case", action="store_true", help="raw mode only")
    parser.add_argument(
        "--char-ngrams",
        type=int,
        default=None,
        metavar="N",
        help="score within-word character N-grams (with word-dispersion and grouping)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=300,
        help="n-gram mode: best n-grams per transcriber to analyse, group and write",
    )
    parser.add_argument(
        "--support-share",
        type=float,
        default=0.8,
        help="n-gram mode: share of an n-gram's excess its supporting words must cover",
    )
    parser.add_argument(
        "--group-jaccard",
        type=float,
        default=0.5,
        help="n-gram mode: supporting-word overlap needed to join a group",
    )
    parser.add_argument(
        "--text",
        choices=["raw", "label"],
        default="raw",
        help="raw transcript tokens, or the training labels (clean_text + normalize_text)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts, sessions = collect_counts(args.raw_dir, args.metadata, args.keep_case, args.text)
    if not counts:
        sys.exit("no transcripts found")
    if args.char_ngrams is not None:
        run_ngrams(args, counts, sessions)
        return
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
