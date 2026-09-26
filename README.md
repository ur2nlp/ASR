# ASR Adaptation Framework

Config-driven framework for adapting pretrained speech models to a new
language. Fork it, point a config at your audio, train.

Two objectives are supported and you get whichever the model config names:
**CTC** (wav2vec2, HuBERT, w2v-BERT), which builds a character vocabulary from
your transcripts, and **sequence-to-sequence** (Whisper), which uses the
checkpoint's own tokenizer.

## Setup

```bash
conda env create -f environment.yml
conda activate asr
```

## 1. Point a dataset config at your data

The common case is a local corpus of matched audio/transcript files: each audio
file has a sibling text file of the same stem (`utt001.wav` ↔ `utt001.txt`).
Copy the template and edit `id` and `path` — the name you give the file is what
you will type as `dataset=`, and `zulu` below is just this guide's example.

```bash
cp configs/dataset/example_paired.yaml configs/dataset/zulu.yaml
```

```yaml
# configs/dataset/zulu.yaml
type: paired
id: zulu                  # used in cache + output directory paths
path: data/zulu/raw       # directory of .wav/.txt pairs (relative to repo root)
audio_ext: .wav           # change for .flac/.mp3/.ogg corpora
transcript_ext: .txt
recursive: false          # true if pairs live in nested (e.g. per-speaker) subfolders
```

The directory may be flat (auto-split into train/dev via `dataset.dev_size`) or
contain `train/`, `dev/`, `test/` subdirectories to use a fixed split. Shared
fields (`cache_dir`, `audio_column`, `text_column`, `dev_size`,
`max_audio_length_seconds`) inherit their defaults from `configs/main.yaml`.

Other `type`s: `huggingface` (a Hub dataset), `audiofolder` (HF's audio-folder
loader), and `concat` (combine several sources).

## 2. Pick a model

A model config is a checkpoint plus the architecture it belongs to:

```yaml
# configs/model/whisper-small.yaml
type: whisper                          # architecture: selects the code path
pretrained_name: openai/whisper-small  # any checkpoint of that architecture
short_name: whisper-small              # what appears in output paths
```

**Any checkpoint of a supported architecture is a config file, not a code
change.** To train `whisper-medium` instead of `whisper-small`, copy the config
and edit two lines:

```bash
cp configs/model/whisper-small.yaml configs/model/whisper-medium.yaml
# pretrained_name: openai/whisper-medium
# short_name: whisper-medium
```

The same goes for any wav2vec2, HuBERT or w2v-BERT checkpoint on the Hub — a
community XLS-R fine-tune, an MMS checkpoint, or one of your own earlier runs.

Four architectures are wired up, each shipping with one config as a starting
point:

| `type` | objective | ships as | vocabulary |
|---|---|---|---|
| `wav2vec2` | CTC | `xls-r` — `facebook/wav2vec2-xls-r-300m` | built from your transcripts |
| `hubert` | CTC | `hubert-large` — `facebook/hubert-large-ll60k` | built from your transcripts |
| `w2v_bert` | CTC | `w2vbert2` — `facebook/w2v-bert-2.0` | built from your transcripts |
| `whisper` | seq2seq | `whisper-small` — `openai/whisper-small` | the checkpoint's own |

A *fifth* architecture is a code change — see [Contributing back](#contributing-back).

Whisper configs carry one field with no safe default: `language`, the token
prefixed to every transcript. Whisper only has tokens for its 100 pretraining
languages, so if yours is absent, name a close proxy whose token you are willing
to repurpose. Leaving it null fails fast at startup and explains why.

## 3. Train

```bash
python -m src dataset=zulu model=xls-r training=ctc-basic
```

`ctc-basic` and `whisper-basic` are tuned single-GPU presets (effective batch 60
on an L40S). Override anything on the command line:

```bash
python -m src dataset=zulu model=w2vbert2 training.learning_rate=1e-5
```

Checkpoints land in `models/{id}/{model_short_name}_{training_name}/`, with the
best under `best-checkpoint/`. Two runs sharing a model and training config
would overwrite each other, so give each run in a sweep an **experiment id**,
appended to that path:

```bash
python -m src dataset=zulu experiment_id=lr1e-5 training.learning_rate=1e-5
# → models/zulu/xlsr300m_ctc-basic_lr1e-5/
```

> **macOS note:** CTC loss has no native MPS kernel. To run locally on Apple
> Silicon, force the CPU fallback: `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src ...`.
> Linux/CUDA hardware needs no such flag.

### Optional: replace the vocabulary (FOCUS)

A multilingual checkpoint's tokenizer often splits an unseen language into far
too many pieces. `focus=basic` trains a fresh SentencePiece vocabulary over your
transcripts and initializes its embeddings with FOCUS ([Dobler & de Melo,
2023](https://arxiv.org/abs/2305.14481)): shared tokens keep their pretrained
vectors, and the rest become weighted combinations of those.

```bash
python -m src dataset=zulu model=whisper-small training=whisper-basic focus=basic
```

Seq2seq only. A CTC model already derives its vocabulary from your transcripts
and errors if you enable this. The learned vocabulary size is `focus.vocab_size`;
the checkpoint's special-token block is kept on top of it.

### Optional: extra held-out eval sets

To track additional test sets (dialects, recording conditions) alongside the
automatic dev split, point an `external_eval` config at directories of
`.wav`/`.txt` pairs. Each is reported as `eval_{name}_wer` / `eval_{name}_cer`;
the dev set still drives best-checkpoint selection.

```bash
cp configs/external_eval/example.yaml configs/external_eval/zulu_conditions.yaml
python -m src dataset=zulu external_eval=zulu_conditions
```

## 4. Evaluate and decode

`--test_data` is a directory `datasets.load_from_disk` can read — a held-out
split you saved yourself, not one the training pipeline emits.

```bash
python -m tools.eval \
    --model_dir models/zulu/xlsr300m_ctc-basic/best-checkpoint \
    --test_data data/zulu/test
```

Plot metric curves from a run's `trainer_state.json`, spot-check files
interactively, or decode a directory to JSONL:

```bash
python -m tools.training_plot --metric eval_wer \
    --state-file models/zulu/xlsr300m_ctc-basic/best-checkpoint/trainer_state.json

python -m tools.transcribe --model_dir models/zulu/xlsr300m_ctc-basic/best-checkpoint

python -m tools.decode \
    --model_dir models/zulu/xlsr300m_ctc-basic/best-checkpoint \
    --audio_dir path/to/audio/ \
    --output predictions.jsonl
```

### Optional: an n-gram LM for CTC decoding

```bash
python -m tools.train_lm \
    --text_source data/zulu/untokenized \
    --order 3 \
    --output lm/zulu_3gram.arpa

python -m tools.eval \
    --model_dir models/zulu/xlsr300m_ctc-basic/best-checkpoint \
    --test_data data/zulu/test \
    --lm_arpa lm/zulu_3gram.arpa
```

## Experiment tracking

Runs are keyed by **experiment id** — the `experiment_id` from §3, which names
both the output directory and every record below.

Three files per run live under `outputs/`:

| path | holds |
|---|---|
| `outputs/configs/{id}.yaml` | the config the run was launched with |
| `outputs/trainer_states/{id}.json` | the metric history HuggingFace wrote |
| `outputs/registry.yaml` | one row per run: extracted params plus your notes |

### Naming runs and configs

A run's checkpoint directory is assembled from four config values:

```
models/{dataset.id}/{model.short_name}_{training.name}[_{focus vocab}]_{experiment_id}
```

Each is a knob, so they compound. Three habits keep the result readable.

**Don't restate an outer level.** `dataset.id` is already a directory, so a
corpus name inside `model.short_name` or `training.name` is paid for twice.

This is where the config *file* name and the name in the path come apart, and
they are already separate fields. The filename is what you type
(`training=<your_preset>`) and should say what the preset is for; `name:` is
what lands in the path, where the corpus is overhead:

```yaml
# configs/training/zulu-whisper.yaml   <- descriptive, for the command line
name: whisper                          <- terse, for the path
```

**Keep `_` for joints and `-` inside components.** The path joins components
with `_`, so a component containing `_` hides its own boundaries:
`whisper_small_zu_zulu_whisper` has no visible seams, while
`whisper-small_zulu-whisper` does.

**Let `experiment_id` name what varies** — `lr1e-5`, `bs64`, `frozen-encoder` —
rather than repeating the model, which is already two segments to its left.

Applied together:

```
zulu/whisper-medium-zu_zulu_whisper_focus-v4k-whisper-medium-zu_whisper-medium1
zulu/whisper-medium_whisper_focus-v4k_lr1e-5
```

The FOCUS segment shortened on its own: the run path uses the vocabulary id
(`focus-v4k`) rather than the full tokenizer id, which ends with the model slug
`model.short_name` already contributed.

Renaming these changes where a run writes, so do it between runs —
`preempt_resume` finds checkpoints by path. The cost differs sharply by value:

| value | what a rename costs |
| --- | --- |
| `training.name` | nothing; it appears only in the run directory |
| `experiment_id` | rename `outputs/configs/{id}.yaml`, `outputs/trainer_states/{id}.json` and the registry row |
| `model.short_name` | moves the processed and FOCUS caches — use `tools/migrate_model_short_name.py` |
| `dataset.id` | relocates the whole `data/{id}` cache tree; not worth it |

```bash
python -m tools.migrate_model_short_name <old_short_name> <new_short_name>          # plan
python -m tools.migrate_model_short_name <old_short_name> <new_short_name> --apply
```

It renames both cache directories and rewrites the `focus_tokenizer_id` the
processed cache records, which a plain `mv` would leave pointing at the old
model. Reversible by running with the names swapped.

A separator-only change needs no migration: `_slugify` normalises `_` to `-`, so
`whisper_small_zu` and `whisper-small-zu` produce the same cache directories and
only the run directory name differs.

### Pulling runs off a cluster

`fetch_results.sh` inventories the remote, works out what is missing or stale,
and copies only that. No host or path is baked in, so set them in your shell:

```bash
export ASR_REMOTE=<ssh_host_or_alias>          # e.g. a Host entry in ~/.ssh/config
export ASR_MODEL_DIRS=<remote_models_dir>      # colon-separated for several roots

bash scripts/fetch_results.sh
```

Runs are keyed by the `experiment_id` inside each `training_config.yaml`, not by
directory path, so the same run inventoried from different roots lands in one
place. Finished runs are never re-fetched; an in-progress run has only its
trainer state refreshed, since a config cannot change mid-run.

To see what it would do without moving anything:

```bash
INV="ssh $ASR_REMOTE 'bash -s' < tools/remote_inventory.sh -- -b $ASR_MODEL_DIRS"
eval "$INV" | python -m tools.fetch_diff --dry-run
```

### Registering and annotating runs

`extract` reads configs and upserts a row per run. Safe to re-run; it updates
rather than duplicates.

```bash
python -m tools.registry extract outputs/configs/<your_run_id>.yaml

# --pattern takes a regex over paths; this takes every id starting with 'whisper'
python -m tools.registry extract --pattern 'outputs/configs/whisper.*\.yaml'
```

Parameters come from the config's `model`, `training`, `dataset` and `focus`
sections automatically — including `model.language`, which matters when the
config names a deliberate proxy. What only you can supply is why the run existed
and what it showed:

```bash
python -m tools.registry annotate <your_run_id> \
    --note "whisper-small, FOCUS 4k, lr 1e-5, effective batch 60" \
    --observation "WER plateaus ~step 12k; dev loss still falling" \
    --era <your_era> --group <your_group>
```

`--status manually_closed` retires a run, which also stops `fetch_results.sh`
re-fetching it.

### Reading the registry

```bash
python -m tools.registry show                       # everything
python -m tools.registry show --era <your_era> --group <your_group>
python -m tools.registry diff whisper5 whisper9     # only what differs
python -m tools.registry verify                     # rows still match outputs/configs/
python -m tools.registry debt                       # runs on disk with no row, rows with no note
```

`diff` is the one to reach for when comparing a sweep — it prints only the
parameters that vary and lists the rest as constant, so a forty-field config
collapses to the handful you actually changed. Ids below are from this fork's
own sweep, as an illustration of the output:

```
Run       effective_batch  vocab_size  mask_time_prob
whisper5                2        4096            0.05
whisper9                4        2048            0.07
```

`debt --strict` exits non-zero, which makes it usable as a pre-commit or CI
check that no run went un-annotated.

### Plotting

`training_plot.py` reads trainer states directly — no registry required. It
works on a fetched mirror as well as the local `models/` tree from §4.

```bash
# one run, several metrics
python -m tools.training_plot --metrics loss eval_wer eval_cer \
    --state-file outputs/trainer_states/<your_run_id>.json

# compare runs; --state-pattern is a regex over paths (ids here are examples)
python -m tools.training_plot --metric eval_wer \
    --state-pattern "outputs/trainer_states/whisper(5|9)\.json"

# discover what a run actually logged
python -m tools.training_plot --list-metrics --state-file outputs/trainer_states/<your_run_id>.json
```

Metric names are regexes, so with external eval sets `--metric "eval_.*_wer"`
draws every per-set WER series at once.

Useful when the defaults fight you:

| flag | does |
|---|---|
| `--output plot.png` | save instead of opening a window |
| `--ylim 0 1` | shared y-limits across panels |
| `--ylims eval_wer:0:0.8` | per-metric limits; repeatable, wins over `--ylim` |
| `--run-names ctc whisper` | legend labels instead of file paths |
| `--exclude-pattern` | drop runs the state pattern swept up |
| `--x-axis epoch` | plot against epochs rather than steps |
| `--dark` | light-on-dark, for slides |

## Testing

```bash
pytest tests/
```

## Project structure

```
src/
├── __main__.py         # Hydra entry point, training orchestration
├── sources/            # One module per dataset type, registered by its `type`
├── data.py             # Dataset dispatcher, feature extraction, caching
├── processors.py       # ModelSpec registry, processor construction
├── models.py           # Model setup (dropout, freezing, vocab resize)
├── focus.py            # FOCUS vocabulary replacement and embedding init
├── vocab.py            # Character vocab + CTC tokenizer
├── preprocessing.py    # Text normalization
├── collators.py        # Batch padding for CTC and seq2seq
├── metrics.py          # WER/CER computation
├── callbacks.py        # Freeze/unfreeze, early stopping
├── decoding.py         # pyctcdecode + KenLM
└── artifact_configs.py # Config tracking for reproducibility
```

The caching layer under `artifact_configs.py` comes from
[`lapt-core`](https://github.com/ur2nlp/LAPT/tree/main/packages/lapt-core),
shared with a sibling project rather than reimplemented here.

## Contributing back

New checkpoints need nothing from this repo — they are a config file, as in §2.

A new *architecture* does: a `ModelSpec` entry in `src/processors.py` naming its
model, feature-extractor and collator classes, its input column and its
objective; a config in `configs/model/`; and sometimes an architecture-specific
quirk handled in `src/models.py`.

If you work one out, please open a PR. That is the contribution this repo most
wants upstream — engineering the next fork would otherwise repeat.
