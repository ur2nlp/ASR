# ASR Adaptation Framework

Modular, config-driven framework for fine-tuning pre-trained CTC-based ASR
models on new languages. Fork the repo, point a config at your data, and train.

## Supported Models

| Config (`model=`) | HuggingFace checkpoint |
|---|---|
| `xls-r` | `facebook/wav2vec2-xls-r-300m` (default) |
| `hubert-large` | `facebook/hubert-large-ll60k` |
| `w2vbert2` | `facebook/w2v-bert-2.0` |

## Setup

```bash
conda env create -f environment.yml
conda activate asr
```

## Usage

### 1. Point a dataset config at your data

The common case is a local corpus of matched audio/transcript files: each
audio file has a sibling text file of the same stem holding its transcription
(`utt001.wav` ↔ `utt001.txt`). Copy the template and edit `id` and `path`:

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

The directory may be flat (auto-split into train/dev via `dataset.dev_size`)
or contain `train/`, `dev/`, `test/` subdirectories to use a fixed split.
Shared fields (`cache_dir`, `audio_column`, `text_column`, `dev_size`,
`max_audio_length_seconds`) inherit their defaults from `configs/main.yaml`.

Other dataset `type`s are also supported: `huggingface` (a Hub dataset),
`audiofolder` (HF's audio-folder loader), and `concat` (combine sources).

### 2. Train

```bash
python -m src dataset=zulu model=xls-r training=l40_basic
```

Override any config value on the command line:

```bash
python -m src dataset=zulu model=w2vbert2 training.learning_rate=1e-5
```

Checkpoints are written to `models/{id}/{model_short_name}_{training_name}/`
(e.g. `models/zulu/xlsr300m_l40_basic/`), with the best checkpoint under
`best-checkpoint/`. Two runs with the same model and training config would
overwrite each other, so give each run in a sweep an **experiment id**, which is
appended as a suffix to that path:

```bash
python -m src dataset=zulu experiment_id=lr1e-5 training.learning_rate=1e-5
# → models/zulu/xlsr300m_l40_basic_lr1e-5/
```

> **macOS note:** CTC loss has no native MPS kernel. To run locally on Apple
> Silicon, force the CPU fallback: `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src ...`.
> The lab's Linux/CUDA hardware needs no such flag.

### Monitor extra held-out eval sets during training (optional)

To track performance on additional test sets (e.g. multiple dialects or
recording conditions) alongside the automatic dev split, point an
`external_eval` config at directories of `.wav`/`.txt` pairs. Each set is
reported during training as `eval_{name}_wer` / `eval_{name}_cer`; the dev set
still drives best-checkpoint selection.

```bash
cp configs/external_eval/example.yaml configs/external_eval/zulu_conditions.yaml
python -m src dataset=zulu external_eval=zulu_conditions
```

### 3. Evaluate

```bash
python -m tools.eval \
    --model_dir models/zulu/xlsr300m_l40_basic/best-checkpoint \
    --test_data data/zulu/processed/test
```

Plot metric curves (WER/CER/loss) from a run's `trainer_state.json`:

```bash
python -m tools.training_plot --metric eval_wer \
    --state-file models/zulu/xlsr300m_l40_basic/best-checkpoint/trainer_state.json
```

Spot-check individual files interactively (type audio paths, toggle LM decoding
live):

```bash
python -m tools.transcribe --model_dir models/zulu/xlsr300m_l40_basic/best-checkpoint
```

### 4. Decode audio files

```bash
python -m tools.decode \
    --model_dir models/zulu/xlsr300m_l40_basic/best-checkpoint \
    --audio_dir path/to/audio/ \
    --output predictions.jsonl
```

### 5. Train a language model (optional)

```bash
python -m tools.train_lm \
    --text_source data/zulu/untokenized \
    --order 3 \
    --output lm/zulu_3gram.arpa
```

Then boost decoding with it during eval:

```bash
python -m tools.eval \
    --model_dir models/zulu/xlsr300m_l40_basic/best-checkpoint \
    --test_data data/zulu/processed/test \
    --lm_arpa lm/zulu_3gram.arpa
```

## Experiment Tracking

Runs are tracked by **experiment id** — the `experiment_id` from §2, which names
both the output directory and every record below.

```bash
python -m src dataset=<your_dataset> experiment_id=<your_run_id> training.learning_rate=1e-5
```

Three files per run live under `outputs/`:

| path | holds |
|---|---|
| `outputs/configs/{id}.yaml` | the config the run was launched with |
| `outputs/trainer_states/{id}.json` | the metric history HuggingFace wrote |
| `outputs/registry.yaml` | one row per run: extracted params plus your notes |

### Pulling runs off a cluster

`fetch_results.sh` inventories the remote, works out what is missing or stale,
and copies only that. No host or path is baked into the repository, so set them
in your shell:

```bash
export ASR_REMOTE=<ssh_host_or_alias>          # e.g. a Host entry in ~/.ssh/config
export ASR_MODEL_DIRS=<remote_models_dir>      # colon-separated for several roots
```

```bash
bash scripts/fetch_results.sh
```

Runs are keyed by the `experiment_id` inside each `training_config.yaml`, not by
their directory path, so the same run inventoried from different roots lands in
one place. Finished runs are never re-fetched; an in-progress run has only its
trainer state refreshed, since a config cannot change mid-run.

To see what it would do without moving anything:

```bash
INV="ssh $ASR_REMOTE 'bash -s' < tools/remote_inventory.sh -- -b $ASR_MODEL_DIRS"
eval "$INV" | python -m tools.fetch_diff --dry-run
```

### Registering and annotating runs

`extract` reads configs and upserts a row per run. It is safe to re-run; it
updates rather than duplicates.

```bash
python -m tools.registry extract outputs/configs/<your_run_id>.yaml

# --pattern takes a regex over paths; this takes every id starting with 'whisper'
python -m tools.registry extract --pattern 'outputs/configs/whisper.*\.yaml'
```

Parameters are pulled from the config's `model`, `training`, `dataset` and
`focus` sections automatically — including `model.language`, which matters here
because Whisper has no Zulu token and the config names a deliberate proxy. What
only you can supply is why the run existed and what it showed:

```bash
python -m tools.registry annotate <your_run_id> \
    --note "whisper-small, FOCUS 4k, lr 1e-5, effective batch 32" \
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
collapses to the handful you actually changed. Run ids below are from this
fork's own sweep, as an illustration of the output:

```
Run       effective_batch  vocab_size  mask_time_prob
whisper5                2        4096            0.05
whisper9                4        2048            0.07
```

`debt --strict` exits non-zero, which makes it usable as a pre-commit or CI check
that no run went un-annotated.

### Plotting

`training_plot.py` reads trainer states directly — no registry required. It works
on a fetched mirror as well as on the local `models/` tree shown in §3.

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

Metric names are regexes, so with external eval sets (§2) `--metric "eval_.*_wer"`
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

## Project Structure

```
src/
├── __main__.py         # Hydra entry point, training orchestration
├── models.py           # Model setup (dropout, freezing, vocab resize)
├── data.py             # Dataset dispatcher + collator
├── processors.py       # ModelSpec registry, processor construction
├── vocab.py            # Character vocab + CTC tokenizer
├── preprocessing.py    # Text normalization
├── metrics.py          # WER/CER computation
├── callbacks.py        # Freeze/unfreeze, early stopping
├── decoding.py         # pyctcdecode + KenLM
└── artifact_configs.py # Config tracking for reproducibility
```

## Adding a New Model Type

1. Add a `ModelSpec` entry in `src/processors.py`
2. Add a config in `configs/model/`
3. Handle any architecture-specific quirks in `src/models.py`
