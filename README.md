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
overwrite each other, so give each run in a sweep a **codename**:

```bash
python -m src dataset=zulu model_name=zulu_lr1e-5 training.learning_rate=1e-5
# → models/zulu_lr1e-5/
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
