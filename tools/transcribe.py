"""Interactively transcribe individual audio files with a trained CTC model.

The ASR analog of a REPL: point it at a checkpoint, then type audio file paths
one at a time to hear back the transcription. Decoding parameters (greedy vs
KenLM beam search, beam width, LM weights) can be toggled live so you can spot
-check a model without re-launching. For batch transcription to JSONL use
`tools/decode.py`; for corpus WER/CER use `tools/eval.py`.

Usage:
    # Interactive session (greedy decoding)
    python -m tools.transcribe --model_dir models/zulu/xlsr300m_l40_basic/best-checkpoint

    # Start with KenLM decoding enabled
    python -m tools.transcribe --model_dir <dir> --lm_arpa lm/zulu_3gram.arpa

    # One-shot: transcribe a single file and exit
    python -m tools.transcribe --model_dir <dir> --audio utt001.wav

Interactive commands:
    <path>            transcribe the audio file at <path>
    /lm <arpa_path>   enable KenLM beam-search decoding with the given ARPA
    /nolm             switch back to greedy (argmax) decoding
    /alpha <float>    set the LM weight (rebuilds the decoder)
    /beta <float>     set the word-insertion bonus (rebuilds the decoder)
    /beam <int>       set the beam width
    /help             show this help
    /quit, /exit      leave the session (Ctrl-D also works)
"""

import argparse
import sys

import librosa
import torch
from transformers import AutoModelForCTC, AutoProcessor

from src.decoding import build_processor_with_lm, decode_logits


TARGET_SAMPLING_RATE = 16000


def resolve_device(requested: str) -> torch.device:
    """Resolve a device string, auto-detecting CUDA/MPS when requested."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Transcriber:
    """Holds model/processor state and current decoding configuration."""

    def __init__(
        self,
        model_dir: str,
        device: torch.device,
        lm_arpa: str | None = None,
        alpha: float = 0.5,
        beta: float = 1.5,
        beam_width: int = 100,
    ) -> None:
        print(f"Loading model and processor from {model_dir}", file=sys.stderr)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForCTC.from_pretrained(model_dir).to(device)
        self.model.eval()
        self.device = device

        self.lm_arpa = lm_arpa
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width
        self.processor_with_lm = None
        if lm_arpa is not None:
            self._rebuild_lm()

    def _rebuild_lm(self) -> None:
        """Rebuild the LM decoder from the current LM settings."""
        self.processor_with_lm = build_processor_with_lm(
            self.processor,
            arpa_path=self.lm_arpa,
            alpha=self.alpha,
            beta=self.beta,
            beam_width=self.beam_width,
        )

    def enable_lm(self, arpa_path: str) -> None:
        """Enable KenLM beam-search decoding with the given ARPA file."""
        self.lm_arpa = arpa_path
        self._rebuild_lm()

    def disable_lm(self) -> None:
        """Switch back to greedy (argmax) decoding."""
        self.lm_arpa = None
        self.processor_with_lm = None

    def transcribe(self, audio_path: str) -> str:
        """Transcribe one audio file with the current decoding configuration."""
        audio, _ = librosa.load(audio_path, sr=TARGET_SAMPLING_RATE, mono=True)

        # the feature extractor emits input_values (wav2vec2/HuBERT) or
        # input_features (W2V-BERT); both flow through as model kwargs
        inputs = self.processor.feature_extractor(
            audio,
            sampling_rate=TARGET_SAMPLING_RATE,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        if self.processor_with_lm is not None:
            predictions = decode_logits(
                logits.cpu().float().numpy(),
                self.processor_with_lm,
                beam_width=self.beam_width,
            )
            return predictions[0]

        predicted_ids = logits.argmax(dim=-1)
        return self.processor.tokenizer.batch_decode(predicted_ids, group_tokens=True)[0]

    def describe_config(self) -> str:
        """Return a short human-readable summary of the decoding config."""
        if self.processor_with_lm is not None:
            return (
                f"LM decoding (arpa={self.lm_arpa}, alpha={self.alpha}, "
                f"beta={self.beta}, beam={self.beam_width})"
            )
        return "greedy decoding"


HELP_TEXT = """\
Commands:
  <path>            transcribe the audio file at <path>
  /lm <arpa_path>   enable KenLM beam-search decoding
  /nolm             switch back to greedy decoding
  /alpha <float>    set the LM weight (rebuilds the decoder)
  /beta <float>     set the word-insertion bonus (rebuilds the decoder)
  /beam <int>       set the beam width
  /help             show this help
  /quit, /exit      leave the session"""


def handle_command(transcriber: Transcriber, line: str) -> bool:
    """Handle a slash command. Returns False if the session should end."""
    parts = line.split()
    command = parts[0]
    argument = parts[1] if len(parts) > 1 else None

    if command in ("/quit", "/exit"):
        return False

    if command == "/help":
        print(HELP_TEXT, file=sys.stderr)
    elif command == "/lm":
        if argument is None:
            print("Usage: /lm <arpa_path>", file=sys.stderr)
        else:
            transcriber.enable_lm(argument)
    elif command == "/nolm":
        transcriber.disable_lm()
        print("Switched to greedy decoding", file=sys.stderr)
    elif command in ("/alpha", "/beta", "/beam"):
        if argument is None:
            print(f"Usage: {command} <value>", file=sys.stderr)
        else:
            _set_lm_param(transcriber, command, argument)
    else:
        print(f"Unknown command: {command} (try /help)", file=sys.stderr)

    return True


def _set_lm_param(transcriber: Transcriber, command: str, argument: str) -> None:
    """Update a decoding parameter and rebuild the LM decoder if needed."""
    if command == "/beam":
        transcriber.beam_width = int(argument)
    elif command == "/alpha":
        transcriber.alpha = float(argument)
    elif command == "/beta":
        transcriber.beta = float(argument)

    if transcriber.lm_arpa is not None:
        transcriber._rebuild_lm()
    print(f"Config: {transcriber.describe_config()}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactively transcribe audio files")
    parser.add_argument("--model_dir", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--audio",
        default=None,
        help="Transcribe a single file and exit (non-interactive)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--lm_arpa", default=None, help="Path to a KenLM ARPA file")
    parser.add_argument("--alpha", type=float, default=0.5, help="LM weight")
    parser.add_argument("--beta", type=float, default=1.5, help="Word-insertion bonus")
    parser.add_argument("--beam_width", type=int, default=100, help="Beam width")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}", file=sys.stderr)

    transcriber = Transcriber(
        model_dir=args.model_dir,
        device=device,
        lm_arpa=args.lm_arpa,
        alpha=args.alpha,
        beta=args.beta,
        beam_width=args.beam_width,
    )

    # one-shot mode
    if args.audio is not None:
        print(transcriber.transcribe(args.audio))
        return

    print(f"Ready ({transcriber.describe_config()}). Type /help for commands.", file=sys.stderr)
    while True:
        try:
            line = input("audio> ").strip()
        except EOFError:
            print(file=sys.stderr)
            break

        if not line:
            continue

        if line.startswith("/"):
            if not handle_command(transcriber, line):
                break
            continue

        try:
            print(transcriber.transcribe(line))
        except FileNotFoundError:
            print(f"File not found: {line}", file=sys.stderr)
        except Exception as error:
            print(f"Error transcribing {line}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
