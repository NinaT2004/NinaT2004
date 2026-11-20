#!/usr/bin/env python3
"""HCR streaming keyword spotting event node (Windows / Jetson).

No motor commands are issued. The robot's controller owns task state and must
explicitly update this node's state. Voice STOP is not a hardware E-stop.

Examples:
  python voice_node.py --simulate
  python voice_node.py --list-devices
  python voice_node.py --device 1
  python voice_node.py --device reSpeaker --debug-all-keywords

Console commands:
  :state IDLE|LISTENING|EXECUTING|PAUSED
  :help
  :quit
"""

import argparse
import json
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
KEYWORDS_FILE = BASE_DIR / "keywords.txt"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1600

STATES = {"IDLE", "LISTENING", "EXECUTING", "PAUSED"}
WAKE_WORDS = {"NAILONG", "MILK_DRAGON"}
END_WORDS = {"FINISH", "THANK_YOU"}


def log(message):
    print(message, file=sys.stderr, flush=True)


def emit(event, keyword, state):
    print(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "voice_kws",
        "event": event,
        "keyword": keyword,
        "state": state,
    }, ensure_ascii=False), flush=True)


class VoiceController:
    """Application-level filtering; does NOT change KWS model sensitivity."""

    def __init__(self, cooldown_seconds=1.0, debug_all=False):
        self.state = "IDLE"
        self.cooldown_seconds = cooldown_seconds
        self.debug_all = debug_all
        self._last_accepted = {}  # (state, keyword) -> monotonic timestamp

    def set_state(self, new_state):
        new_state = new_state.strip().upper()
        if new_state not in STATES:
            raise ValueError(f"Invalid state: {new_state}")
        if new_state != self.state:
            self.state = new_state
            self._last_accepted.clear()
            log(f"[STATE] {self.state} (externally set; not robot-confirmed)")

    def _event_for(self, keyword):
        if self.state == "IDLE":
            return "WAKE" if keyword in WAKE_WORDS else None
        if self.state == "LISTENING":
            if keyword == "STOP":
                return "CANCEL_INTERACTION_REQUEST"
            if keyword == "OKAY":
                return "CONFIRM"
            if keyword in END_WORDS:
                return "END_INTERACTION_REQUEST"
            return None
        if self.state == "EXECUTING":
            return "STOP_REQUEST" if keyword == "STOP" else None
        if self.state == "PAUSED":
            if keyword == "OKAY":
                return "RESUME_REQUEST"
            if keyword in END_WORDS:
                return "CANCEL_TASK_REQUEST"
        return None

    def on_keyword(self, keyword):
        keyword = keyword.strip().upper().replace(" ", "_")
        if not keyword:
            return

        event = self._event_for(keyword)
        if event is None:
            if self.debug_all:
                log(f"[IGNORED] {keyword} (state={self.state})")
            return

        # Debounce repeated accepted detections; never debounce STOP requests.
        now = time.monotonic()
        key = (self.state, keyword)
        if keyword != "STOP" and now - self._last_accepted.get(key, -1e30) < self.cooldown_seconds:
            if self.debug_all:
                log(f"[COOLDOWN] {keyword} (state={self.state})")
            return
        self._last_accepted[key] = now

        log(f"[KWS] {keyword} (state={self.state})")
        if event == "WAKE":
            self.state = "LISTENING"
            self._last_accepted.clear()
            log("[STATE] LISTENING")
        emit(event, keyword, self.state)


def choose_device(sd, requested, samplerate):
    devices = sd.query_devices()
    if requested is None:
        index = sd.default.device[0]
    elif str(requested).isdigit():
        index = int(requested)
    else:
        matched = [i for i, d in enumerate(devices)
                   if d["max_input_channels"] > 0 and requested.lower() in d["name"].lower()]
        if len(matched) != 1:
            raise RuntimeError(f"Input device match for {requested!r}: {matched}; specify its numeric index")
        index = matched[0]
    if index is None or index < 0 or index >= len(devices) or devices[index]["max_input_channels"] < 1:
        raise RuntimeError(f"Invalid input device: {index}")
    sd.check_input_settings(device=index, samplerate=samplerate, channels=1, dtype="float32")
    return index, devices[index]["name"]


def create_spotter(sherpa_onnx, args):
    files = {
        "tokens": args.model_dir / "tokens.txt",
        "encoder": args.model_dir / "encoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        "decoder": args.model_dir / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        "joiner": args.model_dir / "joiner-epoch-13-avg-2-chunk-16-left-64.onnx",
    }
    for p in (*files.values(), args.keywords):
        if not p.is_file():
            raise FileNotFoundError(f"Missing: {p}")
    return sherpa_onnx.KeywordSpotter(
        **{k: str(v) for k, v in files.items()},
        keywords_file=str(args.keywords),
        num_threads=2,
        max_active_paths=args.max_active_paths,
        keywords_score=args.score,
        keywords_threshold=args.threshold,
        num_trailing_blanks=args.trailing_blanks,
        provider="cpu",
    )


def handle_console(command, controller):
    command = command.strip()
    if not command:
        return False
    if command.lower() == ":quit":
        return True
    if command.lower() == ":help":
        log("Commands: :state IDLE|LISTENING|EXECUTING|PAUSED, :quit")
    elif command.lower().startswith(":state "):
        try:
            controller.set_state(command.split(None, 1)[1])
        except ValueError as exc:
            log(f"[ERROR] {exc}")
    else:
        controller.on_keyword(command)
    return False


def console_reader(commands):
    try:
        for line in sys.stdin:
            commands.put(line.strip())
    except (OSError, ValueError):
        pass


def run_simulation(args):
    controller = VoiceController(args.cooldown, args.debug_all_keywords)
    log("[SIMULATION] Enter keywords or :state EXECUTING; use :quit to exit")
    for line in sys.stdin:
        if handle_console(line, controller):
            break


def run_live(args):
    import numpy as np
    import sounddevice as sd
    import sherpa_onnx

    device, name = choose_device(sd, args.device, args.sample_rate)
    log(f"[MIC] {device}: {name}")
    spotter = create_spotter(sherpa_onnx, args)
    stream = spotter.create_stream()
    controller = VoiceController(args.cooldown, args.debug_all_keywords)

    commands = queue.Queue()
    threading.Thread(target=console_reader, args=(commands,), daemon=True).start()
    log("[READY] Listening. Ctrl+C to exit; :help for console commands")
    log("[SAFETY] Emits events only; no motor commands are sent")

    with sd.InputStream(device=device, samplerate=args.sample_rate,
                        channels=1, dtype="float32", blocksize=args.chunk_size) as mic:
        while True:
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break
                if handle_console(command, controller):
                    return

            samples, overflowed = mic.read(args.chunk_size)
            if overflowed:
                log("[WARN] Audio overflow")
            audio = np.asarray(samples[:, 0], dtype=np.float32).copy()
            stream.accept_waveform(args.sample_rate, audio)
            while spotter.is_ready(stream):
                spotter.decode_stream(stream)
            result = spotter.get_result(stream)
            keyword = result if isinstance(result, str) else getattr(result, "keyword", "")
            if keyword:
                controller.on_keyword(keyword)
                spotter.reset_stream(stream)


def main():
    parser = argparse.ArgumentParser(description="HCR KWS event node; does not control motors")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", default=None, help="Input device index or unique substring")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--keywords", type=Path, default=KEYWORDS_FILE)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--score", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--max-active-paths", type=int, default=4)
    parser.add_argument("--trailing-blanks", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--debug-all-keywords", action="store_true",
                        help="Show ignored detections and cooldown rejections")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        print("Default device:", sd.default.device)
        return
    if args.simulate:
        run_simulation(args)
        return
    if args.sample_rate <= 0 or args.chunk_size <= 0 or args.cooldown < 0:
        parser.error("sample rate and chunk size must be positive; cooldown >= 0")
    run_live(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("[EXIT] Stopped")
    except Exception as exc:
        log(f"[ERROR] {exc}")
        sys.exit(1)
