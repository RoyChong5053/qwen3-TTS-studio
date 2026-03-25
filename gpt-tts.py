#!/usr/bin/env python3

"""
Qwen3 TTS 小说批量转换工具
支持:
 - CLI
 - Textual TUI
 - 自动分块
 - MP3输出
"""

import sys
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import soundfile as sf

from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Button, Label, Log, ProgressBar
)
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.binding import Binding
from textual import on, work


# ================================
# 配置
# ================================

BASE_DIR = Path("/home/roychong/app/qwen3-TTS-studio")

MODELS = {
    "1.7B-CustomVoice": BASE_DIR / "Qwen3-TTS-12Hz-1.7B-CustomVoice",
}

CONFIG_FILE = Path("~/.config/qwen3-tts/settings.json").expanduser()

DEFAULT_CONFIG = {
    "model": "1.7B-CustomVoice",
    "speaker": "Vivian",
    "language": "Chinese",
    "chunk_size": 200,
    "bitrate": "192k",
    "volume": 1.5,
    "txt_dir": str(Path.home() / "novel-tts/txt"),
    "mp3_dir": str(Path.home() / "novel-tts/mp3"),
}


# ================================
# config
# ================================

def load_config():
    if CONFIG_FILE.exists():
        return {**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text())}
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ================================
# 分块
# ================================

PUNCTS = set("。！？!?…;\n")


def smart_split(text: str, max_chars=200):

    text = text.strip()
    chunks = []

    start = 0
    n = len(text)

    while start < n:

        end = min(start + max_chars, n)

        if end == n:
            chunks.append(text[start:])
            break

        cut = -1

        for i in range(end - 1, start, -1):
            if text[i] in PUNCTS:
                cut = i + 1
                break

        if cut == -1:
            cut = end

        chunks.append(text[start:cut].strip())

        start = cut

    return [c for c in chunks if c]


# ================================
# 音频
# ================================

def concat_wavs(wavs, sr, out):

    silence = np.zeros(int(sr * 0.3))

    arr = []

    for w in wavs:
        arr.append(w)
        arr.append(silence)

    combined = np.concatenate(arr)

    sf.write(out, combined, sr)


def wav_to_mp3(wav, mp3, bitrate, volume):

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav),
        "-af",
        f"volume={volume}",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(mp3),
    ]

    return subprocess.run(cmd).returncode == 0


# ================================
# TTS
# ================================

class TTSEngine:

    def __init__(self, cfg, log=print):

        self.cfg = cfg
        self.log = log

        self.model = None

    def load(self):

        from qwen_tts import Qwen3TTSModel

        path = MODELS[self.cfg["model"]]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.log(f"加载模型 → {device}")

        self.model = Qwen3TTSModel.from_pretrained(
            str(path),
            device_map=device
        )

    def generate(self, text):

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            speaker=self.cfg["speaker"],
            language=self.cfg["language"]
        )

        return wavs[0], sr


# ================================
# 处理文件
# ================================

def process_file(engine, cfg, txt_file, log=print):

    text = txt_file.read_text()

    chunks = smart_split(text, cfg["chunk_size"])

    log(f"分块: {len(chunks)}")

    wavs = []

    sr = 24000

    for i, chunk in enumerate(chunks):

        log(f"{i+1}/{len(chunks)}")

        w, sr = engine.generate(chunk)

        wavs.append(w)

    tmp = Path(tempfile.mkdtemp())

    combined = tmp / "combined.wav"

    concat_wavs(wavs, sr, combined)

    mp3 = Path(cfg["mp3_dir"]) / (txt_file.stem + ".mp3")

    wav_to_mp3(combined, mp3, cfg["bitrate"], cfg["volume"])

    return mp3


# ================================
# TUI
# ================================

class MainScreen(Screen):

    BINDINGS = [
        Binding("r", "run", "Run"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:

        yield Header()

        with Container():

            yield Button("扫描文件", id="scan")
            yield Button("开始转换", id="run")

            yield ProgressBar(id="bar")

            yield Log(id="log")

        yield Footer()

    def log(self, msg):

        self.query_one(Log).write_line(msg)

    @on(Button.Pressed, "#scan")
    def scan(self, event):

        cfg = self.app.cfg

        txt_dir = Path(cfg["txt_dir"])

        self.files = list(txt_dir.glob("*.txt"))

        self.log(f"发现 {len(self.files)} 个文件")

    @on(Button.Pressed, "#run")
    def run(self, event):

        self.run_batch()

    @work(thread=True)
    def run_batch(self):

        cfg = self.app.cfg

        engine = TTSEngine(cfg, log=self.log)

        engine.load()

        files = getattr(self, "files", [])

        bar = self.query_one("#bar", ProgressBar)

        bar.update(total=len(files))

        for i, f in enumerate(files):

            self.log(f"处理 {f.name}")

            process_file(engine, cfg, f, self.log)

            bar.update(progress=i + 1)


# ================================
# APP
# ================================

class TTSApp(App):

    def __init__(self):

        super().__init__()

        self.cfg = load_config()

    def on_mount(self):

        self.push_screen(MainScreen())


# ================================
# CLI
# ================================

def cli():

    cfg = load_config()

    txt_dir = Path(cfg["txt_dir"])

    mp3_dir = Path(cfg["mp3_dir"])

    mp3_dir.mkdir(parents=True, exist_ok=True)

    files = list(txt_dir.glob("*.txt"))

    print("文件数:", len(files))

    engine = TTSEngine(cfg)

    engine.load()

    for f in files:

        print("处理", f)

        process_file(engine, cfg, f)


# ================================
# main
# ================================

if __name__ == "__main__":

    if "--cli" in sys.argv:

        cli()

    else:

        TTSApp().run()
