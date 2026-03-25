#!/usr/bin/env python3
"""
qwen3-tts.py - 小说文字转语音批量处理脚本 (Qwen3-TTS)
已修复 TUI 运行问题，并加入语气一致性优化（固定随机种子 & 强制闭合标点）
"""

import os
import re
import sys
import json
import time
import shutil
import random
import logging
import tempfile
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── TUI 依赖 ────────────────────────────────────────────────────────────────
try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.widgets import (
        Header, Footer, Button, Label, Input, Select,
        ProgressBar, Log, Static, Switch, Checkbox
    )
    from textual.reactive import reactive
    from textual import work, on
    from textual.screen import Screen, ModalScreen
    from textual.binding import Binding
except ImportError:
    print("请先安装 textual: pip install textual")
    sys.exit(1)

# ── 核心 TTS 依赖 ────────────────────────────────────────────────────────────
try:
    import torch
    import soundfile as sf
    import numpy as np
except ImportError:
    print("请安装依赖: pip install torch soundfile numpy")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/home/roychong/app/qwen3-TTS-studio")

MODELS = {
    "1.7B-CustomVoice": BASE_DIR / "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6B-CustomVoice": BASE_DIR / "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7B-Base":        BASE_DIR / "Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B-Base":        BASE_DIR / "Qwen3-TTS-12Hz-0.6B-Base",
}

SPEAKERS = {
    "Vivian":    "活泼、略带个性的年轻女声 (中文)",
    "Serena":    "温柔、亲切的年轻女声 (中文)",
    "Uncle_Fu":  "低沉醇厚的成熟男声 (中文)",
    "Dylan":     "清朗自然的北京男声 (中文/北京话)",
    "Eric":      "爽朗略带沙哑的成都男声 (中文/四川话)",
    "Ryan":      "充满活力、节奏感强的男声 (英文)",
    "Aiden":     "明亮的美式男声 (英文)",
    "Ono_Anna":  "轻盈灵动的日文女声 (日语)",
    "Sohee":     "情感丰富的韩语温柔女声 (韩语)",
}

LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean"]

CONFIG_FILE = Path("~/.config/qwen3-tts/settings.json").expanduser()
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = {
    "model":        "1.7B-CustomVoice",
    "speaker":      "Vivian",
    "language":     "Chinese",
    "instruct":     "",
    "chunk_size":   200,
    "seed":         42,         # 新增：随机种子，稳定音色
    "bitrate":      "192k",
    "volume":       1.5,
    "txt_dir":      str(Path.home() / "novel-tts/txt"),
    "mp3_dir":      str(Path.home() / "novel-tts/mp3"),
    "log_dir":      str(Path.home() / "novel-tts/logs"),
    "keep_wav":     False,
    "device":       "auto",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(saved)
            return cfg
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# 中文标点符号集
ZH_PUNCTS = set("。！？…；\n")
ZH_PUNCTS_SOFT = set("，、：「」『』【】《》〈〉")
ALL_PUNCTS = ZH_PUNCTS | ZH_PUNCTS_SOFT | set(".!?;,:")

def smart_split(text: str, max_chars: int = 200) -> list[str]:
    """
    智能分块：尽量在标点符号后切割，保证 chunk ≤ max_chars。
    【优化】强制句尾闭合，防止 TTS 语气悬空漂移。
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = start + max_chars

        if end >= n:
            chunks.append(text[start:].strip())
            break

        cut = -1
        for i in range(end - 1, start + max_chars // 3, -1):
            if text[i] in ZH_PUNCTS:
                cut = i + 1
                break

        if cut == -1:
            for i in range(end - 1, start + max_chars // 4, -1):
                if text[i] in ZH_PUNCTS_SOFT:
                    cut = i + 1
                    break

        if cut == -1:
            cut = end

        chunk = text[start:cut].strip()
        if chunk:
            # 【语气优化】如果分段结尾没有标点，强行加个句号，让 TTS 产生自然的降调收尾
            if chunk[-1] not in ALL_PUNCTS:
                chunk += "。"
            chunks.append(chunk)
        start = cut

    return [c for c in chunks if c]

def wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str, volume: float) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-af", f"volume={volume}",
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        str(mp3_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0

def concat_wavs(wav_files: list[Path], out_path: Path, sr: int) -> bool:
    arrays = []
    for wf in wav_files:
        data, _ = sf.read(str(wf))
        arrays.append(data)
        arrays.append(np.zeros(int(sr * 0.3), dtype=data.dtype))
    combined = np.concatenate(arrays)
    sf.write(str(out_path), combined, sr)
    return True

def fix_seed(seed: int):
    """【语气优化】固定随机种子，保证每次推理的音色和情感路径一致"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ═══════════════════════════════════════════════════════════════════════════════
# TTS 引擎
# ═══════════════════════════════════════════════════════════════════════════════

class TTSEngine:
    def __init__(self, cfg: dict, log_fn=None):
        self.cfg = cfg
        self.log = log_fn or print
        self._model = None
        self._loaded_model_key = None

    def _get_device(self):
        d = self.cfg.get("device", "auto")
        if d == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return d

    def load_model(self):
        from qwen_tts import Qwen3TTSModel  # type: ignore
        key = self.cfg["model"]
        model_path = MODELS.get(key)
        if not model_path or not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        if self._loaded_model_key == key and self._model is not None:
            self.log(f"模型已加载: {key}")
            return

        device = self._get_device()
        self.log(f"正在加载模型 {key} → {device} ...")
        dtype = torch.bfloat16 if "cuda" in device else torch.float32
        kwargs = dict(device_map=device, dtype=dtype)

        if "cuda" in device:
            try:
                import flash_attn  # noqa
                kwargs["attn_implementation"] = "flash_attention_2"
                self.log("⚡ 使用 Flash Attention 2")
            except ImportError:
                self.log("⚠️ flash_attn 未安装，使用标准 attention")

        self._model = Qwen3TTSModel.from_pretrained(str(model_path), **kwargs)
        self._loaded_model_key = key
        self.log(f"✅ 模型加载完成")

# ═══════════════════════════════════════════════════════════════════════════════
# TUI - 设定弹窗
# ═══════════════════════════════════════════════════════════════════════════════

class SettingsScreen(ModalScreen):
    CSS = """
    SettingsScreen { align: center middle; }
    #settings-box { width: 70; height: auto; background: $surface; border: thick $primary; padding: 1 2; }
    .setting-row { height: 3; margin: 0 0 1 0; }
    .setting-label { width: 18; padding-top: 1; color: $text-muted; }
    .setting-input { width: 1fr; }
    #btn-row { height: 3; align: right middle; margin-top: 1; }
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg.copy()

    def compose(self) -> ComposeResult:
        cfg = self.cfg
        model_opts = [(k, k) for k in MODELS]
        spk_opts   = [(k, k) for k in SPEAKERS]
        lang_opts  = [(l, l) for l in LANGUAGES]

        with Container(id="settings-box"):
            yield Label("⚙️  设定", classes="setting-label")
            with Horizontal(classes="setting-row"):
                yield Label("模型", classes="setting-label")
                yield Select(model_opts, value=cfg["model"], id="sel-model", classes="setting-input")
            with Horizontal(classes="setting-row"):
                yield Label("说话人", classes="setting-label")
                yield Select(spk_opts, value=cfg["speaker"], id="sel-speaker", classes="setting-input")
            with Horizontal(classes="setting-row"):
                yield Label("语言", classes="setting-label")
                yield Select(lang_opts, value=cfg["language"], id="sel-lang", classes="setting-input")
            with Horizontal(classes="setting-row"):
                yield Label("语气指令", classes="setting-label")
                yield Input(value=cfg.get("instruct",""), placeholder="例: 用愤怒的语气说", id="inp-instruct", classes="setting-input")
            with Horizontal(classes="setting-row"):
                yield Label("分块字数", classes="setting-label")
                yield Input(value=str(cfg["chunk_size"]), id="inp-chunk", classes="setting-input")
            with Horizontal(classes="setting-row"):
                yield Label("随机种子", classes="setting-label")
                yield Input(value=str(cfg.get("seed", 42)), id="inp-seed", classes="setting-input")
            with Horizontal(classes="setting-row"):
                yield Label("音量倍率", classes="setting-label")
                yield Input(value=str(cfg["volume"]), id="inp-volume", classes="setting-input")
            with Horizontal(classes="setting-row"):
                yield Label("MP3 比特率", classes="setting-label")
                yield Input(value=cfg["bitrate"], id="inp-bitrate", classes="setting-input")
            with Horizontal(classes="setting-row"):
                yield Label("TXT 目录", classes="setting-label")
                yield Input(value=cfg["txt_dir"], id="inp-txtdir", classes="setting-input")
            with Horizontal(classes="setting-row"):
                yield Label("MP3 输出目录", classes="setting-label")
                yield Input(value=cfg["mp3_dir"], id="inp-mp3dir", classes="setting-input")
            with Horizontal(classes="setting-row"):
                yield Label("保留 WAV", classes="setting-label")
                yield Switch(value=cfg.get("keep_wav", False), id="sw-keepwav")
            with Horizontal(id="btn-row"):
                yield Button("取消", variant="default", id="btn-cancel")
                yield Button("保存", variant="primary", id="btn-save")

    @on(Button.Pressed, "#btn-cancel")
    def cancel(self):
        self.dismiss(None)

    @on(Button.Pressed, "#btn-save")
    def save(self):
        def q(widget_id):
            return self.query_one(widget_id)
        try:
            self.cfg["model"]      = q("#sel-model").value
            self.cfg["speaker"]    = q("#sel-speaker").value
            self.cfg["language"]   = q("#sel-lang").value
            self.cfg["instruct"]   = q("#inp-instruct").value
            self.cfg["chunk_size"] = int(q("#inp-chunk").value)
            self.cfg["seed"]       = int(q("#inp-seed").value)
            self.cfg["volume"]     = float(q("#inp-volume").value)
            self.cfg["bitrate"]    = q("#inp-bitrate").value
            self.cfg["txt_dir"]    = q("#inp-txtdir").value
            self.cfg["mp3_dir"]    = q("#inp-mp3dir").value
            self.cfg["keep_wav"]   = q("#sw-keepwav").value
        except ValueError:
            return
        self.dismiss(self.cfg)

# ═══════════════════════════════════════════════════════════════════════════════
# TUI - 主界面
# ═══════════════════════════════════════════════════════════════════════════════

class MainScreen(Screen):
    CSS = """
    MainScreen { layout: vertical; }
    #top-bar { height: 5; background: $surface; border-bottom: solid $primary; padding: 0 2; layout: horizontal; align: left middle; }
    #cfg-info { width: 1fr; color: $text-muted; }
    #btn-settings, #btn-load { margin-left: 1; }
    #mid-area { height: 1fr; layout: horizontal; }
    #file-panel { width: 30; border-right: solid $primary; padding: 1; }
    #file-list { height: 1fr; overflow-y: auto; }
    .file-item { height: 2; padding: 0 1; border-bottom: solid $surface-lighten-1; }
    .file-item:hover { background: $surface-lighten-2; }
    #right-panel { width: 1fr; padding: 1; layout: vertical; }
    #log-area { height: 1fr; border: solid $primary; overflow-y: auto; padding: 1; }
    #progress-area { height: 6; padding: 0 1; }
    #chunk-label { color: $text-muted; margin-bottom: 1; }
    #btn-row { height: 3; layout: horizontal; align: right middle; margin-top: 1; }
    #status-bar { height: 2; background: $surface-darken-1; padding: 0 2; color: $text-muted; }
    """

    BINDINGS = [
        Binding("s", "open_settings", "设定"),
        Binding("r", "run_all", "开始转换"),
        Binding("q", "quit", "退出"),
    ]

    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.engine = TTSEngine(self.cfg)
        self._running = False
        self._files: list[Path] = []

    # 【修复 TUI 快捷键崩溃】：补充缺失的 action 方法
    def action_open_settings(self):
        self.on_settings_pressed()

    def action_run_all(self):
        self.on_run_pressed()

    def action_quit(self):
        self.app.exit()

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="top-bar"):
            yield Label(self._cfg_summary(), id="cfg-info")
            yield Button("⚙ 设定 [S]", id="btn-settings")
            yield Button("📂 扫描文件 [R]", id="btn-load")
            yield Button("▶ 开始转换", variant="primary", id="btn-run")
        with Container(id="mid-area"):
            with Container(id="file-panel"):
                yield Label("📚 待处理文件")
                with ScrollableContainer(id="file-list"):
                    yield Label("(扫描目录后显示)", id="empty-hint")
            with Container(id="right-panel"):
                yield Log(id="log-area", auto_scroll=True)
                with Container(id="progress-area"):
                    yield Label("当前分块: —", id="chunk-label")
                    yield ProgressBar(id="chunk-bar", show_eta=False)
                    yield Label("文件进度:", id="file-label")
                    yield ProgressBar(id="file-bar", show_eta=False)
        with Container(id="btn-row"):
            yield Button("停止", variant="error", id="btn-stop", disabled=True)
        yield Static(self._cfg_summary(), id="status-bar")
        yield Footer()

    def _cfg_summary(self):
        c = self.cfg
        return (f"模型:{c['model']}  说话人:{c['speaker']}  语言:{c['language']}  "
                f"分块:{c['chunk_size']}字  种子:{c.get('seed', 42)}")

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.query_one("#log-area", Log).write_line(f"[{ts}] {msg}")

    def _refresh_cfg_ui(self):
        summary = self._cfg_summary()
        self.query_one("#cfg-info", Label).update(summary)
        self.query_one("#status-bar", Static).update(summary)

    def _scan_files(self):
        txt_dir = Path(self.cfg["txt_dir"])
        if not txt_dir.exists():
            self._log(f"❌ 目录不存在: {txt_dir}")
            return
        files = sorted(txt_dir.glob("*.txt"))
        self._files = files
        fl = self.query_one("#file-list", ScrollableContainer)
        fl.remove_children()
        if not files:
            fl.mount(Label("(没有找到 .txt 文件)", id="empty-hint"))
            self._log(f"⚠️  目录内没有 txt 文件: {txt_dir}")
            return
        for f in files:
            fl.mount(Label(f"📄 {f.name}", classes="file-item"))
        self._log(f"📚 找到 {len(files)} 个文件 → {txt_dir}")

    @on(Button.Pressed, "#btn-settings")
    def on_settings_pressed(self):
        def on_close(new_cfg):
            if new_cfg:
                self.cfg = new_cfg
                save_config(self.cfg)
                self.engine.cfg = new_cfg
                self._refresh_cfg_ui()
                self._log("✅ 设定已保存")
        self.app.push_screen(SettingsScreen(self.cfg), on_close)

    @on(Button.Pressed, "#btn-load")
    def on_load_pressed(self):
        self._scan_files()

    @on(Button.Pressed, "#btn-run")
    def on_run_pressed(self):
        if self._running: return
        if not self._files: self._scan_files()
        if not self._files:
            self._log("❌ 没有找到 txt 文件，请先检查目录设定")
            return
        self._start_batch()

    @on(Button.Pressed, "#btn-stop")
    def on_stop_pressed(self):
        self._running = False
        self._log("⛔ 用户请求停止，等待当前文件完成…")

    def _start_batch(self):
        self._running = True
        self.query_one("#btn-run", Button).disabled = True
        self.query_one("#btn-stop", Button).disabled = False
        threading.Thread(target=self._batch_worker, daemon=True).start()

    def _batch_worker(self):
        mp3_dir = Path(self.cfg["mp3_dir"])
        log_dir = Path(self.cfg["log_dir"])
        mp3_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_logger = logging.getLogger("tts_file")
        file_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        file_logger.addHandler(fh)

        self.app.call_from_thread(self._log, f"📝 日志: {log_file}")
        self.app.call_from_thread(self._log, "🔧 加载模型中...")

        try:
            self.engine.load_model()
        except Exception as e:
            self.app.call_from_thread(self._log, f"❌ 模型加载失败: {e}")
            self.app.call_from_thread(self._finish_batch, 0, 0)
            return

        total = len(self._files)
        success = fail = 0
        fb = self.query_one("#file-bar", ProgressBar)
        self.app.call_from_thread(fb.update, total=total, progress=0)

        for idx, txt_file in enumerate(self._files):
            if not self._running: break
            self.app.call_from_thread(self._log, f"\n{'─'*50}")
            self.app.call_from_thread(self._log, f"📖 [{idx+1}/{total}] {txt_file.name}")
            
            if self._process_file(txt_file, mp3_dir, file_logger):
                success += 1
                self.app.call_from_thread(self._log, f"✅ {txt_file.stem}.mp3 完成")
            else:
                fail += 1
            self.app.call_from_thread(fb.update, progress=idx+1)

        file_logger.removeHandler(fh)
        self.app.call_from_thread(self._finish_batch, success, fail)

    def _process_file(self, txt_file: Path, mp3_dir: Path, file_logger) -> bool:
        try:
            text = txt_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            self.app.call_from_thread(self._log, f"⚠️ 读取失败: {e}")
            return False

        if not text: return False
        chunks = smart_split(text, self.cfg["chunk_size"])
        total_chunks = len(chunks)
        self.app.call_from_thread(self._log, f"   分块数: {total_chunks}  总字数: {len(text)}")

        cb = self.query_one("#chunk-bar", ProgressBar)
        cl = self.query_one("#chunk-label", Label)
        self.app.call_from_thread(cb.update, total=total_chunks, progress=0)

        chunk_wavs = []
        sr = 24000
        tmp_dir = Path(tempfile.mkdtemp(prefix="qwen3tts_"))
        seed = self.cfg.get("seed", 42)

        try:
            for i, chunk in enumerate(chunks):
                if not self._running: return False

                # 【语气优化】每次生成前锁定随机种子
                fix_seed(seed)
                
                preview = chunk[:25] + "…" if len(chunk) > 25 else chunk
                self.app.call_from_thread(cl.update, f"分块 {i+1}/{total_chunks}: 「{preview}」")
                self.app.call_from_thread(cb.update, progress=i)

                spk, lang, inst = self.cfg["speaker"], self.cfg["language"], self.cfg.get("instruct", "")
                is_custom = "CustomVoice" in self.cfg["model"]

                if is_custom:
                    wavs, sr = self.engine._model.generate_custom_voice(
                        text=chunk, language=lang if lang != "Auto" else None,
                        speaker=spk, instruct=inst if inst else None
                    )
                else:
                    wavs, sr = self.engine._model.generate(
                        text=chunk, language=lang if lang != "Auto" else None
                    )

                chunk_path = tmp_dir / f"chunk_{i:04d}.wav"
                sf.write(str(chunk_path), wavs[0], sr)
                chunk_wavs.append(chunk_path)
                self.app.call_from_thread(cb.update, progress=i+1)

            self.app.call_from_thread(self._log, "   🔗 合并音频...")
            combined_wav = tmp_dir / "combined.wav"
            concat_wavs(chunk_wavs, combined_wav, sr)

            mp3_path = mp3_dir / (txt_file.stem + ".mp3")
            self.app.call_from_thread(self._log, "   🎵 转换 MP3...")
            if not wav_to_mp3(combined_wav, mp3_path, self.cfg["bitrate"], self.cfg["volume"]):
                return False

            if self.cfg.get("keep_wav"):
                keep_path = mp3_dir / (txt_file.stem + ".wav")
                # 【修复 TUI 崩溃】用 shutil.move 替代 rename，解决跨磁盘移动报错
                shutil.move(str(combined_wav), str(keep_path))

            return True

        except Exception as e:
            self.app.call_from_thread(self._log, f"❌ 处理失败: {e}")
            return False
        finally:
            for wf in chunk_wavs:
                try: wf.unlink()
                except: pass
            try:
                (tmp_dir / "combined.wav").unlink(missing_ok=True)
                tmp_dir.rmdir()
            except: pass

    def _finish_batch(self, success: int, fail: int):
        self._running = False
        self.query_one("#btn-run", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True
        self._log(f"\n{'═'*50}\n🎉 批量完成！成功: {success}  失败: {fail}")

# ═══════════════════════════════════════════════════════════════════════════════
# App & CLI 入口
# ═══════════════════════════════════════════════════════════════════════════════

class Qwen3TTSApp(App):
    TITLE = "Qwen3-TTS 小说转语音"
    CSS = "Screen { background: $background; }"
    SCREENS = {"main": MainScreen}

    def on_mount(self):
        self.push_screen(MainScreen())

def cli_mode():
    import argparse
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Qwen3 TTS 小说批量转换 (CLI)")
    parser.add_argument("--txt",     default=cfg["txt_dir"])
    parser.add_argument("--mp3",     default=cfg["mp3_dir"])
    parser.add_argument("--model",   default=cfg["model"], choices=list(MODELS))
    parser.add_argument("--speaker", default=cfg["speaker"], choices=list(SPEAKERS))
    parser.add_argument("--lang",    default=cfg["language"])
    parser.add_argument("--chunk",   type=int, default=cfg["chunk_size"])
    parser.add_argument("--seed",    type=int, default=cfg.get("seed", 42))
    parser.add_argument("--instruct",default=cfg.get("instruct",""))
    args = parser.parse_args()

    cfg.update(vars(args))
    mp3_dir = Path(cfg["mp3_dir"])
    mp3_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(cfg["txt_dir"]).glob("*.txt"))

    if not files:
        print(f"❌ 目录内没有 txt 文件: {cfg['txt_dir']}")
        return

    engine = TTSEngine(cfg, log_fn=print)
    engine.load_model()

    for i, f in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {f.name}")
        text = f.read_text(encoding="utf-8").strip()
        if not text: continue
        chunks = smart_split(text, cfg["chunk_size"])

        wavs_list = []
        sr = 24000
        tmp = Path(tempfile.mkdtemp())
        for j, chunk in enumerate(chunks):
            # 【语气优化】CLI 模式同样锁定种子
            fix_seed(cfg["seed"])
            
            print(f"  [{j+1}/{len(chunks)}] {chunk[:30]}…")
            if "CustomVoice" in cfg["model"]:
                wavs, sr = engine._model.generate_custom_voice(
                    text=chunk, language=cfg["language"] if cfg["language"] != "Auto" else None,
                    speaker=cfg["speaker"], instruct=cfg.get("instruct") or None,
                )
            else:
                wavs, sr = engine._model.generate(text=chunk)
            
            wp = tmp / f"{j:04d}.wav"
            sf.write(str(wp), wavs[0], sr)
            wavs_list.append(wp)

        combined = tmp / "combined.wav"
        concat_wavs(wavs_list, combined, sr)
        wav_to_mp3(combined, mp3_dir / (f.stem + ".mp3"), cfg["bitrate"], cfg["volume"])
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  ✅ {mp3_dir / (f.stem + '.mp3')}")

if __name__ == "__main__":
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        cli_mode()
    else:
        app = Qwen3TTSApp()
        app.run()
