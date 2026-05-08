"""Novel reading module - long text TTS with tone consistency."""

import re
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

BASE_DIR = Path("/home/roychong/app/qwen3-TTS-studio")

MODELS = {
    "1.7B-CustomVoice": BASE_DIR / "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6B-CustomVoice": BASE_DIR / "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7B-Base":        BASE_DIR / "Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B-Base":        BASE_DIR / "Qwen3-TTS-12Hz-0.6B-Base",
}

ZH_SENTENCE_END = set("。！？…；\n")
ZH_CLAUSE_END = set("，、：「」『』【】《》〈〉")
ALL_PUNCTS = ZH_SENTENCE_END | ZH_CLAUSE_END | set(".!?;,")


def smart_split(
    text: str,
    max_chars: int = 500,
    min_chars: int = 50,
) -> list[str]:
    """
    Smart chunking for novel text.
    Priority: paragraph break > sentence end > clause end.

    Each chunk will be a complete prosodic unit (ending with sentence-ending punctuation).
    """
    text = text.strip()
    if not text:
        return []

    # First split by paragraph breaks
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    for para in paragraphs:
        # Split paragraph into sentences
        sentences = re.split(r"(?<=[。！？…；])", para)
        sentences = [s.strip() for s in sentences if s.strip()]

        current = ""
        for sent in sentences:
            if not current:
                current = sent
            elif len(current) + len(sent) + 1 <= max_chars:
                current = f"{current}{sent}"
            else:
                # Save current chunk
                if current and current[-1] not in ZH_SENTENCE_END:
                    current += "。"
                chunks.append(current)
                current = sent

        if current:
            if current[-1] not in ZH_SENTENCE_END:
                current += "。"
            chunks.append(current)

    # Merge small chunks (less than min_chars) with adjacent ones
    if len(chunks) <= 1:
        return [c for c in chunks if c]

    merged = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        # Merge forward if too small
        while i + 1 < len(chunks) and len(chunk) < min_chars:
            i += 1
            chunk = f"{chunk}{chunks[i]}"
        merged.append(chunk.strip())
        i += 1

    return [c for c in merged if c]


def get_last_sentence(text: str, max_len: int = 30) -> str:
    """Extract the last complete sentence from text for context anchoring."""
    for punct in ZH_SENTENCE_END:
        idx = text.rfind(punct)
        if idx >= 0:
            sentence = text[idx + 1 :].strip()
            if not sentence:
                sentence = text[max(0, idx - max_len) : idx + 1].strip()
            return sentence[:max_len]
    return text[-max_len:].strip()


def normalize_rms(audio: np.ndarray, target_rms: float = 0.15) -> np.ndarray:
    """Normalize audio to target RMS level."""
    if len(audio) == 0:
        return audio
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-6:
        return audio
    return audio * (target_rms / rms)


def crossfade_audio(
    audio1: np.ndarray, audio2: np.ndarray, sr: int, fade_ms: int = 30
) -> np.ndarray:
    """Concatenate two audio arrays with crossfade."""
    fade_samples = int(sr * fade_ms / 1000)
    if len(audio1) < fade_samples or len(audio2) < fade_samples:
        return np.concatenate([audio1, audio2])
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    audio1_end = audio1[-fade_samples:] * fade_out
    audio2_start = audio2[:fade_samples] * fade_in
    crossfaded = audio1_end + audio2_start
    return np.concatenate([audio1[:-fade_samples], crossfaded, audio2[fade_samples:]])


def wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k", volume: float = 1.5) -> bool:
    """Convert WAV to MP3 with volume adjustment."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-af", f"volume={volume}",
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        str(mp3_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


class NovelReader:
    """Long text TTS with tone consistency across chunks."""

    def __init__(
        self,
        model_name: str,
        speaker: str,
        language: str = "Chinese",
        instruct: str = "",
        seed: int = 42,
        temperature: float = 0.25,
        top_k: int = 30,
        top_p: float = 0.85,
        repetition_penalty: float = 1.0,
        chunk_size: int = 500,
        bitrate: str = "192k",
        volume: float = 1.5,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self.instruct = instruct
        self.seed = seed
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.chunk_size = chunk_size
        self.bitrate = bitrate
        self.volume = volume
        self.device = device

        self._model = None
        self._loaded_model_key = None

    def _get_device(self):
        if self.device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return self.device

    def load_model(self):
        from qwen_tts import Qwen3TTSModel

        key = self.model_name
        model_path = MODELS.get(key)

        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        if self._loaded_model_key == key and self._model is not None:
            return self._model

        device = self._get_device()
        dtype = torch.bfloat16 if "cuda" in device else torch.float32
        kwargs = dict(device_map=device, dtype=dtype)

        if "cuda" in device:
            try:
                import flash_attn  # noqa
                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass

        self._model = Qwen3TTSModel.from_pretrained(str(model_path), **kwargs)
        self._loaded_model_key = key
        return self._model

    def _fix_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def synthesize(
        self,
        text: str,
        progress_fn=None,
    ) -> tuple[list[np.ndarray], int, list[str]]:
        """
        Synthesize long text with tone consistency.

        Returns:
            (wav_arrays, sample_rate, chunk_texts)
        """
        model = self.load_model()
        chunks = smart_split(text, max_chars=self.chunk_size)
        total = len(chunks)

        if progress_fn:
            progress_fn(0, total, f"分块完成: {total} 块")

        wav_arrays = []
        sr = 24000
        prev_last_sentence = ""

        for i, chunk in enumerate(chunks):
            if progress_fn:
                preview = chunk[:25] + "…" if len(chunk) > 25 else chunk
                progress_fn(i, total, f"[{i+1}/{total}] {preview}")

            # Build instruct with context anchoring
            context_instruct = self.instruct or ""
            if prev_last_sentence and i > 0:
                context_instruct = f"接续上文：'{prev_last_sentence}'。{context_instruct}"

            # Fix seed for consistency
            self._fix_seed()

            # Generate
            wavs, sr = model.generate_custom_voice(
                text=chunk,
                speaker=self.speaker,
                language=self.language if self.language != "Auto" else None,
                instruct=context_instruct if context_instruct else None,
                non_streaming_mode=True,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )

            audio = wavs[0].astype(np.float32)
            # RMS normalization
            audio = normalize_rms(audio, target_rms=0.15)
            wav_arrays.append(audio)

            # Save last sentence for next chunk's context
            prev_last_sentence = get_last_sentence(chunk)

        if progress_fn:
            progress_fn(total, total, "合成完成")

        return wav_arrays, sr, chunks

    def synthesize_to_mp3(
        self,
        text: str,
        output_mp3: Path,
        progress_fn=None,
    ) -> tuple[bool, str, list[str]]:
        """
        Full pipeline: split -> synthesize -> concat -> mp3.

        Returns:
            (success, output_path, chunk_texts)
        """
        try:
            wav_arrays, sr, chunks = self.synthesize(text, progress_fn)

            if not wav_arrays:
                return False, "没有生成任何音频", chunks

            # Concatenate with crossfade
            merged = wav_arrays[0]
            for audio in wav_arrays[1:]:
                merged = crossfade_audio(merged, audio, sr)

            # Save temp WAV then convert to MP3
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = Path(tmp.name)
                sf.write(str(tmp_wav), merged, sr)

            ok = wav_to_mp3(tmp_wav, output_mp3, self.bitrate, self.volume)
            tmp_wav.unlink(missing_ok=True)

            return ok, str(output_mp3) if ok else "MP3 转换失败", chunks

        except Exception as e:
            return False, str(e), []

    def synthesize_file_to_mp3(
        self,
        txt_path: Path,
        output_mp3: Optional[Path] = None,
        progress_fn=None,
    ) -> tuple[bool, str, list[str]]:
        """Read a .txt file and convert to MP3."""
        if output_mp3 is None:
            output_mp3 = txt_path.with_suffix(".mp3")

        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            return False, "文件内容为空", []

        return self.synthesize_to_mp3(text, output_mp3, progress_fn)
