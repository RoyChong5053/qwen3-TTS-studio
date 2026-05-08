# Changelog

All notable changes to the Qwen3-TTS Studio project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added - Novel Reading Feature (2026-05-08)

#### New Module: `audio/novel_reader.py`
- **`smart_split()` function** - Semantic chunking for novel text
  - Priority: paragraph breaks > sentence endings (。！？) > clause endings (，、)
  - Ensures each chunk ends with sentence-ending punctuation
  - Configurable `max_chars` (default 500) and `min_chars` (default 50)
  - Merges small chunks with adjacent ones for better consistency

- **`NovelReader` class** - Core novel reading engine with tone consistency
  - **Context anchoring**: Uses `instruct` parameter to carry previous chunk's last sentence
  - **Fixed random seed**: `torch.manual_seed(seed)` applied before each chunk
  - **Low temperature**: Default 0.25 (vs 0.3 for regular TTS)
  - **Low top_k**: Default 30 (vs 50 for regular TTS)
  - **RMS normalization**: Matches volume levels across chunks (target RMS = 0.15)
  - **Crossfade concatenation**: 30ms crossfade between chunks
  - **Direct MP3 output**: Uses ffmpeg for WAV→MP3 conversion with volume adjustment

- **Supporting functions**:
  - `normalize_rms()` - Audio volume normalization
  - `crossfade_audio()` - Smooth audio transitions
  - `get_last_sentence()` - Extract last sentence for context anchoring
  - `wav_to_mp3()` - FFmpeg-based audio conversion

#### Modified: `qwen_tts_ui.py`
- **New Tab: "📖 Novel Reader"** (inserted between Personas and Podcast tabs)
  - Text input (multi-line) or TXT file upload
  - Settings panel:
    - Model selection (1.7B/0.6B-CustomVoice)
    - Voice selection (preset speakers)
    - Language selection
    - Voice style instruction
    - Chunk size slider (100-1000 chars, default 500)
    - Random seed (default 42)
    - Temperature slider (0.1-1.0, default 0.25)
  - Progress tracking with chunk-level status
  - WAV preview audio player
  - MP3 download link

- **New function: `generate_novel()`** (line ~1543)
  - Orchestrates the full novel generation pipeline
  - Handles both text input and file upload
  - Reports progress via Gradio's `gr.Progress()`
  - Returns: WAV preview path, status message, MP3 download path
  - Error handling with user-friendly messages

#### Novel Reading Tone Drift Solution (6-Layer Defense)
Implemented to address "语气漂移" (tone drift) in long text TTS:

| Layer | Method | Implementation Location |
|-------|--------|----------------------|
| 1 | Sentence-boundary chunking | `smart_split()` in `novel_reader.py` |
| 2 | Context anchoring via instruct | `NovelReader.synthesize()` |
| 3 | Fixed random seed | `NovelReader._fix_seed()` |
| 4 | Low temperature + low top_k | `NovelReader.__init__()` defaults |
| 5 | RMS normalization | `normalize_rms()` in `novel_reader.py` |
| 6 | Crossfade concatenation | `crossfade_audio()` in `novel_reader.py` |

### Technical Details
- **Model path resolution**: Uses `MODELS` dict in `novel_reader.py` for correct model path mapping
- **Progress callback**: Custom `_progress()` function passed to `NovelReader.synthesize_to_mp3()`
- **Temporary file handling**: Uses `tempfile.NamedTemporaryFile` for intermediate WAV/MP3 files
- **Error handling**: Graceful cleanup of temp files on failure

### Files Changed
- `audio/novel_reader.py` (NEW - 322 lines)
- `qwen_tts_ui.py` (MODIFIED - added ~80 lines for Novel Reader tab + ~60 lines for `generate_novel()` function)

### Migration Notes
- The new Novel Reader Tab supersedes the standalone novel TTS scripts (`my-novel-tts.py`, `gemini-tts.py`, `grok-tts.py`) with better integration
- Existing functionality (Preset Voices, Clone Voice, etc.) remains unchanged
- No database schema changes required
- No new pip dependencies (uses existing `qwen-tts`, `numpy`, `soundfile`)

---

## [Previous Releases]

### [Pre-2026-05-08] - Initial Project State
- Preset voice generation with CustomVoice and Base models
- Voice cloning from audio samples (multi-sample support)
- Voice design with natural language descriptions
- Podcast generation with AI-generated scripts (outline → transcript → audio)
- Persona management for multi-speaker scenarios
- Generation history with search and favorites
- Dark theme UI with custom CSS
- Standalone novel TTS scripts (Textual TUI versions)
- Batch processing support
- Audio quality checks (truncation detection, silence detection)
- FlashAttention 2 support for faster inference on CUDA
- Model caching with LRU eviction
- Dynamic max_new_tokens calculation based on text length

---

*Changelog maintained by: AI Agent (opencode)*
*Format: Keep a Changelog 1.0.0*
