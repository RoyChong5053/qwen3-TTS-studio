# Qwen3-TTS Studio - Project Knowledge Document

> **Purpose**: This document provides a comprehensive overview of the project structure, key modules, and implementation details. It is designed so that a new AI agent session can quickly understand the codebase and make modifications effectively.

---

## 1. Project Overview

**Qwen3-TTS Studio** is a web-based Text-to-Speech (TTS) application built with Gradio, supporting:
- Preset voice generation (CustomVoice & Base models)
- Voice cloning from audio samples
- Voice design with natural language descriptions
- Podcast generation with AI-generated scripts
- **Novel reading** (long text TTS with tone consistency)

**Tech Stack**: Python 3.12, Gradio 4.x, PyTorch, Qwen3-TTS models (0.6B/1.7B)

---

## 2. Directory Structure

```
/home/roychong/app/qwen3-TTS-studio/
├── qwen_tts_ui.py              # Main Gradio web UI (6300+ lines)
├── audio/                      # Core TTS processing modules
│   ├── __init__.py
│   ├── model_loader.py         # Model loading, device detection, FlashAttention
│   ├── generator.py           # Dialogue-level TTS, text chunking, crossfade
│   ├── batch.py               # Batch processing with retry logic
│   ├── combiner.py            # Audio combination (moviepy)
│   ├── novel_reader.py        # Novel reading with tone consistency  ← NEW
│   └── embedding_utils.py     # Voice clone embedding combination
├── ui/                        # Gradio UI components
│   ├── content_input.py
│   ├── voice_cards.py
│   ├── progress.py
│   ├── draft_preview.py
│   ├── draft_editor.py
│   └── persona.py
├── podcast/                    # Podcast generation module
│   ├── orchestrator.py        # Main podcast orchestration
│   ├── models.py              # Pydantic models (Outline, Transcript, etc.)
│   ├── outline.py             # AI outline generation
│   ├── transcript.py          # AI transcript generation
│   ├── script_parser.py      # Custom script parsing
│   ├── llm_client.py         # LLM client (OpenAI, Ollama, etc.)
│   ├── prompts.py             # LLM prompt templates
│   └── session.py            # Session management
├── storage/                   # Data persistence
│   ├── history.py
│   ├── persona.py
│   ├── persona_models.py
│   └── voice.py
├── static/                    # Static assets
│   └── dark-theme.css
├── my-novel-tts.py           # Standalone novel TTS (Textual TUI)
├── gemini-tts.py              # Novel TTS with seed & forced punctuation
├── grok-tts.py               # Optimized novel TTS with instruct prefix
├── gpt-tts.py                # Simplified novel TTS
├── config.py                  # Environment variable loading
├── requirements.txt
├── tts_settings.json          # Persisted TTS parameters
├── README.md
├── PROJECT_KNOWLEDGE.md      # This file
├── CHANGELOG.md               # Change history
└── Qwen3-TTS-*/              # Model weights (local)
```

---

## 3. Key Modules Explained

### 3.1 Main UI (`qwen_tts_ui.py`)

The main entry point. Contains:
- **Gradio Tab Structure** (lines ~2780-6300):
  - `Preset Voices` - Generate speech with preset speakers
  - `Clone Voice` - Clone voice from audio samples
  - `Voice Design` - Design voice with text descriptions
  - `Saved Voices` - Use previously saved voices
  - `Personas` - Manage speaker personas
  - **`📖 Novel Reader`** - Long text TTS with tone consistency ← NEW
  - `Podcast` - AI-generated podcast episodes
  - `📋 History` - Generation history browser

- **Key Functions**:
  - `generate_custom_voice()` (line 1435) - Preset voice generation
  - `generate_voice_design()` (line 1543) - Voice design generation
  - `generate_novel()` (line 1543) - **Novel reading generation** ← NEW
  - `estimate_max_tokens()` (line 680) - Token estimation for long text
  - `get_model()` - Model loading with caching

- **Constants**:
  - `MAX_CHARS = 2000` - Max characters per generation (line 444)
  - `LANGUAGES` - Supported language list

### 3.2 Audio Processing (`audio/`)

#### `model_loader.py`
- `get_model(model_name)` - Load/cache Qwen3TTSModel
- Device detection (CUDA/MPS/CPU)
- FlashAttention 2 support
- LRU eviction for model cache

#### `generator.py`
- `generate_dialogue_audio()` - Generate single dialogue audio
- `generate_transcript_audio()` - Generate full podcast transcript
- `_split_text_into_chunks()` - **Sentence-based chunking** (line 162)
  - Splits at sentence boundaries (。！？)
  - Target: 120 chars, Max: 150 chars, Min: 50 chars
- `_crossfade_audio()` - Smooth audio transitions
- `_check_trailing_silence()` - Detect truncated audio
- `_calculate_dynamic_max_tokens()` - Adaptive token limits

#### `novel_reader.py` ← NEW
- `smart_split()` - **Semantic chunking for novels**
  - Priority: Paragraph break > sentence end > clause end
  - Ensures each chunk ends with sentence-ending punctuation
- `NovelReader` class - Core novel reading engine
  - Context anchoring (instruct carries previous chunk's last sentence)
  - Fixed random seed for consistency
  - Low temperature (0.25) + low top_k (30)
  - RMS normalization across chunks
  - Crossfade concatenation
- `synthesize_to_mp3()` - Full pipeline: split → synthesize → concat → MP3

### 3.3 Standalone Novel TTS Scripts

These are independent Textual TUI applications:

| Script | Key Features |
|--------|----------------|
| `my-novel-tts.py` | Basic smart split, Textual TUI, batch processing |
| `gemini-tts.py` | + Fixed seed, forced trailing punctuation |
| `grok-tts.py` | + Instruct prefix, larger chunks (300 chars) |
| `gpt-tts.py` | Simplified version |

**Note**: The new `audio/novel_reader.py` and Novel Reader Tab in the main UI supersede these standalone scripts with better integration and more features.

---

## 4. TTS Generation Flow

### 4.1 Preset Voice Generation (Existing)

```
User Input (text, speaker, params)
    ↓
generate_custom_voice() [qwen_tts_ui.py:1435]
    ↓
get_model(model_name) [audio/model_loader.py]
    ↓
model.generate_custom_voice(
    text, speaker, language, instruct,
    temperature, top_k, top_p,
    repetition_penalty, max_new_tokens, ...
)
    ↓
Save WAV + Save to History
    ↓
Return audio path + status
```

### 4.2 Novel Reading Generation (NEW)

```
User Input (text/file, speaker, params)
    ↓
generate_novel() [qwen_tts_ui.py:1543]
    ↓
NovelReader.synthesize_to_mp3() [audio/novel_reader.py]
    ↓
smart_split(text, chunk_size) → list of chunks
    ↓
For each chunk:
  1. Build context_instruct = "接续上文：'{last_sentence}'"
  2. Fix random seed (torch.manual_seed)
  3. model.generate_custom_voice(chunk, instruct=context_instruct, ...)
  4. Normalize RMS of audio
  5. Store audio array
    ↓
Crossfade concatenation of all audio arrays
    ↓
Convert to MP3 (ffmpeg)
    ↓
Return MP3 path + WAV preview + status
```

---

## 5. Tone Drift Solution (Novel Reading)

When generating long text, "tone drift" (语气漂移) occurs because each chunk is generated independently. The solution uses **6 layers of defense**:

| Layer | Method | Implementation |
|-------|--------|----------------|
| 1 | **Sentence-boundary chunking** | `smart_split()` ensures chunks end with 。！？ |
| 2 | **Context anchoring** | `instruct` parameter carries previous chunk's last sentence |
| 3 | **Fixed random seed** | `torch.manual_seed(seed)` for all chunks |
| 4 | **Low temperature** | `temperature=0.25`, `top_k=30` |
| 5 | **RMS normalization** | Match volume levels across chunks |
| 6 | **Crossfade** | Smooth transitions between chunks (30ms fade) |

---

## 6. Key Configuration

### 6.1 Model Paths

Models are stored locally in the project root:
```
Qwen3-TTS-12Hz-1.7B-CustomVoice/
Qwen3-TTS-12Hz-1.7B-Base/
Qwen3-TTS-12Hz-1.7B-VoiceDesign/
Qwen3-TTS-12Hz-0.6B-CustomVoice/
Qwen3-TTS-12Hz-0.6B-Base/
Qwen3-TTS-Tokenizer-12Hz/
```

### 6.2 Supported Speakers (Preset Voices)

See `audio/novel_reader.py` MODELS dict and `qwen_tts_ui.py` `_get_preset_speaker_choices()`.

### 6.3 TTS Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.3 | Sampling temperature (lower = more consistent) |
| `top_k` | 50 | Top-k sampling |
| `top_p` | 0.85 | Top-p sampling |
| `repetition_penalty` | 1.0 | Penalty for repeated tokens |
| `max_new_tokens` | auto | Max tokens (auto-calculated) |
| `subtalker_temperature` | 0.3 | Sub-talker temperature |
| `subtalker_top_k` | 50 | Sub-talker top-k |
| `subtalker_top_p` | 0.85 | Sub-talker top-p |

**Novel Reader Defaults** (optimized for consistency):
- `temperature`: 0.25
- `top_k`: 30
- `chunk_size`: 500
- `seed`: 42

---

## 7. Common Tasks Guide

### 7.1 Adding a New UI Tab

1. Locate the Tabs section in `qwen_tts_ui.py` (~line 2780)
2. Add a new `with gr.TabItem("Tab Name", id="tabid"):` block
3. Create the generation function (reference `generate_novel()` as example)
4. Wire up the button click event with `.then()` chain

### 7.2 Modifying Text Chunking

- **For podcast/dialogue**: Edit `_split_text_into_chunks()` in `audio/generator.py`
- **For novel reading**: Edit `smart_split()` in `audio/novel_reader.py`

### 7.3 Adding a New Model

1. Download model weights to project root
2. Add to `MODELS` dict in relevant files:
   - `audio/novel_reader.py` (for Novel Reader)
   - `qwen_tts_ui.py` (for UI radio buttons)
3. Update UI choices in the appropriate Tab

### 7.4 Debugging TTS Issues

1. **Check model loading**: Look for "Model loaded" messages
2. **Check chunking**: Enable progress output in `generate_novel()`
3. **Check audio quality**: Use `_check_trailing_silence()` in `audio/generator.py`
4. **Check tone drift**: Verify seed is fixed, context anchoring is working

---

## 8. Important Implementation Details

### 8.1 Context Anchoring in Novel Reader

The `instruct` parameter is used to carry forward context:
```python
if prev_last_sentence and i > 0:
    context_instruct = f"接续上文：'{prev_last_sentence}'。{context_instruct}"
```

This tells the model to maintain continuity with the previous chunk's ending.

### 8.2 Smart Split Algorithm

Priority order:
1. **Paragraph breaks** (`\n\n`) - split first
2. **Sentence ends** (。！？…；) - split at punctuation
3. **Clause ends** (，、：「」 etc.) - fallback split
4. **Hard cut** at `max_chars` if no punctuation found

Each chunk is forced to end with sentence-ending punctuation (appends "。" if missing).

### 8.3 Gradio Progress Pattern

The project uses Gradio's `gr.Progress()` for progress tracking:
```python
def generate_novel(..., progress=gr.Progress()):
    progress(0.1, desc="Processing...")
    # ... work ...
    progress(1.0, desc="Complete!")
```

---

## 9. Dependencies

See `requirements.txt` for full list. Key dependencies:
- `torch` - PyTorch
- `gradio` - Web UI framework
- `qwen-tts` - Qwen3-TTS model library
- `soundfile` - Audio file I/O
- `numpy` - Array operations
- `textual` - TUI framework (for standalone scripts)
- `moviepy` - Audio/video editing (podcast combiner)
- `ffmpeg` - Audio conversion (system dependency)

---

## 10. Quick Reference

**Main entry point**: `python qwen_tts_ui.py`

**Novel Reader module**: `audio/novel_reader.py`

**Key function for novel generation**: `generate_novel()` in `qwen_tts_ui.py`

**Text chunking for novels**: `smart_split()` in `audio/novel_reader.py`

**Model loading**: `get_model()` in `audio/model_loader.py`

**Tone drift solution**: 6-layer defense in `audio/novel_reader.py` (see Section 5)

---

*Last updated: 2026-05-08*
*Author: AI Agent (opencode)*
