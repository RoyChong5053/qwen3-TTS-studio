# Dark Theme Backup

This backup preserves the dark theme modifications for Qwen3-TTS Studio.

## Files

- `dark-theme.css` - The dark theme CSS variables and styles
- `qwen_tts_ui.py` - Main UI file (patched to load external CSS)
- `persona.py` - Persona UI component
- `progress.py` - Progress indicator UI component
- `draft_preview.py` - Draft preview UI component

## Usage

After `git pull` overwrites your changes, run:

```bash
cd /home/roychong/app/qwen3-TTS-studio
./backup/restore-dark-theme.sh
```

The script will:
1. Copy `dark-theme.css` to `static/` directory
2. Patch `qwen_tts_ui.py` to load CSS from file (if needed)
3. Restore modified UI component files

## Manual Restore

If the script fails, manually copy files:

```bash
cp backup/dark-theme.css static/
cp backup/qwen_tts_ui.py .       # Only if major conflicts
cp backup/persona.py ui/
cp backup/progress.py ui/
cp backup/draft_preview.py ui/
```

## Color Scheme

- Background: `#0d0d14` → `#14141f` → `#1a1a2e` → `#2a2a40`
- Text: `#f0f0f5` (bright) / `#8888a0` (muted) / `#555570` (dim)
- Accent: `#8080ff` (blue) / `#50c878` (green) / `#ffb432` (orange)
