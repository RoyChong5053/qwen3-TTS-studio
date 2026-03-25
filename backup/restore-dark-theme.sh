#!/bin/bash
# Restore dark theme after git pull
# This script patches the code instead of overwriting files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CSS_FILE="$PROJECT_DIR/static/dark-theme.css"
MAIN_FILE="$PROJECT_DIR/qwen_tts_ui.py"

echo "=== Dark Theme Restore Script ==="
echo ""

# Step 1: Ensure static directory exists
mkdir -p "$PROJECT_DIR/static"

# Step 2: Copy CSS file
echo "[1/3] Restoring dark-theme.css..."
cp "$SCRIPT_DIR/dark-theme.css" "$CSS_FILE"

# Step 3: Check if patch is needed
echo "[2/3] Checking qwen_tts_ui.py..."
if grep -q 'custom_css = (Path' "$MAIN_FILE"; then
    echo "       Already patched (external CSS loading)"
else
    echo "       Patching to use external CSS..."
    
    # Check if it has the old inline CSS
    if grep -q 'custom_css = """' "$MAIN_FILE"; then
        echo "       Found inline CSS, replacing with external load..."
        
        # Find the line numbers
        START_LINE=$(grep -n 'custom_css = """' "$MAIN_FILE" | head -1 | cut -d: -f1)
        
        # Find the closing """
        AWK_SCRIPT='
        /^custom_css = """/ {
            start = NR
            in_css = 1
            next
        }
        in_css && /^"""/ {
            print NR
            exit
        }
        '
        END_LINE=$(awk "$AWK_SCRIPT" "$MAIN_FILE")
        
        if [ -z "$END_LINE" ]; then
            echo "       ERROR: Could not find end of custom_css"
            exit 1
        fi
        
        # Create patched file
        head -n $((START_LINE - 1)) "$MAIN_FILE" > "$MAIN_FILE.tmp"
        echo 'custom_css = (Path(__file__).parent / "static" / "dark-theme.css").read_text()' >> "$MAIN_FILE.tmp"
        tail -n +$((END_LINE + 1)) "$MAIN_FILE" >> "$MAIN_FILE.tmp"
        mv "$MAIN_FILE.tmp" "$MAIN_FILE"
        
        echo "       Patched successfully"
    else
        echo "       WARNING: Could not find custom_css definition"
        echo "       Manual intervention may be required"
    fi
fi

# Step 4: Update other UI files
echo "[3/3] Restoring UI component files..."
cp "$SCRIPT_DIR/persona.py" "$PROJECT_DIR/ui/persona.py" 2>/dev/null && echo "       persona.py" || echo "       persona.py (skipped)"
cp "$SCRIPT_DIR/progress.py" "$PROJECT_DIR/ui/progress.py" 2>/dev/null && echo "       progress.py" || echo "       progress.py (skipped)"
cp "$SCRIPT_DIR/draft_preview.py" "$PROJECT_DIR/ui/draft_preview.py" 2>/dev/null && echo "       draft_preview.py" || echo "       draft_preview.py (skipped)"

echo ""
echo "=== Dark theme restored! ==="
echo ""
echo "Modified files:"
echo "  - static/dark-theme.css (new)"
echo "  - qwen_tts_ui.py (1 line patched)"
echo "  - ui/persona.py"
echo "  - ui/progress.py"
echo "  - ui/draft_preview.py"
