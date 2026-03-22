#!/bin/bash
# -------------------------------------------------------
# Build BracketTracker-mac.zip for distribution.
# Upload the resulting zip to GitHub Releases.
# -------------------------------------------------------

set -e
cd "$(dirname "$0")"

OUT_DIR="BracketTracker"
ZIP_NAME="BracketTracker-mac.zip"

rm -rf "$OUT_DIR" "$ZIP_NAME"
mkdir -p "$OUT_DIR/templates"

# App files
cp app.py "$OUT_DIR/"
cp bracket_scraper.py "$OUT_DIR/"
cp espn_scraper.py "$OUT_DIR/"
cp requirements-local.txt "$OUT_DIR/"
cp templates/index.html "$OUT_DIR/templates/"
cp templates/results.html "$OUT_DIR/templates/"

# Launcher scripts
cp install.command "$OUT_DIR/"
cp "Bracket Tracker.command" "$OUT_DIR/"
chmod +x "$OUT_DIR/install.command" "$OUT_DIR/Bracket Tracker.command"

# Build zip
zip -r "$ZIP_NAME" "$OUT_DIR"
rm -rf "$OUT_DIR"

echo ""
echo "Created $ZIP_NAME"
echo "Upload it to: https://github.com/nstrimple/march-madness-autoresearch/releases"
