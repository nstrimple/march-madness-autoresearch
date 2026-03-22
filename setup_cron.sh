#!/usr/bin/env bash
# Sets up a cron job to run the bracket differential analyzer
# Usage: bash setup_cron.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$(which python3)"
LOG_FILE="$SCRIPT_DIR/bracket_scraper.log"
OUTPUT_TSV="$SCRIPT_DIR/differential_results.tsv"

# Default: run every 2 hours between 8am and 10pm
CRON_SCHEDULE="0 8,10,12,14,16,18,20,22 * * *"

echo "=== CBS Bracket Differential - Cron Setup ==="
echo ""
echo "Script dir : $SCRIPT_DIR"
echo "Python     : $PYTHON"
echo "Log file   : $LOG_FILE"
echo "Schedule   : $CRON_SCHEDULE  (every 2 hrs, 8am-10pm)"
echo ""

# Verify .env exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "ERROR: .env file not found."
    echo "Copy .env.example to .env and fill in your credentials first."
    exit 1
fi

# Build the cron line
CRON_CMD="$CRON_SCHEDULE cd '$SCRIPT_DIR' && '$PYTHON' '$SCRIPT_DIR/bracket_scraper.py' --output '$OUTPUT_TSV' >> '$LOG_FILE' 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -qF "bracket_scraper.py"; then
    echo "Cron job already exists. Updating..."
    # Remove old entry
    crontab -l 2>/dev/null | grep -vF "bracket_scraper.py" | crontab -
fi

# Add new cron entry
(crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

echo "Cron job installed! Verify with: crontab -l"
echo ""
echo "The job will:"
echo "  - Run every 2 hours from 8am to 10pm"
echo "  - Log output to: $LOG_FILE"
echo "  - Save TSV results to: $OUTPUT_TSV"
echo ""
echo "To run immediately for a test:"
echo "  cd '$SCRIPT_DIR' && python3 bracket_scraper.py"
echo ""
echo "To remove the cron job:"
echo "  crontab -l | grep -v 'bracket_scraper.py' | crontab -"
