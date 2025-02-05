#!/usr/bin/env bash

# Source user's bash profile to get the same environment as interactive shell
source /home/quantfiction/.bashrc

# Set absolute paths
REPO_PATH="/home/quantfiction/repositories/crypto_trading"
VENV_PATH="/home/quantfiction/snap/code/179/.local/share/virtualenvs/crypto_trading-pCflR77s"
LOG_PATH="$REPO_PATH/logs"

# Ensure log directory exists
mkdir -p "$LOG_PATH"

# Set environment variables
export PYTHONPATH="$REPO_PATH"
export AMBERDATA_API_KEY=$(grep AMBERDATA_API_KEY "$REPO_PATH/.env" | cut -d '=' -f2 | tr -d '"' | tr -d "'" | tr -d ' ')

# Use the virtual environment's Python directly instead of relying on pipenv
PYTHON_PATH="$VENV_PATH/bin/python"

# Run the script
cd "$REPO_PATH"
$PYTHON_PATH scripts/ingest_amberdata_exchange_reference.py >> "$LOG_PATH/amberdata_exchange_reference_ingest.log" 2>&1

# Check if script ran successfully
if [ $? -eq 0 ]; then
    echo "$(date) - Script executed successfully" >> "$LOG_PATH/amberdata_exchange_reference_ingest.log"
else
    echo "$(date) - Script execution failed" >> "$LOG_PATH/amberdata_exchange_reference_ingest.log"
fi
