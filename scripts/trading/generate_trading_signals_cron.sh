#!/usr/bin/env bash

# Source user's bash profile to get the same environment as interactive shell
source /home/quantfiction/.bashrc

# Set absolute paths
REPO_PATH="/home/quantfiction/repositories/crypto_trading"
VENV_PATH="/home/quantfiction/.local/share/virtualenvs/crypto_trading-pCflR77s"
LOG_PATH="$REPO_PATH/logs"

# Ensure log directory exists
mkdir -p "$LOG_PATH"

# Set environment variables
export PYTHONPATH="$REPO_PATH"
# AWS credentials for S3 uploads
export AWS_ACCESS_KEY_ID=$(grep AWS_ACCESS_KEY_ID "$REPO_PATH/.env" | cut -d '=' -f2 | tr -d '"' | tr -d "'" | tr -d ' ')
export AWS_SECRET_ACCESS_KEY=$(grep AWS_SECRET_ACCESS_KEY "$REPO_PATH/.env" | cut -d '=' -f2 | tr -d '"' | tr -d "'" | tr -d ' ')

# Use the virtual environment's Python directly instead of relying on pipenv
PYTHON_PATH="$VENV_PATH/bin/python"

# Run the script
cd "$REPO_PATH"
$PYTHON_PATH scripts/trading/generate_trading_signals.py >> "$LOG_PATH/trading_signals.log" 2>&1

# Check if script ran successfully
if [ $? -eq 0 ]; then
    echo "$(date) - Trading signals generated successfully" >> "$LOG_PATH/trading_signals.log"
else
    echo "$(date) - Trading signal generation failed" >> "$LOG_PATH/trading_signals.log"
fi