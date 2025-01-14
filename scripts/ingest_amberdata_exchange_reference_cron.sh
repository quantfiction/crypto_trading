#!/usr/bin/env bash

# Set environment variables
export PYTHONPATH=/home/quantfiction/repositories/crypto_trading
export AMBERDATA_API_KEY=$(grep AMBERDATA_API_KEY /home/quantfiction/repositories/crypto_trading/.env | cut -d '=' -f2)

# Navigate to project directory
cd /home/quantfiction/repositories/crypto_trading

# Get pipenv virtualenv Python path
PYTHON_PATH=$(pipenv --venv)/bin/python

# Run the script using pipenv's Python
$PYTHON_PATH scripts/ingest_amberdata_exchange_reference.py >> /home/quantfiction/repositories/crypto_trading/logs/amberdata_ingest.log 2>&1

# Check if script ran successfully
if [ $? -eq 0 ]; then
    echo "$(date) - Script executed successfully" >> /home/quantfiction/repositories/crypto_trading/logs/amberdata_ingest.log
else
    echo "$(date) - Script execution failed" >> /home/quantfiction/repositories/crypto_trading/logs/amberdata_ingest.log
fi
