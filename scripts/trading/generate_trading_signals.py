#!/usr/bin/env python3
"""
Trading Signal Generator Script

This script generates trading signals based on technical analysis of market data.
It's designed to be run as a cron job, typically at market close (4 PM EST).

Example crontab entry:
0 16 * * 1-5 /path/to/venv/bin/python /path/to/generate_trading_signals.py
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from crypto_trading.common.db.handler import DatabaseHandler
from crypto_trading.common.utils.logging import setup_logging
from crypto_trading.trading.strategy.config import TradingStrategyConfig
from crypto_trading.trading.strategy.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


def save_outputs(
    signals: Dict[str, pd.DataFrame],
    biases: pd.DataFrame,
    config: TradingStrategyConfig,
) -> None:
    """
    Save signal outputs to configured destinations.

    Args:
        signals: Dictionary of signal DataFrames
        biases: Market biases DataFrame
        config: Trading strategy configuration
    """
    output_config = config.get_output_config()

    # Save to local files if enabled
    if output_config["local"]["enabled"]:
        try:
            output_path = Path(output_config["local"]["path"])
            output_path.mkdir(parents=True, exist_ok=True)

            # Save biases
            biases.to_csv(output_path / "market_biases.csv")
            logger.info(f"Saved market biases to {output_path / 'market_biases.csv'}")

            # Save signals
            for name, df in signals.items():
                file_path = output_path / f"{name}.csv"
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {name} to {file_path}")

        except Exception as e:
            logger.error(f"Error saving local output: {str(e)}")
            raise

    # Upload to S3 if enabled
    if output_config["s3"]["enabled"]:
        try:
            s3_config = output_config["s3"]
            s3_client = boto3.client("s3")

            # Upload biases
            biases_buffer = biases.to_csv().encode()
            s3_client.put_object(
                Bucket=s3_config["bucket"],
                Key="market_biases.csv",
                Body=biases_buffer,
                ACL=s3_config["acl"],
                ContentType=s3_config["content_type"],
                ContentDisposition=s3_config["content_disposition"],
            )
            logger.info(f"Uploaded market biases to S3 bucket {s3_config['bucket']}")

            # Upload signals
            for name, df in signals.items():
                csv_buffer = df.to_csv(index=False).encode()
                s3_client.put_object(
                    Bucket=s3_config["bucket"],
                    Key=f"{name}.csv",
                    Body=csv_buffer,
                    ACL=s3_config["acl"],
                    ContentType=s3_config["content_type"],
                    ContentDisposition=s3_config["content_disposition"],
                )
                logger.info(f"Uploaded {name} to S3 bucket {s3_config['bucket']}")

        except Exception as e:
            logger.warning(f"Error uploading to S3: {str(e)}")
            logger.warning("Continuing with local files only")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate Trading Signals")
    parser.add_argument(
        "--config",
        default="references/trading_strategy_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    try:
        # Initialize components
        config = TradingStrategyConfig(args.config)
        log_config = config.get_logging_config()
        setup_logging(
            level=(
                logging.DEBUG
                if args.debug
                else getattr(logging, log_config.get("level", "INFO"))
            ),
            log_format=log_config.get("format"),
            log_file=log_config.get("file"),
        )

        logger.info("Starting signal generation")
        start_time = datetime.now()

        # Setup components
        db_handler = DatabaseHandler()
        signal_generator = SignalGenerator(db_handler, config)

        # Generate signals
        data = signal_generator.fetch_data()
        logger.info(f"Fetched data for {len(data['instrument'].unique())} instruments")

        features = signal_generator.generate_features(data)
        logger.info("Generated technical features")

        biases, signals = signal_generator.generate_signals(features)
        logger.info("Generated trading signals")
        logger.info(f"Market Biases:\n{biases.to_string()}")

        # Log signal counts
        for name, df in signals.items():
            logger.info(f"{name.capitalize()}: {len(df)} signals generated")

        # Save outputs
        save_outputs(signals, biases, config)

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Signal generation completed successfully in {execution_time:.2f} seconds"
        )

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
