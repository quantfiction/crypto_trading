#!/usr/bin/env python3
"""
Script to test database connections for multiple database types.
"""

import logging
import os
from typing import Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text

from crypto_trading.common.db.handler import DatabaseHandler
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_database_connection(db_config: Dict[str, Any], db_name: str, test_query: str):
    """Tests the database connection and executes a simple query."""
    try:
        db_handler = DatabaseHandler(db_config)
        result = db_handler.query_to_df(test_query)
        logger.info(f"Successfully connected to {db_name} and executed test query.")
        logger.info(f"Result:\n{result.to_string()}")
    except Exception as e:
        logger.error(f"Failed to connect to {db_name} or execute test query: {e}")


def main():
    try:
        load_dotenv()

        # DuckDB configuration
        duckdb_config = {"db_type": "duckdb", "path": "data/crypto_data.db"}
        # Test DuckDB connection
        try:
            db_handler = DatabaseHandler(duckdb_config)
            query = "SELECT * FROM amberdata.exchange_reference LIMIT 5"
            result = db_handler.query_to_df(query)
            logger.info("Successfully connected to DuckDB and executed test query.")
            logger.info(f"DuckDB Result:\n{result.to_string()}")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB or execute test query: {e}")

        # Postgres configuration
        postgres_config = {
            "db_type": "postgresql",
            "host": os.environ["SMA_DATA_HOST"],
            "user": os.environ["SMA_DATA_USER"],
            "password": os.environ["SMA_DATA_PASSWORD"],
            "database": os.environ["SMA_DATA_DB"],
        }
        test_database_connection(
            postgres_config, "Postgres", "SELECT * FROM sma_tweets_copy_dev LIMIT 5"
        )

        # Singlestore configuration
        singlestore_config = {
            "db_type": "singlestore",
            "host": os.environ["SINGLESTORE_HOST"],
            "user": os.environ["SINGLESTORE_USER"],
            "password": os.environ["SINGLESTORE_PASSWORD"],
            "database": os.environ["SINGLESTORE_DB"],
        }
        test_database_connection(
            singlestore_config,
            "Singlestore",
            "SELECT * FROM social.labs_near_tweets LIMIT 5",
        )

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
