import logging
from pathlib import Path
import pandas as pd
from duckdb import DuckDBPyConnection
from crypto_trading.common.db.handler import DatabaseHandler


class BaseIngestor:
    """Base class for data ingestion operations"""

    def __init__(self, db_path: Path):
        self.db_handler = DatabaseHandler(db_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_schema(
        self, cursor: DuckDBPyConnection, schema: str = "amberdata"
    ) -> None:
        """Create a schema if it doesn't exist"""
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    def register_dataframe(
        self, cursor: DuckDBPyConnection, df: pd.DataFrame, name: str = "temp_df"
    ) -> None:
        """Register a temporary DataFrame in DuckDB"""
        cursor.register(name, df)

    def store_data(
        self, df: pd.DataFrame, table_name: str, schema: str = "amberdata"
    ) -> None:
        """
        Store DataFrame in database with proper schema creation and error handling.
        Subclasses should override this method to implement specific storage logic.
        """
        raise NotImplementedError("Subclasses must implement store_data method")

    def process_batch(
        self, items: list, batch_size: int, process_fn, description: str = "items"
    ) -> None:
        """Process items in batches with proper logging and error handling"""
        total_items = len(items)
        for i in range(0, total_items, batch_size):
            batch = items[i : i + batch_size]
            try:
                process_fn(batch)
                self.logger.info(
                    f"Processed batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size} "
                    f"({len(batch)} {description})"
                )
            except Exception as e:
                self.logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                continue

    def validate_data(self, df: pd.DataFrame, required_columns: list) -> bool:
        """Validate DataFrame has required columns"""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
