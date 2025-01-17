import logging
import time
from typing import List, Dict, Optional
import duckdb
from duckdb import DuckDBPyConnection
from contextlib import contextmanager
import pandas as pd
import requests
import re
from dotenv import dotenv_values, find_dotenv
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AmberdataIngestor:
    def __init__(self):
        self.config = self._load_config()
        self.base_url = "https://api.amberdata.com/markets/futures/exchanges/reference"
        self.headers = {
            "accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "x-api-key": self.config.get("AMBERDATA_API_KEY"),
        }
        self.data_path = Path(__file__).parent.parent / "data"
        self.db_path = self.data_path / "crypto_data.db"

    @contextmanager
    def db_connection(self) -> DuckDBPyConnection:  # type: ignore
        """Context manager for handling DuckDB connections"""
        conn = None
        try:
            conn = duckdb.connect(str(self.db_path), read_only=False)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _load_config(self) -> Dict:
        """Load and validate configuration"""
        config = dotenv_values(find_dotenv())
        if not config.get("AMBERDATA_API_KEY"):
            raise ValueError("AMBERDATA_API_KEY not found in .env file")
        return config

    @staticmethod
    def camel_to_snake(name: str) -> str:
        """Convert camelCase to snake_case"""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    @staticmethod
    def convert_df_columns_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to snake_case"""
        df.columns = [AmberdataIngestor.camel_to_snake(col) for col in df.columns]
        return df

    @staticmethod
    def convert_df_columns_to_datetime(
        df: pd.DataFrame, columns: List[str], unit: str = "ms"
    ) -> pd.DataFrame:
        """Convert specified columns to datetime with error handling"""
        for column in columns:
            if column in df.columns:
                try:
                    # Convert to numeric first to handle potential string values
                    df[column] = pd.to_numeric(df[column], errors="coerce")
                    # Handle potential overflow by capping values
                    max_timestamp = 2**53 - 1  # JavaScript max safe integer
                    df[column] = df[column].where(df[column] <= max_timestamp, pd.NaT)
                    df[column] = pd.to_datetime(df[column], unit=unit, errors="coerce")
                except Exception as e:
                    logger.warning(f"Error converting {column} to datetime: {e}")
                    df[column] = pd.NaT
        return df

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _fetch_data(self, url: str, params: Dict) -> Optional[Dict]:
        """Fetch data from API with retry logic"""
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def ingest_exchange_reference_data(self):
        """Main method to ingest exchange reference data"""
        params = {
            "exchange": "binance,bybit",
            "includeInactive": "true",
        }

        list_instruments = []
        url = self.base_url
        total_records = 0

        while url:
            try:
                response = self._fetch_data(url, params)
                payload = response.get("payload", {})
                metadata = payload.get("metadata", {})
                data = payload.get("data", {})

                chunk_instruments = pd.DataFrame(data)
                list_instruments.append(chunk_instruments)
                total_records += len(chunk_instruments)

                url = metadata.get("next")
                # time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Error processing data: {e}")
                break

        logger.info(f"Total records fetched: {total_records}")

        if list_instruments:
            datetime_cols = ["listing_timestamp", "contract_expiration_timestamp"]
            # Filter out empty DataFrames and handle all-NA columns
            valid_instruments = [df for df in list_instruments if not df.empty]
            if valid_instruments:
                # Get common columns that have at least some non-NA values
                common_cols = set.intersection(
                    *[set(df.columns[df.notna().any()]) for df in valid_instruments]
                )

                df_instruments = (
                    pd.concat(
                        [df[list(common_cols)] for df in valid_instruments],
                        axis=0,
                        ignore_index=True,
                    )
                    .pipe(self.convert_df_columns_to_snake_case)
                    .pipe(self.convert_df_columns_to_datetime, columns=datetime_cols)
                )
            else:
                logger.warning("No valid instruments found after filtering")
                return

            self._store_data(df_instruments)

    def _store_data(self, df: pd.DataFrame):
        """Store data in DuckDB with transaction management"""
        try:
            with self.db_connection() as conn:
                with conn.cursor() as cursor:
                    # Start transaction
                    cursor.execute("BEGIN TRANSACTION")

                    try:
                        # Drop existing table if exists
                        cursor.execute(
                            "DROP TABLE IF EXISTS amberdata.exchange_reference"
                        )

                        # Register temporary dataframe
                        cursor.register("temp_df", df)

                        # Create new table with data
                        cursor.execute(
                            """
                            CREATE TABLE amberdata.exchange_reference AS
                            SELECT * FROM temp_df
                            ORDER BY exchange, instrument
                        """
                        )

                        # Commit transaction
                        cursor.execute("COMMIT")
                        logger.info("Data successfully stored in DuckDB")

                    except Exception as e:
                        # Rollback on error
                        cursor.execute("ROLLBACK")
                        logger.error(f"Error storing data: {e}")
                        raise
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise


if __name__ == "__main__":
    try:
        ingestor = AmberdataIngestor()
        ingestor.ingest_exchange_reference_data()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise
