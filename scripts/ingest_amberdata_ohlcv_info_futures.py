import logging
import duckdb
from duckdb import DuckDBPyConnection
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from crypto_trading.amberdata import AmberdataHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AmberdataOHLCVInfoIngestor:
    def __init__(self):
        self.handler = AmberdataHandler()
        self.data_path = Path(__file__).parent.parent / "data"
        self.db_path = self.data_path / "crypto_data.db"
        self.exchanges = ["binance", "bybit"]  # Default exchanges to process

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

    def _create_table(self, conn: DuckDBPyConnection):
        """Create the OHLCV info table if it doesn't exist"""
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS amberdata.ohlcv_info_futures (
                    exchange VARCHAR,
                    instrument VARCHAR,
                    trading_start_date TIMESTAMP,
                    trading_end_date TIMESTAMP,
                    active BOOLEAN,
                    updated_at TIMESTAMP
                );
            """
        )

        conn.execute(
            """
                CREATE UNIQUE INDEX IF NOT EXISTS ohlcv_info_futures_unique_idx
                ON amberdata.ohlcv_info_futures (exchange, instrument);
            """
        )

    def _store_data(self, df: pd.DataFrame):
        """Store OHLCV info data in DuckDB"""
        try:
            with self.db_connection() as conn:
                with conn.cursor() as cursor:
                    # Start transaction
                    cursor.execute("BEGIN TRANSACTION")

                    try:
                        # Register temporary dataframe
                        cursor.register("temp_df", df)

                        # Create schema if not exists
                        cursor.execute("CREATE SCHEMA IF NOT EXISTS amberdata")
                        self._create_table(cursor)

                        # Upsert new data
                        cursor.execute(
                            """
                                INSERT INTO amberdata.ohlcv_info_futures
                                SELECT 
                                    exchange, 
                                    instrument, 
                                    trading_start_date, 
                                    trading_end_date, 
                                    active,
                                    updated_at
                                FROM temp_df
                                ON CONFLICT (exchange, instrument) DO UPDATE SET
                                    trading_start_date = excluded.trading_start_date,
                                    trading_end_date = excluded.trading_end_date,
                                    active = excluded.active,
                                    updated_at = excluded.updated_at;
                        """
                        )

                        # Commit transaction
                        cursor.execute("COMMIT")
                        logger.info("OHLCV info successfully updated in DuckDB")

                    except Exception as e:
                        # Rollback on error
                        cursor.execute("ROLLBACK")
                        logger.error(f"Error updating OHLCV info: {e}")
                        raise
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise

    def ingest_ohlcv_info(self):
        """Main method to ingest OHLCV information for futures contracts"""
        try:
            for exchange in self.exchanges:
                logger.info(f"Processing OHLCV info for exchange: {exchange}")
                try:
                    # Fetch all OHLCV info (handler handles pagination internally)
                    response = self.handler.get_ohlcv_info_futures(
                        exchange=exchange,
                        include_inactive=True,
                        time_format="ms",
                        time_interval="days",
                    )

                    # Use the DataFrame directly from the handler
                    df_ohlcv_info = response[
                        [
                            "exchange",
                            "instrument",
                            "trading_start_date",
                            "trading_end_date",
                            "active",
                        ]
                    ].assign(updated_at=pd.Timestamp.now(tz="UTC"))

                    # Store data
                    self._store_data(df_ohlcv_info)

                except Exception as e:
                    logger.error(f"Error processing OHLCV info for {exchange}: {e}")

        except Exception as e:
            logger.error(f"Error ingesting OHLCV info: {e}")
            raise


if __name__ == "__main__":
    try:
        ingestor = AmberdataOHLCVInfoIngestor()
        ingestor.ingest_ohlcv_info()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise
