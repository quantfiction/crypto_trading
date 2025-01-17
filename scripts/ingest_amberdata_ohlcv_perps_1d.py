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


class AmberdataOHLCVIngestor:
    def __init__(self):
        self.handler = AmberdataHandler()
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

    def ingest_ohlcv_data(self, days: int = 61):
        """Main method to ingest OHLCV data for all perpetual contracts"""
        try:
            # Calculate date range
            end_date = pd.Timestamp.now(tz="UTC")
            start_date = end_date - pd.Timedelta(days=days)

            # Get list of perpetual contracts
            with self.db_connection() as conn:
                # Query active perpetual contracts from database
                perpetual_contracts = conn.execute(
                    """
                        SELECT er.exchange, er.instrument 
                        FROM amberdata.exchange_reference er
                        JOIN amberdata.ohlcv_info_futures oif
                            ON er.exchange = oif.exchange
                            AND er.instrument = oif.instrument
                        WHERE er.contract_period = 'perpetual'
                            AND oif.trading_end_date > CAST(? AS TIMESTAMP)
                    """,
                    [start_date.strftime("%Y-%m-%d %H:%M:%S")],
                ).fetch_df()

                if perpetual_contracts.empty:
                    logger.warning("No active perpetual contracts found in database")
                    return

            # Group contracts by exchange
            grouped_contracts = perpetual_contracts.groupby("exchange")

            # Process each exchange's contracts in batches of 50
            batch_size = 50
            for exchange, contracts in grouped_contracts:
                total_instruments = len(contracts)
                for i in range(0, total_instruments, batch_size):
                    batch = contracts[i : i + batch_size]
                    instruments = batch["instrument"].tolist()

                    try:
                        # Format instruments as single comma-separated string
                        instrument_str = ",".join(instruments)

                        # Fetch OHLCV data for batch
                        df_ohlcv = self.handler.get_ohlcv_data_futures(
                            exchange=exchange,
                            instrument=instrument_str,
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d"),
                            time_interval="days",
                        )

                        if not df_ohlcv.empty:
                            # Store data
                            self._store_data(df_ohlcv)
                            logger.info(
                                f"Successfully ingested OHLCV data for batch {i//batch_size + 1} "
                                f"({len(instruments)} instruments) on {exchange}"
                            )
                        else:
                            logger.warning(
                                f"No OHLCV data found for batch {i//batch_size + 1} on {exchange}"
                            )
                    except Exception as e:
                        logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error ingesting OHLCV data: {e}")
            raise

    def _store_data(self, df: pd.DataFrame):
        """Store data in DuckDB with transaction management"""
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

                        # Create new table with proper primary key
                        cursor.execute(
                            """
                                CREATE TABLE IF NOT EXISTS amberdata.ohlcv_perps_1d (
                                    instrument VARCHAR,
                                    open DOUBLE PRECISION,
                                    high DOUBLE PRECISION,
                                    low DOUBLE PRECISION,
                                    close DOUBLE PRECISION,
                                    volume DOUBLE PRECISION,
                                    exchange VARCHAR,
                                    datetime TIMESTAMP,
                                    -- PRIMARY KEY (exchange, instrument, datetime)
                                )
                            """
                        )

                        cursor.execute(
                            """
                                CREATE UNIQUE INDEX IF NOT EXISTS ohlcv_perps_1d_unique_idx
                                ON amberdata.ohlcv_perps_1d (exchange, instrument, datetime);
                            """
                        )

                        # Upsert new data
                        cursor.execute(
                            """
                            INSERT INTO amberdata.ohlcv_perps_1d (instrument, open, high, low, close, volume, exchange, datetime)
                            SELECT instrument, open, high, low, close, volume, exchange, datetime FROM temp_df
                            ON CONFLICT (exchange, instrument, datetime) DO UPDATE SET
                                open = excluded.open,
                                high = excluded.high,
                                low = excluded.low,
                                close = excluded.close,
                                volume = excluded.volume
                        """
                        )

                        # Commit transaction
                        cursor.execute("COMMIT")
                        logger.info("OHLCV data successfully updated in DuckDB")

                    except Exception as e:
                        # Rollback on error
                        cursor.execute("ROLLBACK")
                        logger.error(f"Error updating OHLCV data: {e}")
                        raise
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise


if __name__ == "__main__":
    try:
        ingestor = AmberdataOHLCVIngestor()
        ingestor.ingest_ohlcv_data()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise
