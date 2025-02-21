"""Ingestor for Amberdata exchange reference data"""

import logging
from pathlib import Path
import pandas as pd
from sqlalchemy import text

from crypto_trading.ingestion.ingestor import BaseIngestor
from crypto_trading.ingestion.providers.amberdata import AmberdataHandler


class AmberdataOHLCVIngestor(BaseIngestor):
    """Ingestor for Amberdata OHLCV data"""

    def __init__(self, data_path: Path = None):
        if data_path is None:
            data_path = Path(__file__).parents[3] / "data"
        db_config = {
            "db_type": "duckdb",
            "path": f"{data_path}/crypto_data.db",
        }
        super().__init__(db_config)
        self.handler = AmberdataHandler()

    def create_ohlcv_table(self, connection):
        """Create OHLCV table with proper schema"""
        self.create_schema(connection, "amberdata")
        connection.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS amberdata.ohlcv_perps_1d (
                instrument VARCHAR,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                exchange VARCHAR,
                datetime TIMESTAMP
            )
            """
            )
        )
        connection.execute(
            text(
                """
            CREATE UNIQUE INDEX IF NOT EXISTS ohlcv_perps_1d_unique_idx
            ON amberdata.ohlcv_perps_1d (exchange, instrument, datetime);
            """
            )
        )

    def store_data(
        self,
        df: pd.DataFrame,
        table_name: str = "ohlcv_perps_1d",
        schema: str = "amberdata",
    ) -> None:
        """Store OHLCV data with proper upsert logic"""
        required_columns = [
            "instrument",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "exchange",
            "datetime",
        ]
        if not self.validate_data(df, required_columns):
            return

        def transaction_operations(connection):
            self.create_schema(connection, schema)
            self.create_ohlcv_table(connection)
            self.register_dataframe(connection, df)
            connection.execute(
                text(
                    f"""
                INSERT INTO {schema}.{table_name}
                SELECT instrument, open, high, low, close, volume, exchange, datetime
                FROM temp_df
                ON CONFLICT (exchange, instrument, datetime) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume
                """
                )
            )

        self.db_handler.execute_transaction([transaction_operations])

    def get_active_contracts(self, start_date: pd.Timestamp) -> pd.DataFrame:
        """Get list of active perpetual contracts"""
        query = text(
            """
            SELECT er.exchange, er.instrument 
            FROM amberdata.exchange_reference er
            JOIN amberdata.ohlcv_info_futures oif
                ON er.exchange = oif.exchange
                AND er.instrument = oif.instrument
            WHERE er.contract_period = 'perpetual'
                AND oif.trading_end_date > CAST(:start_date AS TIMESTAMP)
            """
        )
        with self.db_handler.engine.connect() as connection:
            result = connection.execute(
                query, {"start_date": start_date.strftime("%Y-%m-%d %H:%M:%S")}
            )
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df

    def fetch_ohlcv_data(
        self,
        exchange: str,
        instruments: list,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a batch of instruments"""
        instrument_str = ",".join(instruments)
        return self.handler.get_ohlcv_data_futures(
            exchange=exchange,
            instrument=instrument_str,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            time_interval="days",
        )

    def ingest_ohlcv_data(self, days: int = 61) -> None:
        """Main method to ingest OHLCV data for all perpetual contracts"""
        try:
            # Calculate date range
            end_date = pd.Timestamp.now(tz="UTC")
            start_date = end_date - pd.Timedelta(days=days)

            # Get active contracts
            contracts_df = self.get_active_contracts(start_date)
            if contracts_df.empty:
                self.logger.warning("No active perpetual contracts found")
                return

            # Process each exchange's contracts
            for exchange, group in contracts_df.groupby("exchange"):
                instruments = group["instrument"].tolist()

                def process_batch(batch: list):
                    df = self.fetch_ohlcv_data(exchange, batch, start_date, end_date)
                    if not df.empty:
                        self.store_data(df)

                self.process_batch(instruments, 50, process_batch, "instruments")

        except Exception as e:
            self.logger.error(f"Error ingesting OHLCV data: {e}")
            raise


class AmberdataOHLCVInfoIngestor(BaseIngestor):
    """Ingestor for Amberdata OHLCV information"""

    def __init__(self, data_path: Path = None):
        if data_path is None:
            data_path = Path(__file__).parents[3] / "data"
        db_config = {
            "db_type": "duckdb",
            "path": f"{data_path}/crypto_data.db",
        }
        super().__init__(db_config)
        self.handler = AmberdataHandler()
        self.exchanges = ["binance", "bybit"]

    def create_info_table(self, connection):
        """Create OHLCV info table with proper schema"""
        connection.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS amberdata.ohlcv_info_futures (
                exchange VARCHAR,
                instrument VARCHAR,
                trading_start_date TIMESTAMP,
                trading_end_date TIMESTAMP,
                active BOOLEAN,
                updated_at TIMESTAMP
            )
            """
            )
        )
        connection.execute(
            text(
                """
            CREATE UNIQUE INDEX IF NOT EXISTS ohlcv_info_futures_unique_idx
            ON amberdata.ohlcv_info_futures (exchange, instrument);
            """
            )
        )

    def store_data(
        self,
        df: pd.DataFrame,
        table_name: str = "ohlcv_info_futures",
        schema: str = "amberdata",
    ) -> None:
        """Store OHLCV info with proper upsert logic"""
        required_columns = [
            "exchange",
            "instrument",
            "trading_start_date",
            "trading_end_date",
            "active",
            "updated_at",
        ]
        if not self.validate_data(df, required_columns):
            return

        def transaction_operations(connection):
            self.create_schema(connection, schema)
            self.create_info_table(connection)
            self.register_dataframe(connection, df)
            connection.execute(
                text(
                    f"""
                INSERT INTO {schema}.{table_name}
                SELECT exchange, instrument, trading_start_date, trading_end_date, active, updated_at
                FROM temp_df
                ON CONFLICT (exchange, instrument) DO UPDATE SET
                    trading_start_date = excluded.trading_start_date,
                    trading_end_date = excluded.trading_end_date,
                    active = excluded.active,
                    updated_at = excluded.updated_at
                """
                )
            )

        self.db_handler.execute_transaction([transaction_operations])


class AmberdataExchangeReferenceIngestor(BaseIngestor):
    """Ingestor for Amberdata exchange reference data"""

    def __init__(self, data_path: Path = None):
        if data_path is None:
            data_path = Path(__file__).parents[3] / "data"
        db_config = {
            "db_type": "duckdb",
            "path": f"{data_path}/crypto_data.db",
        }
        super().__init__(db_config)
        self.handler = AmberdataHandler()
        self.exchanges = ["binance", "bybit"]

    def store_data(
        self,
        df: pd.DataFrame,
        table_name: str = "exchange_reference",
        schema: str = "amberdata",
    ) -> None:
        """Store exchange reference data with complete table refresh"""

        def transaction_operations(connection):
            self.create_schema(connection, schema)
            # Drop existing table
            connection.execute(text(f"DROP TABLE IF EXISTS {schema}.{table_name}"))
            # Register temporary dataframe
            self.register_dataframe(connection, df)
            # Create new table with data
            connection.execute(
                text(
                    f"""
                CREATE TABLE {schema}.{table_name} AS
                SELECT * FROM temp_df
                ORDER BY exchange, instrument
                """
                )
            )

        self.db_handler.execute_transaction([transaction_operations])

    def ingest_exchange_reference(self) -> None:
        """Main method to ingest exchange reference data"""
        try:
            exchange_list = ",".join(self.exchanges)
            response = self.handler.get_exchange_reference_futures(
                exchange=exchange_list, include_inactive=True
            )

            if not response.empty:
                self.store_data(response)
                self.logger.info(
                    f"Successfully ingested exchange reference data for {exchange_list}"
                )
            else:
                self.logger.warning("No exchange reference data found")

        except Exception as e:
            self.logger.error(f"Error ingesting exchange reference data: {e}")
            raise
