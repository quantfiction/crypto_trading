import logging
import time
from typing import List, Dict, Optional, Union
from urllib.parse import urljoin
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import dotenv_values, find_dotenv
import pandas as pd
import re


class AmberdataHandler:
    def __init__(self):
        self.config = self._load_config()
        self.base_url = "https://api.amberdata.com"
        self.headers = {
            "accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "x-api-key": self.config.get("AMBERDATA_API_KEY"),
        }
        self.logger = logging.getLogger(__name__)

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
        df.columns = [AmberdataHandler.camel_to_snake(col) for col in df.columns]
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
                    logging.warning(f"Error converting {column} to datetime: {e}")
                    df[column] = pd.NaT
        return df

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _fetch_data(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Fetch data from API with retry logic"""
        url = urljoin(self.base_url, endpoint)
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise

    def get_exchange_reference_futures(
        self, exchange: str, include_inactive: bool = True
    ) -> pd.DataFrame:
        """Get exchange reference data for futures"""
        endpoint = "markets/futures/exchanges/reference"
        params = {
            "exchange": exchange,
            "includeInactive": "true" if include_inactive else "false",
        }

        list_instruments = []
        url = endpoint
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
                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                self.logger.error(f"Error processing data: {e}")
                break

        self.logger.info(f"Total records fetched: {total_records}")

        if list_instruments:
            datetime_cols = ["listing_timestamp", "contract_expiration_timestamp"]
            df_instruments = (
                pd.concat(list_instruments, axis=0, ignore_index=True)
                .pipe(self.convert_df_columns_to_snake_case)
                .pipe(self.convert_df_columns_to_datetime, columns=datetime_cols)
            )
            return df_instruments
        return pd.DataFrame()

    def get_ohlcv_info_futures(
        self,
        exchange: Optional[str] = None,
        include_inactive: bool = True,
        time_format: str = "iso8601",
        time_interval: str = "days",
        url: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get OHLCV information for futures contracts

        Args:
            exchange: Exchange name (optional if url is provided)
            include_inactive: Include inactive instruments
            time_format: Time format for timestamps
            time_interval: Time interval for OHLCV data
            url: Direct URL for pagination (optional)

        Returns:
            DataFrame containing OHLCV information
        """
        list_info = []
        total_records = 0
        current_url = url or f"markets/futures/ohlcv/information"

        while current_url:
            try:
                params = {}
                if not url:
                    if not exchange:
                        raise ValueError("Exchange must be provided when not using URL")
                    params = {
                        "exchange": exchange,
                        "includeInactive": "true" if include_inactive else "false",
                        "timeFormat": time_format,
                        "timeInterval": time_interval,
                    }

                response = self._fetch_data(current_url, params)
                payload = response.get("payload", {})
                data = payload.get("data", [])

                if data:
                    chunk_info = pd.DataFrame(data)
                    list_info.append(chunk_info)
                    total_records += len(chunk_info)

                current_url = payload.get("metadata", {}).get("next")
                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                self.logger.error(f"Error processing data: {e}")
                break

        self.logger.info(f"Total records fetched: {total_records}")

        if list_info:
            df_info = (
                pd.concat(list_info, axis=0, ignore_index=True)
                .pipe(self.convert_df_columns_to_snake_case)
                .pipe(
                    self.convert_df_columns_to_datetime,
                    columns=["start_date", "end_date"],
                )
                .rename(
                    columns={
                        "start_date": "trading_start_date",
                        "end_date": "trading_end_date",
                    }
                )
            )
            return df_info
        return pd.DataFrame()

    def get_ohlcv_data_futures(
        self,
        exchange: str,
        instrument: str,
        start_date: str,
        end_date: str,
        time_interval: str = "days",
    ) -> pd.DataFrame:
        """Get OHLCV data for futures instruments"""
        endpoint = f"/market/futures/ohlcv/exchange/{exchange}/historical"
        params = {
            "instrument": instrument,
            "startDate": start_date,
            "endDate": end_date,
            "timeInterval": time_interval,
        }

        try:
            response = self._fetch_data(endpoint, params)
            if response.get("payload", {}).get("data"):
                data = response["payload"]["data"]
                df_ohlcv_data_futures = (
                    pd.DataFrame(data)
                    .rename(columns={"timestamp": "datetime"})
                    .pipe(self.convert_df_columns_to_snake_case)
                    .pipe(self.convert_df_columns_to_datetime, ["datetime"])
                    .assign(exchange=exchange)
                )

                return df_ohlcv_data_futures
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV data: {e}")
            raise
