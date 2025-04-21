import logging
import time
from typing import List, Dict, Optional, Union
from urllib.parse import urljoin
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import dotenv_values, find_dotenv
import pandas as pd
import re

# Configure logging to display debug messages
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()


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
        logging.debug(self.headers)

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
                    # Convert to datetime and immediately localize to UTC
                    df[column] = pd.to_datetime(
                        df[column], unit=unit, errors="coerce"
                    ).dt.tz_localize("UTC")
                except Exception as e:
                    logging.warning(f"Error converting {column} to datetime: {e}")
                    df[column] = pd.NaT  # Keep NaT as timezone-naive
        return df

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _fetch_data(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Fetch data from API with retry logic"""
        url = urljoin(self.base_url, endpoint)
        try:
            response = requests.get(url, headers=self.headers, params=params)
            self.logger.debug(f"API response: {response.status_code} - {response.text}")
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
        instrument: Optional[str] = None,
        include_inactive: bool = True,
    ) -> pd.DataFrame:
        """Get OHLCV information for futures contracts from Amberdata.

        Fetches information about available OHLCV data ranges for futures
        instruments, including trading start/end dates. Handles pagination
        automatically and formats data for the ohlcv_info_futures table.

        Args:
            exchange (Optional[str]): Comma-separated list of exchanges to filter by.
                                      Defaults to all exchanges.
            instrument (Optional[str]): Instrument symbol to filter by.
            include_inactive (bool): Whether to include inactive instruments.
                                     Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing OHLCV information with columns:
                          exchange, instrument, trading_start_date,
                          trading_end_date, active. Returns empty DataFrame on error
                          or if no data is found.
        """
        list_info = []
        total_records = 0
        # Start with the relative path for the first request
        current_url_or_path = "/markets/futures/ohlcv/information"
        # Build initial parameters dynamically
        initial_params = {"includeInactive": str(include_inactive).lower()}
        if exchange:
            initial_params["exchange"] = exchange
        if instrument:
            initial_params["instrument"] = instrument

        self.logger.info(f"Fetching OHLCV info with params: {initial_params}")

        while current_url_or_path:
            try:
                response_json = None
                # Use requests.get directly for absolute pagination URLs
                if current_url_or_path.startswith("http"):
                    self.logger.debug(
                        f"Fetching paginated data from: {current_url_or_path}"
                    )
                    response = requests.get(
                        current_url_or_path, headers=self.headers, timeout=30
                    )  # Added timeout
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                    response_json = response.json()
                # Use _fetch_data for the initial relative path request
                else:
                    self.logger.debug(
                        f"Fetching initial data from path: {current_url_or_path}"
                    )
                    # _fetch_data handles retries and base URL joining
                    response_json = self._fetch_data(
                        current_url_or_path, initial_params
                    )

                if not response_json:
                    self.logger.warning("Received empty response, stopping pagination.")
                    break  # Exit if response is unexpectedly empty

                payload = response_json.get("payload", {})
                data = payload.get("data", [])

                if data:
                    chunk_info = pd.DataFrame(data)
                    list_info.append(chunk_info)
                    total_records += len(chunk_info)
                    self.logger.debug(
                        f"Fetched {len(chunk_info)} records, total: {total_records}"
                    )
                else:
                    self.logger.debug("No data found in this page.")

                # Get the full 'next' URL from metadata for pagination
                current_url_or_path = payload.get("metadata", {}).get("next")
                if current_url_or_path:
                    self.logger.debug(f"Next page URL found: {current_url_or_path}")
                    time.sleep(0.5)  # Rate limiting
                else:
                    self.logger.debug("No next page URL found, pagination complete.")
                    current_url_or_path = None  # End of pagination

            except requests.exceptions.RequestException as e:
                self.logger.error(f"HTTP request failed during OHLCV info fetch: {e}")
                break  # Exit loop on HTTP error
            except Exception as e:
                # Catch other potential errors (e.g., JSON decoding, processing)
                self.logger.error(
                    f"Error processing data fetching OHLCV info: {e}", exc_info=True
                )
                break  # Exit loop on processing error

        self.logger.info(f"Total OHLCV info records fetched: {total_records}")

        if not list_info:
            self.logger.warning("No OHLCV info records were fetched.")
            return pd.DataFrame()  # Return empty DataFrame if no data fetched

        try:
            df_info = (
                pd.concat(list_info, axis=0, ignore_index=True)
                .pipe(self.convert_df_columns_to_snake_case)
                # Ensure 'start_date' and 'end_date' exist before conversion
                .pipe(
                    lambda df: self.convert_df_columns_to_datetime(
                        df,
                        columns=[
                            col
                            for col in ["start_date", "end_date"]
                            if col in df.columns
                        ],
                        unit="ms",  # API default is ms
                    )
                )
                .rename(
                    columns={
                        "start_date": "trading_start_date",
                        "end_date": "trading_end_date",
                    }
                )
            )

            # Add the 'active' column based on trading_end_date
            # Active if end date is NaT (null) or in the future
            if "trading_end_date" in df_info.columns:
                now = pd.Timestamp.now(tz="UTC")  # Use timezone-aware comparison
                df_info["active"] = (df_info["trading_end_date"].isna()) | (
                    df_info["trading_end_date"] > now
                )
            else:
                # If end_date wasn't in the response, cannot determine active status reliably
                self.logger.warning(
                    "Column 'end_date' not found in API response. Cannot determine 'active' status."
                )
                df_info["active"] = pd.NA  # Assign NA if calculation isn't possible

            # Select and reorder columns to match the target schema
            target_columns = [
                "exchange",
                "instrument",
                "trading_start_date",
                "trading_end_date",
                "active",
            ]
            # Ensure all target columns exist, adding missing ones as NA
            for col in target_columns:
                if col not in df_info.columns:
                    self.logger.warning(
                        f"Target column '{col}' not found in source data. Adding as NA."
                    )
                    df_info[col] = pd.NA

            # Filter out rows where essential columns might be missing after fetch/conversion
            # Important to do this *after* adding potentially missing columns
            initial_rows = len(df_info)
            df_info.dropna(subset=["exchange", "instrument"], inplace=True)
            if len(df_info) < initial_rows:
                self.logger.warning(
                    f"Dropped {initial_rows - len(df_info)} rows due to missing 'exchange' or 'instrument'."
                )

            return df_info[target_columns]

        except Exception as e:
            self.logger.error(
                f"Failed to process concatenated OHLCV info data: {e}", exc_info=True
            )
            return pd.DataFrame()  # Return empty DataFrame on processing error

    def get_ohlcv_data_futures(
        self,
        exchange: str,
        instrument: str,
        start_date: str,
        end_date: str,
        time_interval: str = "days",
    ) -> pd.DataFrame:
        """Get OHLCV data for futures instruments"""
        endpoint = f"/markets/futures/ohlcv/exchange/{exchange}/historical"
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
