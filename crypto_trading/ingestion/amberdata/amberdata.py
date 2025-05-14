import logging
import time
from datetime import timedelta as td
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
        df: pd.DataFrame,
        columns: List[str],
        unit: Optional[str] = None,  # Make unit optional
    ) -> pd.DataFrame:
        """Convert specified columns to datetime, handling numeric and string types."""
        for column in columns:
            if column in df.columns:
                # Skip if column is already all NaT to avoid dtype issues
                if df[column].isnull().all():
                    continue
                try:
                    col_dtype = df[column].dtype
                    if pd.api.types.is_datetime64_any_dtype(col_dtype):
                        # Already datetime, ensure it's timezone-naive
                        if df[column].dt.tz is not None:
                            df[column] = df[column].dt.tz_localize(None)
                    elif pd.api.types.is_numeric_dtype(col_dtype):
                        # Handle numeric types (e.g., Unix timestamps)
                        # unit parameter is crucial here if data is numeric
                        if unit is None:
                            logging.warning(
                                f"Numeric column {column} requires 'unit' for conversion. Skipping conversion."
                            )
                            # Keep numeric or coerce to NaT if needed elsewhere, but don't convert here without unit
                            continue  # Skip conversion for this column
                        # Convert to timezone-naive datetime
                        df[column] = pd.to_datetime(
                            df[column], unit=unit, errors="coerce"  # Removed utc=True
                        )
                    elif pd.api.types.is_string_dtype(
                        col_dtype
                    ) or pd.api.types.is_object_dtype(col_dtype):
                        # Handle string types (e.g., ISO8601) - infer format
                        # Convert to timezone-naive datetime
                        # Convert string/object, inferring format
                        dt_series = pd.to_datetime(df[column], errors="coerce")
                        # Explicitly make it timezone-naive if timezone was inferred
                        if dt_series.dt.tz is not None:
                            df[column] = dt_series.dt.tz_localize(None)
                        else:
                            df[column] = dt_series
                    else:
                        logging.warning(
                            f"Unhandled dtype {col_dtype} for column {column}. Coercing to NaT."
                        )
                        df[column] = pd.Series([pd.NaT] * len(df), index=df.index)

                except Exception as e:
                    # Log error and coerce column to NaT to prevent downstream issues
                    logging.warning(
                        f"Error converting {column} to datetime: {e}. Coercing to NaT."
                    )
                    df[column] = pd.Series([pd.NaT] * len(df), index=df.index)
        return df

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _fetch_data(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Fetch data from API with retry logic"""
        url = urljoin(self.base_url, endpoint)
        try:
            response = requests.get(url, headers=self.headers, params=params)
            self.logger.debug(f"API response: {response.status_code}")
            response.raise_for_status()  # Raise HTTP errors first
            json_data = response.json()  # Parse JSON response
            payload = json_data.get("payload", {})  # Access payload from JSON
            data_list = payload.get("data", [])  # Access data from payload
            self.logger.info(
                f"Number of records retrieved: {len(data_list)}"
            )  # Log count
            return json_data  # Return parsed JSON
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
        initial_params = {
            "includeInactive": str(include_inactive).lower(),
            "timeInterval": "minutes",
        }
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
                cutoff = pd.Timestamp.now(tz=None) - td(
                    hours=24
                )  # Use timezone-aware comparison
                df_info["active"] = (df_info["trading_end_date"].isna()) | (
                    df_info["trading_end_date"] > cutoff
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
        """Get OHLCV data for multiple futures instruments using the batch endpoint."""
        endpoint = f"/markets/futures/batch-ohlcv/{exchange}"
        params = {
            "instruments": instrument,  # Corrected parameter name
            "startDate": start_date,
            "endDate": end_date,
            "timeInterval": time_interval,
            "timeFormat": "iso8601",  # Explicitly set timeFormat
        }
        self.logger.info(
            f"Fetching batch OHLCV data from {endpoint} for exchange {exchange}"
        )
        self.logger.debug(f"Batch OHLCV params: {params}")

        all_instrument_data = []

        try:
            response = self._fetch_data(endpoint, params)
            payload = response.get("payload", {})
            # Expecting a flat list of records based on example response
            data_list = payload.get("data", [])

            if not data_list:
                self.logger.warning(
                    f"No data returned in payload's data list for batch request to {endpoint}"
                )
                return pd.DataFrame()

            self.logger.debug(f"Type of data_list: {type(data_list)}")
            # self.logger.debug(
            #     f"Content of data_list (first 500 chars): {str(data_list)[:500]}"
            # )

            # Create DataFrame directly from the list
            df_ohlcv_data_futures = (
                pd.DataFrame(data_list)
                # Rename the timestamp column based on example response
                .rename(columns={"exchangeTimestamp": "datetime"})
                .pipe(self.convert_df_columns_to_snake_case)
                # Ensure 'datetime' column exists before conversion
                .pipe(
                    lambda df: self.convert_df_columns_to_datetime(
                        df,
                        # Ensure 'datetime' column exists after rename before conversion
                        columns=[col for col in ["datetime"] if col in df.columns],
                        # Use 'iso' unit for pd.to_datetime when source is ISO8601 string
                        # Or let pandas infer if format is consistent
                        # Let's try letting pandas infer first. If issues, specify format.
                        # unit="ms", # unit='ms' is for integer timestamps
                    )
                )
                .assign(exchange=exchange)  # Add exchange column
            )

            # Reorder columns to a standard format if needed, ensuring essential ones are present
            final_columns = [
                "exchange",
                "instrument",
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            # Add missing columns as NA
            for col in final_columns:
                if col not in df_ohlcv_data_futures.columns:
                    self.logger.warning(
                        f"Column '{col}' missing in batch OHLCV response. Adding as NA."
                    )
                    df_ohlcv_data_futures[col] = pd.NA

            return df_ohlcv_data_futures[
                final_columns
            ]  # Return with consistent column order

        except Exception as e:
            self.logger.error(
                f"Failed to fetch or process batch OHLCV data: {e}", exc_info=True
            )
            # Do not re-raise here, return empty DataFrame as per original logic flow
            return pd.DataFrame()
