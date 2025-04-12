import logging
import os
from datetime import datetime, timedelta
import time

from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from crypto_trading.common.db.handler import DatabaseHandler
from .bybit import BybitAPI

# Load environment variables
load_dotenv()
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BybitIngestor:
    def __init__(self, db_handler: DatabaseHandler):
        self.db_handler = db_handler
        self.bybit_api = BybitAPI()

    def get_start_datetime(self) -> datetime:
        """
        Determines the starting datetime for data retrieval.
        """
        logging.info("Determining start datetime for backfilling")

        latest_execution_time = self.db_handler.query_latest_timestamp(
            "live_trading.executions_bybit", "exec_time"
        )
        latest_order_time = self.db_handler.query_latest_timestamp(
            "live_trading.orders_bybit", "created_time"
        )

        if latest_execution_time is None or latest_order_time is None:
            logging.info("One or both databases are empty. Defaulting to 730 days ago.")
            start_datetime = datetime.now() - timedelta(days=730)
        else:
            start_datetime = max(latest_execution_time, latest_order_time)
            logging.info(
                f"Latest execution time: {latest_execution_time}, Latest order time: {latest_order_time}"
            )

        logging.info(f"Start datetime: {start_datetime}")
        return start_datetime

    def fetch_data(self, endpoint: str, params: dict) -> list:
        """
        Fetches data from the Bybit API with retry and rate limiting.
        Handles pagination by accumulating results across multiple pages.
        """
        # Validate time range parameters
        if "start_time" in params and "end_time" in params:
            time_range = params["end_time"] - params["start_time"]
            if time_range > 604800:  # 7 days in seconds
                logging.warning(
                    f"Time range exceeds 7 days ({time_range} seconds). "
                    "Consider breaking into smaller chunks."
                )

        logging.info(f"Fetching data from endpoint: {endpoint} with params: {params}")

        all_data = []
        cursor = None
        retries = 3
        delay = 1
        page_count = 0

        while True:
            try:
                # Add cursor to params if available
                if cursor:
                    params["cursor"] = cursor

                # Ensure limit is set
                params["limit"] = params.get("limit", 100)

                # Fetch data based on endpoint
                if endpoint == "orders":
                    data = self.bybit_api.fetch_bybit_orders(**params)
                elif endpoint == "executions":
                    data = self.bybit_api.fetch_bybit_executions(**params)
                else:
                    logging.error(f"Unknown endpoint: {endpoint}")
                    return []

                if data:
                    page_count += 1
                    logging.info(
                        f"Page {page_count}: Fetched {len(data)} records "
                        f"(total: {len(all_data) + len(data)})"
                    )

                    # Write data to database
                    if endpoint == "orders":
                        self._write_orders(data)
                    elif endpoint == "executions":
                        self._write_executions(data)

                    # Accumulate results
                    all_data.extend(data)

                    # Check if we have more pages
                    if len(data) < params["limit"]:
                        logging.info(
                            f"Received {len(data)} records, which is less than "
                            f"the limit of {params['limit']}. Assuming end of data."
                        )
                        break

                    # Add delay between requests to avoid rate limiting
                    time.sleep(0.1)

                    # Reset retry counter after successful fetch
                    retries = 3
                    delay = 1
                else:
                    logging.info("No more data received from API")
                    break

            except Exception as e:
                logging.error(
                    f"Error fetching Bybit data: {e}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay *= 2
                retries -= 1

                if retries == 0:
                    logging.error("Max retries reached. Unable to fetch Bybit data.")
                    break

        logging.info(f"Fetched {len(all_data)} total records from {endpoint} endpoint")
        return all_data

    def _write_orders(self, orders: list) -> None:
        """
        Writes orders data to the database.
        """
        try:
            self.db_handler.bulk_insert(
                table="live_trading.orders_bybit",
                data=orders,
                conflict_columns=["order_id"],
                update_columns=[
                    "order_link_id",
                    "symbol",
                    "price",
                    "qty",
                    "side",
                    "order_status",
                    "avg_price",
                    "leaves_qty",
                    "leaves_value",
                    "cum_exec_qty",
                    "cum_exec_value",
                    "cum_exec_fee",
                    "updated_time",
                ],
            )
            logging.info(f"Successfully wrote {len(orders)} orders to database")
        except Exception as e:
            logging.error(f"Error writing orders to database: {e}")

    def _write_executions(self, executions: list) -> None:
        """
        Writes executions data to the database.
        """
        try:
            self.db_handler.bulk_insert(
                table="live_trading.executions_bybit",
                data=executions,
                conflict_columns=["exec_id"],
                update_columns=[
                    "symbol",
                    "order_type",
                    "order_id",
                    "exec_time",
                    "exec_price",
                    "exec_qty",
                    "exec_value",
                    "exec_fee",
                    "fee_rate",
                    "is_maker",
                ],
            )
            logging.info(f"Successfully wrote {len(executions)} executions to database")
        except Exception as e:
            logging.error(f"Error writing executions to database: {e}")
