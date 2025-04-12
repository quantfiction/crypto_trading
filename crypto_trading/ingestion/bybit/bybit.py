import os
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from crypto_trading.ingestion.ingestor import BaseIngestor

# Load environment variables
load_dotenv()
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")


class BybitAPI:
    """Handles direct API interactions with Bybit"""

    def __init__(self):
        self.client = self._get_bybit_client()

    def _get_bybit_client(self):
        """
        Returns a Bybit Unified Trading client with timeout configuration.
        """
        try:
            logging.info("Initializing Bybit client with API key and timeout")
            client = HTTP(
                testnet=False,
                api_key=BYBIT_API_KEY,
                api_secret=BYBIT_API_SECRET,
                timeout=10,  # 10 second timeout
            )
            # Test connection
            logging.info("Testing Bybit client connection...")
            test_response = client.get_server_time()
            if test_response["retCode"] == 0:
                logging.info(
                    "Successfully created and tested Bybit client with 10s timeout"
                )
                return client
            else:
                logging.error(f"Bybit client test failed: {test_response['retMsg']}")
                return None
        except Exception as e:
            logging.error(f"Error creating Bybit client: {e}", exc_info=True)
            return None

    def fetch_bybit_orders(
        self,
        symbol: str = None,
        base_coin: str = None,
        settle_coin: str = None,
        order_id: str = None,
        order_link_id: str = None,
        order_filter: str = None,
        order_status: str = None,
        start_time: int = None,
        end_time: int = None,
        category: str = "linear",
        limit: int = 100,  # Bybit's maximum page size
        cursor: str = None,
    ) -> tuple[list, str | None]:
        """
        Fetches order data from the Bybit API and formats it for insertion into the database.
        Supports both real-time and historical data retrieval.
        """
        is_backfill = start_time is not None and end_time is not None
        logging_prefix = "Backfilling" if is_backfill else "Fetching"

        logging.info(
            f"{logging_prefix} Bybit orders for symbol: {symbol}, baseCoin: {base_coin}, settleCoin: {settle_coin}, orderId: {order_id}, orderLinkId: {order_link_id}, orderFilter: {order_filter}, orderStatus: {order_status}, start_time: {start_time}, end_time: {end_time}, category: {category}, limit: {limit}, cursor: {cursor}"
        )
        try:
            all_orders = []
            start_time_local = time.time()
            while True:
                params = {
                    "category": category,
                    "limit": limit,
                    **{
                        k: v
                        for k, v in {
                            "symbol": symbol,
                            "baseCoin": base_coin,
                            "settleCoin": settle_coin,
                            "orderId": order_id,
                            "orderLinkId": order_link_id,
                            "orderFilter": order_filter,
                            "orderStatus": order_status,
                            "startTime": start_time,
                            "endTime": end_time,
                            "cursor": cursor,
                        }.items()
                        if v is not None
                    },
                }

                retries = 3
                delay = 1
                while retries > 0:
                    try:
                        response = self.client.get_order_history(**params)
                        if response["retCode"] == 0:
                            orders = response["result"]["list"]
                            formatted_orders = self._format_orders(orders)
                            # Accumulate orders across all pages
                            all_orders.extend(formatted_orders)

                            # Check if we have more pages to fetch
                            next_cursor = response["result"].get("nextPageCursor")
                            if not next_cursor:
                                logging.info(
                                    f"Completed fetching {len(all_orders)} Bybit orders"
                                )
                                break

                            cursor = next_cursor
                            logging.info(
                                f"Fetched {len(all_orders)} orders so far, fetching next page..."
                            )

                            # Add delay to avoid rate limiting
                            time.sleep(0.1)  # Reverted delay back to 0.1 second
                        else:
                            logging.error(
                                f"Error fetching Bybit orders: {response['retMsg']}"
                            )
                            break
                    except Exception as e:
                        logging.error(
                            f"Error fetching Bybit orders: {e}. Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                        delay *= 2
                        retries -= 1
                logging.error("Max retries reached. Unable to fetch Bybit orders.")
                break
        except Exception as e:
            logging.error(f"Error fetching Bybit orders: {e}")
        finally:
            end_time = time.time()
            duration = end_time - start_time_local
            logging.info(
                f"Finished fetching Bybit orders. Fetched {len(all_orders)} orders in {duration:.2f} seconds."
            )
        return all_orders

    def _format_orders(self, orders):
        """
        Formats the raw order data into a structured format.
        """
        formatted_orders = []
        for order in orders:
            formatted_order = {
                "order_id": order.get("orderId"),
                "order_link_id": order.get("orderLinkId"),
                "block_trade_id": order.get("blockTradeId"),
                "symbol": order.get("symbol"),
                "price": (float(order.get("price")) if order.get("price") else None),
                "qty": (float(order.get("qty")) if order.get("qty") else None),
                "side": order.get("side"),
                "is_leverage": order.get("isLeverage"),
                "position_idx": (
                    int(order.get("positionIdx")) if order.get("positionIdx") else None
                ),
                "order_status": order.get("orderStatus"),
                "cancel_type": order.get("cancelType"),
                "reject_reason": order.get("rejectReason"),
                "avg_price": (
                    float(order.get("avgPrice")) if order.get("avgPrice") else None
                ),
                "leaves_qty": (
                    float(order.get("leavesQty")) if order.get("leavesQty") else None
                ),
                "leaves_value": (
                    float(order.get("leavesValue"))
                    if order.get("leavesValue")
                    else None
                ),
                "cum_exec_qty": (
                    float(order.get("cumExecQty")) if order.get("cumExecQty") else None
                ),
                "cum_exec_value": (
                    float(order.get("cumExecValue"))
                    if order.get("cumExecValue")
                    else None
                ),
                "cum_exec_fee": (
                    float(order.get("cumExecFee")) if order.get("cumExecFee") else None
                ),
                "time_in_force": order.get("timeInForce"),
                "order_type": order.get("orderType"),
                "stop_order_type": order.get("stopOrderType"),
                "order_iv": order.get("orderIv"),
                "trigger_price": (
                    float(order.get("triggerPrice"))
                    if order.get("triggerPrice")
                    else None
                ),
                "take_profit": (
                    float(order.get("takeProfit")) if order.get("takeProfit") else None
                ),
                "stop_loss": (
                    float(order.get("stopLoss")) if order.get("stopLoss") else None
                ),
                "tp_trigger_by": order.get("tpTriggerBy"),
                "sl_trigger_by": order.get("slTriggerBy"),
                "trigger_direction": (
                    int(order.get("triggerDirection"))
                    if order.get("triggerDirection")
                    else None
                ),
                "trigger_by": order.get("triggerBy"),
                "last_price_on_created": (
                    float(order.get("lastPriceOnCreated"))
                    if order.get("lastPriceOnCreated")
                    else None
                ),
                "reduce_only": bool(order.get("reduceOnly")),
                "close_on_trigger": bool(order.get("closeOnTrigger")),
                "smp_type": order.get("smpType"),
                "smp_group": (
                    int(order.get("smpGroup")) if order.get("smpGroup") else None
                ),
                "smp_order_id": order.get("smpOrderId"),
                "tpsl_mode": order.get("tpslMode"),
                "tp_limit_price": (
                    float(order.get("tpLimitPrice"))
                    if order.get("tpLimitPrice")
                    else None
                ),
                "sl_limit_price": (
                    float(order.get("slLimitPrice"))
                    if order.get("slLimitPrice")
                    else None
                ),
                "place_type": order.get("placeType"),
                "slippage_tolerance_type": order.get("slippageToleranceType"),
                "slippage_tolerance": (
                    float(order.get("slippageTolerance"))
                    if order.get("slippageTolerance")
                    else None
                ),
                "created_time": (
                    int(order.get("createdTime")) if order.get("createdTime") else None
                ),
                "updated_time": (
                    int(order.get("updatedTime")) if order.get("updatedTime") else None
                ),
            }
            formatted_orders.append(formatted_order)
        return formatted_orders

    def _format_executions(self, executions):
        """
        Formats the raw execution data into a structured format.
        """
        formatted_executions = []
        for execution in executions:
            formatted_execution = {
                "symbol": execution.get("symbol"),
                "order_type": execution.get("orderType"),
                "underlying_price": execution.get("underlyingPrice"),
                "order_link_id": execution.get("orderLinkId"),
                "side": execution.get("side"),
                "index_price": execution.get("indexPrice"),
                "order_id": execution.get("orderId"),
                "stop_order_type": execution.get("stopOrderType"),
                "leaves_qty": execution.get("leavesQty"),
                "exec_time": (
                    int(execution.get("execTime"))
                    if execution.get("execTime")
                    else None
                ),
                "fee_currency": execution.get("feeCurrency"),
                "is_maker": bool(execution.get("isMaker")),
                "exec_fee": (
                    float(execution.get("execFee"))
                    if execution.get("execFee")
                    else None
                ),
                "fee_rate": (
                    float(execution.get("feeRate"))
                    if execution.get("feeRate")
                    else None
                ),
                "exec_id": execution.get("execId"),
                "trade_iv": execution.get("tradeIv"),
                "block_trade_id": execution.get("blockTradeId"),
                "mark_price": (
                    float(execution.get("markPrice"))
                    if execution.get("markPrice")
                    else None
                ),
                "exec_price": (
                    float(execution.get("execPrice"))
                    if execution.get("execPrice")
                    else None
                ),
                "mark_iv": execution.get("markIv"),
                "order_qty": (
                    float(execution.get("orderQty"))
                    if execution.get("orderQty")
                    else None
                ),
                "order_price": (
                    float(execution.get("orderPrice"))
                    if execution.get("orderPrice")
                    else None
                ),
                "exec_value": (
                    float(execution.get("execValue"))
                    if execution.get("execValue")
                    else None
                ),
                "exec_qty": (
                    float(execution.get("execQty"))
                    if execution.get("execQty")
                    else None
                ),
                "closed_size": execution.get("closedSize"),
                "seq": (int(execution.get("seq")) if execution.get("seq") else None),
            }
            formatted_executions.append(formatted_execution)
        return formatted_executions

    def fetch_bybit_executions(
        self,
        symbol: str = None,
        order_id: str = None,
        order_link_id: str = None,
        base_coin: str = None,
        start_time: int = None,
        end_time: int = None,
        exec_type: str = None,
        limit: int = 100,  # Bybit's maximum page size
        cursor: str = None,
        category: str = "linear",
    ) -> tuple[list, str | None]:
        """
        Fetches execution data from the Bybit API and formats it for insertion into the database.
        Supports both real-time and historical data retrieval.

        Args:
            symbol: Optional trading pair symbol filter
            start_time: Optional start timestamp in milliseconds (default: 730 days ago)
            end_time: Optional end timestamp in milliseconds (default: current time)
        """
        is_backfill = start_time is not None and end_time is not None
        logging_prefix = "Backfilling" if is_backfill else "Fetching"

        # Set default time range if not provided
        if start_time is None:
            start_time = int(time.time() * 1000) - (
                730 * 24 * 60 * 60 * 1000
            )  # 730 days ago
        if end_time is None:
            end_time = int(time.time() * 1000)  # Current time

        # Validate time range
        if start_time >= end_time:
            logging.error(
                f"Invalid time range: start_time {start_time} >= end_time {end_time}"
            )
            return [], None

        logging.info(
            f"{logging_prefix} Bybit executions for symbol: {symbol}, orderId: {order_id}, orderLinkId: {order_link_id}, baseCoin: {base_coin}, start_time: {start_time}, end_time: {end_time}, execType: {exec_type}, limit: {limit}, cursor: {cursor}, category: {category}"
        )
        all_executions = []
        start_time_local = time.time()

        try:
            while True:
                executions, next_cursor = self._fetch_executions_page(
                    symbol,
                    order_id,
                    order_link_id,
                    base_coin,
                    start_time,
                    end_time,
                    exec_type,
                    limit,
                    cursor,
                    category,
                )
                if not executions:
                    break

                all_executions.extend(executions)

                # # Stop the loop after 300 records for testing
                # if len(all_executions) >= 300:
                #     logging.info("Reached 300 records, stopping for testing.")
                #     break

                if not next_cursor:
                    break

                cursor = next_cursor
                logging.info(
                    f"Fetched {len(all_executions)} executions so far, fetching next page..."
                )

        except Exception as e:
            logging.error(f"Error fetching Bybit executions: {e}")
        finally:
            end_time = time.time()
            duration = end_time - start_time_local
            logging.info(
                f"Finished fetching Bybit executions. Fetched {len(all_executions)} executions in {duration:.2f} seconds."
            )
        # Log total and unique executions
        total_executions = len(all_executions)
        unique_executions = len(set(exec["execId"] for exec in all_executions))
        logging.info(
            f"Total executions: {total_executions}, Unique executions: {unique_executions}"
        )

        return all_executions, cursor

    def _fetch_executions_page(
        self,
        symbol: str,
        order_id: str,
        order_link_id: str,
        base_coin: str,
        start_time: int,
        end_time: int,
        exec_type: str,
        limit: int,
        cursor: str,
        category: str,
    ) -> tuple[list, str | None]:
        """
        Fetches a single page of execution data from the Bybit API.
        """
        params = {
            "category": category,
            "limit": limit,
            **{
                k: v
                for k, v in {
                    "symbol": symbol,
                    "orderId": order_id,
                    "orderLinkId": order_link_id,
                    "baseCoin": base_coin,
                    "startTime": start_time,
                    "endTime": end_time,
                    "execType": exec_type,
                    "cursor": cursor,
                }.items()
                if v is not None
            },
        }

        retries = 3
        delay = 0.1
        while retries > 0:
            try:
                logging.info(f"Fetching executions with params: {params}")
                response = self.client.get_executions(**params)
                if response["retCode"] == 0:
                    executions = response["result"]["list"]
                    next_cursor = response["result"].get("nextPageCursor")
                    return executions, next_cursor
                else:
                    logging.error(
                        f"Error fetching Bybit executions: {response['retMsg']}"
                    )
                    return [], None
            except Exception as e:
                logging.error(
                    f"Error fetching Bybit executions: {e}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay *= 2
                retries -= 1
        logging.error("Max retries reached. Unable to fetch Bybit executions.")
        return [], None


if __name__ == "__main__":
    """Test the Bybit API client"""
    import time
    import csv

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        api = BybitAPI()
        start = int(time.time() * 1000) - (30 * 24 * 60 * 60 * 1000)  # 24 hours ago
        end = int(time.time() * 1000)  # Now

        logging.info("Testing Bybit executions API")
        executions, cursor = api.fetch_bybit_executions(start_time=start, end_time=end)
        logging.info(f"Fetched {len(executions)} executions")

        # Save executions to CSV
        with open("executions2.csv", "w", newline="") as csvfile:
            fieldnames = executions[0].keys() if executions else []
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for execution in executions:
                writer.writerow(execution)

        logging.info("Executions saved to executions.csv")

    except Exception as e:
        logging.error(f"Test failed: {e}")
