import os
import logging
import duckdb
from dotenv import load_dotenv
from crypto_trading.ingestion.bybit.bybit_ingestor import BybitIngestor
from crypto_trading.common.db.handler import DatabaseHandler

# Load environment variables
load_dotenv()
DUCKDB_PATH = "/data/crypto_data.db"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def ingest_bybit_data():
    """
    Ingests Bybit order and execution data into DuckDB.
    """
    try:
        # Database configuration
        db_config = {"db_type": "duckdb", "path": DUCKDB_PATH}
        db_handler = DatabaseHandler(db_config)

        # Create Bybit ingestor instance
        ingestor = BybitIngestor(db_handler)

        # Fetch orders and executions
        orders = ingestor.fetch_data(endpoint="orders", params={"symbol": "BTCUSDT"})
        executions = ingestor.fetch_data(
            endpoint="executions", params={"symbol": "BTCUSDT"}
        )

        # Insert order data
        for order in orders:
            insert_order_query = f"""
                INSERT INTO live_trading.orders_bybit VALUES (
                    '{order.get('order_id')}', '{order.get('order_link_id')}', '{order.get('block_trade_id')}',
                    '{order.get('symbol')}', {order.get('price')}, {order.get('qty')}, '{order.get('side')}',
                    '{order.get('is_leverage')}', {order.get('position_idx')}, '{order.get('order_status')}',
                    '{order.get('cancel_type')}', '{order.get('reject_reason')}', {order.get('avg_price')},
                    {order.get('leaves_qty')}, {order.get('leaves_value')}, {order.get('cum_exec_qty')},
                    {order.get('cum_exec_value')}, {order.get('cum_exec_fee')}, '{order.get('time_in_force')}',
                    '{order.get('order_type')}', '{order.get('stop_order_type')}', '{order.get('order_iv')}',
                    {order.get('trigger_price')}, {order.get('take_profit')}, {order.get('stop_loss')},
                    '{order.get('tp_trigger_by')}', '{order.get('sl_trigger_by')}', {order.get('trigger_direction')},
                    '{order.get('trigger_by')}', {order.get('last_price_on_created')}, {order.get('reduce_only')},
                    {order.get('close_on_trigger')}, '{order.get('smp_type')}', {order.get('smp_group')},
                    '{order.get('smp_order_id')}', '{order.get('tpsl_mode')}', {order.get('tp_limit_price')},
                    {order.get('sl_limit_price')}, '{order.get('place_type')}', '{order.get('slippage_tolerance_type')}',
                    {order.get('slippage_tolerance')}, {order.get('created_time')}, {order.get('updated_time')}
                )
            """
            db_handler.execute(insert_order_query)

        # Insert execution data
        for execution in executions:
            insert_execution_query = f"""
                INSERT INTO live_trading.executions_bybit VALUES (
                    '{execution.get('symbol')}', '{execution.get('order_type')}', '{execution.get('underlying_price')}',
                    '{execution.get('order_link_id')}', '{execution.get('side')}', '{execution.get('index_price')}',
                    '{execution.get('order_id')}', '{execution.get('stop_order_type')}', '{execution.get('leaves_qty')}',
                    {execution.get('exec_time')}, '{execution.get('fee_currency')}', {execution.get('is_maker')},
                    {execution.get('exec_fee')}, {execution.get('fee_rate')}, '{execution.get('exec_id')}',
                    '{execution.get('trade_iv')}', '{execution.get('block_trade_id')}', {execution.get('mark_price')},
                    {execution.get('exec_price')}, '{execution.get('mark_iv')}', {execution.get('order_qty')},
                    {execution.get('order_price')}, {execution.get('exec_value')}, {execution.get('exec_qty')},
                    '{execution.get('closed_size')}', {execution.get('seq')}
                )
            """
            db_handler.execute(insert_execution_query)

        logging.info("Bybit data ingestion completed")

    except Exception as e:
        logging.error(f"Error during Bybit data ingestion: {e}")


if __name__ == "__main__":
    ingest_bybit_data()
