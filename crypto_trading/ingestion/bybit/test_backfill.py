import os
from datetime import datetime, timedelta
from pathlib import Path
from crypto_trading.common.db.handler import DatabaseHandler
from crypto_trading.ingestion.bybit.bybit_ingestor import BybitIngestor
from crypto_trading.common.utils.logging import setup_logging


def test_backfill():
    # Initialize logging
    setup_logging()

    # Initialize database handler with test config
    data_path = Path(__file__).parents[3] / "data"
    db_config = {"db_type": "duckdb", "path": f"{data_path}/crypto_data.db"}
    db_handler = DatabaseHandler(db_config)

    # Create ingestor instance
    ingestor = BybitIngestor(db_handler)

    # Test parameters - fetch last 7 days of data
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

    print("\n=== Testing orders backfill ===")
    orders = ingestor.fetch_data(
        endpoint="orders",
        params={
            "start_time": start_time,
            "end_time": end_time,
            "limit": 100,  # Larger sample
        },
    )
    print(f"Fetched {len(orders)} orders")

    print("\n=== Testing executions backfill ===")
    executions = ingestor.fetch_data(
        endpoint="executions",
        params={
            "start_time": start_time,
            "end_time": end_time,
            "limit": 100,  # Larger sample
        },
    )
    print(f"Fetched {len(executions)} executions")

    # Verify data in database
    print("\n=== Verifying database entries ===")
    orders_count = db_handler.query(
        "SELECT COUNT(*) FROM live_trading.orders_bybit WHERE created_time BETWEEN ? AND ?",
        (start_time, end_time),
    )[0][0]

    executions_count = db_handler.query(
        "SELECT COUNT(*) FROM live_trading.executions_bybit WHERE exec_time BETWEEN ? AND ?",
        (start_time, end_time),
    )[0][0]

    print(f"Orders in database: {orders_count}")
    print(f"Executions in database: {executions_count}")

    # Verify counts match
    if len(orders) == orders_count:
        print("Orders count matches database entries")
    else:
        print(f"Orders count mismatch: fetched {len(orders)} vs stored {orders_count}")

    if len(executions) == executions_count:
        print("Executions count matches database entries")
    else:
        print(
            f"Executions count mismatch: fetched {len(executions)} vs stored {executions_count}"
        )

    # Cleanup test data
    print("\n=== Cleaning up test data ===")
    db_handler.execute(
        "DELETE FROM live_trading.orders_bybit WHERE created_time BETWEEN ? AND ?",
        (start_time, end_time),
    )
    db_handler.execute(
        "DELETE FROM live_trading.executions_bybit WHERE exec_time BETWEEN ? AND ?",
        (start_time, end_time),
    )
    print("Test data cleaned up")


if __name__ == "__main__":
    test_backfill()
