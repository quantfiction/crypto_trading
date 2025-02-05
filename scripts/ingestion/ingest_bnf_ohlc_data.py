import asyncio
import asyncpg
import aiohttp
import backoff
import ccxt.async_support as ccxt
import time
from datetime import datetime, timezone
import polars as pl
import pandas as pd
import requests
import logging

from crypto_trading.ingestion.providers.amberdata import (
    get_exchange_reference_futures,
    get_ohlcv_info_futures,
)
from dotenv import find_dotenv, dotenv_values

# Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Get Credentials
config = dotenv_values(find_dotenv())
user = config.get("USERNAME_PG")
password = config.get("PASSWORD_PG")
ad_api_key = config.get("AMBERDATA_API_KEY")

# DB Params
uri = f"postgresql://{user}:{password}@localhost:5432/crypto_data"
db_params = {
    "database": "crypto_data",
    "user": user,
    "password": password,
    "host": "localhost",
}

# Get All Symbols from Amberdata
logging.info("Getting Instrument Data")
df_exchange_reference = get_exchange_reference_futures(exchange="binance")
df_ohlcv_info = get_ohlcv_info_futures(exchange="binance")
df_instrument_data_merged = pd.merge(
    df_exchange_reference,
    df_ohlcv_info.drop("active", axis="columns"),
    on=["instrument", "exchange"],
)

perps = df_instrument_data_merged.query(
    'contractType == "perpetual" and marginType != "inverse"'
)

# Get Existing Symbol Info
query_symbol_info_mv = """
    SELECT
        symbol,
        last_datetime
    FROM perps_last_datetime
    WHERE exchange = 'binance'
"""
df_symbol_info = pl.read_database_uri(query=query_symbol_info_mv, uri=uri)


# Functions
def convert_datetime_to_utc_timestamp(datetime_val):
    return int(datetime_val.replace(tzinfo=timezone.utc).timestamp() * 1000)


async def insert_ohlc_data(conn, table_name, exchange, symbol, ohlc_data):
    """
    Asynchronously inserts OHLC data into a dynamically specified PostgreSQL table.

    Parameters:
    - db_params: A dictionary containing database connection parameters.
    - table_name: The name of the table to insert data into.
    - exchange: The name of the exchange (e.g., 'Binance').
    - symbol: The trading pair or symbol (e.g., 'BTC/USDT').
    - ohlc_data: The OHLC data from the Binance API.
    """

    # SQL query to insert data, using safe string formatting for the table name
    insert_query = f"""
    INSERT INTO {table_name} (exchange, symbol, datetime, open, high, low, close, base_volume, quote_volume, base_buy_volume, quote_buy_volume)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    ON CONFLICT DO NOTHING
    """

    for data in ohlc_data:
        # Convert timestamp to datetime
        datetime_val = datetime.utcfromtimestamp(int(data[0]) / 1000.0)

        # Prepare data tuple
        data_tuple = (
            exchange,
            symbol,
            datetime_val,
            data[1],  # open
            data[2],  # high
            data[3],  # low
            data[4],  # close
            data[5],  # base_volume
            data[7],  # quote_volume (index 6 is close time, so we use index 7)
            data[9],  # base_buy_volume
            data[10],  # quote_buy_volume
        )

        # Execute the insert query
        await conn.execute(insert_query, *data_tuple)


async def fetch_segment(client, symbol, start_time, timeframe="1m", limit=1500):
    logging.info(f"Starting request for symbol {symbol}")
    segment = await client.fapipublic_get_klines(
        {
            "symbol": symbol,
            "startTime": start_time,
            "limit": limit,
            "interval": timeframe,
        }
    )
    logging.info(f"Retrieved {len(segment)} bars for symbol {symbol}")
    return segment


async def fetch_data_for_symbol(client, symbol, start_time, end_time, semaphore):
    # Assuming last_datetime is already in UTC
    # start_time = int(last_datetime.replace(tzinfo=timezone.utc).timestamp() * 1000)

    # # Current time in UTC
    # end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    # # print(last_datetime, start_time, end_time)
    limit = 1500
    all_klines = []

    # Calculate the number of segments needed based on the time range and limit
    time_range = max(end_time - start_time, 0)
    one_minute = 60000  # milliseconds
    segments = int(time_range) // int(limit * one_minute)

    tasks = []
    logging.info(f"Fetching {segments + 1} segments for {symbol}")
    for i in range(segments + 1):
        segment_start = start_time + i * limit * one_minute
        async with semaphore:
            task = asyncio.create_task(
                fetch_segment(
                    client=client, symbol=symbol, start_time=segment_start, limit=limit
                )
            )
            tasks.append(task)

    # Wait for all tasks to complete and collect results
    results = await asyncio.gather(*tasks)
    for result in results:
        all_klines.extend(result)

    return all_klines


async def process_symbol(
    pool, table_name, client, exchange_name, symbol, start_time, end_time, semaphore
):
    """
    Fetches OHLC data for a given symbol and inserts it into the database.
    """
    # Fetch the data for the symbol
    async with pool.acquire() as conn:
        try:
            ohlc_data = await fetch_data_for_symbol(
                client, symbol, start_time, end_time, semaphore
            )

            # Insert the fetched data into the database
            logging.info(f"Inserting {len(ohlc_data)} rows for {symbol}")
            await insert_ohlc_data(conn, table_name, exchange_name, symbol, ohlc_data)
            logging.info(f"Inserted {len(ohlc_data)} rows for {symbol}")
        except ccxt.BadSymbol as e:
            logging.error(f"An error occurred: {e}. Skipping symbol {symbol}.")


async def fetch_and_store_all_symbols(
    client, pool, table_name, instrument_data, dict_existing_symbols, semaphore_num=100
):

    semaphore = asyncio.Semaphore(semaphore_num)

    tasks = []
    for _, row in instrument_data.iterrows():
        symbol = row["instrument"]
        instrument_start_time = row["startDate"]
        instrument_end_time = row["endDate"]

        # If instrument_start_time is NaN, use the default start time
        if pd.isna(instrument_start_time):
            instrument_start_time = convert_datetime_to_utc_timestamp(
                datetime(2019, 9, 1)
            )
        else:
            instrument_start_time += 1000 * 60 * 5  # add 5 minutes

        last_datetime = dict_existing_symbols.get(symbol, datetime(2019, 9, 1))

        # Convert last_datetime to UTC Unix timestamp in milliseconds
        last_datetime = convert_datetime_to_utc_timestamp(last_datetime)

        now = int(datetime.utcnow().timestamp() * 1000)
        start_time = max(instrument_start_time, last_datetime)
        end_time = min(now, instrument_end_time)

        task = asyncio.create_task(
            process_symbol(
                pool,
                table_name,
                client,
                "binance",
                symbol,
                start_time,
                end_time,
                semaphore,
            )
        )
        tasks.append(task)

    await asyncio.gather(*tasks)


async def main(instrument_data, dict_existing_symbols):
    client = ccxt.binanceusdm(
        {
            # 'rateLimit': 1200,  # Binance rate limit in milliseconds
            "enableRateLimit": True,
        }
    )
    client.throttle.config["maxCapacity"] = 1e09
    table_name = "ohlc_1m_swaps_test"

    pool = await asyncpg.create_pool(**db_params)

    await fetch_and_store_all_symbols(
        client, pool, table_name, instrument_data, dict_existing_symbols
    )

    await client.close()
    await pool.close()


if __name__ == "__main__":
    dict_existing_symbols = {
        row["symbol"]: row["last_datetime"] for row in df_symbol_info.to_dicts()
    }
    ref_time = time.time()
    asyncio.run(
        main(instrument_data=perps.iloc[:], dict_existing_symbols=dict_existing_symbols)
    )
    print(f"Got results in {time.time() - ref_time:.2f}s")
