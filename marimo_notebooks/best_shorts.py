import marimo

__generated_with = "0.14.1"
app = marimo.App(width="medium", layout_file="layouts/best_shorts.grid.json")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Imports""")
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from datetime import datetime, timedelta
    from dotenv import find_dotenv, dotenv_values
    return datetime, dotenv_values, find_dotenv, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Config / Params""")
    return


@app.function
def get_db_uri(config):
    host = config.get('S2_HOST')
    user = config.get('S2_USER')
    password = config.get('S2_PASSWORD')
    database = "market"
    return f"mysql://{user}:{password}@{host}:3306/{database}"


@app.cell
def _(dotenv_values, find_dotenv):
    config = dotenv_values(find_dotenv())
    uri = get_db_uri(config)
    return (uri,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Inputs""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Database Functions""")
    return


@app.cell
def _(pl):
    def get_ohlcv_query(exchange, quote, start_date, end_date):
        # Dates are expected as string yyyy-mm-dd
        return f"""
            SELECT 
                datetime,
                symbol,
                open,
                high,
                low,
                close,
                volume as base_volume,
                quote_volume,
                volume_buy as base_volume_buy,
                quote_volume_buy
            FROM market.cc_ohlc_price_spot_1d
            WHERE 
                exchange = '{exchange}'
                and quote = '{quote}'
                AND datetime >= '{start_date} 00:00:00'
                AND datetime < '{end_date} 23:59:59'
        """

    def get_df_from_query(query, uri):
        return pl.read_database_uri(query=query, uri=uri, engine="connectorx")

    def fetch_ohlcv(query, uri):
        return pl.read_database_uri(query=query, uri=uri, engine="connectorx")

    def postprocess_ohlcv(df):
        df = (
            df
            .with_columns(
                (pl.col('quote_volume') / pl.col('base_volume')).alias('vwap')
            )
            .with_columns(
                (pl.col('quote_volume_buy') * 2 - pl.col('quote_volume')).alias('quote_volume_delta')
            )
            .filter(pl.col('vwap').is_not_nan())
            .sort('datetime')
        )
        return df
    return fetch_ohlcv, get_ohlcv_query, postprocess_ohlcv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Get Data""")
    return


@app.cell
def _(fetch_ohlcv, get_ohlcv_query, postprocess_ohlcv, uri):
    query = get_ohlcv_query(
        exchange='binance',
        quote='USDT',
        start_date='2021-01-01',
        end_date='2021-12-31',
    )
    df_raw = fetch_ohlcv(query, uri)
    df_ohlc = postprocess_ohlcv(df_raw)
    df_ohlc.head()
    return (df_ohlc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Top Performance""")
    return


@app.cell
def _(df_ohlc, pl):
    # Ensure 'datetime' is a datetime type, sort, and compute log returns per symbol
    df_symbols = df_ohlc.sort(["symbol", "datetime"])

    df_returns = df_symbols.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("log_return")
    )
    df_returns.head()
    return (df_returns,)


@app.cell
def _(df_returns, np, pl):
    window_n = 30
    annual_factor = np.sqrt(365)  # Assuming daily data

    # Calculate rolling mean, std, and Sharpe per symbol
    df_sharpe = (
        df_returns
        .with_columns([
            pl.col("log_return").rolling_mean(window_n).over("symbol").alias("roll_return_mean"),
            pl.col("log_return").rolling_std(window_n).over("symbol").alias("roll_return_std"),
        ])
        .with_columns(
            (pl.col("roll_return_mean") / pl.col("roll_return_std") * annual_factor).alias("sharpe_30d")
        )
    )
    df_sharpe.select(["symbol", "datetime", "sharpe_30d"]).tail(10)
    return (df_sharpe,)


@app.cell
def _(df_sharpe, pl):
    split_date = pl.datetime(2021, 11, 10)
    end_date = pl.datetime(2021, 12, 10)

    # Group by symbol and get minimum sharpe after split date
    df_after = (
        df_sharpe
        .filter(pl.col("datetime") >= split_date)
        .filter(pl.col("datetime") <= end_date)
        .group_by("symbol")
        .agg(
            pl.col("sharpe_30d").min().alias("min_sharpe_after")
        )
        .drop_nans()
        .sort("min_sharpe_after")
    )

    df_after
    return (df_after,)


@app.cell
def _(df_after, mo):
    import plotly.graph_objs as go

    fig_sharpe = go.Figure(
        data=[
            go.Bar(
                x=df_after["symbol"].to_list(),
                y=df_after["min_sharpe_after"].to_list(),
                marker_color='crimson',
                hoverinfo="x+y",
            )
        ]
    )

    fig_sharpe.update_layout(
        title="Lowest 30d Sharpe Ratio per Asset after 2025-11-10",
        xaxis_title="Symbol",
        yaxis_title="Minimum 30d Sharpe Ratio (post-2025-11-10)",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    mo.ui.plotly(fig_sharpe)
    return (go,)


@app.cell
def _(datetime, df_sharpe, pl):
    # Define the two target dates
    date_pre  = datetime(2021, 11, 10)
    date_post = datetime(2021, 12, 10)

    # 1. Get trailing 30d Sharpe on Nov 10, 2021
    df_pre = (
        df_sharpe
        .filter(pl.col("datetime") == date_pre)
        .select(["symbol", "sharpe_30d"])
        .rename({"sharpe_30d": "sharpe_30d_pre"})
    )

    # 2. Get trailing 30d Sharpe on Dec 10, 2021
    df_post = (
        df_sharpe
        .filter(pl.col("datetime") == date_post)
        .select(["symbol", "sharpe_30d"])
        .rename({"sharpe_30d": "sharpe_30d_post"})
    )

    # 3. Merge them for comparison
    df_compare = df_pre.join(df_post, on="symbol", how="inner")
    df_compare
    return (df_compare,)


@app.cell
def _(df_compare, go, mo):
    fig_trailing = go.Figure(
        data=[
            go.Scatter(
                x=df_compare["sharpe_30d_pre"].to_list(),
                y=df_compare["sharpe_30d_post"].to_list(),
                text=df_compare["symbol"].to_list(),
                mode="markers+text",
                textposition="top right",
                marker=dict(size=10, color="darkorange", line=dict(width=1, color="black")),
            )
        ]
    )

    fig_trailing.update_layout(
        title="30d Sharpe on 2021-11-10 vs. 2021-12-10 (Trailing)",
        xaxis_title="30d Sharpe ON 2021-11-10",
        yaxis_title="30d Sharpe ON 2021-12-10",
        template="plotly_white",
        showlegend=False
    )

    mo.ui.plotly(fig_trailing)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
