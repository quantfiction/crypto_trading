import marimo

__generated_with = "0.14.1"
app = marimo.App(width="medium", layout_file="layouts/monday_dump.grid.json")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Imports""")
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from datetime import datetime, timedelta
    from dotenv import find_dotenv, dotenv_values
    return datetime, dotenv_values, find_dotenv, mo, pl, timedelta


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


@app.cell
def _(mo):
    exchanges = ["bybit", "binance"]
    dd_exchange = mo.ui.dropdown(
        exchanges, value=exchanges[0], label="Exchange"
    )
    return (dd_exchange,)


@app.cell
def _(dd_exchange, get_df_from_query, uri):
    query_instruments = f"""
        select distinct instrument
        from market.cc_futures_ohlcv_1m
        where market = '{dd_exchange.value}'
    """
    # 4. Fetch the instrument list for the selected exchange
    list_instruments = [x[0] for x in get_df_from_query(query_instruments, uri).rows()]
    return (list_instruments,)


@app.cell
def _(datetime, list_instruments, mo, timedelta):
    default_symbol = list_instruments[0] if list_instruments else ""
    dd_symbol = mo.ui.dropdown(
        sorted(list_instruments), value=default_symbol, label="Symbol"
    )

    dt_start = mo.ui.date(label="Start Date", value=(datetime.utcnow() - timedelta(days=30)).date())
    dt_end = mo.ui.date(label="End Date", value=datetime.utcnow().date())
    return dt_end, dt_start


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Database Functions""")
    return


@app.cell
def _(pl):
    def get_ohlcv_query(exchange, symbol, start_date, end_date):
        # Dates are expected as string yyyy-mm-dd
        return f"""
            SELECT 
                datetime,
                open,
                high,
                low,
                close,
                volume as base_volume,
                quote_volume,
                volume_buy as base_volume_buy,
                quote_volume_buy
            FROM market.cc_futures_ohlcv_1m
            WHERE 
                market = '{exchange}'
                AND instrument = '{symbol}'
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
    return fetch_ohlcv, get_df_from_query, get_ohlcv_query, postprocess_ohlcv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Get Data""")
    return


@app.cell
def _(fetch_ohlcv, get_ohlcv_query, postprocess_ohlcv, uri):
    symbol = 'BTCUSDT'
    query = get_ohlcv_query(
        exchange='binance',
        symbol=symbol,
        start_date='2025-06-01',
        end_date='2025-08-12',
    )
    df_raw = fetch_ohlcv(query, uri)
    df_ohlc = postprocess_ohlcv(df_raw)
    df_ohlc.head()
    return df_ohlc, symbol


@app.cell
def _(df_ohlc, pl):
    df_ohlc_1d = (
        df_ohlc
        .with_columns(pl.col('datetime').dt.date().alias('date'))
        .group_by('date')
        .agg([
            pl.col('open').first().alias('open'),
            pl.col('high').max().alias('high'),
            pl.col('low').min().alias('low'),
            pl.col('close').last().alias('close'),
            pl.col('quote_volume').sum().alias('quote_volume'),
            pl.col('quote_volume_buy').sum().alias('quote_volume_buy'),
        ])
        .sort('date')
    )
    return


@app.cell
def _(df_ohlc, dt_end, dt_start, pl):
    # Resampled OHLC 
    def best_bar_interval(start_date, end_date, target_bars=100):
        # List of standardized intervals with minutes in each
        allowed = [
            ("15m", 15), ("30m", 30), ("1h", 60), ("2h", 120), ("4h", 240),
            ("6h", 360), ("12h", 720), ("1d", 1440), ("3d", 4320), ("7d", 10080)
        ]
        total_minutes = (end_date - start_date).days * 24 * 60
        # Pick the smallest interval where num_bars <= target
        for iso, mins in allowed:
            n_bars = total_minutes // mins
            if n_bars <= target_bars:
                return iso
        # Default to largest if all else fails
        return allowed[-1][0]

    _bar_duration = best_bar_interval(dt_start.value, dt_end.value, target_bars=100)

    df_ohlc_resampled = (
        df_ohlc
        .with_columns(pl.col('datetime').dt.truncate('30m').alias('bar_time'))
        .group_by('bar_time')
        .agg([
            pl.col('open').first().alias('open'),
            pl.col('high').max().alias('high'),
            pl.col('low').min().alias('low'),
            pl.col('close').last().alias('close'),
            pl.col('quote_volume').sum().alias('quote_volume'),
            pl.col('quote_volume_buy').sum().alias('quote_volume_buy'),
        ])
        .sort('bar_time')
    )
    df_ohlc_resampled.head()
    return (df_ohlc_resampled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Plot""")
    return


@app.cell
def _():
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    return go, make_subplots


@app.cell
def _(datetime, df_ohlc_resampled, go, make_subplots, symbol):
    def extract_relative_change_window_hours(df, anchor_date, window_bars=48, price_col='close', time_col='bar_time'):
        # Find the anchor index
        anchor_idx = df[time_col].to_list().index(anchor_date)
        # Slice
        start = max(0, anchor_idx - window_bars)
        end = min(len(df), anchor_idx + window_bars + 1)
        window = df.slice(start, end - start)
        anchor_price = df[anchor_idx, price_col]
        rel_index = list(range(start - anchor_idx, end - anchor_idx))
        # Convert to hours (30min per bar)
        hours_from_anchor = [i * 0.5 for i in rel_index]
        relative_change = (window[price_col] - anchor_price) / anchor_price
        return {
            'hours_from_anchor': hours_from_anchor,
            'relative_change': relative_change.to_list(),
            'bar_time': window[time_col].to_list()
        }

    # Set your parameters
    anchor_dates = ['2025-06-09 00:00:00', '2025-08-11 00:00:00']
    window_bars = 144 # 1 day before/after (30-min bars)

    # Convert the anchor_dates to datetime objects matching your DataFrame's type
    anchor_dates_dt = [datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in anchor_dates]

    # Then call your function as:
    rc1 = extract_relative_change_window_hours(df_ohlc_resampled, anchor_dates_dt[0], window_bars)
    rc2 = extract_relative_change_window_hours(df_ohlc_resampled, anchor_dates_dt[1], window_bars)

    # Plotly chart with improved axis formatting
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"{anchor_dates[0]}",
            f"{anchor_dates[1]}"
        ),
        vertical_spacing=0.12
    )

    fig.add_trace(
        go.Scatter(
            x=rc1['hours_from_anchor'],
            y=rc1['relative_change'],
            mode='lines+markers',
            name=f"Relative Change ({anchor_dates[0]})",
            hovertemplate="Hours offset: %{x}<br>Rel. Change: %{y:.2%}<br>Date: %{customdata}",
            customdata=rc1["bar_time"]
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=rc2['hours_from_anchor'],
            y=rc2['relative_change'],
            mode='lines+markers',
            name=f"Relative Change ({anchor_dates[1]})",
            hovertemplate="Hours offset: %{x}<br>Rel. Change: %{y:.2%}<br>Date: %{customdata}",
            customdata=rc2["bar_time"]
        ),
        row=2, col=1
    )

    for row in [1,2]:
        fig.add_hline(
            y=0, 
            line=dict(color='white', dash='dash', width=1),
            row=row, col=1
        )

        fig.add_vline(
            x=0, 
            line=dict(color='white', dash='dash', width=1),
            row=row, col=1
        )

    # Update axis formatting
    fig.update_xaxes(title_text="Hours from Anchor", row=2, col=1)
    fig.update_yaxes(title_text="Relative Change", tickformat=".2%", row=1, col=1)
    fig.update_yaxes(title_text="Relative Change", tickformat=".2%", row=2, col=1)
    fig.update_layout(
        height=700, width=900, 
        title=f"{symbol}, 2nd Monday of the Month",
        showlegend=False
    )

    fig
    return


@app.cell
def _(df_ohlc_resampled):
    df_ohlc_resampled
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
