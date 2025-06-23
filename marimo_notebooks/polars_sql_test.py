import marimo

__generated_with = "0.14.1"
app = marimo.App(layout_file="layouts/polars_sql_test.grid.json")


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
        list_instruments, value=default_symbol, label="Symbol"
    )

    dt_start = mo.ui.date(label="Start Date", value=(datetime.utcnow() - timedelta(days=30)).date())
    dt_end = mo.ui.date(label="End Date", value=datetime.utcnow().date())
    return dd_symbol, dt_end, dt_start


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
def _(
    dd_exchange,
    dd_symbol,
    dt_end,
    dt_start,
    fetch_ohlcv,
    get_ohlcv_query,
    postprocess_ohlcv,
    uri,
):
    query = get_ohlcv_query(
        exchange=dd_exchange.value,
        symbol=dd_symbol.value,
        start_date=dt_start.value.strftime("%Y-%m-%d"),
        end_date=dt_end.value.strftime("%Y-%m-%d"),
    )
    df_raw = fetch_ohlcv(query, uri)
    df_ohlc = postprocess_ohlcv(df_raw)
    df_ohlc.head()
    return (df_ohlc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Volume Profile Functions""")
    return


@app.cell
def _():
    import numpy as np
    from quant_helper.features.volume_profile import (
        get_xr, 
        get_kdy, 
        get_ticks_per_sample, 
        get_lv_levels,
    )
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    return (
        get_kdy,
        get_lv_levels,
        get_ticks_per_sample,
        get_xr,
        go,
        make_subplots,
        np,
    )


@app.cell
def _(
    get_kdy,
    get_lv_levels,
    get_ticks_per_sample,
    get_xr,
    go,
    make_subplots,
    np,
):
    def prepare_volume_profile(df_ohlc, num_samples=500):
        volume_ = df_ohlc['base_volume']
        n = len(volume_)
        decay_rate = 1 / (0.5 * n)
        weights = np.exp(-decay_rate * np.arange(n))[::-1]
        weighted_volume = volume_ * weights
        xr = get_xr(df_ohlc['vwap'], num_samples=num_samples)
        ticks_per_sample = get_ticks_per_sample(xr, num_samples=num_samples)
        kdy_raw = get_kdy(price=df_ohlc['vwap'], volume=volume_, num_samples=num_samples)
        lv_levels_raw = get_lv_levels(xr, kdy_raw, ticks_per_sample)[0]
        kdy_weighted = get_kdy(price=df_ohlc['vwap'], volume=weighted_volume, num_samples=num_samples)
        lv_levels_weighted = get_lv_levels(xr, kdy_weighted, ticks_per_sample)[0]
        return xr, kdy_raw, kdy_weighted, lv_levels_raw, lv_levels_weighted


    def plot_volume_profile(df_ohlc_resampled, profile_results, symbol):
        xr, kdy_raw, kdy_weighted, lv_levels_raw, lv_levels_weighted = profile_results
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.4, 0.6],
            horizontal_spacing=0.05,
            shared_yaxes=True,
            shared_xaxes='columns',
        )
        fig.add_trace(
            go.Scatter(x=kdy_raw, y=xr, line={'color':'red'}, name='Raw Volume'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=kdy_weighted, y=xr, line={'color':'blue'}, name='Weighted Volume'),
            row=1, col=1
        )
        # Use 'bar_time' instead of 'date'
        fig.add_trace(
            go.Candlestick(
                x=df_ohlc_resampled['bar_time'],
                open=df_ohlc_resampled['open'],
                high=df_ohlc_resampled['high'],
                low=df_ohlc_resampled['low'],
                close=df_ohlc_resampled['close'],
                showlegend=False,
            ),
            row=1, col=2
        )
        for lvl in lv_levels_raw:
            fig.add_hline(lvl, line={'width':1, 'dash':'dash', 'color':'red'})
        for lvl in lv_levels_weighted:
            fig.add_hline(lvl, line={'width':1, 'dash':'dash', 'color':'blue'})
        fig.update_xaxes(rangeslider={'visible':False}, row=1, col=2)
        fig.update_layout(title=symbol, height=600)
        return fig
    return plot_volume_profile, prepare_volume_profile


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
    return (df_ohlc_1d,)


@app.cell
def _(df_ohlc, dt_end, dt_start, pl):
    # Resampled OHLC 
    def best_bar_interval(start_date, end_date, target_bars=60):
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

    _bar_duration = best_bar_interval(dt_start.value, dt_end.value, target_bars=60)

    df_ohlc_resampled = (
        df_ohlc
        .with_columns(pl.col('datetime').dt.truncate(_bar_duration).alias('bar_time'))
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
    return (df_ohlc_resampled,)


@app.cell
def _(df_ohlc, df_ohlc_resampled, pl):
    # Calculate anchored VWAP at 1-minute resolution
    price = df_ohlc['vwap']
    volume = df_ohlc['base_volume']

    # cum_qv = (price * volume).cumsum()
    # cum_vol = volume.cumsum()
    # anchored_vwap = (cum_qv / cum_vol).alias('anchored_vwap')

    df_min_vwap = (
        df_ohlc.with_columns([
            (pl.col("vwap") * pl.col("base_volume")).cumsum().alias("cum_qv"),
            pl.col("base_volume").cumsum().alias("cum_vol")
        ])
        .with_columns(
            (pl.col("cum_qv") / pl.col("cum_vol")).alias("anchored_vwap")
        )
    )

    # Downsample anchored VWAP to the resampled ohlc time grid
    df_vwap_resampled = (
        df_min_vwap
        .with_columns(pl.col('datetime').dt.truncate(_bar_duration).alias('bar_time'))
        .group_by('bar_time')
        .agg([
            pl.col('anchored_vwap').last().alias('anchored_vwap'),
        ])
        .sort('bar_time')
    )

    # Join VWAP and OHLC on bar_time for plotting
    df_plot = df_ohlc_resampled.join(df_vwap_resampled, on='bar_time', how='left')
    df_plot.tail()
   
    return


@app.cell
def _(df_ohlc_1d):
    df_ohlc_1d.tail()
    return


@app.cell
def _(dd_exchange, dd_symbol, dt_end, dt_start, mo):
    mo.hstack([dd_exchange, dd_symbol, dt_start, dt_end])
    return


@app.cell
def _(
    dd_symbol,
    df_ohlc,
    df_ohlc_resampled,
    mo,
    plot_volume_profile,
    prepare_volume_profile,
):
    profile_results = prepare_volume_profile(df_ohlc)
    mo.ui.plotly(plot_volume_profile(df_ohlc_resampled, profile_results, dd_symbol.value))
    return


@app.cell
def _(df_ohlc_resampled):
    df_ohlc_resampled
    return


@app.cell
def _(df_ohlc_1d):
    (df_ohlc_1d['quote_volume_buy'] * 2 - df_ohlc_1d['quote_volume']).cum_sum()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
