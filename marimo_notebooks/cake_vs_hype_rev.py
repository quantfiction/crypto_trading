import marimo

__generated_with = "0.14.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go
    import numpy as np
    from datetime import datetime as dt
    return go, np


@app.cell
def _():
    from PIL import Image
    return


@app.cell
def _():
    # data_type = 'Fees'
    data_type = 'Revenue'
    return (data_type,)


@app.cell
def _(data_type):
    import requests
    import pandas as pd

    def fetch_llama_revenue(protocol: str):
        # Fetch DefiLlama summary fees endpoint for a protocol
        url = f"https://api.llama.fi/summary/fees/{protocol}?dataType=daily{data_type}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data

    # Example usage for both protocols
    # PancakeSwap's protocol id is usually "pancakeswap"
    # Hyperliquid's protocol id is "parent#hyperliquid"
    protocols = {
        "PancakeSwap": "pancakeswap",
        "Hyperliquid": "hyperliquid"
    }

    # Fetch all data into a dict for next step
    llama_data = {name: fetch_llama_revenue(pid) for name, pid in protocols.items()}

    llama_data.keys()
    return llama_data, pd, requests


@app.cell
def _(llama_data):
    list(llama_data['PancakeSwap'].keys())
    return


@app.cell
def _(llama_data, pd):
    from datetime import datetime

    def chart_to_df(chart, protocol_name):
        # Each item: [timestamp, revenue]
        df = pd.DataFrame(chart, columns=["timestamp", protocol_name])
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df[["date", protocol_name]]
        return df

    # Extract and convert to DataFrame for each protocol
    ps_df = chart_to_df(llama_data["PancakeSwap"]["totalDataChart"], "PancakeSwap")
    hl_df = chart_to_df(llama_data["Hyperliquid"]["totalDataChart"], "Hyperliquid")

    # Merge on the date for aligned comparison
    revenue_comparison = pd.merge(ps_df, hl_df, on="date", how="outer").sort_values("date")
    revenue_comparison["PancakeSwap_30d"] = revenue_comparison["PancakeSwap"].interpolate().rolling(30).sum() * 12
    revenue_comparison["Hyperliquid_30d"] = revenue_comparison["Hyperliquid"].interpolate().rolling(30).sum() * 12

    start_date = '2025-04-01'
    revenue_comparison = revenue_comparison.query('date >= @start_date')
    revenue_comparison.tail(8)  # Show most recent days
    return datetime, revenue_comparison, start_date


@app.cell
def _(
    base64,
    data_type,
    datetime,
    go,
    llama_data,
    np,
    pd,
    revenue_comparison,
    start_date,
):
    def format_usd_short(value):
        """Format a dollar value using k, M, B with 3 significant digits."""
        thresholds = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")]
        for thresh, suffix in thresholds:
            if abs(value) >= thresh:
                scaled = value / thresh
                # 3 significant figures
                if scaled >= 100:
                    fmt = "${:,.0f}{}".format(scaled, suffix)
                elif scaled >= 10:
                    fmt = "${:,.1f}{}".format(scaled, suffix)
                else:
                    fmt = "${:,.2f}{}".format(scaled, suffix)
                return fmt
        return "${:,.2f}".format(value)

    # Plot using base64 logos for layout_image
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=revenue_comparison["date"], y=revenue_comparison["PancakeSwap_30d"],
        mode="lines", name="PancakeSwap",
        line=dict(width=2, color="#d0884e"),
    ))
    fig.add_trace(go.Scatter(
        x=revenue_comparison["date"], y=revenue_comparison["Hyperliquid_30d"],
        mode="lines", name="Hyperliquid",
        line=dict(width=2, color="#96fce4"),
    ))

    # Find most recent aligned data for logo positioning
    latest_row = revenue_comparison.dropna(subset=["PancakeSwap_30d", "Hyperliquid_30d"]).iloc[-1]

    # Get the most recent aligned date and values
    final_date = latest_row["date"]
    final_ps = latest_row["PancakeSwap_30d"]
    final_hl = latest_row["Hyperliquid_30d"]
    img_size = np.mean([np.mean([final_ps, final_hl]), np.max([final_ps, final_hl])]) / 8

    fig.add_layout_image(
        dict(
            source=llama_data["PancakeSwap"]["logo"],
            x=latest_row["date"],
            y=latest_row["PancakeSwap_30d"],
            xref="x", yref="y",
            sizex=(datetime(2023, 1, 20) - datetime(2023, 1, 10)).total_seconds() * 1000,
            sizey=(img_size),
            xanchor="left", yanchor="middle", opacity=0.8, layer="above"
        )
    )

    fig.add_layout_image(
        dict(
            source=llama_data["Hyperliquid"]["logo"],
            x=latest_row["date"],
            y=latest_row["Hyperliquid_30d"],
            xref="x", yref="y",
            sizex=(datetime(2023, 1, 20) - datetime(2023, 1, 10)).total_seconds() * 1000,
            sizey=(img_size),
            xanchor="left", yanchor="middle", opacity=0.8, layer="above"
        )
    )

    # Read and encode the watermark image
    with open("../references/QF_Logo_big_background.png", "rb") as img_file:
        img_bytes = img_file.read()
        watermark_base64 = base64.b64encode(img_bytes).decode()
        watermark_url = f"data:image/png;base64,{watermark_base64}"

    # Add watermark as a 'background' layout image to the chart
    fig.add_layout_image(
        dict(
            source=watermark_url,
            xref="paper", yref="paper",
            x=0.5, y=0.5,  # Centered
            sizex=1.0, sizey=0.8,  # Full width, most of height
            xanchor="center", yanchor="middle",
            layer="below",  # Put the watermark below chart data
            opacity=0.08,
        )
    )

    fig.add_annotation(
        text="Source: DefiLlama | Chart: @quantfiction",
        font=dict(size=13, color="#888"),
        showarrow=False, xref="paper", yref="paper", x=1.06, y=-0.13, xanchor="right"
    )


    # Add PancakeSwap final value
    fig.add_annotation(
        text=f"<b>{format_usd_short(final_ps)}</b>",
        x=final_date, 
        y=final_ps,
        xref="x", yref="y",
        xanchor="left", yanchor="middle",
        font=dict(size=17, color="#d0884e", family="Roboto, Arial, sans-serif"),
        showarrow=False,
        align="left",
        # offset to the right
        xshift=65,
    )

    # Add Hyperliquid final value
    fig.add_annotation(
        text=f"<b>{format_usd_short(final_hl)}</b>",
        x=final_date, 
        y=final_hl,
        xref="x", yref="y",
        xanchor="left", yanchor="middle",
        font=dict(size=17, color="#44d9b9", family="Roboto, Arial, sans-serif"),
        showarrow=False,
        align="left",
        # offset to the right
        xshift=65,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Annualized {data_type}: PancakeSwap vs. Hyperliquid</b>",
            x=0.05, xanchor="left", font=dict(size=16),
        ),
        xaxis_title="",
        yaxis_title=f"Ann. 30-Day Rolling {data_type} (USD)",
        legend_title="Protocol",
        xaxis={
            'range':[pd.to_datetime(start_date),pd.to_datetime('7/15/2025')],
        },
        yaxis={
            'tickprefix':'$',
        },
        height=600,
        width=900,
        showlegend=False,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        margin = {
            't':50,
            'b':50,
            'l':50,
            'r':50,
        },
        plot_bgcolor="#0f0f0f",
        paper_bgcolor="#0f0f0f"
    )

    fig
    return


@app.cell
def _(go, llama_data):
    fig_test = go.Figure()

    fig_test.add_layout_image(
        source=llama_data.get('Hyperliquid').get('logo'),
        x=1, y=1,
        xref="paper", yref="paper",
        sizex=0.1, sizey=0.1,
        xanchor="right", yanchor="top"
    )
    return


@app.cell
def _(llama_data, requests):
    import base64
    from io import BytesIO

    def image_url_to_base64(url):
        response = requests.get(url)
        img_bytes = BytesIO(response.content)
        encoded = base64.b64encode(img_bytes.read()).decode()
        return f"data:image/png;base64,{encoded}"

    # Transform both logos to base64
    ps_logo_base64 = image_url_to_base64(llama_data["PancakeSwap"]["logo"])
    hl_logo_base64 = image_url_to_base64(llama_data["Hyperliquid"]["logo"])
    ps_logo_base64[:40], hl_logo_base64[:40]  # show snippet to verify
    return (base64,)


@app.cell
def _(revenue_comparison):
    revenue_comparison.tail(30)
    return


@app.cell
def _(llama_data, pd):
    pd.json_normalize(llama_data['Hyperliquid']['totalDataChartBreakdown'][-1]).sum(axis=1)
    return


@app.cell
def _(revenue_comparison):
    revenue_comparison.iloc[-1]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
