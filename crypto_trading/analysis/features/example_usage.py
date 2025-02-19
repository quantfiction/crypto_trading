import pandas as pd
import numpy as np
from feature_generator import (
    generate_features,
    calc_forward_return,
    calc_ema,
    calc_range_percentage,
    calc_pos_in_range,
    calc_cmema,
    calc_aroon,
)


def example_single_instrument():
    """Example of feature generation for a single instrument."""
    print("\n=== Single Instrument Example ===")

    # Create sample data for one instrument
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100)
    base = 100 + np.arange(100) * 0.1

    data = {
        "datetime": dates,
        "close": base + np.random.normal(0, 0.1, 100),
        "high": base + 1 + np.random.normal(0, 0.1, 100),
        "low": base - 1 + np.random.normal(0, 0.1, 100),
        "volume": np.random.uniform(900000, 1100000, 100),
    }
    df = pd.DataFrame(data)

    # Define features with chaining
    # First calculate range percentage, then its EMA
    features = [
        {
            "name": "range_perc",
            "func": calc_range_percentage,
            "params": {},  # Uses default column names
        },
        {
            "name": "range_perc_ema",
            "func": calc_ema,
            "params": {
                "col": "range_perc",  # Uses output from previous feature
                "lookback": 5,
            },
        },
        {
            "name": "fwd_return_1d",
            "func": calc_forward_return,
            "params": {"lookforward": 1, "log_return": True},
        },
        {
            "name": "pos_in_range_10",
            "func": calc_pos_in_range,
            "params": {"lookback": 10},
        },
        {
            "name": "cmema_5_20",
            "func": calc_cmema,
            "params": {
                "fast_lookback": 5,
                "slow_lookback": 20,
                "normalize_by": "range_perc_ema",  # Uses the EMA we calculated earlier
            },
        },
        {
            "name": "aroon_10",
            "func": calc_aroon,
            "params": {
                "lookback": 30,
            },
        },
    ]

    # Generate features
    result_df = generate_features(df, features)  # No group_col needed

    print("\nSample of single instrument results:")
    print(result_df.head())
    print("\nFeature correlations:")
    print(
        result_df.select_dtypes(include=[np.number])
        .corr()["fwd_return_1d"]
        .sort_values(ascending=False)
    )


def example_multi_instrument():
    """Example of feature generation for multiple instruments."""
    print("\n=== Multi-Instrument Example ===")

    # Create sample data for multiple instruments
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=50)
    instruments = ["BTC-USD", "ETH-USD"]

    all_data = []
    for inst in instruments:
        base = 100 + np.arange(50) * (0.2 if inst == "BTC-USD" else 0.1)
        data = {
            "datetime": dates,
            "instrument": inst,
            "close": base + np.random.normal(0, 0.1, 50),
            "high": base + 1 + np.random.normal(0, 0.1, 50),
            "low": base - 1 + np.random.normal(0, 0.1, 50),
            "volume": np.random.uniform(900000, 1100000, 50),
        }
        all_data.append(pd.DataFrame(data))

    df = pd.concat(all_data, ignore_index=True)

    # Define features - same as single instrument case
    features = [
        {"name": "range_perc", "func": calc_range_percentage, "params": {}},
        {
            "name": "range_perc_ema",
            "func": calc_ema,
            "params": {"col": "range_perc", "lookback": 5},
        },
        {
            "name": "fwd_return_1d",
            "func": calc_forward_return,
            "params": {"lookforward": 1, "log_return": True},
        },
        {
            "name": "pos_in_range_10",
            "func": calc_pos_in_range,
            "params": {"lookback": 10},
        },
        {
            "name": "cmema_5_20",
            "func": calc_cmema,
            "params": {
                "fast_lookback": 5,
                "slow_lookback": 20,
                "normalize_by": "range_perc_ema",
            },
        },
    ]

    # Generate features with grouping
    result_df = generate_features(df, features, group_col="instrument")

    print("\nSample of multi-instrument results:")
    # Show first few rows for each instrument
    for inst in instruments:
        print(f"\n{inst}:")
        inst_df = result_df[result_df["instrument"] == inst]
        print(inst_df.tail())
        print(f"\nFeature correlations for {inst}:")
        print(
            inst_df.select_dtypes(include=[np.number])
            .corr()["fwd_return_1d"]
            .sort_values(ascending=False)
        )


if __name__ == "__main__":
    example_single_instrument()
    example_multi_instrument()
