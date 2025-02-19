from typing import Dict, Any, Callable, Optional, Union, List
import pandas as pd
import numpy as np
import logging
from time import time

logger = logging.getLogger(__name__)


def generate_features(
    df: pd.DataFrame, features: List[Dict], group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate features from a DataFrame, optionally grouping by instrument.
    Features are generated in order, allowing subsequent features to use columns
    created by previous transformations.

    Args:
        df: Input DataFrame containing OHLC data
        features: List of feature definitions, each containing:
            - name: Feature name
            - func: Function to generate the feature
            - params: Parameters for the function
            - group_by_instrument: Whether to apply grouping (defaults to True)
        group_col: Column name to group by (e.g., 'instrument')

    Returns:
        DataFrame with added feature columns
    """
    result = df.copy()

    def apply_func(
        data: pd.DataFrame, func: Callable, group_by_instrument: bool = True, **kwargs
    ) -> pd.Series:
        """Apply function to data, with or without grouping."""
        try:
            if group_col and group_by_instrument:
                # Pre-sort by group and datetime to optimize rolling operations
                if "datetime" in data.columns:
                    data = data.sort_values([group_col, "datetime"])
                groups = data.groupby(group_col, group_keys=False)
                result = groups.apply(lambda x: func(x, **kwargs))
                if isinstance(result.index, pd.MultiIndex):
                    result = result.reset_index(level=0, drop=True)
                return result
            return func(data, **kwargs)
        except Exception as e:
            logger.error(f"Error in function {func.__name__}: {str(e)}")
            raise

    start_time = time()
    for feature in features:
        try:
            name = feature["name"]
            func = feature["func"]
            params = feature.get("params", {})
            group_by_instrument = feature.get(
                "group_by_instrument", True
            )  # Default to True for backward compatibility

            feature_start = time()
            result[name] = apply_func(
                result, func, group_by_instrument=group_by_instrument, **params
            )
            feature_time = time() - feature_start

            if feature_time > 0.1:  # Log slow features
                logger.info(f"Feature {name} took {feature_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Failed to generate feature {name}: {str(e)}")
            continue

    total_time = time() - start_time
    logger.info(f"Total feature generation time: {total_time:.2f} seconds")
    return result


def calc_forward_return(
    df: pd.DataFrame, lookforward: int = 1, col: str = "close", log_return: bool = True
) -> pd.Series:
    """Calculate forward returns."""
    if col not in df.columns:
        raise ValueError(f"Column {col} not found")

    prices = df[col].values  # Convert to numpy array for speed
    if log_return:
        return pd.Series(np.log(np.roll(prices, -lookforward) / prices), index=df.index)
    return pd.Series(np.roll(prices, -lookforward) / prices - 1, index=df.index)


def calc_ema(df: pd.DataFrame, col: str = "close", lookback: int = 5) -> pd.Series:
    """Calculate exponential moving average of any column."""
    if col not in df.columns:
        raise ValueError(f"Column {col} not found")

    # Use numpy array for calculations
    values = df[col].values
    alpha = 2 / (lookback + 1)

    # Initialize output array
    result = np.empty_like(values)
    result[0] = values[0]

    # Vectorized EMA calculation
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]

    return pd.Series(result, index=df.index)


def calc_range_percentage(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    ref_col: str = "close",
) -> pd.Series:
    """Calculate range as percentage of reference price."""
    for col in [high_col, low_col, ref_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found")

    # Vectorized calculation using numpy
    high = df[high_col].values
    low = df[low_col].values
    ref = df[ref_col].values

    return pd.Series((high - low) / ref, index=df.index)


def _get_rolling_windows(values: np.ndarray, lookback: int) -> np.ndarray:
    """
    Create rolling windows array for efficient calculations.

    Args:
        values: Input array of values
        lookback: Size of the rolling window

    Returns:
        Array of shape (n, lookback) containing rolling windows
    """
    n = len(values)
    windows = np.full((n, lookback), np.nan)

    for i in range(lookback - 1, n):
        windows[i] = values[i - lookback + 1 : i + 1]

    return windows


def _rolling_window_extremes(
    values: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate rolling window maximum and minimum values efficiently.

    Args:
        values: Input array of values
        lookback: Size of the rolling window

    Returns:
        Tuple of (rolling_max, rolling_min) arrays
    """
    windows = _get_rolling_windows(values, lookback)

    if windows.size == 0:
        return np.array([np.nan] * len(values)), np.array([np.nan] * len(values))

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Calculate max and min
        rolling_max = np.nanmax(windows, axis=1)
        rolling_min = np.nanmin(windows, axis=1)

    return rolling_max, rolling_min


def get_bar_triangle(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """
    Calculate bar triangle pattern - bars contained within previous highs and lows.

    Args:
        df: DataFrame containing OHLC data
        lookback: Number of bars to look back for highest high and lowest low

    Returns:
        Series of boolean values indicating where bars are contained within the range
    """
    for col in ["high", "low"]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found")

    high = df["high"].values
    low = df["low"].values

    # Calculate rolling max/min excluding current bar
    highest_high = _rolling_window_extremes(high, lookback)[0]
    lowest_low = _rolling_window_extremes(low, lookback)[1]

    # Vectorized comparison
    bar_triangle = (high < highest_high) & (low > lowest_low)
    return pd.Series(bar_triangle, index=df.index)


def calc_pos_in_range(
    df: pd.DataFrame,
    lookback: int = 10,
    high_col: str = "high",
    low_col: str = "low",
    price_col: str = "close",
) -> pd.Series:
    """Calculate position within price range."""
    for col in [high_col, low_col, price_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found")

    # Use vectorized operations with helper function
    high = df[high_col].values
    low = df[low_col].values
    price = df[price_col].values

    highest_high = _rolling_window_extremes(high, lookback)[0]
    lowest_low = _rolling_window_extremes(low, lookback)[1]

    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator[denominator == 0] = np.nan

    return pd.Series((price - lowest_low) / denominator, index=df.index)


def calc_cmema(
    df: pd.DataFrame,
    fast_lookback: int = 5,
    slow_lookback: int = 20,
    col: str = "close",
    normalize_by: Optional[str] = None,
) -> pd.Series:
    """Calculate cross-MA momentum indicator."""
    if col not in df.columns:
        raise ValueError(f"Column {col} not found")
    if normalize_by and normalize_by not in df.columns:
        raise ValueError(f"Normalize column {normalize_by} not found")

    # Calculate EMAs using vectorized function
    ema_fast = calc_ema(df, col=col, lookback=fast_lookback)
    ema_slow = calc_ema(df, col=col, lookback=slow_lookback)

    result = ema_fast / ema_slow - 1

    if normalize_by:
        result = result / df[normalize_by]
    return result


def get_highest_high(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """
    Calculate rolling highest high using vectorized operations.

    Args:
        df: DataFrame containing OHLC data
        lookback: Number of periods to look back

    Returns:
        Series with rolling highest high values
    """
    if "high" not in df.columns:
        raise ValueError("Column 'high' not found")

    return pd.Series(
        _rolling_window_extremes(df["high"].values, lookback)[0], index=df.index
    )


def get_lowest_low(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """
    Calculate rolling lowest low using vectorized operations.

    Args:
        df: DataFrame containing OHLC data
        lookback: Number of periods to look back
        lookback: Number of periods to look back

    Returns:
        Series with rolling lowest low values
    """
    if "low" not in df.columns:
        raise ValueError("Column 'low' not found")

    return pd.Series(
        _rolling_window_extremes(df["low"].values, lookback)[1], index=df.index
    )


def shift_values(df: pd.DataFrame, col: str, shift: int = 1) -> pd.Series:
    """
    Shift values in a column by a specified number of periods.

    Args:
        df: DataFrame containing the data
        col: Column name to shift
        shift: Number of periods to shift (positive = backward, negative = forward)

    Returns:
        Series with shifted values

    Raises:
        ValueError: If column not found in DataFrame
    """
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in DataFrame")

    return df[col].shift(shift)


def calc_aroon(
    df: pd.DataFrame, lookback: int = 14, high_col: str = "high", low_col: str = "low"
) -> pd.Series:
    """Calculate Aroon Oscillator using vectorized operations."""
    for col in [high_col, low_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found")

    high = df[high_col].values
    low = df[low_col].values

    # Get rolling windows for high and low values
    window_highs = _get_rolling_windows(high, lookback)
    window_lows = _get_rolling_windows(low, lookback)

    # Calculate days since highest high and lowest low using vectorized operations
    high_days = (lookback - 1 - np.argmax(window_highs, axis=1)) * (
        100 / (lookback - 1)
    )
    low_days = (lookback - 1 - np.argmin(window_lows, axis=1)) * (100 / (lookback - 1))
    if window_highs.size == 0 or window_lows.size == 0:
        return pd.Series(np.array([np.nan] * len(df)), index=df.index)
    return pd.Series(low_days - high_days, index=df.index)
