from typing import Union, List, Optional
import pandas as pd
import numpy as np


def min_max_norm(
    x: Union[pd.Series, np.ndarray, float, int],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Union[pd.Series, np.ndarray, float]:
    """
    Normalize data to range [0, 1] using min-max normalization.

    Args:
        x: Input data to normalize (array, series, or single numeric value)
        min_value: Optional minimum value for normalization. If None, uses min(x)
        max_value: Optional maximum value for normalization. If None, uses max(x)

    Returns:
        Normalized data with same type as input (except numeric inputs return float)

    Raises:
        ValueError: If max_value <= min_value or if input contains NaN
        TypeError: If input is not numeric, Series, or ndarray
    """
    # Handle single numeric values
    if isinstance(x, (int, float)):
        if pd.isna(x):
            raise ValueError("Input contains NaN value")

        min_val = min_value if min_value is not None else x
        max_val = max_value if max_value is not None else x

        if max_val <= min_val:
            raise ValueError(
                f"max_value ({max_val}) must be greater than min_value ({min_val})"
            )

        return float((x - min_val) / (max_val - min_val))

    # Handle arrays and series
    elif isinstance(x, (pd.Series, np.ndarray)):

        min_val = min_value if min_value is not None else np.min(x)
        max_val = max_value if max_value is not None else np.max(x)

        if max_val <= min_val:
            raise ValueError(
                f"max_value ({max_val}) must be greater than min_value ({min_val})"
            )

        return (x - min_val) / (max_val - min_val)
    else:
        raise TypeError(
            f"Expected numeric value, pd.Series, or np.ndarray, got {type(x)}"
        )


def normalize_0_1(
    df: pd.DataFrame,
    cols: List[str],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Normalize specified columns of a DataFrame to range [0, 1].

    Args:
        df: Input DataFrame
        cols: List of column names to normalize
        min_value: Optional minimum value for normalization. If None, uses min per column
        max_value: Optional maximum value for normalization. If None, uses max per column
        inplace: If True, modify df in place. If False, return a copy

    Returns:
        DataFrame with normalized columns

    Raises:
        ValueError: If any column not in DataFrame or contains NaN
        KeyError: If cols is empty
    """
    if not cols:
        raise KeyError("No columns specified for normalization")

    if not all(col in df.columns for col in cols):
        missing = [col for col in cols if col not in df.columns]
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    result = df if inplace else df.copy()

    for col in cols:
        result[col] = min_max_norm(result[col], min_value, max_value)

    return result
