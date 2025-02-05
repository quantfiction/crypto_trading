import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional


class TimeSeriesFeatures:
    """
    Efficient time series feature computation for multiple instruments.
    Uses groupby.transform() for better performance and maintains alignment.
    """

    def __init__(self, group_col: str = "instrument"):
        self.group_col = group_col

    def _validate_lookback(self, lookback: int) -> None:
        """Validate lookback period."""
        if lookback <= 0:
            raise ValueError("Lookback period must be positive")

    def _validate_dataframe(
        self, df: pd.DataFrame, required_columns: List[str]
    ) -> None:
        """Validate DataFrame has required columns."""
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        if self.group_col not in df.columns:
            raise KeyError(f"Group column '{self.group_col}' not found in DataFrame")

    def _transform_grouped(self, df: pd.DataFrame, func, *args, **kwargs) -> pd.Series:
        """
        Apply transformation to grouped data using transform instead of apply.
        Returns aligned Series with same index as input.
        """
        self._validate_dataframe(df, [self.group_col])
        return df.groupby(self.group_col).transform(func, *args, **kwargs)

    def calc_ema(self, series: pd.Series, lookback: int) -> pd.Series:
        """
        Vectorized EMA calculation.

        Args:
            series: Input price series
            lookback: EMA lookback period

        Returns:
            pd.Series: EMA values
        """
        self._validate_lookback(lookback)
        return series.ewm(span=lookback, adjust=False).mean()

    def calc_returns(
        self,
        df: pd.DataFrame,
        lookforward: int = 1,
        price_col: str = "open",
        log: bool = True,
    ) -> pd.Series:
        """
        Calculate forward returns for each instrument.

        Args:
            df: Input DataFrame
            lookforward: Number of periods to look forward
            price_col: Column to use for price data
            log: If True, compute log returns

        Returns:
            pd.Series: Forward returns
        """
        self._validate_dataframe(df, [price_col])
        self._validate_lookback(lookforward)

        def _returns(group):
            if log:
                return np.log(
                    group[price_col].shift(-1 - lookforward)
                    / group[price_col].shift(-1)
                )
            return (
                group[price_col].shift(-1 - lookforward) / group[price_col].shift(-1)
                - 1
            )

        return self._transform_grouped(df, _returns)

    def calc_pos_in_range(self, df: pd.DataFrame, lookback: int) -> pd.Series:
        """
        Calculate position within price range.

        Args:
            df: Input DataFrame
            lookback: Lookback period for range calculation

        Returns:
            pd.Series: Position in range values (0 to 1)
        """
        self._validate_dataframe(df, ["high", "low", "close"])
        self._validate_lookback(lookback)

        def _pos_in_range(group):
            highest_high = group["high"].rolling(lookback).max()
            lowest_low = group["low"].rolling(lookback).min()
            return (group["close"] - lowest_low) / (highest_high - lowest_low)

        return self._transform_grouped(df, _pos_in_range)

    def calc_cmema(
        self,
        df: pd.DataFrame,
        fast_lookback: int = 25,
        slow_lookback: int = 100,
        price_col: str = "close",
        range_col: str = "range_perc_ema_5",
    ) -> pd.Series:
        """
        Calculate cross-MA momentum indicator.

        Args:
            df: Input DataFrame
            fast_lookback: Fast EMA lookback period
            slow_lookback: Slow EMA lookback period
            price_col: Column to use for price data
            range_col: Column to use for range normalization

        Returns:
            pd.Series: CMEMA values
        """
        self._validate_dataframe(df, [price_col, range_col])
        self._validate_lookback(fast_lookback)
        self._validate_lookback(slow_lookback)
        if fast_lookback >= slow_lookback:
            raise ValueError("fast_lookback must be less than slow_lookback")

        def _cmema(group):
            ema_fast = self.calc_ema(group[price_col], fast_lookback)
            ema_slow = self.calc_ema(group[price_col], slow_lookback)
            ema_diff_perc = ema_fast / ema_slow - 1
            return ema_diff_perc / group[range_col]

        return self._transform_grouped(df, _cmema)

    def calc_aroon(self, df: pd.DataFrame, lookback: int = 25) -> pd.Series:
        """
        Calculate Aroon Oscillator.

        Args:
            df: Input DataFrame
            lookback: Lookback period for Aroon calculation

        Returns:
            pd.Series: Aroon Oscillator values (-100 to 100)
        """
        self._validate_dataframe(df, ["high", "low"])
        self._validate_lookback(lookback)

        def _aroon(group):
            high_days = (
                group["high"]
                .rolling(lookback)
                .apply(lambda x: x.argmax() / (lookback - 1) * 100)
            )
            low_days = (
                group["low"]
                .rolling(lookback)
                .apply(lambda x: x.argmin() / (lookback - 1) * 100)
            )
            return high_days - low_days

        return self._transform_grouped(df, _aroon)

    def compute_features(
        self, df: pd.DataFrame, feature_configs: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Compute all features based on configuration.

        Args:
            df: Input DataFrame with OHLCV data
            feature_configs: Dictionary of feature configurations
                           If None, uses default settings

        Returns:
            pd.DataFrame: Original DataFrame with added feature columns
        """
        if feature_configs is None:
            feature_configs = {
                "returns": {"lookforward": 1},
                "pos_in_range": {"lookbacks": [3, 7, 10, 30, 100]},
                "cmema": [
                    {"fast_lookback": 3, "slow_lookback": 12},
                    {"fast_lookback": 10, "slow_lookback": 40},
                    {"fast_lookback": 25, "slow_lookback": 100},
                ],
                "aroon": {"lookbacks": [10, 30, 60]},
            }

        # Validate input DataFrame has all required columns
        required_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "range_perc_ema_5",
        ]
        self._validate_dataframe(df, required_columns)

        result = df.copy()

        # Forward returns
        result["fwd_return_1"] = self.calc_returns(df, **feature_configs["returns"])

        # Position in range features
        for lb in feature_configs["pos_in_range"]["lookbacks"]:
            result[f"pos_in_range_{lb}d"] = self.calc_pos_in_range(df, lb)

        # CMEMA features
        for config in feature_configs["cmema"]:
            name = f"cmema_{config['fast_lookback']}_{config['slow_lookback']}"
            result[name] = self.calc_cmema(df, **config)

        # Aroon features
        for lb in feature_configs["aroon"]["lookbacks"]:
            result[f"aroon_{lb}"] = self.calc_aroon(df, lb)

        return result
