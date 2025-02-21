"""Signal generation for the trading strategy."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from dotenv import dotenv_values

from crypto_trading.analysis.features.feature_generator import (
    generate_features,
    calc_range_percentage,
    calc_ema,
    calc_pos_in_range,
    calc_cmema,
    calc_aroon,
    get_bar_triangle,
    get_highest_high,
    get_lowest_low,
    shift_values,
)
from crypto_trading.analysis.features.normalize import normalize_0_1, min_max_norm
from crypto_trading.common.db.handler import DatabaseHandler
from crypto_trading.trading.strategy.config import TradingStrategyConfig

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates trading signals based on technical analysis."""

    def __init__(self, db_handler: DatabaseHandler, config: TradingStrategyConfig):
        """
        Initialize the signal generator.

        Args:
            db_handler: Database handler for data access
            config: Trading strategy configuration
        """
        self.db = db_handler
        self.config = config
        self.features_config = config.get_features_config()
        self.signals_config = config.get_signals_config()
        self.highest_volume_exchanges = None

    def fetch_highest_volume_exchanges(self) -> pd.DataFrame:
        """
        Fetch the exchange with highest volume for each instrument.

        Returns:
            DataFrame with instrument and highest volume exchange
        """
        query = """
        WITH LatestDatePerInstrument AS (
            SELECT
                ohlcv.instrument,
                MAX(ohlcv.datetime) AS latest_datetime
            FROM
                amberdata.ohlcv_perps_1d ohlcv
            JOIN
                amberdata.ohlcv_info_futures info
            ON
                ohlcv.exchange = info.exchange
                AND ohlcv.instrument = info.instrument
            JOIN
                amberdata.exchange_reference ref
            ON
                ohlcv.exchange = ref.exchange
                AND ohlcv.instrument = ref.instrument
            WHERE
                info.active = true
                AND ref.exchange_enabled = true
                AND ref.quote_symbol = 'USDT'
            GROUP BY
                ohlcv.instrument
        ),
        VolumeRanked AS (
            SELECT
                ohlcv.instrument,
                ohlcv.exchange,
                ohlcv.volume,
                ROW_NUMBER() OVER (PARTITION BY ohlcv.instrument ORDER BY ohlcv.volume DESC) as volume_rank
            FROM
                amberdata.ohlcv_perps_1d ohlcv
            JOIN
                LatestDatePerInstrument ldi
            ON
                ohlcv.instrument = ldi.instrument
            AND ohlcv.datetime = ldi.latest_datetime
            JOIN
                amberdata.ohlcv_info_futures info
            ON
                ohlcv.exchange = info.exchange
            AND ohlcv.instrument = info.instrument
            JOIN
                amberdata.exchange_reference ref
            ON
                ohlcv.exchange = ref.exchange
            AND ohlcv.instrument = ref.instrument
            WHERE
                info.active = true
                AND ref.exchange_enabled = true
        )
        SELECT
            instrument,
            exchange,
            volume
        FROM
            VolumeRanked
        WHERE
            volume_rank = 1
        ORDER BY
            volume DESC;
        """
        try:
            df = self.db.query_to_df(query)
            self.highest_volume_exchanges = dict(zip(df["instrument"], df["exchange"]))
            return df
        except Exception as e:
            logger.error(f"Error fetching highest volume exchanges: {str(e)}")
            raise

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch required market data from the database.

        Returns:
            DataFrame containing market data
        """
        lookback_days = self.config.get_database_config()["query_lookback_days"]
        query = f"""
            SELECT 
                instrument,
                datetime,
                SUM(open * volume) / SUM(volume) AS open,
                SUM(high * volume) / SUM(volume) AS high,
                SUM(low * volume) / SUM(volume) AS low,
                SUM(close * volume) / SUM(volume) AS close,
                SUM(volume) as volume
            FROM amberdata.ohlcv_perps_1d
            WHERE 
                CAST(datetime AS TIMESTAMP) >= NOW() - INTERVAL {lookback_days} DAY
                AND instrument IN (
                    SELECT instrument
                    FROM amberdata.exchange_reference
                    WHERE
                        exchange_enabled = TRUE
                        AND contract_period = 'perpetual'
                        AND quote_symbol = 'USDT'
                )
            GROUP BY instrument, datetime
            ORDER BY instrument ASC, datetime ASC
        """

        try:
            df = self.db.query_to_df(query)
            df["volume_usd"] = df["volume"] * df[["open", "high", "low", "close"]].mean(
                axis=1
            )
            return df
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            raise

    def _build_feature_definitions(self) -> List[Dict]:
        """
        Build feature definitions based on configuration.

        Returns:
            List of feature definitions
        """
        features = []

        # Range percentage features
        features.extend(
            [
                {
                    "name": "range_perc",
                    "func": calc_range_percentage,
                    "params": {},
                    "group_by_instrument": True,
                },
                {
                    "name": "range_perc_ema_5",
                    "func": calc_ema,
                    "params": {
                        "col": "range_perc",
                        "lookback": 5,
                    },
                    "group_by_instrument": True,
                },
                {
                    "name": "range_perc_ema_5_prev",
                    "func": shift_values,
                    "params": {"col": "range_perc_ema_5", "shift": 1},
                    "group_by_instrument": True,
                },
            ]
        )

        # Position in range features
        for lookback in self.features_config["indicators"]["position_in_range"][
            "periods"
        ]:
            features.append(
                {
                    "name": f"pos_in_range_{lookback}",
                    "func": calc_pos_in_range,
                    "params": {"lookback": lookback},
                    "group_by_instrument": True,
                }
            )

        # EMA cross features
        for fast, slow in self.features_config["indicators"]["ema_cross"]["pairs"]:
            features.append(
                {
                    "name": f"cmema_{fast}_{slow}",
                    "func": calc_cmema,
                    "params": {
                        "fast_lookback": fast,
                        "slow_lookback": slow,
                        "normalize_by": "range_perc_ema_5",
                    },
                    "group_by_instrument": True,
                }
            )

        # Aroon features
        for lookback in self.features_config["indicators"]["aroon"]["periods"]:
            features.append(
                {
                    "name": f"aroon_{lookback}",
                    "func": calc_aroon,
                    "params": {"lookback": lookback},
                    "group_by_instrument": True,
                }
            )

        # Bar pattern features
        features.append(
            {
                "name": "three_bar_triangle",
                "func": get_bar_triangle,
                "params": {"lookback": 3},
                "group_by_instrument": True,
            }
        )

        # High/Low features
        features.extend(
            [
                {
                    "name": "high_3d",
                    "func": get_highest_high,
                    "params": {"lookback": 3},
                    "group_by_instrument": True,
                },
                {
                    "name": "low_3d",
                    "func": get_lowest_low,
                    "params": {"lookback": 3},
                    "group_by_instrument": True,
                },
            ]
        )

        return features

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical features for analysis.

        Args:
            data: Market data DataFrame

        Returns:
            DataFrame with generated features
        """
        try:
            features = self._build_feature_definitions()
            df = generate_features(data, features, "instrument")

            # Calculate range percentage features
            df["range_perc_rel"] = np.log(
                df["range_perc"] / df["range_perc_ema_5_prev"]
            )

            # Calculate range prediction features
            df["range_perc_ema_5_log"] = np.log(df["range_perc_ema_5"])
            df["range_perc_log_pred"] = (
                df["range_perc_ema_5_log"] * 0.82113026 - 0.5240189478074768
            )
            df["range_perc_pred"] = np.exp(df["range_perc_log_pred"])

            return df
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}")
            raise

    def _calculate_market_biases(self, df_recent: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market biases from recent data.

        Args:
            df_recent: Recent market data

        Returns:
            DataFrame with market bias calculations
        """
        # Calculate trend metrics
        trend_features = [
            "pos_in_range_3",
            "pos_in_range_7",
            "pos_in_range_30",
            "aroon_10",
            "aroon_30",
            "cmema_3_12",
        ]

        # Normalize features to 0-1 range
        df_normalized = df_recent.copy()

        # Normalize Aroon indicators
        df_normalized = normalize_0_1(
            df_normalized,
            cols=["aroon_10", "aroon_30"],
            min_value=-100,
            max_value=100,
        )

        # Normalize CMEMA
        df_normalized = normalize_0_1(
            df_normalized, cols=["cmema_3_12"], min_value=-2, max_value=2
        )

        # Calculate mean trend score
        trend_score = df_normalized[trend_features].mean(axis=1)
        avg_trend = trend_score.mean()

        # Calculate range metrics
        range_score = 1 - min_max_norm(
            df_recent["range_perc_rel"], min_value=-1.5, max_value=1.5
        )
        avg_range = range_score.mean()

        # Compile biases
        biases = pd.Series(
            {
                "Bull": avg_trend,
                "Bear": 1 - avg_trend,
                "Trend": avg_range,
                "Chop": 1 - avg_range,
            }
        )

        return biases.to_frame()

    def save_tradingview_watchlists(
        self, signals: Dict[str, pd.DataFrame], output_path: Path
    ) -> None:
        """
        Save TradingView watchlist files.

        Args:
            signals: Dictionary of signal DataFrames
            output_path: Path to save watchlist files
        """
        try:
            if self.highest_volume_exchanges is None:
                self.fetch_highest_volume_exchanges()

            # Create raw directory if it doesn't exist
            raw_path = output_path
            raw_path.mkdir(parents=True, exist_ok=True)

            # Generate watchlist files
            for name, df in signals.items():
                if not df.empty:
                    symbols = [
                        f"{self.highest_volume_exchanges.get(instrument, 'binance').upper()}:{instrument}.P"
                        for instrument in df["instrument"]
                    ]
                    watchlist = ",".join(symbols)

                    # Save to file
                    file_path = raw_path / f"{name.title()}.txt"
                    with open(file_path, "w") as f:
                        f.write(watchlist)
                    logger.info(f"Saved {name} watchlist to {file_path}")

            # Generate combined watchlist for outperformers/underperformers
            sections = {
                "Long-Term Outperformers": signals.get("top_lt_trend", pd.DataFrame()),
                "Long-Term Underperformers": signals.get(
                    "bottom_lt_trend", pd.DataFrame()
                ),
                "Short-Term Outperformers": signals.get("top_st_trend", pd.DataFrame()),
                "Short-Term Underperformers": signals.get(
                    "bottom_st_trend", pd.DataFrame()
                ),
            }

            combined_path = raw_path / "Combined Out&Under-Performers.txt"
            with open(combined_path, "w") as f:
                for section, df in sections.items():
                    if not df.empty:
                        symbols = [
                            f"{self.highest_volume_exchanges.get(instrument, 'binance').upper()}:{instrument}.P"
                            for instrument in df["instrument"]
                        ]
                        f.write(f"### {section}\n")
                        f.write("\n".join(symbols) + "\n\n")
            logger.info(f"Saved combined watchlist to {combined_path}")

        except Exception as e:
            logger.error(f"Error saving TradingView watchlists: {str(e)}")
            raise

    def generate_signals(
        self, features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generate trading signals from features.

        Args:
            features: Feature DataFrame

        Returns:
            Tuple of (market biases, signal DataFrames)
        """
        try:
            # Ensure we have highest volume exchanges
            if self.highest_volume_exchanges is None:
                self.fetch_highest_volume_exchanges()

            signals = {}

            # Filter for latest data
            latest_date = features["datetime"].max()
            df_recent = features[features["datetime"] == latest_date].copy()

            # Add exchange column
            df_recent["exchange"] = df_recent["instrument"].apply(
                lambda x: self.highest_volume_exchanges.get(x)
            )

            # Get output configuration
            output_config = self.config.get_output_config()
            local_enabled = output_config["local"]["enabled"]
            s3_enabled = output_config["s3"]["enabled"]
            output_path = Path(output_config["local"]["path"])

            # Save df_recent to local file and upload to S3
            if local_enabled:
                file_path = output_path / "bnf_recent.csv"
                df_recent.to_csv(file_path, index=False)
                logger.info(f"Saved df_recent to {file_path}")

                if s3_enabled:
                    s3_config = output_config["s3"]
                    env_path = Path(".env")
                    config_env = dotenv_values(env_path)
                    aws_access_key_id = config_env.get("AWS_ACCESS_KEY_ID")
                    aws_secret_access_key = config_env.get("AWS_SECRET_ACCESS_KEY")
                    s3_client = boto3.client(
                        "s3",
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                    )
                    csv_buffer = df_recent.to_csv(index=False).encode()
                    s3_client.put_object(
                        Bucket=s3_config["bucket"],
                        Key="bnf_recent.csv",
                        Body=csv_buffer,
                        ACL=s3_config["acl"],
                        ContentType=s3_config["content_type"],
                        ContentDisposition=s3_config["content_disposition"],
                    )
                    logger.info(
                        f"Uploaded df_recent to S3 bucket {s3_config['bucket']}"
                    )

            # Generate breakout signals
            breakout_config = self.signals_config["breakout"]
            signals["breakouts"] = df_recent[
                (df_recent["three_bar_triangle"] == True)
                & (df_recent["pos_in_range_3"] >= breakout_config["pos_in_range_min"])
                & (df_recent["range_perc_rel"] < breakout_config["range_perc_rel_max"])
                & (df_recent["aroon_10"] >= breakout_config["aroon_min"])
                & (df_recent["aroon_30"] >= breakout_config["aroon_min"])
            ]

            # Generate breakdown signals
            breakdown_config = self.signals_config["breakdown"]
            df_recent["low_distance"] = (
                -(df_recent["low_3d"] / df_recent["close"] - 1)
                / df_recent["range_perc_ema_5"]
            )
            signals["breakdowns"] = df_recent[
                (df_recent["three_bar_triangle"] == True)
                & (df_recent["pos_in_range_3"] <= breakdown_config["pos_in_range_max"])
                & (df_recent["range_perc_rel"] < breakdown_config["range_perc_rel_max"])
                & (df_recent["low_distance"] < breakdown_config["low_distance_max"])
                & (df_recent["aroon_10"] <= breakdown_config["aroon_10_max"])
                & (df_recent["aroon_30"] <= breakdown_config["aroon_30_max"])
            ]

            # Generate rip fade signals
            rip_fade_config = self.signals_config["rip_fade"]
            signals["rip_fades"] = df_recent[
                (df_recent["range_perc_rel"] >= rip_fade_config["range_perc_rel_min"])
                & (
                    df_recent["aroon_10"]
                    <= rip_fade_config["aroon_conditions"]["aroon_10_max"]
                )
                & (
                    df_recent["aroon_30"]
                    <= rip_fade_config["aroon_conditions"]["aroon_30_max"]
                )
            ]

            # Generate dip buy signals
            dip_buy_config = self.signals_config["dip_buy"]
            signals["dip_buys"] = df_recent[
                (df_recent["range_perc_rel"] >= dip_buy_config["range_perc_rel_min"])
                & (
                    df_recent["aroon_10"]
                    >= dip_buy_config["aroon_conditions"]["aroon_10_min"]
                )
                & (
                    df_recent["aroon_30"]
                    >= dip_buy_config["aroon_conditions"]["aroon_30_min"]
                )
            ]

            # Generate trend ranking signals
            trend_features = [
                "range_perc_pred",
                "pos_in_range_3",
                "pos_in_range_7",
                "pos_in_range_30",
                "cmema_3_12",
                "aroon_10",
                "aroon_30",
            ]

            # Short-term trend ranking
            st_trend_rank = (
                df_recent[["instrument"] + trend_features]
                .assign(
                    average_score=lambda x: x[trend_features]
                    .rank(ascending=False)
                    .mean(axis=1)
                )
                .sort_values("average_score", ascending=True)
                .dropna()
            )

            signals["top_st_trend"] = st_trend_rank.head(20)
            signals["bottom_st_trend"] = st_trend_rank.tail(20)

            # Long-term trend signals
            signals["top_lt_trend"] = (
                df_recent.dropna(subset=["pos_in_range_100"])
                .query("pos_in_range_100 >= 0.75")[
                    ["instrument", "range_perc_pred", "pos_in_range_100", "exchange"]
                ]
                .head(20)
            )

            signals["bottom_lt_trend"] = (
                df_recent.dropna(subset=["pos_in_range_100"])
                .query("pos_in_range_100 <= 0.25")[
                    ["instrument", "range_perc_pred", "pos_in_range_100", "exchange"]
                ]
                .head(20)
            )

            # Calculate market biases
            biases = self._calculate_market_biases(df_recent)

            # Save TradingView watchlists
            if local_enabled:
                self.save_tradingview_watchlists(signals, output_path)

            return biases, signals

        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            pass
