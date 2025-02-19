"""Configuration management for the trading strategy."""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class TradingStrategyConfig:
    """Manages configuration loading and validation for the trading strategy."""

    def __init__(self, config_path: str = "references/trading_strategy_config.yaml"):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_paths()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Dictionary containing configuration settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {str(e)}")
            raise

    def _validate_config(self) -> None:
        """
        Validate configuration settings.

        Raises:
            ValueError: If required settings are missing or invalid
        """
        required_sections = ["database", "trading", "features", "signals", "output"]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate specific settings
        if not self.config["trading"]["symbols"]:
            raise ValueError("No trading symbols configured")

        if not self.config["features"]["lookback_periods"]:
            raise ValueError("No lookback periods configured")

        # Validate database settings
        if "connection_string" not in self.config["database"]:
            raise ValueError("Database configuration must include a connection_string")

        # Validate output paths
        if self.config["output"]["local"]["enabled"]:
            if not self.config["output"]["local"]["path"]:
                raise ValueError("Local output path not configured")

        if self.config["output"]["s3"]["enabled"]:
            if not self.config["output"]["s3"]["bucket"]:
                raise ValueError("S3 bucket not configured")

    def _setup_paths(self) -> None:
        """Setup and create necessary directories."""
        if self.config["output"]["local"]["enabled"]:
            output_path = Path(self.config["output"]["local"]["path"])
            output_path.mkdir(parents=True, exist_ok=True)

        # Setup logging directory
        log_file = self.config.get("logging", {}).get("file")
        if log_file:
            log_path = Path(log_file).parent
            log_path.mkdir(parents=True, exist_ok=True)

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration settings."""
        database_config = self.config["database"]
        return database_config

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration settings."""
        return self.config["trading"]

    def get_features_config(self) -> Dict[str, Any]:
        """Get feature generation configuration settings."""
        return self.config["features"]

    def get_signals_config(self) -> Dict[str, Any]:
        """Get signal generation configuration settings."""
        return self.config["signals"]

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration settings."""
        return self.config["output"]

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration settings."""
        return self.config.get(
            "logging",
            {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        )
