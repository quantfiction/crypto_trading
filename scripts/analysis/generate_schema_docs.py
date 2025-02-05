#!/usr/bin/env python3
"""
Script to generate comprehensive YAML schema documentation for the Amberdata database.
Includes schema version tracking, relationship validation, and integrity checks.
"""

import logging
import time
import hashlib
from pathlib import Path
import yaml
from datetime import datetime
import pytz
from typing import Dict, Any, List, Set, Tuple
import duckdb
from crypto_trading.db.handler import DatabaseHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when schema validation fails"""

    pass


class SchemaDocumentationGenerator:
    """Generates YAML documentation for database schema with validation"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.db_handler = DatabaseHandler()
        self.doc_path = (
            Path(__file__).parent.parent / "references" / "amberdata_schema.yaml"
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.known_relationships = {
            ("amberdata.exchange_reference", "amberdata.ohlcv_info_futures"): [
                ("exchange", "exchange"),
                ("instrument", "instrument"),
            ],
            ("amberdata.ohlcv_info_futures", "amberdata.ohlcv_perps_1d"): [
                ("exchange", "exchange"),
                ("instrument", "instrument"),
            ],
        }

    def _get_table_description(self, table: str) -> str:
        """Get description for a table"""
        descriptions = {
            "ohlcv_perps_1d": "Daily OHLCV (Open, High, Low, Close, Volume) data for perpetual futures contracts",
            "ohlcv_info_futures": "Metadata and trading information for futures contracts, including trading dates and status",
            "exchange_reference": "Reference data for exchanges and their supported instruments, including trading limits and contract specifications",
            "spot_ohlc_1h": "Hourly OHLCV data for spot trading pairs",
        }
        return descriptions.get(table, f"Table containing {table} data")

    def _get_column_description(self, table: str, column: str) -> str:
        """Get description for a column"""
        descriptions = {
            # Common fields
            "instrument": "Trading instrument identifier (e.g., BTCUSDT)",
            "exchange": "Exchange where the instrument is traded (e.g., binance, bybit)",
            "datetime": "Timestamp for the data point",
            # OHLCV fields
            "open": "Opening price for the period",
            "high": "Highest price during the period",
            "low": "Lowest price during the period",
            "close": "Closing price for the period",
            "volume": "Trading volume during the period",
            # Contract info fields
            "trading_start_date": "Date when trading begins for the contract",
            "trading_end_date": "Date when trading ends for the contract",
            "active": "Whether the contract is currently active for trading",
            "updated_at": "Last time this record was updated",
            # Exchange reference fields
            "base_symbol": "Base currency of the trading pair (e.g., BTC in BTC/USDT)",
            "quote_symbol": "Quote currency of the trading pair (e.g., USDT in BTC/USDT)",
            "market": "Market type (e.g., spot, futures)",
            "exchange_enabled": "Whether trading is currently enabled on the exchange",
            "contract_period": "Contract period type (e.g., perpetual, quarterly)",
            "contract_size": "Size of one contract in base currency units",
            "contract_settle_type": "Settlement type (e.g., linear, inverse)",
            "contract_settle_symbol": "Currency used for contract settlement",
            "contract_underlying": "Underlying asset for derivative contracts",
            "contract_expiration_timestamp": "When the contract expires (null for perpetual contracts)",
            "listing_timestamp": "When the instrument was first listed on the exchange",
            # Limit fields
            "limits_price_min": "Minimum allowed price for orders",
            "limits_price_max": "Maximum allowed price for orders",
            "limits_volume_min": "Minimum order volume",
            "limits_volume_max": "Maximum order volume",
            "limits_market_min": "Minimum market order value",
            "limits_market_max": "Maximum market order value",
            "limits_leverage_min": "Minimum allowed leverage",
            "limits_leverage_max": "Maximum allowed leverage",
            "limits_leverage_super_max": "Maximum super leverage allowed (if applicable)",
            "limits_cost_min": "Minimum order cost in quote currency",
            "limits_cost_max": "Maximum order cost in quote currency",
            # Precision fields
            "precision_price": "Decimal precision for price values",
            "precision_volume": "Decimal precision for volume values",
            "precision_base": "Decimal precision for base currency",
            "precision_quote": "Decimal precision for quote currency",
            # CamelCase variants for spot_ohlc_1h
            "baseSymbol": "Base currency of the trading pair (e.g., BTC in BTC/USDT)",
            "quoteSymbol": "Quote currency of the trading pair (e.g., USDT in BTC/USDT)",
            "exchangeEnabled": "Whether trading is currently enabled on the exchange",
            "limitsPriceMin": "Minimum allowed price for orders",
            "limitsPriceMax": "Maximum allowed price for orders",
            "limitsVolumeMin": "Minimum order volume",
            "limitsVolumeMax": "Maximum order volume",
            "limitsMarketMin": "Minimum market order value",
            "limitsMarketMax": "Maximum market order value",
            "limitsLeverageMin": "Minimum allowed leverage",
            "limitsLeverageMax": "Maximum allowed leverage",
            "limitsLeverageSuperMax": "Maximum super leverage allowed (if applicable)",
            "limitsCostMin": "Minimum order cost in quote currency",
            "limitsCostMax": "Maximum order cost in quote currency",
            "precisionPrice": "Decimal precision for price values",
            "precisionVolume": "Decimal precision for volume values",
            "precisionBase": "Decimal precision for base currency",
            "precisionQuote": "Decimal precision for quote currency",
            "listingTimestamp": "When the instrument was first listed on the exchange",
            "contractUnderlying": "Underlying asset for derivative contracts",
            "contractExpirationTimestamp": "When the contract expires (null for perpetual contracts)",
            "contractPeriod": "Contract period type (e.g., perpetual, quarterly)",
            "contractSize": "Size of one contract in base currency units",
            "contractSettleType": "Settlement type (e.g., linear, inverse)",
            "contractSettleSymbol": "Currency used for contract settlement",
        }
        return descriptions.get(column, f"Column containing {column} data")

    def _execute_with_retry(self, operation):
        """Execute database operation with retry logic"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return operation()
            except duckdb.IOException as e:
                if "lock" in str(e).lower():
                    last_error = e
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"Database locked, retrying in {self.retry_delay} seconds (attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(self.retry_delay)
                        continue
                raise
            except Exception as e:
                last_error = e
                break

        raise Exception(
            f"Operation failed after {self.max_retries} attempts. Last error: {last_error}"
        )

    def _calculate_schema_hash(self) -> str:
        """Calculate a hash of the current schema structure for version tracking"""
        schema_def = []
        tables_df = self.db_handler.list_tables()

        for _, row in tables_df.iterrows():
            schema, table = row["schema"], row["table"]
            table_schema = self.db_handler.get_table_schema(schema, table)
            schema_def.append(f"{schema}.{table}:")
            for _, col in table_schema.iterrows():
                schema_def.append(f"  {col['column_name']}: {col['data_type']}")

        schema_text = "\n".join(sorted(schema_def))
        return hashlib.sha256(schema_text.encode()).hexdigest()[:8]

    def _validate_relationships(self) -> List[str]:
        """Validate all known relationships and return any issues found"""
        issues = []

        for (table1, table2), columns in self.known_relationships.items():
            schema1, name1 = table1.split(".")
            schema2, name2 = table2.split(".")

            # Check if tables exist
            tables_df = self.db_handler.list_tables()
            if not (
                (tables_df["schema"] == schema1) & (tables_df["table"] == name1)
            ).any():
                issues.append(f"Table {table1} not found")
                continue
            if not (
                (tables_df["schema"] == schema2) & (tables_df["table"] == name2)
            ).any():
                issues.append(f"Table {table2} not found")
                continue

            # Check if columns exist
            schema1_df = self.db_handler.get_table_schema(schema1, name1)
            schema2_df = self.db_handler.get_table_schema(schema2, name2)

            for col1, col2 in columns:
                if col1 not in schema1_df["column_name"].values:
                    issues.append(f"Column {col1} not found in {table1}")
                if col2 not in schema2_df["column_name"].values:
                    issues.append(f"Column {col2} not found in {table2}")

        return issues

    def _detect_potential_relationships(self) -> List[Dict[str, Any]]:
        """Detect potential relationships between tables based on column names"""
        relationships = []
        tables_df = self.db_handler.list_tables()

        # Get all tables and their columns
        table_columns: Dict[str, Set[str]] = {}
        for _, row in tables_df.iterrows():
            schema, table = row["schema"], row["table"]
            schema_df = self.db_handler.get_table_schema(schema, table)
            table_columns[f"{schema}.{table}"] = set(schema_df["column_name"])

        # Look for matching column names between tables
        for table1 in table_columns:
            for table2 in table_columns:
                if table1 >= table2:  # Skip self-joins and duplicates
                    continue

                common_cols = table_columns[table1].intersection(table_columns[table2])
                if common_cols:
                    relationships.append(
                        {
                            "tables": [table1, table2],
                            "potential_join_columns": sorted(common_cols),
                            "confidence": "suggested",  # Mark as suggested vs known
                        }
                    )

        return relationships

    def _check_data_integrity(self, schema: str, table: str) -> List[str]:
        """Check for potential data integrity issues"""
        issues = []

        try:
            # Check for null values in important columns
            schema_df = self.db_handler.get_table_schema(schema, table)
            for _, col in schema_df.iterrows():
                if not col["is_nullable"] and col["column_default"] is None:
                    null_check_query = f"""
                    SELECT COUNT(*) as null_count
                    FROM {schema}.{table}
                    WHERE {col['column_name']} IS NULL
                    """
                    result = self.db_handler.query_to_df(null_check_query)
                    if result.iloc[0]["null_count"] > 0:
                        issues.append(
                            f"Found NULL values in non-nullable column {col['column_name']}"
                        )

            # Check for orphaned records in relationships
            for (table1, table2), columns in self.known_relationships.items():
                if f"{schema}.{table}" == table1:
                    schema2, name2 = table2.split(".")
                    for col1, col2 in columns:
                        orphan_check_query = f"""
                        SELECT COUNT(*) as orphan_count
                        FROM {table1} t1
                        LEFT JOIN {table2} t2
                            ON t1.{col1} = t2.{col2}
                        WHERE t2.{col2} IS NULL
                        """
                        result = self.db_handler.query_to_df(orphan_check_query)
                        if result.iloc[0]["orphan_count"] > 0:
                            issues.append(
                                f"Found orphaned records in {table1} referencing {table2}"
                            )

        except Exception as e:
            issues.append(f"Error checking data integrity: {str(e)}")

        return issues

    def get_table_details(self, schema: str, table: str) -> Dict[str, Any]:
        """Get detailed information about a table including columns and constraints"""

        def _get_details():
            schema_df = self.db_handler.get_table_schema(schema, table)
            info = self.db_handler.get_table_info(schema, table)

            columns = {}
            for _, row in schema_df.iterrows():
                columns[row["column_name"]] = {
                    "type": row["data_type"],
                    "nullable": row["is_nullable"] == "YES",
                    "default": row["column_default"],
                    "description": self._get_column_description(
                        table, row["column_name"]
                    ),
                }

            # Get constraints and relationships
            constraints = self._get_table_constraints(schema, table)
            relationships = self._get_table_relationships(schema, table)
            integrity_issues = self._check_data_integrity(schema, table)

            # Get sample data if table is not empty
            sample_data = []
            if info["row_count"] > 0:
                sample_query = f"""
                SELECT *
                FROM {schema}.{table}
                LIMIT 3
                """
                df = self.db_handler.query_to_df(sample_query)
                # Convert timestamps to ISO format strings
                for col in df.select_dtypes(include=["datetime64"]).columns:
                    df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
                sample_data = df.to_dict("records")

            return {
                "description": self._get_table_description(table),
                "columns": columns,
                "row_count": int(info["row_count"]),
                "constraints": constraints,
                "relationships": relationships,
                "integrity_issues": integrity_issues,
                "sample_data": sample_data,
            }

        return self._execute_with_retry(_get_details)

    def _get_table_constraints(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Get table constraints using DuckDB's system tables"""

        def _get_constraints():
            constraints = []

            # Get table creation SQL
            create_sql_query = f"""
            SELECT sql 
            FROM duckdb_tables() 
            WHERE database_name = 'main' 
            AND schema_name = '{schema}' 
            AND table_name = '{table}'
            """

            create_sql_df = self.db_handler.query_to_df(create_sql_query)
            if not create_sql_df.empty:
                sql = create_sql_df.iloc[0]["sql"].lower()

                # Parse UNIQUE constraints
                if "unique" in sql:
                    for line in sql.split("\n"):
                        if "unique" in line and "(" in line:
                            cols = line.split("(")[1].split(")")[0].split(",")
                            cols = [c.strip() for c in cols]
                            constraints.append(
                                {
                                    "type": "UNIQUE",
                                    "columns": cols,
                                    "name": f'unique_{table}_{"_".join(cols)}',
                                }
                            )

                # Parse PRIMARY KEY constraints
                if "primary key" in sql:
                    for line in sql.split("\n"):
                        if "primary key" in line and "(" in line:
                            cols = line.split("(")[1].split(")")[0].split(",")
                            cols = [c.strip() for c in cols]
                            constraints.append(
                                {
                                    "type": "PRIMARY KEY",
                                    "columns": cols,
                                    "name": f"pk_{table}",
                                }
                            )

            return constraints

        return self._execute_with_retry(_get_constraints)

    def _get_table_relationships(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Get relationships for a specific table"""
        relationships = []
        full_table = f"{schema}.{table}"

        # Add known relationships
        for (table1, table2), columns in self.known_relationships.items():
            if table1 == full_table:
                relationships.append(
                    {"referenced_table": table2, "columns": columns, "type": "known"}
                )
            elif table2 == full_table:
                relationships.append(
                    {
                        "referenced_table": table1,
                        "columns": [(col2, col1) for col1, col2 in columns],
                        "type": "known",
                    }
                )

        return relationships

    def generate_documentation(self) -> Dict[str, Any]:
        """Generate complete schema documentation"""

        def _generate():
            # Calculate schema version
            schema_version = self._calculate_schema_hash()

            # Validate relationships
            relationship_issues = self._validate_relationships()
            if relationship_issues:
                logger.warning("Relationship validation issues found:")
                for issue in relationship_issues:
                    logger.warning(f"  - {issue}")

            # Detect potential new relationships
            potential_relationships = self._detect_potential_relationships()

            doc = {
                "database": {
                    "name": "crypto_data.db",
                    "description": "DuckDB database containing cryptocurrency market data from Amberdata API",
                    "generated_at": datetime.now(pytz.UTC).isoformat(),
                    "schema_version": schema_version,
                    "schemas": {},
                }
            }

            # Document all tables
            tables_df = self.db_handler.list_tables()
            for schema in tables_df["schema"].unique():
                schema_tables = tables_df[tables_df["schema"] == schema]

                doc["database"]["schemas"][schema] = {
                    "description": f"Schema containing {schema}-specific tables",
                    "tables": {},
                }

                for _, row in schema_tables.iterrows():
                    table_name = row["table"]
                    doc["database"]["schemas"][schema]["tables"][table_name] = (
                        self.get_table_details(schema, table_name)
                    )

            # Add relationships section
            doc["relationships"] = {
                "known": [
                    {
                        "name": "active_perpetuals",
                        "description": "Relationship between exchange reference and futures info for active perpetual contracts",
                        "tables": ["exchange_reference", "ohlcv_info_futures"],
                        "type": "JOIN",
                        "conditions": [
                            "exchange_reference.exchange = ohlcv_info_futures.exchange",
                            "exchange_reference.instrument = ohlcv_info_futures.instrument",
                        ],
                        "filters": [
                            "exchange_reference.contract_period = 'perpetual'",
                            "ohlcv_info_futures.active = true",
                        ],
                    }
                ],
                "suggested": potential_relationships,
                "validation_issues": relationship_issues,
            }

            # Add sample queries section
            doc["sample_queries"] = {
                "get_latest_prices": {
                    "description": "Get the most recent prices for all active perpetual contracts",
                    "sql": """
                        SELECT 
                            o.exchange,
                            o.instrument,
                            o.datetime,
                            o.close as price,
                            o.volume
                        FROM amberdata.ohlcv_perps_1d o
                        JOIN amberdata.ohlcv_info_futures i
                            ON o.exchange = i.exchange
                            AND o.instrument = i.instrument
                        WHERE i.active = true
                        AND o.datetime = (
                            SELECT MAX(datetime)
                            FROM amberdata.ohlcv_perps_1d
                        )
                        ORDER BY o.exchange, o.instrument
                    """,
                },
                "get_trading_pairs": {
                    "description": "Get all trading pairs with their specifications",
                    "sql": """
                        SELECT 
                            exchange,
                            instrument,
                            base_symbol,
                            quote_symbol,
                            contract_period,
                            contract_settle_type,
                            limits_leverage_max,
                            precision_price,
                            precision_volume
                        FROM amberdata.exchange_reference
                        WHERE exchange_enabled = true
                        ORDER BY exchange, instrument
                    """,
                },
            }

            # Add maintenance guidelines
            doc["maintenance_guidelines"] = {
                "schema_changes": [
                    "Run this documentation generator after any schema changes",
                    "Compare schema_version to detect structural changes",
                    "Review validation_issues and integrity_issues sections",
                    "Update known_relationships when adding new foreign keys",
                    "Check suggested relationships for potential missing constraints",
                ],
                "data_integrity": [
                    "Monitor integrity_issues section for each table",
                    "Address any NULL values in non-nullable columns",
                    "Fix any orphaned records in relationships",
                    "Maintain referential integrity when deleting records",
                ],
                "best_practices": [
                    "Add new tables to appropriate schemas based on data source",
                    "Document all columns with clear descriptions",
                    "Include appropriate constraints and indexes",
                    "Update sample queries when adding new use cases",
                    "Keep relationship documentation in sync with actual foreign keys",
                ],
            }

            return doc

        return self._execute_with_retry(_generate)

    def save_documentation(self, doc: Dict[str, Any]) -> None:
        """Save documentation to YAML file with proper formatting"""

        class CustomDumper(yaml.SafeDumper):
            pass

        def str_presenter(dumper, data):
            if len(data.splitlines()) > 1:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        CustomDumper.add_representer(str, str_presenter)

        with open(self.doc_path, "w") as f:
            yaml.dump(
                doc, f, sort_keys=False, Dumper=CustomDumper, default_flow_style=False
            )
        logger.info(f"Schema documentation saved to {self.doc_path}")


def main():
    try:
        generator = SchemaDocumentationGenerator()
        doc = generator.generate_documentation()
        generator.save_documentation(doc)
        logger.info("Schema documentation generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate schema documentation: {e}")
        raise


if __name__ == "__main__":
    main()
