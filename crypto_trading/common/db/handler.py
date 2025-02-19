import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Callable, Optional, Dict, Union
import duckdb
from duckdb import DuckDBPyConnection
import pandas as pd


class DatabaseHandler:
    """Base class for database operations"""

    DEFAULT_DB_PATH = Path(__file__).parents[3] / "data" / "crypto_data.db"

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def db_connection(self, read_only: bool = False) -> DuckDBPyConnection:
        """Context manager for handling DuckDB connections

        Args:
            read_only: Whether to open the connection in read-only mode
        """
        conn = None
        try:
            conn = duckdb.connect(str(self.db_path), read_only=read_only)
            yield conn
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def execute_transaction(
        self, operations: List[Callable[[DuckDBPyConnection], None]]
    ) -> None:
        """Execute multiple operations in a single transaction"""
        try:
            with self.db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("BEGIN TRANSACTION")
                    try:
                        for operation in operations:
                            operation(cursor)
                        cursor.execute("COMMIT")
                    except Exception as e:
                        cursor.execute("ROLLBACK")
                        self.logger.error(f"Transaction failed: {e}")
                        raise
        except Exception as e:
            self.logger.error(f"Database operation failed: {e}")
            raise

    def query_to_df(self, query: str, params: Optional[List] = None) -> pd.DataFrame:
        """Execute a query and return results as a DataFrame

        Args:
            query: SQL query string
            params: Optional list of parameters for parameterized queries

        Returns:
            DataFrame containing query results
        """
        with self.db_connection(read_only=True) as conn:
            if params:
                return conn.execute(query, params).fetch_df()
            return conn.execute(query).fetch_df()

    def get_table_schema(self, schema: str, table: str) -> pd.DataFrame:
        """Get the schema of a table

        Args:
            schema: Schema name
            table: Table name

        Returns:
            DataFrame containing column names, types, and constraints
        """
        query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
        AND table_name = '{table}'
        ORDER BY ordinal_position
        """
        return self.query_to_df(query)

    def get_table_info(self, schema: str, table: str) -> Dict[str, Union[int, str]]:
        """Get information about a table

        Args:
            schema: Schema name
            table: Table name

        Returns:
            Dictionary containing table statistics
        """
        # Get row count
        row_count = self.query_to_df(
            f"SELECT COUNT(*) as count FROM {schema}.{table}"
        ).iloc[0]["count"]

        # Get column information
        columns = self.get_table_schema(schema, table)
        column_count = len(columns)
        column_types = columns["data_type"].value_counts().to_dict()

        return {
            "row_count": row_count,
            "column_count": column_count,
            "column_types": column_types,
            "schema": schema,
            "table": table,
        }

    def list_tables(self, schema: Optional[str] = None) -> pd.DataFrame:
        """List all tables in the database or specific schema

        Args:
            schema: Optional schema name to filter tables

        Returns:
            DataFrame containing table information
        """
        query = """
        SELECT 
            table_schema as schema,
            table_name as table,
            table_type as type
        FROM information_schema.tables
        WHERE table_schema != 'information_schema'
        """
        if schema:
            query += f" AND table_schema = '{schema}'"
        query += " ORDER BY table_schema, table_name"
        return self.query_to_df(query)

    def get_latest_timestamp(
        self, schema: str, table: str, timestamp_col: str
    ) -> Optional[pd.Timestamp]:
        """Get the latest timestamp from a table

        Args:
            schema: Schema name
            table: Table name
            timestamp_col: Name of timestamp column

        Returns:
            Latest timestamp or None if table is empty
        """
        query = f"""
        SELECT MAX({timestamp_col}) as max_ts
        FROM {schema}.{table}
        """
        result = self.query_to_df(query)
        return result.iloc[0]["max_ts"] if not result.empty else None

    def get_date_range_stats(
        self, schema: str, table: str, timestamp_col: str, grouping: str = "month"
    ) -> pd.DataFrame:
        """Get statistics about data coverage across a date range

        Args:
            schema: Schema name
            table: Table name
            timestamp_col: Name of timestamp column
            grouping: Time grouping ('day', 'month', 'year')

        Returns:
            DataFrame with count of records per time period
        """
        date_trunc = {"day": "day", "month": "month", "year": "year"}.get(
            grouping.lower(), "month"
        )

        query = f"""
        SELECT 
            DATE_TRUNC('{date_trunc}', {timestamp_col}) as period,
            COUNT(*) as record_count
        FROM {schema}.{table}
        GROUP BY 1
        ORDER BY 1
        """
        return self.query_to_df(query)
