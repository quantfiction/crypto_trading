"""Database handler using SQLAlchemy for multiple database types."""

import logging
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from typing import Dict, Any, List, Union
import os

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """
    Database handler using SQLAlchemy for multiple database types.
    Supports DuckDB, PostgreSQL, and Singlestore (MySQL).
    """

    def __init__(self, config: dict):
        """
        Initialize the database handler with a SQLAlchemy engine.

        Args:
            config: A dictionary containing the database connection parameters.
                    Must include a 'db_type' key and database-specific parameters.
                    Example:
                    {
                        "db_type": "duckdb",
                        "path": "/path/to/crypto_data.db"
                    }
                    {
                        "db_type": "postgresql",
                        "host": "host",
                        "user": "user",
                        "password": "password",
                        "database": "database"
                    }
                    {
                        "db_type": "singlestore",
                        "host": "host",
                        "user": "user",
                        "password": "password",
                        "database": "database"
                    }
        """
        self.config = config
        connection_string = self.get_connection_string()

        try:
            self.engine = create_engine(connection_string, pool_pre_ping=True)
            # pool_pre_ping=True helps to check the connection before use
            # and recycle connections if they are no longer valid.
        except Exception as e:
            logger.error(f"Error creating SQLAlchemy engine: {str(e)}")
            raise

    def get_connection_string(self) -> str:
        """
        Generates the connection string based on the database type.
        """
        db_type = self.config.get("db_type")
        if not db_type:
            raise ValueError("Database configuration must include a 'db_type'.")

        if db_type == "duckdb":
            db_path = self.config.get("path")
            if not db_path:
                raise ValueError("DuckDB configuration must include a 'path'.")
            return f"duckdb:///{db_path}"
        elif db_type == "postgresql":
            host = self.config.get("host")
            user = self.config.get("user")
            password = self.config.get("password")
            database = self.config.get("database")
            if not all([host, user, password, database]):
                raise ValueError(
                    "PostgreSQL configuration must include host, user, password, and database."
                )
            return f"postgresql://{user}:{password}@{host}/{database}"
        elif db_type == "singlestore":
            host = self.config.get("host")
            user = self.config.get("user")
            password = self.config.get("password")
            database = self.config.get("database")
            if not all([host, user, password, database]):
                raise ValueError(
                    "Singlestore configuration must include host, user, password, and database."
                )
            return f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")

    def query_to_df(self, query: str) -> pd.DataFrame:
        """
        Execute a query and return the results as a Pandas DataFrame.

        Args:
            query: The SQL query to execute.

        Returns:
            A Pandas DataFrame containing the results of the query.
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def execute(self, query: Union[str, text]) -> None:
        """
        Execute a non-query SQL statement (e.g., INSERT, UPDATE, DELETE).

        Args:
            query: The SQL statement to execute.
        """
        try:
            with self.engine.connect() as connection:
                if isinstance(query, str):
                    query = text(query)
                connection.execute(query)
                connection.commit()  # For SQLAlchemy versions < 2.0
        except Exception as e:
            logger.error(f"Error executing statement: {str(e)}")
            raise

    def list_tables(self) -> pd.DataFrame:
        """
        List all tables in the database.

        Returns:
            A Pandas DataFrame containing the schema and table names.
        """
        try:
            with self.engine.connect() as connection:
                query = text(
                    """
                    SELECT table_schema as schema, table_name as table
                    FROM information_schema.tables
                    WHERE table_type = 'BASE TABLE'
                    AND table_schema NOT IN ('pg_catalog', 'information_schema')
                    """
                )
                result = connection.execute(query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            raise

    def get_table_schema(self, schema: str, table: str) -> pd.DataFrame:
        """
        Get the schema for a specific table.

        Args:
            schema: The schema name.
            table: The table name.

        Returns:
            A Pandas DataFrame containing the column name, data type, and other information.
        """
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table, schema=schema)
            column_data = []
            for column in columns:
                column_data.append(
                    {
                        "column_name": column["name"],
                        "data_type": str(column["type"]),
                        "is_nullable": "YES" if column["nullable"] else "NO",
                        "column_default": (
                            str(column["default"]) if column["default"] else None
                        ),
                    }
                )
            df = pd.DataFrame(column_data)
            return df
        except Exception as e:
            logger.error(f"Error getting table schema for {schema}.{table}: {str(e)}")
            raise

    def get_table_info(self, schema: str, table: str) -> Dict[str, Any]:
        """
        Get information about a specific table.

        Args:
            schema: The schema name.
            table: The table name.

        Returns:
            A dictionary containing information about the table, such as the row count.
        """
        try:
            with self.engine.connect() as connection:
                query = text(f"""SELECT COUNT(*) FROM "{schema}"."{table}" """)
                result = connection.execute(query)
                row_count = result.scalar()
                return {"row_count": row_count}
        except Exception as e:
            logger.error(f"Error getting table info for {schema}.{table}: {str(e)}")
            raise

    def execute_transaction(self, operations: List[callable]) -> None:
        """
        Execute a series of operations within a single transaction.

        Args:
            operations: A list of callable objects (functions) that take a database
                connection as an argument and perform database operations.
        """
        try:
            with self.engine.connect() as connection:
                trans = connection.begin()
                try:
                    for operation in operations:
                        operation(connection)
                    trans.commit()
                except Exception as e:
                    trans.rollback()
                    logger.error(f"Transaction rolled back due to error: {str(e)}")
                    raise
        except Exception as e:
            logger.error(f"Error executing transaction: {str(e)}")
            raise
