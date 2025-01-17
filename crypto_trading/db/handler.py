import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Callable
import duckdb
from duckdb import DuckDBPyConnection


class DatabaseHandler:
    """Base class for database operations"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def db_connection(self) -> DuckDBPyConnection:
        """Context manager for handling DuckDB connections"""
        conn = None
        try:
            conn = duckdb.connect(str(self.db_path), read_only=False)
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
