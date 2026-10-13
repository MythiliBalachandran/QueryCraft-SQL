
import sqlite3

import pytest


def test_read_only_connection_blocks_data_changes(tmp_path):
    """Verify SQLite read-only mode prevents database modifications."""
    database_path = tmp_path / "readonly.db"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT)"
        )
        connection.execute(
            "INSERT INTO students (name) VALUES ('Alice')"
        )

    uri = database_path.as_uri() + "?mode=ro"

    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")

        result = connection.execute(
            "SELECT name FROM students"
        ).fetchone()

        assert result == ("Alice",)

        with pytest.raises(sqlite3.OperationalError):
            connection.execute(
                "INSERT INTO students (name) VALUES ('Bob')"
            )

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT name FROM students ORDER BY id"
        ).fetchall()

    assert rows == [("Alice",)]
