
import sqlite3

import pytest

from sql.validator import validate_query_against_database


@pytest.fixture
def database_path(tmp_path):
    """Create a small SQLite database for validator tests."""
    path = tmp_path / "test_students.db"

    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE Students (
                StudentID INTEGER PRIMARY KEY,
                Name TEXT,
                Gender TEXT
            );

            CREATE TABLE Marks (
                StudentID INTEGER,
                Math INTEGER,
                FOREIGN KEY (StudentID)
                    REFERENCES Students(StudentID)
            );

            INSERT INTO Students VALUES (1, 'Alice', 'Female');
            INSERT INTO Students VALUES (2, 'Bob', 'Male');

            INSERT INTO Marks VALUES (1, 85);
            INSERT INTO Marks VALUES (2, 92);
            """
        )

    return str(path)


def test_rejects_column_from_wrong_table(database_path):
    query = "SELECT AVG(Math) FROM Marks GROUP BY Gender"

    is_valid, message = validate_query_against_database(
        query, database_path
    )

    assert not is_valid
    assert "Gender" in message


def test_accepts_correct_join_and_grouping(database_path):
    query = """
        SELECT S.Gender, AVG(M.Math) AS average_math
        FROM Students AS S
        JOIN Marks AS M ON S.StudentID = M.StudentID
        GROUP BY S.Gender
    """

    is_valid, message = validate_query_against_database(
        query, database_path
    )

    assert is_valid, message


def test_rejects_unknown_column(database_path):
    query = "SELECT UnknownColumn FROM Students"

    is_valid, message = validate_query_against_database(
        query, database_path
    )

    assert not is_valid
    assert "UnknownColumn" in message


def test_rejects_unknown_table(database_path):
    query = "SELECT Name FROM UnknownTable"

    is_valid, message = validate_query_against_database(
        query, database_path
    )

    assert not is_valid
    assert "UnknownTable" in message


def test_rejects_delete_statement(database_path):
    query = "DELETE FROM Students"

    is_valid, message = validate_query_against_database(
        query, database_path
    )

    assert not is_valid
    assert "SELECT" in message or "read-only" in message


def test_accepts_simple_select(database_path):
    query = "SELECT Name FROM Students"

    is_valid, message = validate_query_against_database(
        query, database_path
    )

    assert is_valid, message
