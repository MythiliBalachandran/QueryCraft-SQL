
import sqlite3
from unittest.mock import patch

from app import attempt_sql_generation


def test_rejects_unrelated_query_for_salary_question(tmp_path):
    database_path = tmp_path / "students.db"

    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            CREATE TABLE Students (
                StudentID INTEGER PRIMARY KEY,
                Name TEXT,
                Age INTEGER,
                Gender TEXT
            )
        """)

    with patch(
        "app.get_response_from_llm_model",
        return_value=(
            "test prompt",
            "SELECT Students.Name FROM Students",
        ),
    ):
        generated_sql, message, success = attempt_sql_generation(
            llm_model=None,
            table_schema="""
                CREATE TABLE Students (
                    StudentID INTEGER PRIMARY KEY,
                    Name TEXT,
                    Age INTEGER,
                    Gender TEXT
                );
            """,
            question="Show the salary of every student.",
            database_path=str(database_path),
            max_retries=1,
        )

    print("GENERATED SQL:", repr(generated_sql))
    print("SUCCESS:", success)
    print("MESSAGE:", message)

    assert success is False, (
        "The app accepted a query that retrieves names instead of salaries."
    )
    assert generated_sql == ""

