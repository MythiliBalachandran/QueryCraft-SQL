from unittest.mock import patch

from app import attempt_sql_generation


def test_unanswerable_response_stops_generation():
    with patch(
        "app.get_response_from_llm_model",
        return_value=("test prompt", "UNANSWERABLE"),
    ):
        generated_sql, message, success = attempt_sql_generation(
            llm_model=None,
            table_schema="CREATE TABLE Students (StudentID INTEGER);",
            question="Show students from a table that does not exist.",
            database_path="unused.db",
            max_retries=3,
        )

    assert generated_sql == ""
    assert success is False
    assert "cannot be answered" in message


def test_salary_question_does_not_accept_unrelated_query():
    with patch(
        "app.get_response_from_llm_model",
        return_value=(
            "test prompt",
            "SELECT Students.Salary FROM Students",
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
            database_path="unused.db",
            max_retries=1,
        )

    assert success is False
    assert generated_sql == ""