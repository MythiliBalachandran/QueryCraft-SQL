
"""
Evaluation runner for QueryCraft AI.

Runs natural-language questions through the existing SQL generation
and validation pipeline, compares successful query results against
expected SQL results, and records latency and rejection behavior.
"""

import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

import yaml

# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import attempt_sql_generation, initialize_llm
from sql.schema_introspector import get_database_schema


EVALUATION_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVALUATION_DIR / "questions.json"
REPORT_FILE = EVALUATION_DIR / "evaluation_report.csv"


def load_configuration():
    """Load the existing project configuration."""
    config_path = PROJECT_ROOT / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    database_path = config.get("database", {}).get("path")
    if not database_path:
        raise ValueError("Database path is missing from config.yaml.")

    database_path = Path(database_path)
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path

    database_path = database_path.resolve()

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Database file not found: {database_path}"
        )

    return config, database_path


def open_readonly_connection(database_path):
    """Open SQLite in read-only mode."""
    uri = database_path.as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def fetch_rows(connection, sql):
    """Execute a SELECT query and return comparable result rows."""
    cursor = connection.execute(sql)

    if cursor.description is None:
        raise ValueError("Query did not return a result set.")

    columns = [item[0] for item in cursor.description]
    rows = cursor.fetchall()

    # Column names are included so different selected fields
    # cannot accidentally be treated as equivalent.
    return columns, rows


def results_match(generated_result, expected_result):
    """Compare result columns and rows, ignoring row order."""
    if generated_result is None or expected_result is None:
        return False

    generated_columns, generated_rows = generated_result
    expected_columns, expected_rows = expected_result

    if len(generated_columns) != len(expected_columns):
        return False

    if sorted(name.lower() for name in generated_columns) != sorted(
        name.lower() for name in expected_columns
    ):
        return False

    # Sorting allows equivalent results returned in a different order.
    # repr() gives a stable sortable representation for mixed SQLite values.
    return sorted(map(repr, generated_rows)) == sorted(
        map(repr, expected_rows)
    )


def main():
    config, database_path = load_configuration()

    with (QUESTIONS_FILE).open("r", encoding="utf-8") as file:
        questions = json.load(file)

    llm_config = config.get("llm", {})
    model_name = llm_config.get("model_name", "llama3.2:3b")
    base_url = llm_config.get("base_url", "http://localhost:11434")
    max_retries = max(1, int(llm_config.get("max_retries", 3)))

    print(f"Model: {model_name}")
    print(f"Database: {database_path}")
    print(f"Questions: {len(questions)}")
    print("-" * 65)

    llm_model = initialize_llm(base_url, model_name)
    if llm_model is None:
        raise RuntimeError(
            "Could not initialize Ollama. Check that Ollama is running "
            "and the configured model is available."
        )

    with open_readonly_connection(database_path) as connection:
        schema = get_database_schema(connection.cursor())

    records = []

    for index, item in enumerate(questions, start=1):
        question_id = item["id"]
        question = item["question"]
        expected_behavior = item["expected_behavior"]
        expected_sql = item.get("expected_sql")

        start = time.perf_counter()

        generated_sql = ""
        message = ""
        generation_success = False
        actual_result = None
        expected_result = None
        validation_error = ""
        result_correct = False

        try:
            generated_sql, message, generation_success = (
                attempt_sql_generation(
                    llm_model=llm_model,
                    table_schema=schema,
                    question=question,
                    database_path=str(database_path),
                    max_retries=max_retries,
                )
            )

            if expected_behavior == "answer":
                if generation_success and generated_sql:
                    with open_readonly_connection(
                        database_path
                    ) as connection:
                        actual_result = fetch_rows(
                            connection, generated_sql
                        )
                        expected_result = fetch_rows(
                            connection, expected_sql
                        )

                    result_correct = results_match(
                        actual_result, expected_result
                    )

            elif expected_behavior == "reject":
                # A rejected query is correct for unsupported requests.
                result_correct = not generation_success

            elif expected_behavior == "clarify_or_reject":
                # This version accepts a safe rejection as the expected
                # behavior. Clarification support can be added later.
                result_correct = not generation_success

        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            result_correct = False

        latency_seconds = time.perf_counter() - start

        if expected_behavior == "answer":
            if generation_success and actual_result is not None:
                status = (
                    "PASS" if result_correct
                    else "WRONG_RESULT"
                )
            elif generation_success:
                status = "EXECUTION_ERROR"
            else:
                status = "GENERATION_OR_VALIDATION_FAILURE"

        elif result_correct:
            status = "PASS_REJECTED"

        elif generation_success:
            status = "UNEXPECTED_ACCEPTANCE"

        else:
            status = "REJECTION_NEEDS_REVIEW"

        records.append({
            "id": question_id,
            "category": item.get("category", ""),
            "question": question,
            "expected_behavior": expected_behavior,
            "generation_success": generation_success,
            "generated_sql": generated_sql,
            "result_correct": result_correct,
            "status": status,
            "latency_seconds": round(latency_seconds, 3),
            "message": str(message),
            "error": validation_error,
        })

        print(
            f"[{index:02d}/{len(questions)}] "
            f"{question_id} | {status} | "
            f"{latency_seconds:.2f}s"
        )

    with REPORT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    total = len(records)
    correct = sum(row["result_correct"] for row in records)
    answer_cases = [
        row for row in records
        if row["expected_behavior"] == "answer"
    ]
    reject_cases = [
        row for row in records
        if row["expected_behavior"] in ("reject", "clarify_or_reject")
    ]

    answer_correct = sum(
        row["result_correct"] for row in answer_cases
    )
    rejection_correct = sum(
        row["result_correct"] for row in reject_cases
    )
    average_latency = sum(
        row["latency_seconds"] for row in records
    ) / total if total else 0

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Total questions: {total}")
    print(
        f"Overall expected-behavior accuracy: "
        f"{correct}/{total} ({correct / total:.1%})"
        if total else "No questions evaluated."
    )
    print(
        f"Answer cases correct: "
        f"{answer_correct}/{len(answer_cases)}"
    )
    print(
        f"Unsupported/ambiguous cases handled as expected: "
        f"{rejection_correct}/{len(reject_cases)}"
    )
    print(f"Average latency: {average_latency:.2f} seconds")
    print(f"Detailed report: {REPORT_FILE}")

    print(
        "\nReview WRONG_RESULT, EXECUTION_ERROR, "
        "UNEXPECTED_ACCEPTANCE, and REJECTION_NEEDS_REVIEW "
        "before drawing conclusions."
    )


if __name__ == "__main__":
    main()
