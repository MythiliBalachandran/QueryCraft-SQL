
"""
SQL query validation module.

Provides basic read-only SQL validation and schema-aware validation
using SQLite's own query preparation.
"""

import re
import sqlite3
from typing import Optional, Tuple


DANGEROUS_KEYWORDS = [
    "DROP",
    "DELETE",
    "TRUNCATE",
    "ALTER",
    "INSERT",
    "UPDATE",
    "CREATE",
    "MODIFY",
    "REPLACE",
    "ATTACH",
    "DETACH",
    "VACUUM",
    "REINDEX",
]


def is_select_query(query: str) -> bool:
    """
    Allow SELECT queries and read-only queries beginning with WITH.
    """
    cleaned = query.strip().lstrip("\ufeff")
    return bool(re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE))


def check_dangerous_keywords(
    query: str,
) -> Tuple[bool, Optional[str]]:
    """
    Reject potentially dangerous SQL operations.

    This is a basic keyword check, not a complete SQL security boundary.
    """
    for keyword in DANGEROUS_KEYWORDS:
        pattern = rf"\b{re.escape(keyword)}\b"
        if re.search(pattern, query, re.IGNORECASE):
            return False, (
                f"Query contains potentially dangerous keyword: {keyword}"
            )

    return True, None


def validate_sql_syntax(
    query: str,
) -> Tuple[bool, Optional[str]]:
    """
    Perform basic syntax sanity checks.
    SQLite preparation is used separately for actual syntax validation.
    """
    if not query or not query.strip():
        return False, "Query is empty"

    if query.count("(") != query.count(")"):
        return False, "Unmatched parentheses"

    # Check single-quote balance while allowing escaped SQL quotes ('').
    quote_count = 0
    i = 0

    while i < len(query):
        if query[i] == "'":
            if i + 1 < len(query) and query[i + 1] == "'":
                i += 2
                continue
            quote_count += 1
        i += 1

    if quote_count % 2 != 0:
        return False, "Unmatched single quotes"

    return True, None


def validate_query(
    query: str,
    allow_unsafe: bool = False,
) -> Tuple[bool, str]:
    """
    Perform basic read-only query validation.

    Note:
        This function does not know the database schema. Use
        validate_query_against_database() before executing a query.
    """
    if not query or not query.strip():
        return False, "Query is empty"

    cleaned = query.strip()

    if not is_select_query(cleaned):
        return False, "Only SELECT or read-only WITH queries are supported"

    # Reject multiple statements, ignoring a single trailing semicolon.
    without_trailing_semicolon = cleaned.rstrip().rstrip(";").rstrip()

    if ";" in without_trailing_semicolon:
        return False, "Multiple SQL statements are not supported"

    if not allow_unsafe:
        is_safe, message = check_dangerous_keywords(cleaned)
        if not is_safe:
            return False, message or "Potentially unsafe SQL"

    is_valid, message = validate_sql_syntax(cleaned)
    if not is_valid:
        return False, message or "Invalid SQL syntax"

    return True, "Basic query checks passed"


def validate_query_against_database(
    query: str,
    database_path: str,
) -> Tuple[bool, str]:
    """
    Validate a query against the actual SQLite database schema.

    Opens the database in read-only mode and asks SQLite to prepare the
    query without executing it. This catches unknown tables, unknown
    columns, and SQLite syntax errors.

    This function must be called before query execution.
    """
    is_valid, message = validate_query(query)

    if not is_valid:
        return False, message

    uri = f"file:{database_path.replace(chr(92), '/')}?mode=ro"

    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            connection.execute("PRAGMA query_only = ON")

            # EXPLAIN prepares the query and checks schema references
            # without executing the query itself.
            connection.execute("EXPLAIN " + query.rstrip().rstrip(";"))
        finally:
            connection.close()

        return True, "Query syntax and schema checks passed"

    except sqlite3.Error as exc:
        return False, f"SQLite validation failed: {exc}"
