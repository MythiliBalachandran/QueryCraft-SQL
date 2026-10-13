
"""
Prompt templates for SQL generation.

This module contains templates for generating SQL queries from natural
language and for validating queries and summarizing their results.
"""

from typing import List



def get_sql_generation_prompt(table_schema: str, question: str) -> str:
    """
    Generate a schema-aware SQLite query-generation prompt.
    """
    return f"""
You are an expert SQLite text-to-SQL system.

Convert the user's question into ONE correct, read-only SQLite query
using only the database schema provided below.

### DATABASE SCHEMA
{table_schema}

### USER QUESTION
{question}

### QUERY GENERATION PROCESS

Before producing the final output, reason through these steps internally:

1. Identify the exact information requested by the user.
2. Map every requested output field to the table that contains it.
3. Identify the table containing each filter, sorting field, and
   aggregation field.
4. Determine which tables are necessary and join them using the
   relationships supported by the schema.
5. Apply filters, grouping, sorting, and limits exactly as requested.
6. Verify every table alias and column reference before answering.

### STRICT RULES

1. Use only tables and columns explicitly present in the schema.
2. A column may only be referenced through a table or alias that
   actually contains that column.
3. Never assume that related tables contain the same columns.
4. Use JOIN conditions supported by the schema. Do not invent
   relationships.
5. Select only the columns needed to answer the question.
6. Do not add unrelated columns to the SELECT list.
7. When the user asks for the highest or lowest values of a specific
   measure, sort by the column representing that measure.
8. Distinguish between total counts and percentages. For attendance,
   use AttendancePercentage when the question asks for the highest
   attendance percentage or rate.
9. Apply LIMIT only when the user requests a specific number of results
   or the question otherwise requires it.
10. If a requested field or operation cannot be supported by the
    schema, return UNANSWERABLE rather than inventing information.
11. Generate exactly one read-only SELECT query. Never generate
    INSERT, UPDATE, DELETE, DROP, or other data-modifying statements.
12. Do not return Markdown fences, explanations, or multiple queries.
13. Before answering, verify that every selected column, table alias,
    join, filter, and sorting expression is valid for the schema.

### EXAMPLES

Schema:
CREATE TABLE Students (StudentID INTEGER, Name TEXT, Age INTEGER);
CREATE TABLE Marks (StudentID INTEGER, Math INTEGER);

Question:
Show all students with their math marks.

Output:
SELECT S.Name, M.Math FROM Students AS S JOIN Marks AS M ON S.StudentID = M.StudentID;

Question:
List students with their names and age.

Output:
SELECT Name, Age FROM Students;

Question:
Find the top 5 students by attendance percentage and show their math marks.

Schema:
CREATE TABLE Students (StudentID INTEGER, Name TEXT);
CREATE TABLE Marks (StudentID INTEGER, Math INTEGER);
CREATE TABLE Attendance (StudentID INTEGER, AttendancePercentage REAL);

Output:
SELECT S.Name, M.Math FROM Students AS S JOIN Marks AS M ON S.StudentID = M.StudentID JOIN Attendance AS A ON S.StudentID = A.StudentID ORDER BY A.AttendancePercentage DESC LIMIT 5;

Question:
Show students from a table or column that does not exist.

Output:
UNANSWERABLE

### FINAL OUTPUT
Return exactly one of:
- A valid, read-only SQLite SELECT query on one line.
- UNANSWERABLE

Output:
""".strip()



def get_validation_prompt(sql_query: str, table_schema: str) -> str:
    """
    Generate a prompt for reviewing a SQL query.

    Args:
        sql_query: SQL query to validate.
        table_schema: Database schema information.

    Returns:
        Formatted validation prompt.
    """
    return f"""
You are an expert SQLite query reviewer.

DATABASE SCHEMA:
{table_schema}

QUERY TO VALIDATE:
{sql_query}

Check whether all referenced tables and columns exist, joins use appropriate
keys, SQLite syntax is valid, and the query is read-only.

Do not assume a query is correct merely because its syntax looks valid.

Respond with:
VALID
or
INVALID: followed by a concise explanation.
""".strip()


def generate_result_prompt(
    columns: List[str],
    rows: List[tuple],
    question: str,
) -> str:
    """
    Generate a prompt for analyzing and summarizing query results.
    """
    num_rows = len(rows)
    rows_preview = rows[:10]

    sample_data = "\n".join(
        f"  Row {i}: " + " | ".join(
            f"{col}: {val}" for col, val in zip(columns, row)
        )
        for i, row in enumerate(rows_preview, 1)
    )

    if num_rows > 10:
        sample_data += f"\n  ...\nTotal: {num_rows} results"

    return f"""
Analyze these database query results and provide concise, evidence-based
insights.

Original Question: {question}
Column Names: {', '.join(columns)}
Number of Results: {num_rows}

Data (first {min(num_rows, 10)} rows):
{sample_data}

Provide:
1. Key findings or patterns in 2-3 sentences.
2. Notable observations supported by the data.

If no rows were returned, state that clearly. Do not invent facts.
Be direct and avoid obvious statements.
""".strip()
