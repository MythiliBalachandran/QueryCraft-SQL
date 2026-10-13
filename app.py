import re
import sqlite3
from pathlib import Path
from typing import Tuple

import streamlit as st
import yaml

from llm import get_llm_model, get_response_from_llm_model
from llm.inference import get_result_summary
from llm.prompts import generate_result_prompt
from sql import get_database_connection, execute_query
from sql.validator import validate_query, validate_query_against_database
from sql.schema_introspector import get_database_schema
from security.sql_guardrails import SQLGuardrails


def clean_sql_from_response(response: str) -> str:
    """Extract and clean a SQL query from an LLM response."""
    sql = (response or "").strip()

    # Remove Markdown code fences.
    if sql.startswith("```"):
        lines = sql.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql = "\n".join(lines).strip()

    # Remove a leading SQL label, if present.
    sql = re.sub(r"^\s*SQL\s*:\s*", "", sql, flags=re.IGNORECASE)

    # Preserve spaces between SQL tokens.
    return re.sub(r"\s+", " ", sql).strip()


def _schema_column_names(table_schema: str) -> set[str]:
    """Extract column names from CREATE TABLE statements."""
    columns = set()

    for line in (table_schema or "").splitlines():
        line = line.strip().rstrip(",").strip()
        if not line:
            continue
        if line.upper().startswith("CREATE TABLE"):
            continue
        if line.startswith(("(", ")")):
            continue
        if re.match(
            r"^(PRIMARY KEY|FOREIGN KEY|UNIQUE|CHECK|CONSTRAINT)\b",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        match = re.match(
            r'''["`\[]?([A-Za-z_]\w*)["`\]]?\s+
            (?:INTEGER|TEXT|REAL|NUMERIC|BLOB|VARCHAR|CHAR|
            BOOLEAN|DATE|DATETIME|INT|FLOAT|DOUBLE)\b''',
            line,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        if match:
            columns.add(match.group(1).lower())

    return columns


def is_query_relevant(
    question: str,
    query: str,
    table_schema: str,
) -> Tuple[bool, str]:
    """Reject queries that fail targeted checks for explicitly requested fields.

    This is a targeted relevance check, not a complete semantic verifier.
    """
    question_lower = (question or "").lower()
    query_lower = (query or "").lower()
    schema_columns = _schema_column_names(table_schema)

    requested_concepts = {
        "salary": {
            "question_terms": (
                "salary", "salaries", "pay", "wage", "wages", "compensation",
            ),
            "column_aliases": {
                "salary", "salaries", "pay", "wage", "wages", "compensation",
            },
        },
        "email": {
            "question_terms": ("email", "e-mail", "email address"),
            "column_aliases": {"email", "emailaddress", "e_mail"},
        },
        "phone": {
            "question_terms": (
                "phone", "telephone", "mobile number", "contact number",
            ),
            "column_aliases": {
                "phone", "phonenumber", "phone_number", "telephone",
                "mobile", "contactnumber",
            },
        },
        "address": {
            "question_terms": ("address", "home address", "street address"),
            "column_aliases": {"address", "streetaddress", "homeaddress"},
        },
        "department": {
            "question_terms": ("department", "department name"),
            "column_aliases": {
                "department", "departmentname", "department_id",
                "departmentid", "dept",
            },
        },
    }

    for concept, definition in requested_concepts.items():
        question_terms = definition["question_terms"]
        column_aliases = definition["column_aliases"]

        if not any(term in question_lower for term in question_terms):
            continue

        available_columns = schema_columns.intersection(column_aliases)
        if not available_columns:
            return (
                False,
                f"The question requests {concept} information, but the available "
                "database schema does not contain a matching field. The query "
                "was rejected rather than returning unrelated data.",
            )

        query_identifiers = {
            token.lower()
            for token in re.findall(r"[A-Za-z_]\w*", query_lower)
        }
        if not query_identifiers.intersection(available_columns):
            return (
                False,
                f"The question requests {concept} information, but the generated "
                "SQL does not refer to a matching field.",
            )

    return True, "The targeted relevance checks passed."


def question_requests_nonexistent_table(question: str) -> bool:
    """Detect explicit requests that describe a nonexistent table."""
    patterns = [
        r"\btable\s+that\s+does\s+not\s+exist\b",
        r"\bnon[- ]existent\s+table\b",
        r"\bnonexisting\s+table\b",
        r"\bmissing\s+table\b",
        r"\bunknown\s+table\b",
        r"\btable\s+which\s+does\s+not\s+exist\b",
    ]
    return any(re.search(pattern, question or "", re.IGNORECASE) for pattern in patterns)


def attempt_sql_generation(
    llm_model,
    table_schema: str,
    question: str,
    database_path: str,
    max_retries: int = 3,
) -> tuple:
    """Generate SQL and validate it against the configured SQLite database.

    Returns:
        (generated_sql, prompt_or_message, success_flag)
    """
    # Reject the exact type of explicit nonexistent-table request before the LLM
    # can silently replace it with a valid but unrelated table.
    if question_requests_nonexistent_table(question):
        return (
            "",
            "The question requests data from a nonexistent table. "
            "No matching table is available in the database schema.",
            False,
        )

    last_error = "Failed to generate valid SQL after all retries"

    for attempt in range(1, max_retries + 1):
        try:
            llm_prompt, llm_response = get_response_from_llm_model(
                llm_model=llm_model,
                table_schema=table_schema,
                question=question,
            )
            generated_sql = clean_sql_from_response(llm_response)

            if not generated_sql:
                last_error = "The model returned an empty SQL response."

            elif generated_sql.strip().upper() == "UNANSWERABLE":
                return (
                    "",
                    "This question cannot be answered using the available "
                    "database schema. Please check the table and column names.",
                    False,
                )

            else:
                # Step 1: Check relevance before normal SQL validation.
                is_relevant, relevance_message = is_query_relevant(
                    question, generated_sql, table_schema
                )
                if not is_relevant:
                    return "", relevance_message, False

                # Step 2: Perform basic read-only and SQL validation.
                is_valid, validation_msg = validate_query(generated_sql)
                if not is_valid:
                    last_error = validation_msg
                else:
                    # Step 3: Validate SQLite syntax and schema references.
                    is_valid, validation_msg = validate_query_against_database(
                        generated_sql, database_path
                    )
                    if not is_valid:
                        last_error = validation_msg
                    else:
                        # Step 4: Apply the project's existing security checks.
                        is_safe, safety_msg = SQLGuardrails.check_query_safety(
                            generated_sql
                        )
                        if not is_safe:
                            last_error = safety_msg
                        else:
                            if attempt > 1:
                                st.success(
                                    f"SQL passed validation on attempt "
                                    f"{attempt}/{max_retries}."
                                )
                            return generated_sql, llm_prompt, True

            # Retry when the query fails validation.
            if attempt < max_retries:
                st.warning(
                    f"Attempt {attempt}/{max_retries}: {last_error}. "
                    "Generating another query."
                )

        except Exception as exc:
            last_error = f"SQL generation or validation failed: {exc}"
            if attempt < max_retries:
                st.warning(f"Attempt {attempt}/{max_retries}: {last_error}")

    return "", last_error, False


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML."""
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file) or {}
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as exc:
        st.error(f"Invalid configuration file: {exc}")
        return {}


@st.cache_resource
def initialize_llm(base_url: str, model_name: str):
    """Initialize and cache the Ollama model client."""
    try:
        return get_llm_model(ollama_base_url=base_url, model_name=model_name)
    except Exception as exc:
        st.error(f"Failed to connect to Ollama: {exc}")
        return None


def get_db_connection(config):
    """Open a database connection using the project's existing helper."""
    try:
        database_path = config["database"]["path"]
        return get_database_connection(database_path)
    except Exception as exc:
        st.error(f"Failed to connect to database: {exc}")
        return None, None


def main():
    """Run the SQL Assistant with a compact dark-purple interface."""
    st.set_page_config(page_title="QueryCraft AI", page_icon="Q", layout="wide")

    
    # Hide Streamlit's built-in header, Deploy button, and menu
    st.markdown(
        """
        <style>
            header[data-testid="stHeader"] {
                display: none !important;
            }

            div[data-testid="stToolbar"] {
                display: none !important;
            }

            .stAppDeployButton {
                display: none !important;
            }

            #MainMenu {
                visibility: hidden !important;
            }

            footer {
                visibility: hidden !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # UI-only styling; SQL generation, validation, security, and execution are unchanged.
    st.markdown("""
    <style>
    :root { --qc-bg:#100D18; --qc-panel:#171222; --qc-border:#33264D; --qc-purple:#8950F5; --qc-light:#B39AFF; --qc-text:#F4F0FF; --qc-muted:#AAA1BD; }
    .stApp { background:radial-gradient(ellipse at 58% 35%,#24163A 0%,#120E1C 58%,#0D0B13 100%); color:var(--qc-text); }
    [data-testid="stHeader"] { background:rgba(16,13,24,.88); }
    .block-container { max-width:1320px; padding-top:1.8rem; padding-bottom:2.8rem; }
    h1,h2,h3,p,label,.stMarkdown { color:var(--qc-text); }
    h1 { letter-spacing:-.04em; font-weight:700; }
    .qc-brand { display:flex; align-items:center; gap:12px; margin-bottom:1.2rem; }
    .qc-mark { display:flex; align-items:center; justify-content:center; width:42px; height:42px; border-radius:14px; background:linear-gradient(135deg,#A98BFF,#7540E8); color:white; font-size:23px; font-weight:800; box-shadow:0 0 28px rgba(137,80,245,.22); }
    .qc-brand-name { font-size:1.12rem; font-weight:750; color:#F8F5FF; }
    .qc-brand-caption { color:var(--qc-muted); font-size:.78rem; margin-top:2px; }
    .qc-eyebrow { color:var(--qc-light); font-size:.74rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
    .qc-subtitle { color:var(--qc-muted); margin-top:-.55rem; margin-bottom:1.5rem; }
    .qc-card-title { color:#F4F0FF; font-size:1.04rem; font-weight:700; margin-bottom:5px; }
    .qc-card-caption { color:var(--qc-muted); font-size:.84rem; margin-bottom:12px; }
    div[data-testid="stTextArea"] textarea { background:#171321; color:#F4F0FF; border:1px solid #44335F; border-radius:12px; }
    div[data-testid="stTextArea"] textarea:focus { border-color:var(--qc-purple); box-shadow:0 0 0 1px var(--qc-purple); }
    div[data-testid="stTextArea"] label { color:#D9D1EA; }
    div[data-testid="stButton"]>button { background:#211A30; color:#EDE6FF; border:1px solid #403154; border-radius:11px; min-height:2.75rem; font-weight:650; }
    div[data-testid="stButton"]>button:hover { border-color:#9B71FF; color:white; }
    div[data-testid="stButton"]>button[kind="primary"] { background:linear-gradient(100deg,#8D50F5,#7040D6); color:white; border:1px solid #9B70FF; }
    div[data-testid="stCode"] { border:1px solid #3B2A56; border-radius:12px; background:#120F1B; }
    div[data-testid="stAlert"] { border-radius:12px; }
    [data-testid="stDataFrame"] { border:1px solid #3B2A56; border-radius:12px; overflow:hidden; }
    details { background:rgba(25,19,39,.75); border:1px solid #33264D; border-radius:12px; padding:4px 10px; }
    hr { border-color:#33264D; }
    small { color:var(--qc-muted); }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="qc-brand"><div class="qc-mark">Q</div><div><div class="qc-brand-name">QueryCraft AI</div><div class="qc-brand-caption">Private, local natural-language SQL assistant</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="qc-eyebrow">AI WORKSPACE</div>', unsafe_allow_html=True)
    st.title("What would you like to know?")
    st.markdown('<p class="qc-subtitle">Ask a question in plain English. Review the SQL, then explore the results.</p>', unsafe_allow_html=True)

    config = load_config("config.yaml")
    if not config:
        st.error("Failed to load configuration.")
        return

    llm_config = config.get("llm", {})
    database_config = config.get("database", {})
    database_path = database_config.get("path")
    if not database_path:
        st.error("The database path is missing from config.yaml.")
        return

    project_root = Path(__file__).resolve().parent
    resolved_database_path = Path(database_path)
    if not resolved_database_path.is_absolute():
        resolved_database_path = project_root / resolved_database_path
    database_path = str(resolved_database_path.resolve())
    if not Path(database_path).is_file():
        st.error(f"Database file not found: {database_path}")
        st.info("Create the database using the project's database setup script.")
        return

    model_name = llm_config.get("model_name", "llama3.2:3b")
    base_url = llm_config.get("base_url", "http://localhost:11434")
    llm_model = initialize_llm(base_url, model_name)
    if llm_model is None:
        st.error("Failed to initialize the LLM.")
        return

    db_connection = None
    try:
        db_connection, db_cursor = get_db_connection(config)
        if db_cursor is None:
            st.error("Failed to initialize the database.")
            return
        schema = get_database_schema(db_cursor)
        if not schema:
            st.error("Could not retrieve the database schema.")
            return
    except Exception as exc:
        st.error(f"Error retrieving database schema: {exc}")
        return
    finally:
        if db_connection is not None:
            db_connection.close()

    schema_col, workspace_col = st.columns([0.82, 1.8], gap="large")
    with schema_col:
        st.markdown('<div class="qc-card-title">Database schema</div>', unsafe_allow_html=True)
        st.markdown('<div class="qc-card-caption">Tables and available columns</div>', unsafe_allow_html=True)
        st.text_area("Schema", value=schema, height=300, disabled=True, label_visibility="collapsed", key="qc_schema")
        st.markdown('<small>Queries are checked against this schema before execution.</small>', unsafe_allow_html=True)

    with workspace_col:
        st.markdown('<div class="qc-card-title">Ask your database</div>', unsafe_allow_html=True)
        st.markdown('<div class="qc-card-caption">Describe the result you need. You do not need to write SQL.</div>', unsafe_allow_html=True)
        question = st.text_area("Your question", height=180, placeholder="Ask anything about your data.\n\nFor example: Find the average Math marks grouped by gender.", label_visibility="collapsed", key="qc_question")
        submit_col, clear_col = st.columns([1.35, 1])
        with submit_col:
            submit = st.button("Generate SQL and run query", type="primary", use_container_width=True)
        with clear_col:
            clear = st.button("Clear", use_container_width=True)

        st.markdown("<small>Try a question like one of these</small>", unsafe_allow_html=True)
        example_cols = st.columns(3, gap="small")
        examples = [("All students", "Show all students."), ("Average marks", "Find the average Math marks grouped by gender."), ("Attendance", "Show students with the highest attendance.")]
        for example_col, (label, example_question) in zip(example_cols, examples):
            with example_col:
                st.button(
                    label,
                    use_container_width=True,
                    key=f"example_{label}",
                    on_click=lambda q=example_question: st.session_state.update({"qc_question": q}),
                )

    if clear:
        st.session_state["qc_question"] = ""
        st.rerun()
    if submit and not question.strip():
        st.warning("Please enter a question.")
        return
    if not submit:
        st.markdown("---")
        with st.expander("How QueryCraft AI works"):
            st.markdown("Enter a question in plain English. A local language model proposes SQL, the application validates it against the database schema and safety rules, and only then attempts read-only execution.\n\n- Read-only SQL validation and security guardrails\n- SQLite syntax and table/column validation\n- Targeted checks for unsupported fields and explicit nonexistent-table requests\n- Retries when generated SQL fails validation")
       
        return

    max_retries = max(1, int(llm_config.get("max_retries", 3)))
    with st.spinner("Generating and validating SQL..."):
        generated_sql, prompt_or_message, success = attempt_sql_generation(
            llm_model=llm_model, table_schema=schema, question=question.strip(),
            database_path=database_path, max_retries=max_retries,
        )
    if not success:
        st.error("Could not generate a query that passed validation: " f"{prompt_or_message}")
        st.info("Try rephrasing the question or checking the database schema.")
        return

    st.markdown("---")
    st.subheader("Generated SQL")
    st.code(generated_sql, language="sql")
    st.success("SQL passed the configured validation checks.")
 
    exec_connection = None
    try:
        with st.spinner("Executing query..."):
            uri = Path(database_path).as_uri() + "?mode=ro"
            exec_connection = sqlite3.connect(uri, uri=True)
            exec_connection.execute("PRAGMA query_only = ON")
            is_valid, validation_msg = validate_query_against_database(generated_sql, database_path)
            if not is_valid:
                st.error(f"Query failed the final schema check: {validation_msg}")
                return
            sql_response = execute_query(exec_connection.cursor(), generated_sql)
            rows = sql_response.fetchall()
            st.subheader("Query results")
            if not rows:
                st.info("Query executed successfully but returned no rows.")
            else:
                columns = ([description[0] for description in sql_response.description] if sql_response.description else [])
                st.dataframe([dict(zip(columns, row)) for row in rows], use_container_width=True, hide_index=True)
                st.success(f"Query executed successfully Â· {len(rows)} rows returned")
                if llm_config.get("enable_result_formatting", False):
                    with st.expander("AI-generated insights"):
                        try:
                            analysis_prompt = generate_result_prompt(columns, rows, question)
                            with st.spinner("Generating insights..."):
                                summary = get_result_summary(llm_model=llm_model, analysis_prompt=analysis_prompt)
                            st.markdown(summary)
                        except Exception as exc:
                            st.warning(f"Could not generate insights: {exc}")
    except sqlite3.Error as exc:
        st.error(f"Query execution failed: {exc}")
    except Exception as exc:
        st.error(f"An unexpected error occurred: {exc}")
    finally:
        if exec_connection is not None:
            exec_connection.close()

    st.markdown("---")
    st.caption("QueryCraft AI Â· Local LLM Â· SQL is checked against the SQLite schema before execution.")

if __name__ == "__main__":
    main()