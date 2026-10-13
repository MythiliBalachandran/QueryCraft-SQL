# QueryCraft SQL

A natural-language SQL assistant that uses a locally running large language model to translate questions into SQL queries and retrieve results from a relational database.

## Tech Stack

- **Python** — application logic
- **Streamlit** — interactive web interface
- **Ollama** — local LLM runtime
- **Llama 3.2 (3B)** — SQL generation
- **SQLite** — relational database

## How It Works

1. The user enters a question in natural language.
2. The application provides database schema context to the language model.
3. The model generates an SQL query.
4. The application executes the query against the SQLite database.
5. The results are displayed in the interface.

## Example Queries

- Show all students.
- Find students older than 18.
- Display student names and marks.
- Show student attendance details.

## Run Locally

Install Python and [Ollama](https://ollama.com/), then download the model:

```bash
ollama pull llama3.2:3b
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

Make sure Ollama is running before submitting questions.
