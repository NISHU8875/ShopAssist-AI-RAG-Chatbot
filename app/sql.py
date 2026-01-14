import os
import re
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from phi.model.openai import OpenAIChat

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
load_dotenv()

model = OpenAIChat(id="gpt-5-mini")
db_path = Path(__file__).parent / "db.sqlite"

DEBUG = True

# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

SQL_GENERATION_PROMPT = """You are an expert SQL assistant.

You are given a SQLite database schema and a natural language question.
Generate ONE valid SQLite SQL query.

<schema>
table: product

columns:
- product_link (TEXT)
- title (TEXT)
- brand (TEXT)
- price (INTEGER)
- discount (REAL)
- avg_rating (REAL)
- total_ratings (INTEGER)
</schema>

RULES:
- Always use SELECT *
- Brand search must be case-insensitive using LOWER(brand) LIKE LOWER('%value%')
- Never use ILIKE
- Use ORDER BY and LIMIT when the question implies ranking or top results
- Never generate destructive SQL (DROP, DELETE, UPDATE, INSERT, ALTER)
- Output ONLY the SQL inside <SQL></SQL> tags
"""

RESULT_NARRATION_PROMPT = """You are an expert in converting structured product data into natural language.

You will receive:
- QUESTION
- DATA (list of products)

RULES:
- Use ONLY the provided data
- Do not mention databases, tables, or queries
- Format each product on a new line as:

1. Product title: Rs. price (discount% off), Rating: rating <product_link>

- If no products exist, say:
"No products match your request."
"""

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

UNSAFE_SQL_KEYWORDS = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER"}


def _is_safe_sql(sql: str) -> bool:
    sql_upper = sql.upper()
    return not any(keyword in sql_upper for keyword in UNSAFE_SQL_KEYWORDS)


def _extract_sql(text: str) -> str | None:
    """
    Extract first SQL block enclosed in <SQL></SQL>
    """
    match = re.search(r"<SQL>\s*(.*?)\s*</SQL>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------
# LLM Calls
# ---------------------------------------------------------------------

def generate_sql(question: str) -> str:
    messages = [
        {"role": "system", "content": SQL_GENERATION_PROMPT},
        {"role": "user", "content": question},
    ]

    response = model.run(messages)
    sql = _extract_sql(response.content)

    if not sql:
        raise ValueError("LLM failed to generate SQL.")

    if not _is_safe_sql(sql):
        raise ValueError("Unsafe SQL detected.")

    if DEBUG:
        print("\n[DEBUG] Generated SQL:\n", sql)

    return sql


def run_sql(sql: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def narrate_results(question: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "No products match your request."

    data = df.to_dict(orient="records")

    messages = [
        {"role": "system", "content": RESULT_NARRATION_PROMPT},
        {
            "role": "user",
            "content": f"QUESTION: {question}\nDATA: {data}",
        },
    ]

    response = model.run(messages)
    return response.content


# ---------------------------------------------------------------------
# Public Chain
# ---------------------------------------------------------------------

def sql_chain(question: str) -> str:
    try:
        sql = generate_sql(question)
        df = run_sql(sql)
        return narrate_results(question, df)

    except Exception as e:
        if DEBUG:
            print("[ERROR]", e)
        return "Sorry, I couldn't process your request at the moment."


# ---------------------------------------------------------------------
# Local Testing
# ---------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        "Show top 3 shoes in descending order of rating",
        "Show Puma shoes under 3000",
        "Give me highly rated Nike shoes",
        "List products with discount above 40%",
    ]

    for q in tests:
        print("\n" + "=" * 80)
        print("Query:", q)
        print(sql_chain(q))
