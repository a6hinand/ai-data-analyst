import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def clean(code):
    return code.replace("```python", "").replace("```", "").strip()


BLOCKED = [
    r"\bimport\b", r"\bos\b", r"\bsys\b", r"\bsubprocess\b",
    r"\bopen\b", r"\beval\b", r"\bexec\b", r"\bcompile\b",
    r"\bgetattr\b", r"\bglobals\b", r"__", r"read_",
    r"to_csv", r"to_excel", r"to_json", r"to_pickle",
    r"to_sql", r"to_parquet",
]

SAFE_BUILTINS = {
    "len": len, "sum": sum, "min": min, "max": max, "round": round,
    "sorted": sorted, "range": range, "abs": abs, "list": list,
    "dict": dict, "set": set, "str": str, "int": int, "float": float,
    "bool": bool, "tuple": tuple, "zip": zip, "enumerate": enumerate,
    "isinstance": isinstance,
}


def is_safe(code):
    return not any(re.search(p, code) for p in BLOCKED)


def ask(df, question, max_tries=3):
    schema = df.dtypes.to_string()
    prompt = f"""You are a data analyst. A pandas DataFrame named df is already loaded.
Columns and types:
{schema}

Write Python code to answer this question: {question}
Rules:
- Use only pandas (pd) and df.
- Store the final answer in a variable named result.
- If the question needs a column that does not exist in df, do NOT use a different column. Set result = "The data has no column for that."
- Return ONLY code, no explanation, no markdown fences."""

    messages = [{"role": "user", "content": prompt}]
    last_code = ""

    for attempt in range(1, max_tries + 1):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
        )
        raw = response.choices[0].message.content
        code = clean(raw)
        last_code = code
        try:
            if not is_safe(code):
                raise ValueError("Code uses something not allowed. Use only pandas and df.")
            namespace = {"df": df.copy(), "pd": pd, "__builtins__": SAFE_BUILTINS}
            exec(code, namespace)
            return namespace["result"], code, attempt
        except Exception as e:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That code failed with this error: {e}\nFix the code. Return ONLY code.",
            })

    return "Sorry, I could not answer this after several tries.", last_code, max_tries


st.title("AI Data Analyst")

file = st.file_uploader("Upload a CSV file", type="csv")

if file is not None:
    df = pd.read_csv(file, encoding="latin-1")
    st.write("Preview of your data:")
    st.dataframe(df.head())

    question = st.text_input("Ask a question about your data")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            result, code, attempts = ask(df, question)
        st.subheader("Answer")
        st.write(result)
        if isinstance(result, (pd.Series, pd.DataFrame)) and len(result) > 1:
            st.bar_chart(result)
        st.caption(f"Answered in {attempts} attempt(s)")
        with st.expander("Show the code the AI wrote"):
            st.code(code, language="python")