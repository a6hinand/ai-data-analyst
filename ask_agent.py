import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

df = pd.read_csv("data.csv", encoding="latin-1")
schema = df.dtypes.to_string()


def clean(code):
    return code.replace("```python", "").replace("```", "").strip()


def ask(question, max_tries=3):
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

    for attempt in range(1, max_tries + 1):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
        )
        raw = response.choices[0].message.content
        code = clean(raw)

        print(f"--- Attempt {attempt} ---")
        print(code)

        try:
            namespace = {"df": df.copy(), "pd": pd}
            exec(code, namespace)
            return namespace["result"]
        except Exception as e:
            print("Error:", e)
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That code failed with this error: {e}\nFix the code. Return ONLY code.",
            })

    return "Sorry, I could not answer this after several tries."


question = input("Ask a question about your data: ")
print("--- Answer ---")
print(ask(question))