from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import requests
import json
import re
import matplotlib.pyplot as plt
import base64

app = FastAPI()

OPENROUTER_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6ImRpdnlhZGFyc2hpbmlhYmlAZ21haWwuY29tIn0.2Q_tZIXG62WM5jN4WUcaD2szCv7o9cwDxiK2JrEbu6Y"  # your API key here
BASE_URL = "https://aipipe.org/openrouter/v1"
MODEL = "openai/gpt-3.5-turbo"

async def summarize_attachments(files: List[UploadFile]) -> str:
    summaries = []
    for file in files:
        filename = file.filename
        content = await file.read()
        await file.seek(0)

        if filename.lower().endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content))
                summaries.append(f"CSV '{filename}' sample:\n{df.head(5).to_string(index=False)}")
            except Exception as e:
                summaries.append(f"CSV '{filename}' parse error: {e}")

        elif filename.lower().endswith(".json"):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    summaries.append(f"JSON '{filename}' keys: {list(data.keys())}")
                else:
                    summaries.append(f"JSON '{filename}' list with {len(data)} items")
            except Exception as e:
                summaries.append(f"JSON '{filename}' parse error: {e}")

        else:
            summaries.append(f"File '{filename}' not previewed.")
    return "\n".join(summaries)

def clean_llm_json(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\n", "", raw.strip(), flags=re.MULTILINE)
    raw = raw.replace("```", "").strip()
    raw = re.sub(r'"(\d+)\.\s*', '"', raw)
    raw = raw.replace('""data:image', '"data:image')
    # Remove newlines and whitespace inside base64 strings (if any)
    base64_pattern = r'(data:image\/[a-zA-Z]+;base64,)([A-Za-z0-9+/=\n\r\s]+)'
    def clean_base64(m):
        prefix, data = m.groups()
        clean_data = re.sub(r'\s+', '', data)
        return prefix + clean_data
    raw = re.sub(base64_pattern, clean_base64, raw)
    # Add prefix if missing but looks like base64 (less relevant now)
    raw = re.sub(r'"([A-Za-z0-9+/]{50,}={0,2})"', r'"data:image/png;base64,\1"', raw)
    return raw

def call_openrouter_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a data analyst agent. Return ONLY valid JSON in the exact format requested. "
                    "Do NOT include any images or base64 data. Only data and stats."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 1024
    }
    response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"OpenRouter API Error: {response.status_code} - {response.text}")
    return response.json()["choices"][0]["message"]["content"]

def generate_bar_chart_base64(total_sales: int) -> str:
    # Simple example bar chart with total sales
    plt.figure(figsize=(4,3))
    plt.bar(['Total Sales'], [total_sales], color='skyblue')
    plt.ylabel('Sales')
    plt.title('Total Sales')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"

@app.post("/api/")
async def analyze(questions: UploadFile = File(...), files: Optional[List[UploadFile]] = File(None)):
    try:
        questions_text = (await questions.read()).decode("utf-8")
        context_files = await summarize_attachments(files) if files else "No additional files provided."

        prompt = f"""
Task:
{questions_text}

Context from attached files:
{context_files}

IMPORTANT:
- Output must be valid JSON containing data and stats ONLY.
- Do NOT include images or base64 data.
- Do not add explanations or text outside JSON.
"""

        raw_result = call_openrouter_llm(prompt)
        cleaned_result = clean_llm_json(raw_result)
        parsed_result = json.loads(cleaned_result)

        # Now generate image locally from returned data
        total_sales = parsed_result.get("total_sales", 0)
        bar_chart_b64 = generate_bar_chart_base64(total_sales)

        # Insert generated base64 image string into response JSON
        parsed_result["bar_chart"] = bar_chart_b64

        return JSONResponse(content=parsed_result)

    except Exception as e:
        error_output = locals().get("raw_result", "")
        print(f"Exception while processing request: {e}")
        print(f"Raw LLM output during error:\n{error_output}")
        return JSONResponse(content={"error": str(e), "output": error_output}, status_code=500)
