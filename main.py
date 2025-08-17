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
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import re

app = FastAPI()


# Allow CORS from any origin (for dev; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or list of allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== LLM / OpenRouter config ====
OPENROUTER_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6ImRpdnlhZGFyc2hpbmlhYmlAZ21haWwuY29tIn0.2Q_tZIXG62WM5jN4WUcaD2szCv7o9cwDxiK2JrEbu6Y"  # replace with your AIPipe key
BASE_URL = "https://aipipe.org/openrouter/v1"
MODEL = "openai/gpt-3.5-turbo"

# ==== Helper functions ====
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


def parse_questions_file(file_content: str):
    """
    Extract dataset description, wiki URLs, and questions block dynamically.
    """
    lines = file_content.strip().splitlines()
    
    dataset_desc = []
    wiki_urls = []
    questions_block = []
    inside_json = False
    
    for line in lines:
        # capture URLs
        if "http" in line:
            urls = re.findall(r'(https?://\S+)', line)
            wiki_urls.extend(urls)
        # detect JSON block start
        if line.strip().startswith("{"):
            inside_json = True
        if inside_json:
            questions_block.append(line)
        else:
            dataset_desc.append(line)
    
    dataset_description = "\n".join(dataset_desc).strip()
    questions_text = "\n".join(questions_block).strip()
    
    return {
        "dataset_description": dataset_description,
        "wiki_urls": wiki_urls,
        "questions": questions_text
    }


def clean_llm_json(raw: str, max_base64_length=100000) -> str:
    # remove markdown json fences
    raw = re.sub(r"^```(?:json)?\n", "", raw.strip(), flags=re.MULTILINE)
    raw = raw.replace("```", "").strip()
    
    # clean base64
    base64_pattern = r'(data:image\/[a-zA-Z]+;base64,)([A-Za-z0-9+/=\n\r\s]+)'
    def clean_base64(m):
        prefix, data = m.groups()
        clean_data = re.sub(r'\s+', '', data)
        if len(clean_data) > max_base64_length:
            clean_data = clean_data[:max_base64_length]  # truncate to avoid parsing issues
        return prefix + clean_data
    raw = re.sub(base64_pattern, clean_base64, raw)
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
                    "Do NOT include any explanations outside JSON."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 4096
    }
    response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"OpenRouter API Error: {response.status_code} - {response.text}")
    return response.json()["choices"][0]["message"]["content"]


def generate_chart_base64(x, y, xlabel="X", ylabel="Y", title="Chart"):
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64

    plt.figure(figsize=(5,4))
    plt.scatter(x, y, color='skyblue')
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, [m*xi+b for xi in x], 'r--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    return clean_base64_string(b64)

def clean_base64_string(b64: str) -> str:
    """Remove all whitespace/newlines from base64 and return proper data URI."""
    clean_b64 = re.sub(r'\s+', '', b64)
    return f"data:image/png;base64,{clean_b64}"

# ==== FastAPI endpoint ====
@app.post("/api/")
async def analyze(questions: UploadFile = File(...), files: Optional[List[UploadFile]] = File(None)):
    try:
        questions_text = (await questions.read()).decode("utf-8")
        context_files = await summarize_attachments(files) if files else "No additional files provided."
        
        parsed = parse_questions_file(questions_text)
        prompt = f"""
Dataset Description:
{parsed['dataset_description']}

Wikipedia / Reference Links:
{', '.join(parsed['wiki_urls'])}

Attached Files Context:
{context_files}

Questions:
{parsed['questions']}

IMPORTANT:
- Output must be valid JSON containing data and stats ONLY.
- Include any charts as base64 if requested.
"""
        raw_result = call_openrouter_llm(prompt)
        cleaned_result = clean_llm_json(raw_result)
        try:
            parsed_result = json.loads(cleaned_result)
        except json.JSONDecodeError:
            # fallback for debugging
            return JSONResponse(content={"error": "JSON parsing failed", "raw": cleaned_result}, status_code=500)

        # Optionally generate a chart if numeric data present
        if "total_sales" in parsed_result:
            parsed_result["bar_chart"] = generate_chart_base64(
                x=["Total Sales"], y=[parsed_result["total_sales"]],
                xlabel="Metric", ylabel="Value", title="Total Sales"
            )
        
        return JSONResponse(content=parsed_result)

    except Exception as e:
        error_output = locals().get("raw_result", "")
        return JSONResponse(content={"error": str(e), "output": error_output}, status_code=500)
