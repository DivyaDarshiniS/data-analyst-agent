from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional, Union
import pandas as pd
import io
import requests
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import base64
from fastapi.middleware.cors import CORSMiddleware

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
OPENROUTER_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6ImRpdnlhZGFyc2hpbmlhYmlAZ21haWwuY29tIn0.2Q_tZIXG62WM5jN4WUcaD2szCv7o9cwDxiK2JrEbu6Y"   # replace with your API key
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
    lines = file_content.strip().splitlines()
    dataset_desc, wiki_urls, questions_block = [], [], []
    inside_json = False

    for line in lines:
        if "http" in line:
            urls = re.findall(r'(https?://\S+)', line)
            wiki_urls.extend(urls)
        if line.strip().startswith("{") or line.strip().startswith("["):
            inside_json = True
        if inside_json:
            questions_block.append(line)
        else:
            dataset_desc.append(line)

    dataset_description = "\n".join(dataset_desc).strip()
    questions_text = "\n".join(questions_block).strip()
    return {"dataset_description": dataset_description, "wiki_urls": wiki_urls, "questions": questions_text}

def clean_llm_json(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\n", "", raw.strip(), flags=re.MULTILINE)
    raw = raw.replace("```", "").strip()
    base64_pattern = r'(data:image\/[a-zA-Z]+;base64,)([A-Za-z0-9+/=\n\r\s]+)'
    def clean_base64(m): return m.group(1) + re.sub(r'\s+', '', m.group(2))
    return re.sub(base64_pattern, clean_base64, raw)

def call_openrouter_llm(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a data analyst agent. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 2048
    }
    response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"OpenRouter API Error: {response.status_code} - {response.text}")
    return response.json()["choices"][0]["message"]["content"]

def generate_chart_base64(x, y, xlabel="X", ylabel="Y", title="Chart"):
    plt.figure(figsize=(5,4))
    plt.scatter(x, y, color='skyblue')
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, [m*xi + b for xi in x], 'r--')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def normalize_base64_images(parsed_result):
    """
    Ensure all base64 outputs are valid data URIs.
    If a value looks like base64, validate & fix it.
    """
    prefix = "data:image/png;base64,"

    def fix_base64_string(val: str) -> str:
        # Remove whitespace/newlines
        cleaned = re.sub(r"\s+", "", val)

        # If already has prefix, strip and keep only base64 part
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

        # Validate by trying to decode/encode again
        try:
            decoded = base64.b64decode(cleaned, validate=True)
            reencoded = base64.b64encode(decoded).decode("utf-8")
            return prefix + reencoded
        except Exception:
            # Not valid base64, just return as-is
            return val

    if isinstance(parsed_result, dict):
        for k, v in parsed_result.items():
            if isinstance(v, str) and ("iVBOR" in v or "base64" in v):
                parsed_result[k] = fix_base64_string(v)
    elif isinstance(parsed_result, list):
        parsed_result = [
            fix_base64_string(v) if isinstance(v, str) and ("iVBOR" in v or "base64" in v) else v
            for v in parsed_result
        ]
    return parsed_result



# ==== FastAPI endpoint ====
from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse
from typing import List, Optional, Union
import json

@app.post("/api/")
async def analyze(
    questions: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    questions_text: Optional[str] = Body(None)  # allow JSON/raw text instead of file
):
    try:
        if questions is not None:
            # read from uploaded file
            questions_text = (await questions.read()).decode("utf-8")
        elif questions_text is not None:
            # already sent in JSON body
            if isinstance(questions_text, dict):
                # if JSON object, turn into pretty string
                questions_text = json.dumps(questions_text, indent=2)
        else:
            return JSONResponse(
                {"error": "No questions provided (need file or JSON body)"},
                status_code=400
            )

        # ... (rest of your pipeline: parse_questions_file, call LLM, etc.)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
