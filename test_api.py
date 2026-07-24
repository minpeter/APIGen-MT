#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv
import requests
import json
from runtime_config import DEFAULT_API_BASE, DEFAULT_MODEL

load_dotenv()

base_url = os.getenv("OPENAI_API_BASE", DEFAULT_API_BASE)
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("API_MODEL", DEFAULT_MODEL)

payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with just the word 'hello'"}],
    "max_tokens": 20,
    "temperature": 0.7,
}

try:
    print(f"Testing {model} at {base_url}...")
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    print(f"Status: {resp.status_code}")
    data = resp.json()
    if "choices" in data:
        content = data["choices"][0]["message"]["content"]
        print(f"SUCCESS: {content[:100]}")
    else:
        print(f"FAILED: {data}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
