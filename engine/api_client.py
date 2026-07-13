"""OpenAI-compatible API client for DeepSeek with logprobs (imported by query, l2_neural_gate, main).

Uses urllib (stdlib) to POST to DeepSeek /v1/chat/completions.
API key: DEEPSEEK_API_KEY → ANTHROPIC_AUTH_TOKEN → ANTHROPIC_API_KEY → OPENAI_API_KEY → settings.json
No data files written — returns API response dicts.
"""

from __future__ import annotations

import json, os, time, urllib.request, urllib.error
from pathlib import Path
from typing import Optional

from config.defaults import (
    DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_TEMPERATURE, DEEPSEEK_TIMEOUT_S, DEEPSEEK_MAX_RETRIES,
    DEEPSEEK_RETRY_DELAY_S, API_KEY_ENV_VARS,
)


def get_api_key() -> Optional[str]:
    for var in API_KEY_ENV_VARS:
        key = os.environ.get(var, "").strip()
        if key:
            return key
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            cfg = json.loads(settings_path.read_text(encoding="utf-8"))
            for var in ("DEEPSEEK_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
                key = cfg.get("env", {}).get(var, "").strip()
                if key:
                    return key
        except (json.JSONDecodeError, OSError):
            pass
    return None


def call_api(
    messages: list[dict],
    *,
    api_key: Optional[str] = None,
    model: str = DEEPSEEK_MODEL,
    base_url: str = DEEPSEEK_BASE_URL,
    max_tokens: int = DEEPSEEK_MAX_TOKENS,
    temperature: float = DEEPSEEK_TEMPERATURE,
    logprobs: bool = False,
    top_logprobs: int = 20,
    tools: Optional[list[dict]] = None,
    timeout_s: int = DEEPSEEK_TIMEOUT_S,
    max_retries: int = DEEPSEEK_MAX_RETRIES,
) -> Optional[dict]:
    key = api_key or get_api_key()
    if not key:
        raise RuntimeError("No DeepSeek API key found. Set DEEPSEEK_API_KEY env var.")

    url = f"{base_url.rstrip('/')}/chat/completions"
    body: dict = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs
    if tools:
        body["tools"] = tools

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"), headers=headers,
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                body_text = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                body_text = ""
            last_error = f"HTTP {e.code}: {body_text}"
        except Exception as e:
            last_error = str(e)
        if attempt < max_retries:
            time.sleep(DEEPSEEK_RETRY_DELAY_S)

    print(f"[api_client] Failed after {max_retries+1} attempts: {last_error}")
    return None


def extract_logprobs(response: dict) -> Optional[list[dict]]:
    try:
        return response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def extract_text(response: dict) -> str:
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def extract_tool_calls(response: dict) -> list[dict]:
    try:
        return response["choices"][0]["message"].get("tool_calls", [])
    except (KeyError, IndexError, TypeError):
        return []


def extract_usage(response: dict) -> dict:
    try:
        return response.get("usage", {})
    except (KeyError, TypeError):
        return {}


def quick_chat(
    system_prompt: str, user_prompt: str, *,
    api_key: Optional[str] = None, logprobs: bool = False, **kwargs,
) -> Optional[dict]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return call_api(messages, api_key=api_key, logprobs=logprobs, **kwargs)
