"""Provider-agnostic LLM client.

Speaks two wire formats:
  * OpenAI-compatible chat completions (openrouter / openai / custom)
  * Native Anthropic Messages API      (anthropic)

All calls are temperature-0 and expect the model to answer with a single
JSON object; parsing is defensive (strips code fences, finds outermost {}).
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field

import requests

from .config import LLMConfig


@dataclass
class LLMUsage:
    calls: int = 0
    vision_calls: int = 0
    errors: list[str] = field(default_factory=list)


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.usage = LLMUsage()

    # ------------------------------------------------------------- public
    def ask_json(self, prompt: str, system: str = "", images_png: list[bytes] | None = None) -> dict:
        """Send a prompt (optionally with PNG images) and parse a JSON object reply."""
        model = self.cfg.vision_model if images_png else self.cfg.text_model
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                text = self._call(model, prompt, system, images_png or [])
                obj = _extract_json(text)
                self.usage.calls += 1
                if images_png:
                    self.usage.vision_calls += 1
                return obj
            except Exception as e:  # noqa: BLE001 — retry any transport/parse error
                last_err = e
                self.usage.errors.append(f"{type(e).__name__}: {e}")
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM call failed after {self.cfg.max_retries} attempts: {last_err}")

    # ------------------------------------------------------------ private
    def _call(self, model: str, prompt: str, system: str, images: list[bytes]) -> str:
        if self.cfg.provider == "anthropic":
            return self._call_anthropic(model, prompt, system, images)
        return self._call_openai_compat(model, prompt, system, images)

    def _call_openai_compat(self, model: str, prompt: str, system: str, images: list[bytes]) -> str:
        content: list[dict] | str
        if images:
            content = [{"type": "text", "text": prompt}] + [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()},
                }
                for img in images
            ]
        else:
            content = prompt
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        resp = requests.post(
            self.cfg.resolved_base_url() + "/chat/completions",
            headers={
                "Authorization": f"Bearer {self.cfg.api_key()}",
                "Content-Type": "application/json",
                # OpenRouter attribution headers (harmless elsewhere)
                "HTTP-Referer": "https://local.pipeline/narrative-director",
                "X-Title": "AI Narrative Video Director",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            },
            timeout=self.cfg.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"Unexpected LLM response: {json.dumps(data)[:500]}")
        return data["choices"][0]["message"]["content"] or ""

    def _call_anthropic(self, model: str, prompt: str, system: str, images: list[bytes]) -> str:
        content: list[dict] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(img).decode()},
            }
            for img in images
        ]
        content.append({"type": "text", "text": prompt})
        body = {
            "model": model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            body["system"] = system
        resp = requests.post(
            self.cfg.resolved_base_url() + "/v1/messages",
            headers={
                "x-api-key": self.cfg.api_key(),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.cfg.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []))


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of an LLM reply, tolerating fences/preamble."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object found in LLM reply: {text[:300]!r}")
