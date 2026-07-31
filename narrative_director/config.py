"""Configuration loading. All tunables live in config.yaml; API keys in env/.env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROVIDER_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}


@dataclass
class LLMConfig:
    provider: str = "openrouter"
    base_url: str | None = None
    api_key_env: str = "OPENROUTER_API_KEY"
    text_model: str = "anthropic/claude-sonnet-4.5"
    vision_model: str = "anthropic/claude-sonnet-4.5"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 3
    timeout_s: int = 120

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        try:
            return PROVIDER_BASE_URLS[self.provider]
        except KeyError:
            raise ValueError(
                f"Unknown llm.provider '{self.provider}'. "
                f"Use one of {list(PROVIDER_BASE_URLS)} or 'custom' with base_url set."
            )

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"API key env var '{self.api_key_env}' is not set. "
                f"Export it or add it to a .env file next to config.yaml."
            )
        return key


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    transcription: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    editing: dict[str, Any] = field(default_factory=dict)
    hitl: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    root: Path = field(default_factory=Path.cwd)

    @property
    def cache_dir(self) -> Path:
        d = self.root / self.output.get("cache_dir", ".cache")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def output_dir(self) -> Path:
        d = self.root / self.output.get("dir", "output")
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path).resolve()
    load_dotenv(path.parent / ".env")
    raw = yaml.safe_load(path.read_text()) or {}
    llm_raw = raw.get("llm", {}) or {}
    llm = LLMConfig(**{k: v for k, v in llm_raw.items() if k in LLMConfig.__dataclass_fields__})
    return Config(
        llm=llm,
        transcription=raw.get("transcription", {}) or {},
        analysis=raw.get("analysis", {}) or {},
        editing=raw.get("editing", {}) or {},
        hitl=raw.get("hitl", {}) or {},
        output=raw.get("output", {}) or {},
        root=path.parent,
    )
