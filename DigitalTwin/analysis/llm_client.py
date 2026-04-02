"""
LLM Client
===========
Unified thin wrapper around multiple LLM providers.

Supported providers (in recommended order):

* **Gemini (FREE)** -- uses Google's OpenAI-compatible endpoint at
  ``generativelanguage.googleapis.com``.  Get a free API key at
  https://aistudio.google.com (no credit card, 1500 req/day).
  Uses the ``openai`` Python SDK with a custom ``base_url``, so it
  avoids the deprecated ``google-generativeai`` library entirely.
* **Puter (FREE)** -- OpenAI-compatible endpoint at ``api.puter.com``.
  Get a free auth token at https://puter.com/dashboard#account.
  NOTE: may be blocked by corporate/university firewalls.
* **OpenAI** -- requires a paid API key.
* **Anthropic** -- requires a paid API key.

Each provider follows the same interface:
    ``send(system_prompt, messages) -> str``

where *messages* is a list of ``{"role": "user"|"assistant", "content": "..."}``
dicts.

API keys / tokens are supplied at runtime through the dashboard UI and
stored only in memory (never written to disk).
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024

PROVIDER_DEFAULTS: Dict[str, Dict] = {
    "gemini":    {"model": "gemini-2.0-flash",         "label": "Google Gemini (FREE -- aistudio.google.com)"},
    "puter":     {"model": "claude-sonnet-4-5",        "label": "Puter (FREE -- GPT/Claude/Gemini)"},
    "openai":    {"model": "gpt-4o-mini",              "label": "OpenAI GPT (paid)"},
    "anthropic": {"model": "claude-3-5-haiku-latest",  "label": "Anthropic Claude (paid)"},
}

_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_PUTER_BASE_URL = "https://api.puter.com/puterai/openai/v1/"

_GEMINI_FALLBACK_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


class LLMClient:
    """Provider-agnostic LLM caller."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self._cfg = config or LLMConfig()

    @property
    def config(self) -> LLMConfig:
        return self._cfg

    def configure(self, provider: str, api_key: str,
                  model: str = "", temperature: float = 0.7):
        self._cfg.provider = provider
        self._cfg.api_key = api_key
        self._cfg.model = model or PROVIDER_DEFAULTS.get(provider, {}).get("model", "")
        self._cfg.temperature = temperature

    def send(self, system_prompt: str,
             messages: List[Dict[str, str]]) -> str:
        """Send conversation to the configured LLM and return the response."""
        if not self._cfg.api_key:
            return ("[Error] No API key configured.  "
                    "For FREE access: select 'Google Gemini' and paste "
                    "your key from https://aistudio.google.com")

        provider = self._cfg.provider.lower()
        dispatch = {
            "gemini": self._call_gemini_openai,
            "puter": self._call_puter,
            "openai": self._call_openai,
            "anthropic": self._call_anthropic,
        }
        handler = dispatch.get(provider)
        if handler is None:
            return f"[Error] Unknown provider: {provider}"
        try:
            return handler(system_prompt, messages)
        except Exception as exc:
            return f"[Error] {provider} API call failed: {exc}"

    # ------------------------------------------------------------------
    # Helper: OpenAI-compatible call (shared by Gemini, Puter, OpenAI)
    # ------------------------------------------------------------------
    def _call_openai_compat(self, base_url: Optional[str],
                            api_key: str, default_model: str,
                            system_prompt: str,
                            messages: List[Dict[str, str]]) -> str:
        openai = importlib.import_module("openai")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = openai.OpenAI(**kwargs)

        api_msgs = [{"role": "system", "content": system_prompt}]
        api_msgs.extend(messages)

        resp = client.chat.completions.create(
            model=self._cfg.model or default_model,
            messages=api_msgs,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
        )
        return resp.choices[0].message.content

    # ------------------------------------------------------------------
    # Google Gemini (FREE) via OpenAI-compatible endpoint
    # With automatic fallback to other Gemini models on 429 quota errors
    # ------------------------------------------------------------------
    def _call_gemini_openai(self, system_prompt: str,
                            messages: List[Dict[str, str]]) -> str:
        openai_mod = importlib.import_module("openai")
        RateLimitError = openai_mod.RateLimitError

        primary = self._cfg.model or "gemini-2.0-flash"
        models_to_try = [primary] + [m for m in _GEMINI_FALLBACK_MODELS if m != primary]
        last_err = None

        for model_name in models_to_try:
            try:
                saved = self._cfg.model
                self._cfg.model = model_name
                result = self._call_openai_compat(
                    _GEMINI_OPENAI_BASE_URL, self._cfg.api_key,
                    model_name, system_prompt, messages)
                self._cfg.model = saved
                if model_name != primary:
                    return f"[Used {model_name} -- {primary} quota exceeded]\n\n{result}"
                return result
            except RateLimitError as exc:
                last_err = exc
                self._cfg.model = primary
                continue
            except Exception:
                self._cfg.model = primary
                raise

        return (f"[Error] All Gemini models quota-exhausted for today.  "
                f"Daily quota resets at midnight Pacific Time.  "
                f"Last error: {last_err}")

    # ------------------------------------------------------------------
    # Puter (FREE) -- OpenAI-compatible endpoint
    # ------------------------------------------------------------------
    def _call_puter(self, system_prompt: str,
                    messages: List[Dict[str, str]]) -> str:
        return self._call_openai_compat(
            _PUTER_BASE_URL, self._cfg.api_key,
            "claude-sonnet-4-5", system_prompt, messages)

    # ------------------------------------------------------------------
    # OpenAI (paid)
    # ------------------------------------------------------------------
    def _call_openai(self, system_prompt: str,
                     messages: List[Dict[str, str]]) -> str:
        return self._call_openai_compat(
            None, self._cfg.api_key,
            "gpt-4o-mini", system_prompt, messages)

    # ------------------------------------------------------------------
    # Anthropic (paid)
    # ------------------------------------------------------------------
    def _call_anthropic(self, system_prompt: str,
                        messages: List[Dict[str, str]]) -> str:
        anthropic = importlib.import_module("anthropic")
        client = anthropic.Anthropic(api_key=self._cfg.api_key)

        resp = client.messages.create(
            model=self._cfg.model or "claude-3-5-haiku-latest",
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            system=system_prompt,
            messages=messages,
        )
        return resp.content[0].text
