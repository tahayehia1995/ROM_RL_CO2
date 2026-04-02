"""
Chat Manager
=============
Session-scoped conversation manager that ties together the
:class:`ContextBuilder` and :class:`LLMClient`.

Responsibilities:
- Maintains the conversation history (list of user/assistant messages).
- Automatically injects the latest reservoir state context into every
  user message sent to the LLM.
- Clears history on simulation reset.
- Provides serialisable history for the Dash ``dcc.Store``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from .context_builder import ContextBuilder
from .llm_client import LLMClient

if TYPE_CHECKING:
    from DigitalTwin.engine.digital_twin_engine import DigitalTwinEngine


class ChatManager:
    """Manages a single chat session between the user and an LLM."""

    def __init__(self, engine: "DigitalTwinEngine"):
        self._engine = engine
        self._context = ContextBuilder(engine)
        self._llm = LLMClient()
        self._history: List[Dict[str, str]] = []

    @property
    def llm(self) -> LLMClient:
        return self._llm

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def configure_llm(self, provider: str, api_key: str,
                      model: str = "", temperature: float = 0.7):
        self._llm.configure(provider, api_key, model, temperature)

    def send(self, user_text: str) -> str:
        """Send a user message, get an LLM response, and record both."""
        enriched = self._context.build_user_message(user_text)
        self._history.append({"role": "user", "content": user_text})

        system = self._context.build_system_prompt()

        api_messages = []
        for msg in self._history[:-1]:
            api_messages.append(msg)
        api_messages.append({"role": "user", "content": enriched})

        reply = self._llm.send(system, api_messages)
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        """Clear conversation history (called on simulation reset)."""
        self._history.clear()

    def history_for_display(self) -> List[Dict[str, str]]:
        """Return history in a format suitable for UI rendering."""
        return list(self._history)
