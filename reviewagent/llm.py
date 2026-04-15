"""Backward-compatible entry: unified LLM factory lives in adapters."""

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from reviewagent.adapters.llm_factory import create_chat_model

__all__ = [
    "create_chat_model",
    "BaseChatModel",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
]
