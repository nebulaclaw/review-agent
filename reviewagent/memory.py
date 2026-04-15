import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage

from reviewagent.config import get_settings


class ShortTermMemory:
    """Short-term chat history via langchain_core (LangChain 1.x; no langchain.schema / ConversationBufferMemory)."""

    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self._history = InMemoryChatMessageHistory()

    def _trim(self) -> None:
        cap = max(1, self.max_messages) * 2
        msgs = self._history.messages
        while len(msgs) > cap:
            msgs.pop(0)

    def add_user_message(self, message: str) -> None:
        self._history.add_user_message(message)
        self._trim()

    def add_ai_message(self, message: str) -> None:
        self._history.add_ai_message(message)
        self._trim()

    def get_messages(self) -> list[BaseMessage]:
        return list(self._history.messages)

    def load_memory_variables(self) -> dict:
        return {"history": self.get_messages()}

    def save_context(self, input_dict: dict, output_dict: dict) -> None:
        inp = input_dict.get("input")
        if inp is not None:
            self.add_user_message(str(inp))
        out = output_dict.get("output")
        if out is not None:
            self.add_ai_message(str(out))

    def clear(self) -> None:
        self._history.clear()

    def get_messages_count(self) -> int:
        return len(self._history.messages)


class LongTermMemory:
    def __init__(self, persist_directory: str = "data/memory", collection_name: str = "memory"):
        self.enabled = False
        self.memories = []

    def add_memory(self, content: str, metadata: dict = None):
        if self.enabled:
            self.memories.append({"content": content, "metadata": metadata or {}})

    def search(self, query: str) -> list[str]:
        return []

    def clear(self):
        self.memories.clear()

    def as_retriever(self):
        return None


class UnifiedMemory:
    def __init__(self):
        settings = get_settings()

        self.short_term = ShortTermMemory(
            max_messages=settings.memory.short_term_max_messages
        )

        if settings.memory.long_term_enabled and not getattr(
            settings, "offline_mode", False
        ):
            try:
                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings import OpenAIEmbeddings

                self.embeddings = OpenAIEmbeddings()
                self.vectorstore = Chroma(
                    persist_directory=settings.memory.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="memory",
                )
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3},
                )
                self.long_term_enabled = True
            except Exception:
                self.long_term_enabled = False
                self.vectorstore = None
                self.retriever = None
        else:
            self.long_term_enabled = False
            self.vectorstore = None
            self.retriever = None

    def add_turn(self, user_message: str, ai_message: str):
        self.short_term.add_user_message(user_message)
        self.short_term.add_ai_message(ai_message)

        if self.long_term_enabled and self.vectorstore:
            doc = Document(
                page_content=f"User: {user_message}\nAI: {ai_message}",
                metadata={"type": "conversation"},
            )
            self.vectorstore.add_documents([doc])

    def get_context(self) -> list[BaseMessage]:
        messages = self.short_term.get_messages()

        if self.long_term_enabled and self.retriever:
            last_message = messages[-1].content if messages else ""
            if last_message:
                try:
                    docs = self.retriever.get_relevant_documents(last_message)
                    if docs:
                        relevant_memories = [doc.page_content for doc in docs]
                        system_msg = SystemMessage(
                            content=f"Relevant historical context:\n{chr(10).join(relevant_memories)}"
                        )
                        return [system_msg] + messages
                except Exception:
                    pass

        return messages

    def clear(self):
        self.short_term.clear()
        if self.vectorstore:
            try:
                self.vectorstore.delete_collection()
            except Exception:
                pass


_sessions_lock = threading.Lock()
_sessions: Dict[str, UnifiedMemory] = {}

_staging_lock = threading.Lock()
# Server-side paths from /v1/review/file kept until session DELETE (multi-turn re-analysis).
_session_review_staging_paths: Dict[str, List[str]] = {}


def _unlink_review_staging_paths(paths: Iterable[str]) -> None:
    for tmp_path in paths:
        try:
            p = Path(tmp_path)
            if p.exists():
                p.unlink()
            parent = p.parent
            if parent.name.startswith("review_upload_"):
                parent.rmdir()
        except OSError:
            pass


def register_session_review_staging_paths(session_id: str, paths: List[str]) -> None:
    """Replace any prior staged uploads for this session, then own ``paths`` until cleared."""
    sid = str(session_id).strip()
    if not sid:
        return
    with _staging_lock:
        prev = _session_review_staging_paths.pop(sid, None)
    if prev:
        _unlink_review_staging_paths(prev)
    with _staging_lock:
        _session_review_staging_paths[sid] = list(paths)


def clear_session_review_staging(session_id: str) -> None:
    sid = str(session_id).strip()
    if not sid:
        return
    with _staging_lock:
        paths = _session_review_staging_paths.pop(sid, None)
    if paths:
        _unlink_review_staging_paths(paths)


def get_session_review_staging_paths(session_id: str) -> List[str]:
    sid = str(session_id).strip()
    if not sid:
        return []
    with _staging_lock:
        return list(_session_review_staging_paths.get(sid, []))


def get_memory(session_id: Optional[str] = None) -> UnifiedMemory:
    """
    Non-empty session_id: reuse the same UnifiedMemory per id (multi-turn).
    Empty session_id: a fresh instance each time (one-shot review or legacy paths).
    """
    if not session_id or not str(session_id).strip():
        return UnifiedMemory()
    sid = str(session_id).strip()
    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = UnifiedMemory()
        return _sessions[sid]


def clear_session_memory(session_id: str) -> None:
    sid = str(session_id).strip()
    if not sid:
        return
    clear_session_review_staging(sid)
    with _sessions_lock:
        mem = _sessions.pop(sid, None)
    if mem is not None:
        mem.clear()
