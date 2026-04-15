"""Vector knowledge index: ingest Markdown and plain text from paths for review-agent RAG."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from reviewagent.config import get_settings

logger = logging.getLogger(__name__)

# reviewagent/rag/store.py -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}

_cached: Optional[tuple[Any, Any]] = None
_cache_lock = threading.Lock()


def _resolve_repo_path(p: Union[str, Path]) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def invalidate_knowledge_cache() -> None:
    global _cached
    with _cache_lock:
        _cached = None


def _read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1", errors="replace")


def _chunk_texts(
    texts: Sequence[str],
    metadatas: Optional[Sequence[dict]] = None,
) -> tuple[list[str], list[dict]]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    settings = get_settings()
    rag = settings.rag
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag.chunk_size,
        chunk_overlap=rag.chunk_overlap,
        length_function=len,
    )
    out_docs: list[str] = []
    out_meta: list[dict] = []
    for i, text in enumerate(texts):
        meta = dict(metadatas[i]) if metadatas and i < len(metadatas) else {}
        chunks = splitter.split_text(text)
        for j, ch in enumerate(chunks):
            out_docs.append(ch)
            m = {**meta, "chunk": j}
            out_meta.append(m)
    return out_docs, out_meta


def _max_ingest_bytes() -> int:
    kb = get_settings().rag.max_ingest_file_kb
    return 0 if kb <= 0 else kb * 1024


def _collect_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = raw.resolve()
        if p.is_file():
            if p.suffix.lower() in TEXT_SUFFIXES:
                files.append(p)
        elif p.is_dir():
            for sub in sorted(p.rglob("*")):
                if sub.is_file() and sub.suffix.lower() in TEXT_SUFFIXES:
                    files.append(sub)
    return files


def _get_vectorstore():
    from langchain_community.vectorstores import Chroma

    from reviewagent.adapters.llm_factory import create_embeddings_model

    settings = get_settings()
    rag = settings.rag
    emb = create_embeddings_model()
    if emb is None:
        raise RuntimeError("无法创建嵌入模型（请检查 rag.embedding_provider 与 llm 密钥）")
    persist = _resolve_repo_path(rag.persist_directory)
    persist.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(persist),
        embedding_function=emb,
        collection_name=rag.collection_name,
    )


def get_knowledge_retriever() -> Optional[Any]:
    """Return a LangChain Retriever, or None if RAG is disabled or initialization fails."""
    settings = get_settings()
    if not settings.rag.enabled:
        return None
    global _cached
    with _cache_lock:
        if _cached is not None:
            return _cached[1]
        try:
            vs = _get_vectorstore()
            retriever = vs.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.rag.retrieve_k},
            )
            _cached = (vs, retriever)
            return retriever
        except Exception:
            logger.exception("知识库向量索引初始化失败，本进程内 RAG 不可用")
            return None


def retrieve_knowledge_context(query: str, *, max_query_chars: int = 6000) -> str:
    """Retrieve for ``query``; return plain text to merge into the system message, or empty string."""
    if not query or not query.strip():
        return ""
    retriever = get_knowledge_retriever()
    if retriever is None:
        return ""
    q = query.strip()[:max_query_chars]
    try:
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(q)
        else:
            docs = retriever.get_relevant_documents(q)
    except Exception:
        logger.exception("知识库检索失败")
        return ""
    if not docs:
        return ""
    parts: list[str] = []
    for i, d in enumerate(docs, 1):
        src = ""
        if getattr(d, "metadata", None):
            src = str(d.metadata.get("source") or d.metadata.get("path") or "")
        body = (d.page_content or "").strip()
        if not body:
            continue
        if src:
            parts.append(f"[{i}] 来源: {src}\n{body}")
        else:
            parts.append(f"[{i}]\n{body}")
    return "\n\n".join(parts)


def ingest_paths(paths: Sequence[Union[str, Path]]) -> int:
    """
    Chunk .txt/.md under files or directories into Chroma. Returns number of text chunks written.
    """
    settings = get_settings()
    if not settings.rag.enabled:
        raise RuntimeError("请先在 config.yaml 中设置 rag.enabled: true")
    path_objs = [_resolve_repo_path(p) for p in paths]
    files = _collect_files(path_objs)
    if not files:
        return 0
    max_b = _max_ingest_bytes()
    texts: list[str] = []
    metas: list[dict] = []
    for fp in files:
        if max_b > 0 and fp.stat().st_size > max_b:
            logger.warning("跳过过大文件（>max_ingest_file_kb）: %s", fp)
            continue
        try:
            content = _read_text_file(fp)
        except OSError as e:
            logger.warning("无法读取 %s: %s", fp, e)
            continue
        texts.append(content)
        try:
            src_rel = str(fp.relative_to(REPO_ROOT))
        except ValueError:
            src_rel = str(fp)
        metas.append({"source": src_rel})

    if not texts:
        return 0
    chunks, chunk_metas = _chunk_texts(texts, metas)
    vs = _get_vectorstore()
    vs.add_texts(texts=chunks, metadatas=chunk_metas)
    try:
        vs.persist()
    except Exception:
        pass
    invalidate_knowledge_cache()
    logger.info("知识库已索引 %s 个文件，共 %s 个文本块", len(files), len(chunks))
    return len(chunks)


def ingest_configured_directories() -> int:
    """Ingest paths from config.rag.knowledge_dirs (relative to repo root)."""
    settings = get_settings()
    dirs = settings.rag.knowledge_dirs or []
    if not dirs:
        return 0
    resolved = [_resolve_repo_path(d) for d in dirs]
    total = 0
    for d in resolved:
        if d.is_dir():
            total += ingest_paths([d])
        elif d.is_file():
            total += ingest_paths([d])
    return total


def clear_knowledge_index() -> None:
    """Drop the vector collection (requires re-ingest)."""
    settings = get_settings()
    rag = settings.rag
    persist = _resolve_repo_path(rag.persist_directory)
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist))
        try:
            client.delete_collection(rag.collection_name)
        except Exception:
            pass
    except Exception:
        logger.exception("清空 Chroma 集合时出错")
    invalidate_knowledge_cache()
