"""Multi-agent coordination: parallel specialist sub-agents → judge synthesis.

Pattern
-------
1. Parallel sub-agents each analyze ONE text surface (ASR / subtitle / OCR)
   via direct LLM invocation (no tools, no memory writes).
2. A single Judge agent receives all sub-verdicts and emits the final binding
   decision, optionally escalating disagreements.

Usage example::

    tasks = [
        SubAgentTask("audio_asr",   "请审核以下语音转写内容：\\n" + asr_text),
        SubAgentTask("subtitle",    "请审核以下字幕文本：\\n" + subtitle),
    ]
    sub_results = await run_sub_agents_parallel(tasks, orchestrator)
    final = await run_judge_agent(sub_results, orchestrator, context=video_path)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from reviewagent.agent import ReviewOrchestrator

logger = logging.getLogger(__name__)
_MC = "[review.core]"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubAgentTask:
    """One specialist task to run in parallel."""
    name: str           # e.g. "audio_asr", "subtitle", "visual_ocr", "visual_frames"
    prompt: str         # full user-facing prompt for this modality
    content_type: str = "text"
    images: list[str] = field(default_factory=list)  # base64 data-URLs for vision LLMs


@dataclass
class SubAgentResult:
    """Structured output from a single specialist sub-agent."""
    name: str
    verdict: str                           # PASS | WARN | BLOCK | UNKNOWN
    confidence: float
    violations: list[dict[str, Any]]
    summary: str
    raw_response: str
    detail: str = ""                       # Detailed analysis description from the sub-agent
    error: Optional[str] = None
    duration_ms: float = 0.0
    skipped: bool = False


# ---------------------------------------------------------------------------
# Sub-agent system prompt
# ---------------------------------------------------------------------------

_SUB_AGENT_SYSTEM = """你是专注于单一内容类型的内容安全分析子 Agent。
你的唯一任务是对用户提供的文本做内容安全审核，然后**只输出一段合法 JSON**（不含 Markdown 围栏）：
{
  "verdict": "PASS|WARN|BLOCK",
  "confidence": 0.0~1.0,
  "violations": [{"type":"...","content":"...","severity":"high|medium|low"}],
  "summary": "一句话结论（≤40字）",
  "detail": "详细分析：内容实际情况描述、检测发现、判断依据（50~150字）"
}
- verdict 仅限 PASS / WARN / BLOCK / UNKNOWN
- violations 无命中时为 []
- detail 即使 PASS 也须填写，说明内容的实际情况（方便 Judge 综合研判）
- 不要工具调用，不要 Markdown 围栏，不要多余输出"""

# ---------------------------------------------------------------------------
# Judge system prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """你是内容审核仲裁 Agent（Judge）。
你将收到若干子 Agent 对同一媒体内容各模态的审核结论，你的职责是综合所有子结论，输出**最终绑定性裁决**（一段 JSON，不含 Markdown 围栏）：
{
  "verdict": "PASS|WARN|BLOCK",
  "confidence": 0.0~1.0,
  "violations": [...合并所有子结论的 violations...],
  "summary": "一句话最终结论（≤60字）",
  "reasoning": "详细推理：各模态分别检测到了什么内容 → 哪些存在风险 → 综合权衡后为何给出此裁决（100~300字）",
  "modality_analysis": {
    "<子Agent名称>": "该模态实际检测内容及结论描述（中文）"
  },
  "risk_level": "none|low|medium|high",
  "recommendations": "处置建议（1~2句话，PASS时说明可正常发布，WARN时建议人工复核，BLOCK时建议拦截下架）"
}

裁决规则（优先级从高到低）：
- 任意子 Agent BLOCK → 最终大概率 BLOCK（除非有充分降级理由）
- 多个子 Agent WARN → 升为 BLOCK 或保持 WARN
- 全部 PASS → 最终 PASS；若子结论间严重不一致则 WARN

字段规则：
- violations：取各子结论并集，去除完全重复项
- confidence：取加权均值（视觉/音频权重较高）
- risk_level：BLOCK→high，WARN→medium，PASS 且轻微疑虑→low，PASS 无任何疑虑→none
- modality_analysis：每个 key 为子 Agent 名称，value 用简洁中文描述该模态的实际内容和判断
- 不要输出多余文字，仅输出以上 JSON"""

_VISUAL_SUB_AGENT_SYSTEM = """你是专注于**视频画面**的内容安全分析子 Agent。
你将收到若干视频关键帧图片，任务是：
1. 逐一观察每张画面，判断是否含有违规视觉内容（色情/裸露、暴力/血腥、恐怖图像、仇恨符号、违禁物品等）
2. **只输出一段合法 JSON**（不含 Markdown 围栏）：
{
  "verdict": "PASS|WARN|BLOCK",
  "confidence": 0.0~1.0,
  "violations": [{"type":"...","content":"...","severity":"high|medium|low","frame_hint":"第X帧"}],
  "summary": "一句话结论（≤40字）",
  "detail": "逐帧分析：画面内容描述（人物/场景/文字/物品）、发现的元素、安全性判断（50~200字）"
}
- verdict 仅限 PASS / WARN / BLOCK / UNKNOWN
- violations 无命中时为 []
- detail 即使 PASS 也须填写，描述画面的实际内容（方便 Judge 综合研判）
- 不要工具调用，不要 Markdown 围栏，不要多余输出"""


# ---------------------------------------------------------------------------
# Core execution helpers
# ---------------------------------------------------------------------------

def _parse_sub_verdict(raw: str) -> dict[str, Any]:
    """Best-effort JSON parse of a sub-agent response."""
    from reviewagent.review_report import parse_review_json_from_llm_output
    d = parse_review_json_from_llm_output(raw)
    if isinstance(d, dict):
        return d
    return {}


async def run_sub_agent(
    task: SubAgentTask,
    orchestrator: "ReviewOrchestrator",
) -> SubAgentResult:
    """Execute one specialist sub-agent via direct LLM call (no tools, no memory)."""
    from reviewagent.observability import tracing  # noqa: PLC0415

    t0 = time.perf_counter()

    @tracing.span(f"sub_agent.{task.name}")
    async def _traced() -> SubAgentResult:
        try:
            if task.images:
                system = SystemMessage(content=_VISUAL_SUB_AGENT_SYSTEM)
                content_parts: list[dict[str, Any]] = [{"type": "text", "text": task.prompt}]
                for img_url in task.images:
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": img_url}}
                    )
                messages = [system, HumanMessage(content=content_parts)]
            else:
                messages = [
                    SystemMessage(content=_SUB_AGENT_SYSTEM),
                    HumanMessage(content=task.prompt),
                ]
            ai = await orchestrator.llm.ainvoke(messages)
            raw = (getattr(ai, "content", None) or "").strip()
            dur = round((time.perf_counter() - t0) * 1000.0, 2)

            parsed = _parse_sub_verdict(raw)
            return SubAgentResult(
                name=task.name,
                verdict=str(parsed.get("verdict") or "UNKNOWN").upper(),
                confidence=float(parsed.get("confidence") or 0.5),
                violations=list(parsed.get("violations") or []),
                summary=str(parsed.get("summary") or ""),
                detail=str(parsed.get("detail") or ""),
                raw_response=raw,
                duration_ms=dur,
            )
        except Exception as exc:
            dur = round((time.perf_counter() - t0) * 1000.0, 2)
            logger.warning("%s sub_agent name=%s error=%s", _MC, task.name, exc)
            return SubAgentResult(
                name=task.name,
                verdict="UNKNOWN",
                confidence=0.0,
                violations=[],
                summary="",
                raw_response="",
                error=str(exc)[:300],
                duration_ms=dur,
            )

    return await _traced()


async def run_sub_agents_parallel(
    tasks: list[SubAgentTask],
    orchestrator: "ReviewOrchestrator",
) -> list[SubAgentResult]:
    """Run all sub-agent tasks concurrently and return ordered results."""
    if not tasks:
        return []
    results = await asyncio.gather(
        *[run_sub_agent(t, orchestrator) for t in tasks],
        return_exceptions=False,
    )
    for r in results:
        logger.info(
            "%s sub_agent name=%s verdict=%s confidence=%.2f violations=%d dur=%.0fms error=%s",
            _MC,
            r.name,
            r.verdict,
            r.confidence,
            len(r.violations),
            r.duration_ms,
            r.error,
        )
    return list(results)


# ---------------------------------------------------------------------------
# Judge agent
# ---------------------------------------------------------------------------

def _build_judge_prompt(
    sub_results: list[SubAgentResult],
    context: str,
) -> str:
    """Build the judge's user prompt from sub-agent results."""
    lines: list[str] = []
    if context:
        lines.append(f"【审核对象】{context}\n")
    lines.append("【各模态子 Agent 审核结论】\n")
    for r in sub_results:
        entry: dict[str, Any] = {
            "agent": r.name,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "violations": r.violations,
            "summary": r.summary,
        }
        if r.detail:
            entry["detail"] = r.detail
        if r.error:
            entry["error"] = r.error
        if r.skipped:
            entry["skipped"] = True
        lines.append(json.dumps(entry, ensure_ascii=False))
    lines.append("\n请综合以上所有子结论，输出最终裁决 JSON（包含 verdict、confidence、violations、summary、reasoning、modality_analysis、risk_level、recommendations）。")
    return "\n".join(lines)


async def run_judge_agent(
    sub_results: list[SubAgentResult],
    orchestrator: "ReviewOrchestrator",
    *,
    context: str = "",
) -> dict[str, Any]:
    """Synthesize sub-verdicts into a final binding decision."""
    from reviewagent.observability import tracing  # noqa: PLC0415

    t0 = time.perf_counter()

    @tracing.span("judge.synthesize")
    async def _traced() -> dict[str, Any]:
        valid = [r for r in sub_results if not r.error and not r.skipped]
        if not valid:
            payload = {
                "verdict": "UNKNOWN",
                "confidence": 0.0,
                "violations": [],
                "summary": "所有子 Agent 均未能完成分析，请人工复核。",
                "reasoning": "所有检测模态均发生错误或被跳过，无法给出可信裁决。",
                "modality_analysis": {r.name: f"错误: {r.error or '未知'}" for r in sub_results},
                "risk_level": "low",
                "recommendations": "建议人工复核，或排查检测依赖（OCR、ASR 等）是否正常运行。",
            }
            dur = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "success": True,
                "response": json.dumps(payload, ensure_ascii=False),
                "iterations": 0,
                "duration_ms": dur,
            }

        try:
            messages = [
                SystemMessage(content=_JUDGE_SYSTEM),
                HumanMessage(content=_build_judge_prompt(sub_results, context)),
            ]
            ai = await orchestrator.llm.ainvoke(messages)
            raw = (getattr(ai, "content", None) or "").strip()
            dur = round((time.perf_counter() - t0) * 1000.0, 2)

            parsed = _parse_sub_verdict(raw)
            if not parsed:
                raw = _fallback_merge(sub_results)
                parsed = _parse_sub_verdict(raw) or {}

            logger.info(
                "%s judge verdict=%s confidence=%s sub_agents=%d dur=%.0fms",
                _MC,
                parsed.get("verdict"),
                parsed.get("confidence"),
                len(sub_results),
                dur,
            )
            return {
                "success": True,
                "response": json.dumps(parsed, ensure_ascii=False) if parsed else raw,
                "iterations": len(sub_results) + 1,
                "duration_ms": dur,
            }
        except Exception as exc:
            logger.warning("%s judge_agent error=%s", _MC, exc)
            raw = _fallback_merge(sub_results)
            dur = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "success": True,
                "response": raw,
                "iterations": len(sub_results),
                "duration_ms": dur,
                "error": str(exc)[:300],
            }

    return await _traced()


_VERDICT_RISK = {"BLOCK": "high", "WARN": "medium", "PASS": "low", "UNKNOWN": "low"}
_VERDICT_RECS = {
    "BLOCK": "内容存在违规，建议立即拦截或下架，并通知相关审核人员。",
    "WARN":  "内容存疑，建议人工复核后再决定是否发布。",
    "PASS":  "内容合规，可正常发布。",
    "UNKNOWN": "分析结果不确定，建议人工复核。",
}


def _fallback_merge(sub_results: list[SubAgentResult]) -> str:
    """Emergency merge when judge LLM call fails: take the highest severity verdict."""
    _RANK = {"BLOCK": 3, "WARN": 2, "PASS": 1, "UNKNOWN": 0}
    best = max(sub_results, key=lambda r: _RANK.get(r.verdict, 0))
    all_violations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in sub_results:
        for v in r.violations:
            key = f"{v.get('type')}:{v.get('content')}"
            if key not in seen:
                seen.add(key)
                all_violations.append(v)
    modality_analysis = {
        r.name: (r.detail or r.summary or f"verdict={r.verdict}")
        for r in sub_results if not r.error
    }
    payload = {
        "verdict": best.verdict,
        "confidence": best.confidence,
        "violations": all_violations,
        "summary": f"[降级合并] {best.summary}" if best.summary else "子 Agent 结论合并（Judge 调用失败）",
        "reasoning": "Judge 调用失败，自动取各模态中最高严重度的子 Agent 结论作为最终裁决。",
        "modality_analysis": modality_analysis,
        "risk_level": _VERDICT_RISK.get(best.verdict, "low"),
        "recommendations": _VERDICT_RECS.get(best.verdict, "建议人工复核。"),
    }
    return json.dumps(payload, ensure_ascii=False)


__all__ = [
    "SubAgentTask",
    "SubAgentResult",
    "run_sub_agent",
    "run_sub_agents_parallel",
    "run_judge_agent",
]
