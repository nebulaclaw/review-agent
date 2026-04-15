"""Per-request business context (passed into prompts and pipeline_trace)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class BizContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    biz_line: str = Field(default="", description="Business domain, e.g. ugc / ad / dm")
    tenant_id: str = Field(default="", description="Tenant isolation id")
    trust_tier: str = Field(default="", description="Account trust tier")
    audience: str = Field(default="", description="Visibility: public / unlisted / private")
    policy_pack_id: str = Field(default="", description="Compliance policy pack id")


def biz_context_from_payload(data: Optional[dict[str, Any]]) -> BizContext:
    if not data:
        return BizContext()
    return BizContext.model_validate(data)


__all__ = ["BizContext", "biz_context_from_payload"]
