# app/core/config/intent.py
# Cấu hình các lớp PHÂN LOẠI Ý ĐỊNH (Intent Classification)
# Lớp 3.1: Vector Router (So khớp nhanh bằng Embedding)
# Lớp 3 Validator: Chống LLM sai chính tả Intent
# Lớp 3.2: Semantic Router (Phân loại sâu bằng LLM)

from pydantic import BaseModel, Field
from typing import List, Literal
from app.core.config import yaml_data


# ============================================================
# LỚP 3.1: Vector Intent Router (Fast Semantic Search)
# ============================================================
_vr = yaml_data.get("vector_router", {})

class VectorRouterConfig(BaseModel):
    enabled: bool = _vr.get("enabled", True)
    provider: str = _vr.get("provider", "openai")
    model: str = _vr.get("model", "text-embedding-3-small")
    dimensions: int = _vr.get("dimensions", 1536)
    similarity_threshold: float = Field(
        default=_vr.get("similarity_threshold", 0.75),
        ge=0.0, le=1.0
    )


# ============================================================
# LỚP 3 - Validator: Chống LLM sai chính tả Intent
# ============================================================
_iv_val = yaml_data.get("intent_validator", {})

class IntentValidatorConfig(BaseModel):
    enabled: bool = _iv_val.get("enabled", True)
    fallback_intent: str = _iv_val.get("fallback_intent", "KHONG_XAC_DINH")


# ============================================================
# LỚP 3.2: LLM Semantic Router (Deep Intent Classification)
# ============================================================
_sr = yaml_data.get("semantic_router", {})

class SemanticRouterConfig(BaseModel):
    provider: str = _sr.get("provider", "groq")
    model: str = _sr.get("model", "qwen/qwen3-32b")
    temperature: float = _sr.get("temperature", 0.0)
    response_format: Literal["json_object"] = _sr.get("response_format", "json_object")
    allowed_intents: List[str] = _sr.get(
        "allowed_intents",
        ["tuyen_sinh", "hoc_phi", "ky_tuc_xa", "ngoai_le"]
    )
    fallback_out_of_scope: str = _sr.get(
        "fallback_out_of_scope",
        "Câu hỏi nằm ngoài phạm vi hỗ trợ tuyển sinh UFM."
    )
