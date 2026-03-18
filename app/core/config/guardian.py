# Query Flow - Guardian Pipeline Config
# Đọc secrets từ .env, đọc settings từ guardian_config.yaml
# Pydantic models chỉ định nghĩa SCHEMA (cái khung), không hardcode giá trị

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional

# ============================================================
# 1. Load .env (API Keys & Secrets)
# ============================================================
load_dotenv()

# ============================================================
# 2. Load YAML (Non-sensitive settings)
# ============================================================
_YAML_PATH = Path(__file__).parent / "guardian_config.yaml"

def _load_yaml() -> dict:
    """Đọc file YAML config. Trả về dict rỗng nếu file không tồn tại."""
    if _YAML_PATH.exists():
        with open(_YAML_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

_yaml_data = _load_yaml()

# ============================================================
# 3. API Key Config (từ .env)
# ============================================================
class APIKeyConfig(BaseModel):
    """Quản lý API keys cho tất cả các cloud provider."""
    groq_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )
    groq_base_url: str = Field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    google_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )
    google_base_url: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    )
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openrouter_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )

    def get_key(self, provider: str) -> Optional[str]:
        """Lấy API key theo tên provider."""
        return getattr(self, f"{provider}_api_key", None)

    def get_base_url(self, provider: str) -> str:
        """Lấy base URL theo tên provider."""
        return getattr(self, f"{provider}_base_url", "")


# ============================================================
# LỚP 0: Input Validation (Gatekeeper)
# ============================================================
_iv = _yaml_data.get("input_validation", {})

class InputValidationConfig(BaseModel):
    max_input_chars: int = _iv.get("max_input_chars", 800)
    fallback_too_long: str = _iv.get(
        "fallback_too_long",
        "Câu hỏi của bạn quá dài. Vui lòng tóm tắt lại."
    )


# ============================================================
# LỚP 1: Keyword Filter & Normalization
# ============================================================
_kf = _yaml_data.get("keyword_filter", {})

class KeywordFilterConfig(BaseModel):
    banned_regex_patterns: List[str] = _kf.get("banned_regex_patterns", [])
    teencode_map: Dict[str, str] = _kf.get("teencode_map", {})
    noise_regex_pattern: str = _kf.get(
        "noise_regex_pattern",
        r"[\.\\*_\-~!#\$%\^&\(\)\+=\|\\/<>\}\{\[\]:;'\"]"
    )
    fallback_message: str = _kf.get(
        "fallback_message",
        "Câu hỏi chứa nội dung không phù hợp."
    )


# ============================================================
# LỚP 2: Prompt Guard (Injection / Jailbreak Detection)
# ============================================================
_pg = _yaml_data.get("prompt_guard", {})

class PromptGuardConfig(BaseModel):
    provider: str = _pg.get("provider", "groq")
    model: str = _pg.get("model", "meta-llama/llama-prompt-guard-2-86m")
    max_tokens_per_chunk: int = _pg.get("max_tokens_per_chunk", 512)
    safe_threshold: float = Field(
        default=_pg.get("safe_threshold", 0.75),
        ge=0.0, le=1.0
    )
    fallback_unsafe: str = _pg.get(
        "fallback_unsafe",
        "Phát hiện dấu hiệu bất thường. Vui lòng diễn đạt lại."
    )


# ============================================================
# LỚP 3.1: Vector Intent Router (Fast Semantic Search)
# ============================================================
_vr = _yaml_data.get("vector_router", {})

class VectorRouterConfig(BaseModel):
    enabled: bool = _vr.get("enabled", True)
    provider: str = _vr.get("provider", "openai")
    model: str = _vr.get("model", "text-embedding-3-small")
    dimensions: int = _vr.get("dimensions", 512)
    similarity_threshold: float = Field(
        default=_vr.get("similarity_threshold", 0.75),
        ge=0.0, le=1.0
    )


# ============================================================
# LỚP 3.2: LLM Semantic Router (Deep Intent Classification)
# ============================================================
_sr = _yaml_data.get("semantic_router", {})

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


# ============================================================
# LỚP 4: Main Bot (RAG / LLM Generation)
# ============================================================
_mb = _yaml_data.get("main_bot", {})

class MainBotConfig(BaseModel):
    enabled: bool = _mb.get("enabled", True)
    provider: str = _mb.get("provider", "groq")
    model: str = _mb.get("model", "llama-3.1-70b-versatile")
    temperature: float = _mb.get("temperature", 0.2)


# ============================================================
# CẤU HÌNH TỔNG (Pipeline Orchestrator)
# ============================================================
class QueryFlowConfig(BaseModel):
    api_keys: APIKeyConfig = APIKeyConfig()
    input_validation: InputValidationConfig = InputValidationConfig()
    keyword_filter: KeywordFilterConfig = KeywordFilterConfig()
    prompt_guard: PromptGuardConfig = PromptGuardConfig()
    vector_router: VectorRouterConfig = VectorRouterConfig()
    semantic_router: SemanticRouterConfig = SemanticRouterConfig()
    main_bot: MainBotConfig = MainBotConfig()


# Khởi tạo instance config chung
query_flow_config = QueryFlowConfig()