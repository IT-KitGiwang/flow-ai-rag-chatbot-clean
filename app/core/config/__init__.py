# app/core/config/__init__.py
# Phần dùng chung: Load .env, Load YAML, API Keys, Main Bot, Pipeline tổng

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

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

yaml_data = _load_yaml()

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
# LỚP 4: Main Bot (RAG / LLM Generation)
# ============================================================
_mb = yaml_data.get("main_bot", {})

class MainBotConfig(BaseModel):
    enabled: bool = _mb.get("enabled", True)
    provider: str = _mb.get("provider", "groq")
    model: str = _mb.get("model", "llama-3.1-70b-versatile")
    temperature: float = _mb.get("temperature", 0.2)


# ============================================================
# CẤU HÌNH TỔNG (Pipeline Orchestrator)
# ============================================================
from app.core.config.guardian import InputValidationConfig, KeywordFilterConfig, PromptGuardFastConfig, PromptGuardDeepConfig
from app.core.config.intent import VectorRouterConfig, IntentValidatorConfig, SemanticRouterConfig

class QueryFlowConfig(BaseModel):
    api_keys: APIKeyConfig = APIKeyConfig()
    input_validation: InputValidationConfig = InputValidationConfig()
    keyword_filter: KeywordFilterConfig = KeywordFilterConfig()
    prompt_guard_fast: PromptGuardFastConfig = PromptGuardFastConfig()
    prompt_guard_deep: PromptGuardDeepConfig = PromptGuardDeepConfig()
    vector_router: VectorRouterConfig = VectorRouterConfig()
    intent_validator: IntentValidatorConfig = IntentValidatorConfig()
    semantic_router: SemanticRouterConfig = SemanticRouterConfig()
    main_bot: MainBotConfig = MainBotConfig()


# Khởi tạo instance config chung
query_flow_config = QueryFlowConfig()
