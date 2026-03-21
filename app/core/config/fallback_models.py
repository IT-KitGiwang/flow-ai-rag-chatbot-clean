# app/core/config/fallback_models.py
# Pydantic config cho Fallback Models (có provider)
# Đọc từ: fallback_models_config.yaml

from pydantic import BaseModel, Field
from typing import List, Optional
from app.core.config import _load_yaml


_fb_data = _load_yaml("fallback_models_config.yaml")


class ModelEntry(BaseModel):
    """1 model entry: provider + model name."""
    provider: str = "openrouter"
    model: str = ""


class ModelGroupConfig(BaseModel):
    """Nhóm model: 1 primary + N fallbacks, mỗi cái có provider riêng."""
    primary: ModelEntry = ModelEntry()
    fallbacks: List[ModelEntry] = []


class FallbackSettingsConfig(BaseModel):
    max_retries: int = 2
    retry_delay_ms: int = 500
    log_fallback: bool = True


def _parse_model_entry(data) -> ModelEntry:
    """Parse dict hoặc string thành ModelEntry."""
    if isinstance(data, dict):
        return ModelEntry(
            provider=data.get("provider", "openrouter"),
            model=data.get("model", ""),
        )
    elif isinstance(data, str):
        return ModelEntry(provider="openrouter", model=data)
    return ModelEntry()


def _parse_group(data: dict) -> ModelGroupConfig:
    """Parse 1 nhóm model từ YAML."""
    primary = _parse_model_entry(data.get("primary", {}))
    fallbacks_raw = data.get("fallbacks", []) or []
    fallbacks = [_parse_model_entry(f) for f in fallbacks_raw]
    return ModelGroupConfig(primary=primary, fallbacks=fallbacks)


# ── Khởi tạo từ YAML ──
_settings = _fb_data.get("fallback_settings", {})


class FallbackModelsConfig(BaseModel):
    light_models: ModelGroupConfig = _parse_group(_fb_data.get("light_models", {}))
    medium_models: ModelGroupConfig = _parse_group(_fb_data.get("medium_models", {}))
    search_models: ModelGroupConfig = _parse_group(_fb_data.get("search_models", {}))
    embedding_models: ModelGroupConfig = _parse_group(_fb_data.get("embedding_models", {}))
    guard_models: ModelGroupConfig = _parse_group(_fb_data.get("guard_models", {}))
    main_bot_models: ModelGroupConfig = _parse_group(_fb_data.get("main_bot_models", {}))
    settings: FallbackSettingsConfig = FallbackSettingsConfig(
        max_retries=_settings.get("max_retries", 2),
        retry_delay_ms=_settings.get("retry_delay_ms", 500),
        log_fallback=_settings.get("log_fallback", True),
    )

    def get_model_chain(self, group: str) -> List[ModelEntry]:
        """
        Trả về danh sách [primary, fallback1, ...] theo nhóm.
        Mỗi phần tử là ModelEntry(provider, model).
        group: "light" | "medium" | "search" | "embedding" | "guard" | "main_bot"
        """
        group_config = getattr(self, f"{group}_models", None)
        if not group_config:
            return []
        return [group_config.primary] + group_config.fallbacks[:self.settings.max_retries]
