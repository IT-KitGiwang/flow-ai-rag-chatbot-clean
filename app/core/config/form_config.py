"""
Cấu trúc Pydantic cho Form Config.
Đọc file app/core/config/yaml/form_config.yaml.
"""

from typing import List, Optional
from pydantic import BaseModel
import yaml
import os
from app.utils.logger import get_logger
from app.core.config import models_yaml_data

_logger = get_logger(__name__)

_fm_settings = models_yaml_data.get("form", {})


class FormSettings(BaseModel):
    extractor_model: str = _fm_settings.get("extractor", {}).get("model", "qwen/qwen3-30b-a3b")
    extractor_temperature: float = _fm_settings.get("extractor", {}).get("temperature", 0.1)
    extractor_max_tokens: int = _fm_settings.get("extractor", {}).get("max_tokens", 500)
    extractor_timeout: int = _fm_settings.get("extractor", {}).get("timeout_seconds", 10)

    drafter_model: str = _fm_settings.get("drafter", {}).get("model", "qwen/qwen3-30b-a3b")
    drafter_temperature: float = _fm_settings.get("drafter", {}).get("temperature", 0.4)
    drafter_max_tokens: int = _fm_settings.get("drafter", {}).get("max_tokens", 2000)
    drafter_timeout: int = _fm_settings.get("drafter", {}).get("timeout_seconds", 15)

    provider: str = _fm_settings.get("provider", "openrouter")


class FormFieldDef(BaseModel):
    key: str
    label: str
    field_type: str                    # "personal" | "content"
    placeholder: str = ""
    extract_hint: str = ""
    reference_hint: str = ""           # Chỉ dùng cho field_type="content"


class FormTemplateDef(BaseModel):
    id: str
    name: str
    description: str
    keywords: List[str]
    template_file: Optional[str] = None


class FormConfig(BaseModel):
    settings: FormSettings
    fields: List[FormFieldDef]
    forms: List[FormTemplateDef]

    @classmethod
    def load(cls, yaml_path: str) -> "FormConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Khong tim thay {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        # Bỏ qua 'settings' từ form_config.yaml (models đã chuyển sang models_config.yaml)
        # Chỉ load fields và forms.
        fields = data.get("fields", [])
        forms = data.get("forms", [])

        return cls(
            settings=FormSettings(), # Luôn lấy từ models_config.yaml
            fields=[FormFieldDef(**f) for f in fields],
            forms=[FormTemplateDef(**frm) for frm in forms],
        )


# Singleton — Load 1 lần khi import
_YAML_PATH = os.path.join(
    os.path.dirname(__file__), "yaml", "form_config.yaml"
)

try:
    form_cfg = FormConfig.load(_YAML_PATH)
except Exception as e:
    _logger.warning("FormConfig - Loi khi load form_config.yaml: %s", e)
    form_cfg = FormConfig(
        settings=FormSettings(),
        fields=[],
        forms=[],
    )
