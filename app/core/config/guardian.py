# app/core/config/guardian.py
# Cấu hình các lớp BẢO VỆ (Security Layers)
# Lớp 0: Input Validation (Chống DoS)
# Lớp 1: Keyword Filter & Normalization (Chống từ khóa cấm)
# Lớp 2: Prompt Guard (Chống Injection / Jailbreak)

from pydantic import BaseModel, Field
from typing import List, Dict
from app.core.config import yaml_data


# ============================================================
# LỚP 0: Input Validation (Gatekeeper - Chống DoS)
# ============================================================
_iv = yaml_data.get("input_validation", {})

class InputValidationConfig(BaseModel):
    max_input_chars: int = _iv.get("max_input_chars", 800)
    fallback_too_long: str = _iv.get(
        "fallback_too_long",
        "Câu hỏi của bạn quá dài. Vui lòng tóm tắt lại."
    )


# ============================================================
# LỚP 1: Keyword Filter & Normalization
# ============================================================
_kf = yaml_data.get("keyword_filter", {})

class KeywordFilterConfig(BaseModel):
    banned_regex_patterns: List[str] = _kf.get("banned_regex_patterns", [])
    injection_regex_patterns: List[str] = _kf.get("injection_regex_patterns", [])
    teencode_map: Dict[str, str] = _kf.get("teencode_map", {})
    fallback_message: str = _kf.get(
        "fallback_message",
        "Câu hỏi chứa nội dung không phù hợp."
    )
    fallback_injection: str = _kf.get(
        "fallback_injection",
        "Hệ thống phát hiện dấu hiệu can thiệp bất thường."
    )


# ============================================================
# LỚP 2a: Prompt Guard Fast (Llama 86M - Score-based)
# ============================================================
_pgf = yaml_data.get("prompt_guard_fast", {})

class PromptGuardFastConfig(BaseModel):
    provider: str = _pgf.get("provider", "groq")
    model: str = _pgf.get("model", "meta-llama/llama-prompt-guard-2-86m")
    max_tokens_per_chunk: int = _pgf.get("max_tokens_per_chunk", 512)
    score_threshold: float = Field(
        default=_pgf.get("score_threshold", 0.9),
        ge=0.0, le=1.0
    )
    fallback_unsafe: str = _pgf.get(
        "fallback_unsafe",
        "Phát hiện dấu hiệu tấn công rõ ràng."
    )


# ============================================================
# LỚP 2b: Prompt Guard Deep (Qwen 7B - Vietnamese SAFE/UNSAFE)
# ============================================================
_pgd = yaml_data.get("prompt_guard_deep", {})

class PromptGuardDeepConfig(BaseModel):
    provider: str = _pgd.get("provider", "openrouter")
    model: str = _pgd.get("model", "qwen/qwen-2.5-7b-instruct")
    temperature: float = _pgd.get("temperature", 0.0)
    response_format: str = _pgd.get("response_format", "json_object")
    system_prompt: str = _pgd.get(
        "system_prompt",
        'Bạn là hệ thống bảo mật. Trả về JSON: {"status": "SAFE"} hoặc {"status": "UNSAFE"}'
    )
    fallback_unsafe: str = _pgd.get(
        "fallback_unsafe",
        "Phát hiện dấu hiệu bất thường. Vui lòng diễn đạt lại."
    )