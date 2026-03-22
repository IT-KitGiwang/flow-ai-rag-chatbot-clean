"""
Field Extractor — Trích xuất thông tin người dùng từ chat_history.
Sử dụng LLM với Temperature = 0.1 để đảm bảo tính chính xác (Zero Hallucination).
"""

import json
import re
import time
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config.form_config import form_cfg
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def extract_fields(chat_history: list, user_query: str) -> dict:
    """
    Quét lịch sử hội thoại để thu thập giá trị cho các fields trong form_config.

    Trả về Dict: { "ho_ten": "Nguyen Van A", "nganh_hoc": None, ... }
    Giá trị nào không tìm thấy trả về None.
    """
    start_time = time.time()

    # ── Xây dựng danh sách fields cần trích xuất ──
    fields_list_str = ""
    for idx, f in enumerate(form_cfg.fields, 1):
        is_required = (f.field_type == "personal")
        fields_list_str += (
            f"{idx}. {f.key}: {f.extract_hint} "
            f"(Bắt buộc: {'Có' if is_required else 'Không'})\n"
        )

    # ── Xây dựng context từ history ──
    context_str = ""
    if chat_history:
        for msg in chat_history[-6:]:  # Lấy 6 lượt gần nhất
            role = "Tư vấn viên" if msg.get("role") == "assistant" else "Người dùng"
            content = msg.get("content", "").replace("\n", " ")
            context_str += f"{role}: {content}\n"
    context_str += f"Người dùng (hiện tại): {user_query}\n"

    # ── Render System Prompt ──
    # get_system trả raw string, cần replace thủ công vì system_prompt
    # chứa {{ fields_config }} mà PromptManager không render system prompt.
    sys_prompt_raw = prompt_manager.get_system("form_extractor")
    sys_prompt = sys_prompt_raw.replace("{{ fields_config }}", fields_list_str)

    # ── Render User Prompt (Jinja2 qua PromptManager) ──
    user_content = prompt_manager.render_user(
        "form_extractor",
        context=context_str
    )

    # ── Config API calls ──
    class _ExtractorConfig:
        model = form_cfg.settings.extractor_model
        provider = form_cfg.settings.provider
        temperature = form_cfg.settings.extractor_temperature
        max_tokens = form_cfg.settings.extractor_max_tokens
        timeout_seconds = form_cfg.settings.extractor_timeout

    try:
        raw_output = _call_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=_ExtractorConfig(),
            node_key="form",
        )

        # ── Parse JSON — strip markdown code fences nếu LLM trả về ──
        cleaned = raw_output.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        # Fallback: tìm object JSON đầu tiên bằng regex đơn giản
        if not cleaned.startswith("{"):
            match = re.search(r'\{[^{}]*\}', cleaned)
            if match:
                cleaned = match.group(0)

        extracted_data = json.loads(cleaned)

        # ── Chuẩn hoá keys theo config ──
        result = {}
        for f in form_cfg.fields:
            val = extracted_data.get(f.key)
            if val == "" or str(val).lower() in ("null", "none"):
                val = None
            result[f.key] = val

        elapsed = time.time() - start_time
        logger.info("Form Extractor [%.3fs] OK -> %s", elapsed, result)
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            "Form Extractor [%.3fs] Loi: %s -> Fallback Return Empty",
            elapsed, e, exc_info=True,
        )
        return {f.key: None for f in form_cfg.fields}
