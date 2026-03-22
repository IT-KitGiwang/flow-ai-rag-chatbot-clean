"""
Form Drafter — Sinh file Markdown mẫu đơn cuối cùng bằng LLM.

Luồng điền thông tin:
  1. User đã cung cấp info → Điền đúng y nguyên (không sửa, không bịa).
  2. Field loại "content" (nguyện vọng, lý do...) mà user KHÔNG cung cấp
     → LLM viết MẪU THAM KHẢO dựa trên reference_hint.
  3. Field loại "personal" (tên, CCCD, SĐT...) mà user KHÔNG cung cấp
     → Để trống placeholder: [Điền ... tại đây].
"""

import time
import os
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config.form_config import form_cfg
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _load_template_content(filename: str) -> tuple[str, str]:
    """Đọc file biểu mẫu markdown trên ổ đĩa. Trả về (metadata, template_content)."""
    if not filename:
        return "", ""

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )))
    )
    template_path = os.path.join(
        base_dir, "data", "unstructured", "markdown", "maudon", filename
    )

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "-start-" in content:
                parts = content.split("-start-", 1)
                return parts[0].strip(), parts[1].strip()
            return "", content.strip()
    except Exception as e:
        logger.warning("Form Drafter - Khong the doc file mau %s: %s", template_path, e)
        return "", ""


def _build_field_instructions(extracted_fields: dict) -> str:
    """
    Xây dựng block hướng dẫn điền cho LLM dựa trên field_type.

    Trả về text dạng:
        THÔNG TIN NGƯỜI DÙNG ĐÃ CUNG CẤP (ĐIỀN NGUYÊN VĂN):
        - Họ và tên: Nguyễn Văn A

        CÁC TRƯỜNG CẦN VIẾT MẪU THAM KHẢO (vì người dùng chưa cung cấp):
        - Lý do / Mong muốn: hãy viết mẫu tham khảo lý do nộp đơn...

        CÁC TRƯỜNG ĐỂ TRỐNG (thông tin cá nhân, người dùng tự điền):
        - Số CCCD/CMND: [Điền số CCCD/CMND]
    """
    provided_lines = []
    reference_lines = []
    blank_lines = []

    for f in form_cfg.fields:
        val = extracted_fields.get(f.key)

        if val:
            # User đã cung cấp → điền y nguyên
            provided_lines.append(f"- {f.label}: {val}")

        elif f.field_type == "content" and f.reference_hint:
            # Content field trống → LLM viết mẫu tham khảo
            reference_lines.append(
                f"- {f.label}: (Viết mẫu tham khảo — {f.reference_hint})"
            )

        else:
            # Personal field trống → để placeholder
            placeholder = f.placeholder or f"[Điền {f.label.lower()}]"
            blank_lines.append(f"- {f.label}: {placeholder}")

    result = ""

    if provided_lines:
        result += "THONG TIN NGUOI DUNG DA CUNG CAP (DIEN NGUYEN VAN, KHONG DUOC SUA):\n"
        result += "\n".join(provided_lines) + "\n\n"

    if reference_lines:
        result += (
            "CAC TRUONG CAN VIET MAU THAM KHAO "
            "(nguoi dung chua cung cap, hay viet noi dung tham khao phu hop voi boi canh):\n"
        )
        result += "\n".join(reference_lines) + "\n\n"

    if blank_lines:
        result += (
            "CAC TRUONG DE TRONG "
            "(thong tin ca nhan, nguoi dung tu dien, giu nguyen placeholder):\n"
        )
        result += "\n".join(blank_lines) + "\n\n"

    return result


def generate_form(form_metadata: dict, extracted_fields: dict) -> str:
    """
    Soạn thảo văn bản hành chính với template và thông tin người dùng.
    """
    start_time = time.time()

    # ── Đọc nội dung mẫu ──
    metadata, template_content = _load_template_content(
        form_metadata.get("template_file")
    )

    if not template_content:
        template_content = (
            "[Không tìm thấy file mẫu, vui lòng tự sinh bố cục đơn hành chính chung cho "
            + form_metadata.get("name", "Đơn xin việc") + "]"
        )

    # ── Xây dựng block hướng dẫn điền thông tin ──
    info_str = _build_field_instructions(extracted_fields)

    # ── Render Prompts ──
    sys_prompt_raw = prompt_manager.get_system("form_drafter")
    sys_prompt = sys_prompt_raw.replace(
        "{{ form_name }}", form_metadata.get("name", "Mẫu đơn")
    )

    user_content = prompt_manager.render_user(
        "form_drafter",
        template_metadata=metadata,
        template_content=template_content,
        extracted_info=info_str,
        form_name=form_metadata.get("name", "Mẫu đơn"),
    )

    class _DrafterConfig:
        model = form_cfg.settings.drafter_model
        provider = form_cfg.settings.provider
        temperature = form_cfg.settings.drafter_temperature
        max_tokens = form_cfg.settings.drafter_max_tokens
        timeout_seconds = form_cfg.settings.drafter_timeout

    try:
        draft = _call_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=_DrafterConfig(),
            node_key="form",
        )

        elapsed = time.time() - start_time
        logger.info("Form Drafter [%.3fs] %d ky tu sinh ra", elapsed, len(draft))
        return draft.strip()

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Form Drafter [%.3fs] Loi: %s", elapsed, e, exc_info=True)
        return "Xin lỗi bạn, hiện tại hệ thống soạn thảo biểu mẫu đang bị lỗi. Vui lòng thử lại sau."
