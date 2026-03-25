"""
Form Drafter — Sinh file Markdown mẫu đơn cuối cùng bằng LLM.

CHIẾN LƯỢC MỚI (Template-Driven):
  1. Đọc template gốc (file .md) → Đây là "khuôn" chính xác, có đúng tên đơn + đúng field.
  2. Trích xuất thông tin user đã cung cấp → Dạng key-value tự do.
  3. Prompt LLM: "Điền thông tin user vào template, giữ nguyên cấu trúc template,
     field nào user chưa cung cấp thì giữ nguyên placeholder (___)"

  → LLM KHÔNG ĐƯỢC bịa thêm field, KHÔNG ĐƯỢC đổi tên mẫu đơn.
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


def _build_extracted_info_block(extracted_fields: dict) -> str:
    """
    Xây dựng block thông tin user đã cung cấp — dạng đơn giản, liệt kê key-value.
    
    VD:
        THÔNG TIN NGƯỜI DÙNG ĐÃ CUNG CẤP (ĐIỀN NGUYÊN VĂN VÀO MẪU ĐƠN):
        - Họ tên: Nguyễn Văn A
        - Ngành dự tuyển: Quản trị Kinh doanh
    """
    if not extracted_fields:
        return "THÔNG TIN NGƯỜI DÙNG: Chưa cung cấp thông tin cá nhân nào.\n"

    lines = []
    for key, val in extracted_fields.items():
        if val:
            # Chuyển key từ snake_case sang label đẹp hơn
            label = key.replace("_", " ").capitalize()
            lines.append(f"- {label}: {val}")

    if not lines:
        return "THÔNG TIN NGƯỜI DÙNG: Chưa cung cấp thông tin cá nhân nào.\n"

    result = "THONG TIN NGUOI DUNG DA CUNG CAP (DIEN NGUYEN VAN VAO MAU DON, KHONG DUOC SUA):\n"
    result += "\n".join(lines) + "\n"
    return result


def generate_form(form_metadata: dict, extracted_fields: dict) -> str:
    """
    Soạn thảo văn bản hành chính dựa trên template gốc và thông tin người dùng.
    
    Template-Driven: LLM phải bám sát template, KHÔNG được bịa field hay đổi tên đơn.
    """
    start_time = time.time()

    form_name = form_metadata.get("name", "Mẫu đơn")
    template_file = form_metadata.get("template_file")

    # ── Đọc nội dung mẫu ──
    metadata, template_content = _load_template_content(template_file)

    if not template_content:
        template_content = (
            f"[Không tìm thấy file mẫu cho \"{form_name}\". "
            f"Vui lòng soạn đơn hành chính với tiêu đề \"{form_name}\" "
            f"theo đúng chuẩn văn phong hành chính nhà nước.]"
        )

    # ── Xây dựng block thông tin user ──
    info_str = _build_extracted_info_block(extracted_fields)

    # ── Render Prompts ──
    sys_prompt_raw = prompt_manager.get_system("form_drafter")
    sys_prompt = sys_prompt_raw.replace("{{ form_name }}", form_name)

    user_content = prompt_manager.render_user(
        "form_drafter",
        template_metadata=metadata,
        template_content=template_content,
        extracted_info=info_str,
        form_name=form_name,
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
