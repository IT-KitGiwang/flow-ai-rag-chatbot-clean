"""
Form Node — Quản lý Graph nhỏ của PROCEED_FORM.

Input: standalone_query, chat_history
Output: final_response chứa nội dung Biểu mẫu.

Luồng mới:
  - Selector → Extractor → Drafter (luôn luôn draft)
  - Personal fields trống → placeholder [Điền ...]
  - Content fields trống → LLM tự viết mẫu tham khảo
  - Không cần clarification loop nữa.
"""

import time
from app.services.langgraph.state import GraphState
from app.core.config.form_config import form_cfg
from app.services.langgraph.nodes.proceed_form.form_selector import select_form
from app.services.langgraph.nodes.proceed_form.field_extractor import extract_fields
from app.services.langgraph.nodes.proceed_form.form_drafter import generate_form
from app.utils.logger import get_logger

logger = get_logger(__name__)


def form_node(state: GraphState) -> GraphState:
    """
    FORM NODE — Tạo biểu mẫu hành chính.

    Luôn sinh form ngay lập tức:
    - Thông tin user có → điền nguyên văn.
    - Nội dung chung thiếu → LLM viết tham khảo.
    - Thông tin cá nhân thiếu → để placeholder.
    """
    start_time = time.time()
    logger.info("========== BAT DAU FORM NODE ==========")

    # ── Guard: Chỉ chạy khi intent đúng là PROCEED_FORM ──
    intent_action = state.get("intent_action", "")
    if intent_action != "PROCEED_FORM":
        logger.info("Form Node - SKIP (intent_action='%s' != 'PROCEED_FORM')", intent_action)
        return state

    query = state.get("standalone_query", state.get("user_query", ""))
    chat_history = state.get("chat_history", [])

    # Bước 1: Chọn mẫu đơn phù hợp
    target_form = select_form(query, form_cfg.forms)
    logger.info(
        "Form Node - Mau don: %s", target_form.get("name", "N/A")
    )

    # Bước 2: Trích xuất thông tin từ chat history
    extracted = extract_fields(chat_history, query)

    # Log tóm tắt thông tin thu được
    logger.info(
        "Form Node - Extracted %d fields co du lieu: %s",
        len(extracted), list(extracted.keys()),
    )

    # Bước 3: Draft form (luôn luôn, không hỏi lại)
    draft = generate_form(target_form, extracted)

    elapsed = time.time() - start_time
    logger.info("========== HOAN TAT FORM NODE (%.3fs) ==========", elapsed)

    return {
        **state,
        "final_response": draft,
        "response_source": "form_template",
    }
