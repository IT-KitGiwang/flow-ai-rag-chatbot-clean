"""
Care Node — Xử lý nhóm intent Chăm sóc / Khiếu nại / Tâm lý.

Vị trí trong Graph:
  [intent_node] → [care_node] → [response_node] → END

Nhiệm vụ:
  Nhận intent từ Intent Node (HO_TRO_SINH_VIEN hoặc KHIEU_NAI_GOP_Y).
  Gọi Qwen 3.5 Flash (qua OpenRouter) để sinh câu trả lời đồng cảm,
  truyền thông tin liên hệ từ care_config.yaml vào system prompt.

Model: qwen/qwen3.5-flash-02-23 (via OpenRouter)
Chi phí: Cực thấp (~0.001$/query, model nhỏ nhất)
Latency: ~500ms-1s
Fallback: Nếu LLM lỗi, trả thẳng contact info từ config (bypass cứng)
"""

import json
import os
import time
import urllib.request
import urllib.error

from app.services.langgraph.state import GraphState
from app.core.config.care import CareConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Khởi tạo config 1 lần duy nhất khi module load
_care_config = CareConfig()

# OpenRouter API
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _build_system_prompt(intent: str, contact_text: str) -> str:
    """Tạo system prompt cho Qwen dựa trên intent và contact info."""
    if intent.lower() == "khieu_nai_gop_y":
        tone_guide = (
            "Sinh viên đang muốn phản ánh hoặc khiếu nại. "
            "Hãy thể hiện sự lắng nghe, ghi nhận ý kiến, "
            "và hướng dẫn họ gửi phản ánh đến đúng nơi."
        )
    else:
        tone_guide = (
            "Sinh viên đang cần hỗ trợ (tâm lý, học vụ, bảo lưu, nợ môn...). "
            "Hãy thể hiện sự đồng cảm, trấn an, "
            "và hướng dẫn họ đến đúng đơn vị hỗ trợ."
        )

    return (
        "Bạn là trợ lý tư vấn chăm sóc sinh viên của Đại học Tài chính - Marketing (UFM).\n"
        "Xưng hô: Mình - Bạn.\n\n"
        "QUY TẮC:\n"
        "- Trả lời ngắn gọn (3-5 câu), ấm áp và chân thành.\n"
        "- KHÔNG tư vấn chuyên môn (học phí, điểm chuẩn). Chỉ đồng cảm và hướng dẫn liên hệ.\n"
        "- PHẢI dẫn thông tin liên hệ bên dưới vào câu trả lời một cách tự nhiên.\n"
        "- KHÔNG bịa thêm số điện thoại hay email ngoài danh sách.\n"
        "- Kết thúc bằng lời động viên nhẹ nhàng.\n\n"
        f"NGỮ CẢNH: {tone_guide}\n\n"
        f"THÔNG TIN LIÊN HỆ (bắt buộc dẫn vào câu trả lời):\n{contact_text}"
    )


def _call_qwen(system_prompt: str, user_query: str) -> str:
    """Gọi Qwen 3.5 Flash qua OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY chưa được cấu hình")

    payload = {
        "model": _care_config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "temperature": _care_config.temperature,
        "max_tokens": _care_config.max_tokens,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(
        _OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    return body["choices"][0]["message"]["content"].strip()


def _build_fallback_message(contact_text: str) -> str:
    """Tin nhắn dự phòng khi LLM lỗi."""
    return (
        "Mình hiểu bạn đang cần hỗ trợ. "
        "Bạn có thể liên hệ trực tiếp qua các kênh sau:\n\n"
        f"{contact_text}\n\n"
        "Đừng ngại liên hệ nhé, đội ngũ UFM luôn sẵn sàng hỗ trợ bạn!"
    )


def care_node(state: GraphState) -> GraphState:
    """
    Care Node — Gọi Qwen 3.5 Flash sinh câu trả lời đồng cảm.

    Input:
      - state["intent"]: Tên intent (HO_TRO_SINH_VIEN, KHIEU_NAI_GOP_Y)
      - state["standalone_query"]: Câu hỏi của sinh viên

    Output:
      - state["final_response"]: Câu trả lời đồng cảm + thông tin liên hệ
      - state["response_source"]: "care_template"
      - state["next_node"]: "response"
    """
    intent = state.get("intent", "")
    user_query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    # Lấy thông tin liên hệ theo intent
    contact = _care_config.get_contact(intent)
    contact_text = contact.to_text()

    # Gọi Qwen LLM
    try:
        system_prompt = _build_system_prompt(intent, contact_text)
        care_response = _call_qwen(system_prompt, user_query)
        elapsed = time.time() - start_time
        logger.info(
            "Care Node [%.3fs] Qwen OK, intent='%s', %d ky tu",
            elapsed, intent, len(care_response)
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            "Care Node [%.3fs] Qwen loi: %s -> Fallback",
            elapsed, e, exc_info=True
        )
        care_response = _build_fallback_message(contact_text)

    return {
        **state,
        "final_response": care_response,
        "response_source": "care_template",
        "next_node": "response",
    }
