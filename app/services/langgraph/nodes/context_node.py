"""
Context Node — Cối Xay Ngữ Cảnh (Query Reformulation).

Vị trí trong Graph:
  [fast_scan_node] → [context_node] → [contextual_guard_node] → ...

Nhiệm vụ:
  1. Đọc chat_history (10 lượt gần nhất) + user_query
  2. Gọi Gemini Flash Lite để "dịch" câu hỏi lửng lơ
     thành câu hỏi độc lập có đầy đủ ngữ cảnh (standalone_query).
  3. Nếu không có lịch sử → skip reformulation (tiết kiệm 1 API call).

Model: google/gemini-3.1-flash-lite-preview (OpenRouter)
Chi phí: ~$0.000002/query (gần như miễn phí)
Latency: ~200ms

Ví dụ:
  chat_history: [User: "Học phí QTKD?", Bot: "600.000đ/tín chỉ"]
  user_query: "Thế còn ngành Marketing?"
  → standalone_query: "Học phí ngành Marketing là bao nhiêu?"
"""

import json
import time
import urllib.request
import urllib.error
from app.services.langgraph.state import GraphState
from app.core.config import query_flow_config


def _build_history_prompt(chat_history: list, max_turns: int) -> str:
    """
    Xây dựng chuỗi lịch sử hội thoại để đưa vào prompt.
    Chỉ lấy max_turns cặp gần nhất (Sliding Window).
    """
    if not chat_history:
        return ""

    # Cắt lấy max_turns cặp gần nhất (mỗi cặp = 2 messages)
    max_messages = max_turns * 2
    recent = chat_history[-max_messages:]

    lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Cắt bớt message quá dài
        max_tokens = query_flow_config.memory.max_tokens_per_message
        if len(content) > max_tokens:
            content = content[:max_tokens] + "..."

        if role == "user":
            lines.append(f"Người dùng: {content}")
        else:
            lines.append(f"Bot: {content}")

    return "\n".join(lines)


def _call_gemini_api(
    system_prompt: str,
    user_content: str,
    config_section,
) -> str:
    """
    Gọi API LLM (OpenRouter) cho Gemini Flash Lite.
    Dùng chung cho cả Reformulation và Multi-Query.
    """
    api_key = query_flow_config.api_keys.get_key(config_section.provider)
    base_url = query_flow_config.api_keys.get_base_url(config_section.provider)

    if not api_key:
        raise ValueError(
            f"Chưa cấu hình API Key cho provider '{config_section.provider}'"
        )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "UFM-Admission-Bot/1.0",
    }
    data = {
        "model": config_section.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": config_section.temperature,
        "max_tokens": config_section.max_tokens,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config_section.timeout_seconds) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()


def _reformulate_query(user_query: str, chat_history: list) -> str:
    """
    Gọi Gemini Flash Lite để dịch câu hỏi lửng lơ thành câu hỏi độc lập.

    Input:
      user_query: "Thế còn ngành Marketing?"
      chat_history: [{"role": "user", "content": "Học phí QTKD?"}, ...]

    Output:
      "Học phí ngành Marketing là bao nhiêu?"
    """
    config = query_flow_config.query_reformulation

    # Xây prompt lịch sử
    max_turns = query_flow_config.memory.max_history_turns
    history_text = _build_history_prompt(chat_history, max_turns)

    # Ghép user_content
    user_content = (
        f"LỊCH SỬ HỘI THOẠI:\n{history_text}\n\n"
        f"CÂU HỎI MỚI NHẤT CỦA NGƯỜI DÙNG:\n{user_query}"
    )

    return _call_gemini_api(
        system_prompt=config.system_prompt,
        user_content=user_content,
        config_section=config,
    )


# ================================================================
# CONTEXT NODE — Hàm chính cho Graph
# ================================================================
def context_node(state: GraphState) -> GraphState:
    """
    🔄 CONTEXT NODE — Cối Xay Ngữ Cảnh.

    Input:
      - state["user_query"]: Câu hỏi thô (đã qua Fast-Scan)
      - state["chat_history"]: Lịch sử hội thoại (10 lượt gần nhất)

    Output:
      - state["standalone_query"]: Câu hỏi đã reformulate (hoặc giữ nguyên)
      - state["next_node"]: "contextual_guard"

    Logic:
      1. Nếu chat_history RỖNG + skip_if_no_history = true
         → standalone_query = user_query (BỎ QUA API call, tiết kiệm tiền)
      2. Nếu có lịch sử
         → Gọi Gemini Flash Lite dịch câu hỏi
      3. Nếu API lỗi
         → Fallback: standalone_query = user_query (fail-safe)
    """
    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])
    config = query_flow_config.query_reformulation
    start_time = time.time()

    # ── Trường hợp 1: Reformulation bị tắt ──
    if not config.enabled:
        return {
            **state,
            "standalone_query": user_query,
            "next_node": "contextual_guard",
        }

    # ── Trường hợp 2: Không có lịch sử → Skip (tiết kiệm $) ──
    if config.skip_if_no_history and (not chat_history or len(chat_history) == 0):
        elapsed = time.time() - start_time
        print(f"   [Context Node — {elapsed:.3f}s] Không có history → giữ nguyên user_query")
        return {
            **state,
            "standalone_query": user_query,
            "next_node": "contextual_guard",
        }

    # ── Trường hợp 3: Có lịch sử → Gọi Gemini để reformulate ──
    try:
        standalone = _reformulate_query(user_query, chat_history)
        elapsed = time.time() - start_time
        print(f"   [Context Node — {elapsed:.3f}s] Reformulated:")
        print(f"     user_query       : \"{user_query}\"")
        print(f"     standalone_query : \"{standalone}\"")
        return {
            **state,
            "standalone_query": standalone,
            "next_node": "contextual_guard",
        }
    except urllib.error.URLError as e:
        # Timeout hoặc lỗi mạng → Fallback giữ nguyên
        elapsed = time.time() - start_time
        print(f"   [Context Node — {elapsed:.3f}s] ⚠️ API timeout/error: {e}")
        print(f"     Fallback: standalone_query = user_query")
        return {
            **state,
            "standalone_query": user_query,
            "next_node": "contextual_guard",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   [Context Node — {elapsed:.3f}s] ⚠️ Unexpected error: {e}")
        print(f"     Fallback: standalone_query = user_query")
        return {
            **state,
            "standalone_query": user_query,
            "next_node": "contextual_guard",
        }


def context_router(state: GraphState) -> str:
    """Conditional Edge: context → contextual_guard (luôn luôn)."""
    return "contextual_guard"
