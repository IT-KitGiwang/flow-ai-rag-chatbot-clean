"""
Fast-Scan Node — Chốt 1: Chặn thô TRƯỚC khi gọi Gemini.

Vị trí trong Graph:
  [START] → [fast_scan_node] → [context_node] → ...
                              ↘ [END] (nếu bị chặn)

Nhiệm vụ:
  Quét user_query thô bằng Regex + kiểm tra độ dài.
  KHÔNG GỌI API. Chi phí: $0. Thời gian: ~0ms.

Layers:
  Layer 0: Input Validation — Max 800 ký tự (chống DoS/spam)
  Layer 1a: Keyword Filter — Từ cấm nhạy cảm (bạo lực, ma tuý, cờ bạc)
  Layer 1b: Injection Filter — Prompt Injection/Jailbreak pattern

Lợi ích:
  Nếu user gửi rác/tấn công → CHẶN NGAY, không tốn tiền gọi Gemini Flash Lite
  để reformulate cũng không tốn tiền gọi Llama/Qwen để Guard.
"""

import time
from app.services.langgraph.state import GraphState
from app.utils.guardian_utils import GuardianService


def fast_scan_node(state: GraphState) -> GraphState:
    """
    🟢 CHỐT 1: FAST-SCAN — Chặn thô trên user_query gốc.

    Input:  state["user_query"]
    Output: state["fast_scan_passed"], state["fast_scan_blocked_layer"],
            state["fast_scan_message"], state["normalized_query"],
            state["next_node"], (state["final_response"] nếu blocked)
    """
    query = state.get("user_query", "")
    start_time = time.time()

    # ── Chuẩn hóa teencode (lowercase + thay teencode) ──
    normalized = GuardianService.normalize_text(query)

    # ════════════════════════════════════════════════════════
    # LAYER 0: Input Validation — Chống DoS (Max 800 ký tự)
    # ════════════════════════════════════════════════════════
    is_valid, msg = GuardianService.check_layer_0_input_validation(query)
    if not is_valid:
        elapsed = time.time() - start_time
        return {
            **state,
            "normalized_query": normalized,
            "fast_scan_passed": False,
            "fast_scan_blocked_layer": 0,
            "fast_scan_message": f"[Fast-Scan L0 — {elapsed:.3f}s] {msg}",
            "next_node": "end",
            "final_response": msg,
            "response_source": "fast_scan",
        }

    # ════════════════════════════════════════════════════════
    # LAYER 1a: Keyword Filter — Từ cấm nhạy cảm
    # ════════════════════════════════════════════════════════
    is_valid, msg = GuardianService.check_layer_1_keyword_filter(normalized)
    if not is_valid:
        elapsed = time.time() - start_time
        return {
            **state,
            "normalized_query": normalized,
            "fast_scan_passed": False,
            "fast_scan_blocked_layer": 1,
            "fast_scan_message": f"[Fast-Scan L1a — {elapsed:.3f}s] {msg}",
            "next_node": "end",
            "final_response": msg,
            "response_source": "fast_scan",
        }

    # ════════════════════════════════════════════════════════
    # LAYER 1b: Injection Filter — Prompt Injection/Jailbreak
    # ════════════════════════════════════════════════════════
    is_valid, msg = GuardianService.check_layer_1b_injection_filter(normalized)
    if not is_valid:
        elapsed = time.time() - start_time
        return {
            **state,
            "normalized_query": normalized,
            "fast_scan_passed": False,
            "fast_scan_blocked_layer": 1,
            "fast_scan_message": f"[Fast-Scan L1b — {elapsed:.3f}s] {msg}",
            "next_node": "end",
            "final_response": msg,
            "response_source": "fast_scan",
        }

    # ════════════════════════════════════════════════════════
    # ✅ PASS — Cho qua Context Node (Gemini Lite reformulate)
    # ════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    return {
        **state,
        "normalized_query": normalized,
        "fast_scan_passed": True,
        "fast_scan_blocked_layer": None,
        "fast_scan_message": f"[Fast-Scan PASS — {elapsed:.3f}s] Sạch, cho qua Context Node",
        "next_node": "context",
    }


def fast_scan_router(state: GraphState) -> str:
    """Conditional Edge: fast_scan → context (SAFE) hoặc end (BLOCKED)."""
    if state.get("fast_scan_passed", False):
        return "context"
    return "end"
