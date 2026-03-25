"""
Contextual-Guard Node — Chốt 2: Chặn tinh SAU khi có ngữ cảnh.

Vị trí trong Graph:
  [context_node] → [contextual_guard_node] → [intent_node] → ...
                                            ↘ [END] (nếu bị chặn)

Nhiệm vụ:
  Quét standalone_query (đã reformulate, có ngữ cảnh đầy đủ) bằng:
    Layer 2a: Llama 86M Score-based (Groq, ~100ms)
    Layer 2b: Gemini 2.0 Flash (OpenRouter) — SAFE/UNSAFE JSON
  Hai model chạy SONG SONG — ai UNSAFE trước thì CHẶN ngay.
"""

import time
from app.services.langgraph.state import GraphState
from app.utils.guardian_utils import GuardianService
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _get_hotline() -> str:
    """Lazy-load block liên hệ đầy đủ từ contact_info.md (cache sau lần đầu)."""
    from app.core.config.contact_loader import get_contact_block
    return get_contact_block()


def _timed_check_2a(query: str):
    """Wrap Layer 2a với timing chi tiết."""
    t0 = time.time()
    try:
        is_valid, msg = GuardianService.check_layer_2a_prompt_guard_fast(query)
        elapsed = time.time() - t0
        status = "SAFE" if is_valid else "UNSAFE"
        logger.info("  ⏱️ Layer 2a (Groq Llama 86M): %.3fs → %s %s", elapsed, status, f"| {msg}" if msg else "")
        return is_valid, msg, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("  ⏱️ Layer 2a (Groq Llama 86M): %.3fs → ERROR: %s", elapsed, e)
        return True, f"Bỏ qua 2a ({e})", elapsed


def contextual_guard_node(state: GraphState) -> GraphState:
    """
    🔴 CHỐT 2: CONTEXTUAL-GUARD — Chặn tinh trên standalone_query.

    LAYER 2a: Chạy Score-based model (Llama Guard fast) cực nhanh.
    - UNSAFE → block ngay
    - SAFE → PASS sang Intent Node (nơi sẽ check intent + an toàn bằng LLM lớn)
    """
    query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    logger.info("CONTEXTUAL GUARD — Bắt đầu quét Layer 2a (Groq)...")

    # Chỉ chạy _timed_check_2a (Layer 2b đã được gộp vào Intent Classification)
    is_valid, msg, layer_time = _timed_check_2a(query)

    if not is_valid:
        elapsed = time.time() - start_time
        logger.warning(
            "CONTEXTUAL GUARD [%.3fs] BLOCKED by L2a (layer_time=%.3fs)",
            elapsed, layer_time,
        )
        return {
            **state,
            "contextual_guard_passed": False,
            "contextual_guard_blocked_layer": "2a",
            "contextual_guard_message": f"[Contextual-Guard L2a — {elapsed:.3f}s] {msg}",
            "final_response": f"{msg}\n{_get_hotline()}",
            "response_source": "contextual_guard",
        }

    # ════════════════════════════════════════════════════════
    # ✅ PASS — standalone_query an toàn qua chốt 2a
    # ════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    logger.info("CONTEXTUAL GUARD [%.3fs] PASS", elapsed)
    return {
        **state,
        "contextual_guard_passed": True,
        "contextual_guard_blocked_layer": None,
        "contextual_guard_message": f"[Contextual-Guard PASS — {elapsed:.3f}s] standalone_query an toàn",
    }


