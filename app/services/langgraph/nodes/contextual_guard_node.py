"""
Contextual Guard Node — Chốt 2: Chặn tinh SAU khi có ngữ cảnh.

Vị trí trong Graph:
  [context_node] → [contextual_guard_node] → [intent_node] → ...
                                            ↘ [END] (nếu bị chặn)

Nhiệm vụ:
  Quét standalone_query (đã reformulate) bằng Layer 2a (Llama Guard fast).
  UNSAFE → block ngay. SAFE → chuyển sang Intent Node.
  (Layer 2b đã được gộp vào Intent Classification để giảm latency.)
"""

import time

from app.services.langgraph.state import GraphState
from app.core.config.contact_loader import get_contact_block
from app.utils.guardian_utils import GuardianService
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _timed_check_2a(query: str):
    """Chạy Layer 2a (Llama Guard Score-based) với timing."""
    t0 = time.time()
    try:
        is_valid, msg = GuardianService.check_layer_2a_prompt_guard_fast(query)
        elapsed = time.time() - t0
        status = "SAFE" if is_valid else "UNSAFE"
        logger.info("  Layer 2a: %.3fs -> %s %s", elapsed, status, f"| {msg}" if msg else "")
        return is_valid, msg, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("  Layer 2a: %.3fs -> ERROR: %s", elapsed, e)
        return True, f"Bỏ qua 2a ({e})", elapsed


def contextual_guard_node(state: GraphState) -> GraphState:
    """
    Contextual Guard — Quét standalone_query bằng Layer 2a.
    UNSAFE → block + trả contact info. SAFE → pass sang Intent Node.
    """
    query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    logger.info("CONTEXTUAL GUARD — Quet Layer 2a...")

    is_valid, msg, layer_time = _timed_check_2a(query)

    if not is_valid:
        elapsed = time.time() - start_time
        logger.warning(
            "CONTEXTUAL GUARD [%.3fs] BLOCKED by L2a (layer=%.3fs)",
            elapsed, layer_time,
        )
        return {
            **state,
            "contextual_guard_passed": False,
            "contextual_guard_blocked_layer": "2a",
            "contextual_guard_message": f"[Contextual-Guard L2a — {elapsed:.3f}s] {msg}",
            "final_response": f"{msg}\n{get_contact_block()}",
            "response_source": "contextual_guard",
        }

    # PASS
    elapsed = time.time() - start_time
    logger.info("CONTEXTUAL GUARD [%.3fs] PASS", elapsed)
    return {
        **state,
        "contextual_guard_passed": True,
        "contextual_guard_blocked_layer": None,
        "contextual_guard_message": f"[Contextual-Guard PASS — {elapsed:.3f}s] standalone_query an toan",
    }
