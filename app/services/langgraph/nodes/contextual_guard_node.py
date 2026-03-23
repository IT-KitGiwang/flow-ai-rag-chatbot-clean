"""
Contextual-Guard Node — Chốt 2: Chặn tinh SAU khi có ngữ cảnh.

Vị trí trong Graph:
  [context_node] → [contextual_guard_node] → [multi_query_node] → ...
                                            ↘ [END] (nếu bị chặn)

Nhiệm vụ:
  Quét standalone_query (đã reformulate, có ngữ cảnh đầy đủ) bằng:
    Layer 2a: Llama 86M Score-based (Groq, ~100ms)
    Layer 2b: Qwen 7B Vietnamese SAFE/UNSAFE (OpenRouter, ~500ms)
  Hai model chạy SONG SONG — ai phát hiện UNSAFE trước thì CHẶN ngay.

Tại sao phải chạy SAU Context Node?
  Vì user_query thô có thể chỉ là "Thế còn cái đó thì sao?" — trông vô hại.
  Nhưng sau khi Gemini reformulate thành "Làm cách nào để hack hệ thống UFM?"
  thì Contextual-Guard mới phát hiện được ý đồ thực sự.

Chi phí:
  Layer 2a (Groq Llama 86M): ~$0.000001/query
  Layer 2b (OpenRouter Qwen 7B): ~$0.00005/query
  Tổng: ~$0.00005/query — chỉ tốn khi đã qua Fast-Scan
"""

import time
import concurrent.futures
from app.services.langgraph.state import GraphState
from app.utils.guardian_utils import GuardianService
from app.utils.logger import get_logger

logger = get_logger(__name__)


def contextual_guard_node(state: GraphState) -> GraphState:
    """
    🔴 CHỐT 2: CONTEXTUAL-GUARD — Chặn tinh trên standalone_query.

    Chạy Layer 2a (Llama 86M) + Layer 2b (Qwen 7B) SONG SONG
    bằng ThreadPoolExecutor — không dùng asyncio (tránh overhead).
    """
    query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    # ════════════════════════════════════════════════════════
    # LAYER 2: Chạy 2a + 2b SONG SONG bằng ThreadPool (sync)
    # ════════════════════════════════════════════════════════
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_2a = pool.submit(GuardianService.check_layer_2a_prompt_guard_fast, query)
            future_2b = pool.submit(GuardianService.check_layer_2b_prompt_guard_deep, query)

            # Ai xong trước check trước — nếu UNSAFE thì chặn ngay
            for future in concurrent.futures.as_completed([future_2a, future_2b], timeout=8):
                is_valid, msg = future.result()
                if not is_valid:
                    # Cancel task còn lại (best effort)
                    future_2a.cancel()
                    future_2b.cancel()
                    elapsed = time.time() - start_time
                    blocked_layer = "2a" if future is future_2a else "2b"
                    logger.warning("CONTEXTUAL GUARD [%.3fs] BLOCKED by L%s", elapsed, blocked_layer)
                    return {
                        **state,
                        "contextual_guard_passed": False,
                        "contextual_guard_blocked_layer": blocked_layer,
                        "contextual_guard_message": f"[Contextual-Guard L{blocked_layer} — {elapsed:.3f}s] {msg}",
                        "final_response": msg,
                        "response_source": "contextual_guard",
                    }

    except concurrent.futures.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning("CONTEXTUAL GUARD [%.3fs] TIMEOUT → cho qua (SAFE)", elapsed)
        is_valid, msg = True, ""
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("CONTEXTUAL GUARD [%.3fs] ERROR: %s → cho qua (SAFE)", elapsed, e, exc_info=True)
        is_valid, msg = True, ""

    # ════════════════════════════════════════════════════════
    # ✅ PASS — standalone_query an toàn
    # ════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    logger.info("CONTEXTUAL GUARD [%.3fs] PASS", elapsed)
    return {
        **state,
        "contextual_guard_passed": True,
        "contextual_guard_blocked_layer": None,
        "contextual_guard_message": f"[Contextual-Guard PASS — {elapsed:.3f}s] standalone_query an toàn",
    }

