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
import asyncio
from app.services.langgraph.state import GraphState
from app.utils.guardian_utils import GuardianService


def contextual_guard_node(state: GraphState) -> GraphState:
    """
    🔴 CHỐT 2: CONTEXTUAL-GUARD — Chặn tinh trên standalone_query.

    Input:  state["standalone_query"] (từ Context Node)
    Output: state["contextual_guard_passed"], state["contextual_guard_blocked_layer"],
            state["contextual_guard_message"], state["next_node"],
            (state["final_response"] nếu blocked)

    Chỉ chạy khi: state["fast_scan_passed"] == True
                   state["standalone_query"] đã được sinh bởi Context Node
    """
    # Dùng standalone_query (đã có ngữ cảnh) thay vì user_query thô
    query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    # ════════════════════════════════════════════════════════
    # LAYER 2: LLM Prompt Guard (Song song 2a + 2b)
    # 2a: Llama 86M (Groq) — Fast score-based
    # 2b: Qwen 7B (OpenRouter) — Deep Vietnamese check
    # ════════════════════════════════════════════════════════
    try:
        is_valid, msg = asyncio.run(
            GuardianService.check_layer_2_concurrent(query)
        )
    except RuntimeError:
        # Nếu đang chạy trong event loop có sẵn (FastAPI)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    GuardianService.check_layer_2_concurrent(query)
                )
                is_valid, msg = future.result(timeout=15)
        else:
            is_valid, msg = True, ""

    elapsed = time.time() - start_time

    if not is_valid:
        # Xác định layer nào chặn từ message
        blocked_layer = "2a" if "2a" in msg else "2b"
        return {
            **state,
            "contextual_guard_passed": False,
            "contextual_guard_blocked_layer": blocked_layer,
            "contextual_guard_message": f"[Contextual-Guard L{blocked_layer} — {elapsed:.3f}s] {msg}",
            "next_node": "end",
            "final_response": msg,
            "response_source": "contextual_guard",
        }

    # ════════════════════════════════════════════════════════
    # ✅ PASS — standalone_query an toàn
    # Cho phép chạy Multi-Query + Intent Classification
    # ════════════════════════════════════════════════════════
    return {
        **state,
        "contextual_guard_passed": True,
        "contextual_guard_blocked_layer": None,
        "contextual_guard_message": f"[Contextual-Guard PASS — {elapsed:.3f}s] standalone_query an toàn",
        "next_node": "multi_query",
    }


def contextual_guard_router(state: GraphState) -> str:
    """
    Conditional Edge: contextual_guard → multi_query (SAFE) hoặc end (BLOCKED).

    Luồng sau khi PASS:
      contextual_guard → multi_query → embedding → cache → intent → agent
    """
    if state.get("contextual_guard_passed", False):
        return "multi_query"
    return "end"
