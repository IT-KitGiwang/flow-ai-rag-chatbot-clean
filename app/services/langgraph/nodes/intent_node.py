"""
Intent Router Node — Phân loại ý định người dùng bằng Qwen LLM.

Vị trí trong Graph:
  [embedding_node] → [intent_node] → [rag/form/pr/care/response]

Nhiệm vụ:
  Gọi IntentService (Qwen) để xác định intent của standalone_query.
  Ghi kết quả vào State và điều hướng sang Node tương ứng.

Routing:
  PROCEED_RAG    → rag_node   (Tìm kiếm tuyển sinh, học phí, CTĐT...)
  PROCEED_FORM   → form_node  (Yêu cầu mẫu đơn)
  PROCEED_PR     → pr_node    (Thành tích, so sánh, cơ hội việc làm...)
  PROCEED_CARE   → care_node  (Tâm lý, bỏ học, khiếu nại...)
  GREET          → response   (Template chào hỏi — $0, 0ms)
  CLARIFY        → response   (Template hỏi lại — $0, 0ms)
  BLOCK_FALLBACK → response   (Fallback cứng từ YAML — $0, 0ms)
"""

import time
from app.services.langgraph.state import GraphState
from app.services.intent_service import classify_intent
from app.core.config import query_flow_config

# ── Map intent_action → next_node ──
_ACTION_TO_NODE: dict = {
    "PROCEED_RAG":    "rag",
    "PROCEED_FORM":   "form",
    "PROCEED_PR":     "pr",
    "PROCEED_CARE":   "care",
    "GREET":          "response",
    "CLARIFY":        "response",
    "BLOCK_FALLBACK": "response",
}


def intent_node(state: GraphState) -> GraphState:
    """
    🧭 INTENT ROUTER NODE

    Input:
      - state["standalone_query"]: Câu hỏi đã reformulate (từ Context Node)

    Output:
      - state["intent"]:          Tên intent ("HOC_PHI_HOC_BONG", ...)
      - state["intent_summary"]:  Tóm tắt câu hỏi (từ Qwen)
      - state["intent_action"]:   Action routing ("PROCEED_RAG", ...)
      - state["next_node"]:       "rag" / "form" / "pr" / "care" / "response"
      - state["final_response"]:  Ghi sẵn nếu GREET / CLARIFY / BLOCK_FALLBACK
      - state["response_source"]: Nguồn gốc final_response
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    # ── Phân loại intent (Qwen + edge case ngắn) ──
    result = classify_intent(standalone_query=standalone_query)

    intent        = result["intent"]
    intent_summary = result["intent_summary"]
    intent_action  = result["intent_action"]
    next_node      = _ACTION_TO_NODE.get(intent_action, "response")
    elapsed        = time.time() - start_time

    print(f"   [Intent Node — {elapsed:.3f}s] intent='{intent}' action='{intent_action}' → {next_node}")

    # ════════════════════════════════════════════════════════
    # XỬ LÝ CÁC NHÓM KHÔNG CẦN GỌI RAG (Trả ngay lập tức)
    # ════════════════════════════════════════════════════════

    # ── GREET: Chào lại + Mời hỏi ──
    if intent_action == "GREET":
        greet_msg = query_flow_config.response_templates.get_greet()
        print(f"   [Intent Node] 👋 GREET → Trả template ngay")
        return {
            **state,
            "intent": intent,
            "intent_summary": intent_summary,
            "intent_action": intent_action,
            "next_node": "response",
            "final_response": greet_msg,
            "response_source": "greet_template",
        }

    # ── CLARIFY: Hỏi lại nhẹ nhàng ──
    if intent_action == "CLARIFY":
        clarify_msg = query_flow_config.response_templates.get_clarify()
        print(f"   [Intent Node] 🔍 CLARIFY → Mời user nói rõ hơn")
        return {
            **state,
            "intent": intent,
            "intent_summary": intent_summary,
            "intent_action": intent_action,
            "next_node": "response",
            "final_response": clarify_msg,
            "response_source": "clarify_template",
        }

    # ── BLOCK_FALLBACK: Intent nhóm 4 lọt xuống ──
    if intent_action == "BLOCK_FALLBACK":
        semantic_cfg = query_flow_config.semantic_router
        fallback_msg = semantic_cfg.fallbacks.get(
            intent,
            semantic_cfg.fallback_out_of_scope
        ).strip()
        print(f"   [Intent Node] 🚫 BLOCK_FALLBACK → intent='{intent}'")
        return {
            **state,
            "intent": intent,
            "intent_summary": intent_summary,
            "intent_action": intent_action,
            "next_node": "response",
            "final_response": fallback_msg,
            "response_source": "intent_block",
        }

    # ════════════════════════════════════════════════════════
    # PROCEED: Chuyển sang Agent Node (RAG / Form / PR / Care)
    # ════════════════════════════════════════════════════════
    return {
        **state,
        "intent": intent,
        "intent_summary": intent_summary,
        "intent_action": intent_action,
        "next_node": next_node,
        "final_response": state.get("final_response", ""),
        "response_source": state.get("response_source", ""),
    }


def intent_router(state: GraphState) -> str:
    """
    Conditional Edge: intent → rag / form / pr / care / response
    """
    return state.get("next_node", "response")
