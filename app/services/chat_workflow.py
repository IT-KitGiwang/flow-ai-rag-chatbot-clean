"""
Chat Workflow — Production Runner cho LangGraph Pipeline.

Đóng gói toàn bộ luồng xử lý tin nhắn:
  Fast Scan → Context → Guard → Multi-Query → Embedding → RAG → Intent → Agent Dispatch → Response

Input:  (query: str, chat_history: list[dict])
Output: {"response": str, "source": str, "intent": str, "elapsed": float}
"""

import time
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Import tất cả Node functions ──
from app.services.langgraph.nodes.fast_scan_node import fast_scan_node
from app.services.langgraph.nodes.context_node import context_node
from app.services.langgraph.nodes.contextual_guard_node import contextual_guard_node
from app.services.langgraph.nodes.multi_query_node import multi_query_node
from app.services.langgraph.nodes.embedding_node import embedding_node
from app.services.langgraph.nodes.rag_node import rag_node
from app.services.langgraph.nodes.intent_node import intent_node
from app.services.langgraph.nodes.response_node import response_node
from app.services.langgraph.nodes.care_node import care_node

# Sub-graphs
from app.services.langgraph.nodes.proceed_form.graph import form_node
from app.services.langgraph.nodes.proceed_rag_search.graph import proceed_rag_search_pipeline


def _run_node(node_fn, state: dict, node_name: str) -> dict:
    """Chạy 1 node, đo thời gian, log kết quả."""
    t0 = time.time()
    try:
        new_state = node_fn(state)
        elapsed = time.time() - t0
        logger.debug(
            "Pipeline - %s completed in %.3fs | next_node=%s",
            node_name, elapsed, new_state.get("next_node", "?"),
        )
        return new_state
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("Pipeline - %s FAILED after %.3fs: %s", node_name, elapsed, e, exc_info=True)
        return {
            **state,
            "next_node": "end",
            "final_response": f"Xin lỗi, hệ thống gặp lỗi khi xử lý. Vui lòng thử lại sau.",
            "response_source": "error",
        }


def run_chat_pipeline(
    query: str,
    chat_history: Optional[list[dict]] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Chạy toàn bộ LangGraph pipeline từ query → final_response.

    Args:
        query: Tin nhắn của người dùng.
        chat_history: Lịch sử chat (danh sách dict {role, content}).
        session_id: ID phiên chat (cho logging).

    Returns:
        {
            "response": str,
            "source": str,           # "rag_direct" | "form_template" | "care_agent" | ...
            "intent": str,            # "PROCEED_RAG" | "PROCEED_FORM" | ...
            "intent_action": str,
            "blocked": bool,          # True nếu bị chặn bởi guardian
            "blocked_reason": str,
            "elapsed_seconds": float,
        }
    """
    pipeline_start = time.time()
    log_prefix = f"[{session_id[:8]}]" if session_id else "[chat]"

    logger.info("%s Pipeline START | query='%s' history=%d msgs",
                log_prefix, query[:80], len(chat_history or []))

    # ── Khởi tạo state ──
    state = {
        "user_query": query,
        "chat_history": chat_history or [],
        "original_query": "",
        "query_was_summarized": False,
        "normalized_query": "",
        "standalone_query": "",
        "fast_scan_passed": None,
        "fast_scan_blocked_layer": None,
        "fast_scan_message": "",
        "contextual_guard_passed": None,
        "contextual_guard_blocked_layer": None,
        "contextual_guard_message": "",
        "multi_queries": [],
        "query_embeddings": [],
        "rag_context": "",
        "retrieved_chunks": [],
        "rag_confidence_failed": False,
        "top1_cosine_score": 0.0,
        "intent": "",
        "intent_summary": "",
        "intent_action": "",
        "program_level_filter": None,
        "program_name_filter": None,
        "next_node": "",
        "final_response": "",
        "response_source": "",
    }

    # ═══════════════════════════════════════════════
    # NODE 1: FAST SCAN (Regex, 0ms, $0)
    # ═══════════════════════════════════════════════
    state = _run_node(fast_scan_node, state, "FastScan")
    if not state.get("fast_scan_passed"):
        elapsed = time.time() - pipeline_start
        logger.info("%s Pipeline BLOCKED at FastScan (%.3fs)", log_prefix, elapsed)
        return {
            "response": state.get("final_response", ""),
            "source": state.get("response_source", "fast_scan"),
            "intent": "BLOCKED",
            "intent_action": "BLOCK_FALLBACK",
            "blocked": True,
            "blocked_reason": state.get("fast_scan_message", ""),
            "elapsed_seconds": round(elapsed, 3),
        }

    # ═══════════════════════════════════════════════
    # NODE 2: CONTEXT (Reformulate query)
    # ═══════════════════════════════════════════════
    state = _run_node(context_node, state, "Context")

    # ═══════════════════════════════════════════════
    # NODE 3: CONTEXTUAL GUARD (Deep LLM check)
    # ═══════════════════════════════════════════════
    state = _run_node(contextual_guard_node, state, "ContextualGuard")
    if not state.get("contextual_guard_passed"):
        elapsed = time.time() - pipeline_start
        logger.info("%s Pipeline BLOCKED at ContextualGuard (%.3fs)", log_prefix, elapsed)
        return {
            "response": state.get("final_response", ""),
            "source": state.get("response_source", "contextual_guard"),
            "intent": "BLOCKED",
            "intent_action": "BLOCK_FALLBACK",
            "blocked": True,
            "blocked_reason": state.get("contextual_guard_message", ""),
            "elapsed_seconds": round(elapsed, 3),
        }

    # ═══════════════════════════════════════════════
    # NODE 4: MULTI-QUERY (Sinh biến thể)
    # ═══════════════════════════════════════════════
    state = _run_node(multi_query_node, state, "MultiQuery")

    # ═══════════════════════════════════════════════
    # NODE 5: EMBEDDING (BGE-M3 batch)
    # ═══════════════════════════════════════════════
    state = _run_node(embedding_node, state, "Embedding")

    # ═══════════════════════════════════════════════
    # NODE 6: RAG (Hybrid Search DB)
    # ═══════════════════════════════════════════════
    state = _run_node(rag_node, state, "RAG")

    # ═══════════════════════════════════════════════
    # NODE 7: INTENT (Router trung tâm)
    # ═══════════════════════════════════════════════
    state = _run_node(intent_node, state, "Intent")

    # ═══════════════════════════════════════════════
    # NODE 8: AGENT DISPATCH (tùy intent_action)
    # ═══════════════════════════════════════════════
    next_node = state.get("next_node", "response")

    if next_node == "form":
        state = _run_node(form_node, state, "FormAgent")
    elif next_node == "care":
        state = _run_node(care_node, state, "CareAgent")
    elif next_node == "rag_search":
        state = _run_node(proceed_rag_search_pipeline, state, "RAGSearch")

    # ═══════════════════════════════════════════════
    # NODE FINAL: RESPONSE (LLM Sinh câu trả lời)
    # ═══════════════════════════════════════════════
    state = _run_node(response_node, state, "Response")

    elapsed = time.time() - pipeline_start
    logger.info(
        "%s Pipeline DONE (%.3fs) | intent=%s source=%s response=%d chars",
        log_prefix, elapsed,
        state.get("intent", "?"),
        state.get("response_source", "?"),
        len(state.get("final_response", "")),
    )

    return {
        "response": state.get("final_response", ""),
        "source": state.get("response_source", ""),
        "intent": state.get("intent", ""),
        "intent_action": state.get("intent_action", ""),
        "blocked": False,
        "blocked_reason": "",
        "elapsed_seconds": round(elapsed, 3),
    }
