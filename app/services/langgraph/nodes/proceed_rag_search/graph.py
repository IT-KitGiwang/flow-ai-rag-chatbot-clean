"""
Proceed RAG Search — Sub-pipeline Graph (v2: Self-RAG Evaluator).

Luồng TỐI ƯU v2:

  [Evaluator Gate] ── Gemini 3.0 Flash phán đoán DB đủ hay thiếu
      ↓ (YES: DB đủ)              ↓ (NO: DB thiếu, hoặc Intent PR)
      Trả thẳng rag_context       [pr_query_node] → [CACHE] → [web_search]
      → Bỏ qua toàn bộ Web        → Gộp rag_context + web_results
      → Tiết kiệm 5-15s            → final_response

Quy tắc rẽ nhánh:
  - PROCEED_RAG_PR_SEARCH (uy tín, thành tích) → LUÔN chạy Web Search
  - PROCEED_RAG_UFM_SEARCH (thông tin đào tạo) → Chạy Evaluator trước:
      YES → Trả context DB, bỏ Web Search
      NO  → Chạy Web Search bổ sung
"""

import time
from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.proceed_rag_search.pr_query_node import pr_query_node
from app.services.langgraph.nodes.proceed_rag_search.web_search_node import web_search_node
from app.services.langgraph.nodes.proceed_rag_search.search_cache import cache_lookup, cache_save
from app.services.langgraph.nodes.proceed_rag_search.evaluator import evaluate_rag_context
from app.core.config import query_flow_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _merge_context(rag_context: str, web_results: str, web_citations: list) -> str:
    """
    Ghép ngữ cảnh từ DB nội bộ + Web Search thành 1 khối text.
    Không gọi LLM — chỉ format chuỗi Python thuần.
    """
    parts = []

    if rag_context:
        parts.append(f"[DỮ LIỆU NỘI BỘ]\n{rag_context}")

    if web_results:
        parts.append(f"[DỮ LIỆU TỪ WEB]\n{web_results}")
        if web_citations:
            links = "\n".join(f"- [{c['text']}]({c['url']})" for c in web_citations)
            parts.append(f"[NGUỒN TRÍCH DẪN]\n{links}")

    if not parts:
        return ""

    return "\n\n".join(parts)


def proceed_rag_search_pipeline(state: GraphState) -> GraphState:
    """
    🔍 PROCEED RAG SEARCH PIPELINE v2 — Self-RAG Evaluator + Web Search.
    """
    pipeline_start = time.time()
    action = state.get("intent_action", "PROCEED_RAG_PR_SEARCH")
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    rag_context = state.get("rag_context") or ""

    logger.info("PROCEED RAG SEARCH v2 - Nhanh: %s, RAG context: %d ky tu", action, len(rag_context))

    # ── Khởi tạo state ──
    state = {
        **state,
        "ufm_search_queries": None,
        "pr_search_query": None,
        "web_search_results": None,
        "web_search_citations": None,
        "search_cache_hit": False,
        "search_cache_similarity": 0.0,
    }

    # ══════════════════════════════════════════════════════════
    # Bước 0: SELF-RAG EVALUATOR GATE
    # Chỉ áp dụng cho nhánh UFM_SEARCH (thông tin đào tạo nội bộ).
    # Nhánh PR_SEARCH (uy tín, thành tích) LUÔN chạy Web Search.
    # ══════════════════════════════════════════════════════════
    skip_web_search = False

    if action == "PROCEED_RAG_UFM_SEARCH" and rag_context:
        logger.info("Buoc 0: Self-RAG Evaluator...")
        is_sufficient = evaluate_rag_context(
            standalone_query=standalone_query,
            rag_context=rag_context,
        )

        if is_sufficient:
            skip_web_search = True
            logger.info("EVALUATOR: YES -> DB du, bo qua Web Search")
        else:
            logger.info("EVALUATOR: NO -> DB thieu, tiep tuc Web Search")
    elif action == "PROCEED_RAG_PR_SEARCH":
        logger.info("Buoc 0: Nhanh PR -> LUON chay Web Search")
    elif not rag_context:
        logger.info("Buoc 0: Context DB rong -> bat buoc Web Search")

    # ══════════════════════════════════════════════════════════
    # NHÁNH A: DB ĐỦ → Trả thẳng context, kết thúc sớm
    # ══════════════════════════════════════════════════════════
    if skip_web_search:
        state["final_response"] = rag_context
        state["response_source"] = "rag_db_only"
        state["next_node"] = "response"

        elapsed = time.time() - pipeline_start
        logger.info(
            "PROCEED RAG SEARCH - Ket thuc som (%.2fs), DB only, %d ky tu",
            elapsed, len(rag_context)
        )
        return state

    # ══════════════════════════════════════════════════════════
    # NHÁNH B: CẦN WEB SEARCH (DB thiếu hoặc nhánh PR)
    # ══════════════════════════════════════════════════════════

    # ── Bước 1: Sinh Query (PR hoặc UFM) ──
    logger.info("Buoc 1: Sinh query...")
    state = pr_query_node(state)

    # ── Bước 2: SEMANTIC CACHE CHECK ──
    logger.info("Buoc 2: Kiem tra Semantic Cache...")
    cache_hit, cache_sim, cached_results, cached_citations = cache_lookup(
        query_text=standalone_query,
        intent_action=action,
    )
    state["search_cache_hit"] = cache_hit
    state["search_cache_similarity"] = cache_sim

    if cache_hit:
        state["web_search_results"] = cached_results
        state["web_search_citations"] = cached_citations
        logger.info("CACHE HIT -> Bo qua Web Search API")
    else:
        logger.info("Buoc 3: Web Search...")
        state = web_search_node(state)

        # Lưu cache nếu có kết quả
        if state.get("web_search_results"):
            cache_save(
                query_text=standalone_query,
                intent_action=action,
                web_results=state["web_search_results"],
                web_citations=state.get("web_search_citations") or [],
            )

    # ── Bước 4: GỘP NGỮ CẢNH (Python thuần, KHÔNG gọi LLM) ──
    logger.info("Buoc 4: Gop ngu canh...")
    merged = _merge_context(
        rag_context=rag_context,
        web_results=state.get("web_search_results") or "",
        web_citations=state.get("web_search_citations") or [],
    )

    # Ghi vào final_response (đây là NGỮ CẢNH cho LLM chính, không phải câu trả lời)
    state["final_response"] = merged if merged else rag_context
    state["response_source"] = "rag_search_context"
    state["next_node"] = "response"

    elapsed = time.time() - pipeline_start
    logger.info(
        "PROCEED RAG SEARCH - Hoan tat (%.2fs), cache=%s (%.4f), web_citations=%d, ctx=%d ky tu",
        elapsed,
        'HIT' if cache_hit else 'MISS',
        cache_sim,
        len(state.get('web_search_citations') or []),
        len(merged),
    )

    return state

