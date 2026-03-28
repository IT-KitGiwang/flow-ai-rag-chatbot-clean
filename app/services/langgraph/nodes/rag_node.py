"""
RAG Node — Truy xuất Context từ VectorDB nội bộ + LLM Context Curator.

Vị trí trong Graph:
  [embedding_node] → [rag_node] → [response_node] hoặc [rag_search]

Nhiệm vụ:
  1. Nhận query_embeddings từ Embedding Node
  2. Gọi Hybrid Retriever: Vector + BM25 → RRF → Top N Parents
  3. Context Curator (LLM) lọc giữ info liên quan, loại noise
  4. Ghi kết quả đã curate vào state["rag_context"]

Fallback: Nếu DB chưa sẵn sàng → rag_context = ""
"""

import time

from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.services.retriever_service import hybrid_retrieve
from app.core.config import query_flow_config
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)

_EMPTY_RAG = {"rag_context": "", "retrieved_chunks": []}


def _curate_context(standalone_query: str, raw_context: str) -> str:
    """
    Context Curator — LLM lọc ngữ cảnh, giữ info liên quan.
    Nếu LLM lỗi/timeout → trả raw_context nguyên bản (fail-safe).
    """
    config = query_flow_config.context_curator

    try:
        sys_prompt = prompt_manager.get_system("context_curator")
        user_content = prompt_manager.render_user(
            "context_curator",
            standalone_query=standalone_query,
            rag_context=raw_context,
        )

        t0 = time.time()
        curated = _call_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=config,
            node_key="context_evaluator",
        )
        elapsed = time.time() - t0

        curated_clean = (curated or "").strip()
        if not curated_clean:
            logger.warning("Context Curator [%.3fs]: LLM tra rong -> dung raw", elapsed)
            return raw_context

        # LLM trả "KHÔNG CÓ THÔNG TIN LIÊN QUAN" → context rỗng
        if curated_clean.upper() in ("KHÔNG CÓ THÔNG TIN LIÊN QUAN", "NONE", "N/A", "KHÔNG CÓ"):
            logger.info("Context Curator [%.3fs]: DB khong lien quan -> rong", elapsed)
            return ""

        logger.info("Context Curator [%.3fs]: OK, %d->%d chars", elapsed, len(raw_context), len(curated_clean))
        return curated_clean

    except Exception as e:
        logger.error("Context Curator loi: %s -> dung raw", e)
        return raw_context


def rag_node(state: GraphState) -> GraphState:
    """
    RAG Node — Hybrid Retrieval + Context Curator.

    Input:  state["standalone_query"], state["query_embeddings"]
    Output: state["rag_context"], state["retrieved_chunks"]
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    query_embeddings = state.get("query_embeddings", [])
    start_time = time.time()

    # Không có embedding → skip
    if not query_embeddings:
        elapsed = time.time() - start_time
        logger.warning("RAG Node [%.3fs] Khong co embeddings -> skip", elapsed)
        return {**state, **_EMPTY_RAG}

    primary_embedding = query_embeddings[0]

    try:
        program_level = state.get("program_level_filter")
        program_name = state.get("program_name_filter")

        result = hybrid_retrieve(
            query_text=standalone_query,
            query_embedding=primary_embedding,
            program_level=program_level,
            program_name=program_name,
            query_embeddings=query_embeddings,
        )

        raw_context = result["rag_context"]
        retrieved_chunks = result["retrieved_chunks"]
        top1_cosine = result.get("top1_cosine_score", 0.0)
        elapsed_db = time.time() - start_time

        logger.info(
            "RAG Node [%.3fs] Hybrid OK: vec=%d, bm25=%d, parents=%d, ctx=%d chars, top1=%.4f",
            elapsed_db, result['vector_count'], result['bm25_count'],
            len(result['parent_ids']), len(raw_context), top1_cosine
        )

        # Context Curator — lọc ngữ cảnh
        if raw_context:
            curated_context = _curate_context(standalone_query, raw_context)
        else:
            curated_context = ""
            logger.info("RAG Node: Context DB rong -> skip Curator")

        elapsed_total = time.time() - start_time
        logger.info(
            "RAG Node [%.3fs total] Curated: %d chars (raw=%d)",
            elapsed_total, len(curated_context), len(raw_context),
        )

        return {
            **state,
            "rag_context": curated_context,
            "retrieved_chunks": retrieved_chunks,
        }

    except ImportError as e:
        elapsed = time.time() - start_time
        logger.warning("RAG Node [%.3fs] psycopg2 chua cai: %s", elapsed, e)
        return {**state, **_EMPTY_RAG}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("RAG Node [%.3fs] DB error: %s", elapsed, e, exc_info=True)
        return {**state, **_EMPTY_RAG}
