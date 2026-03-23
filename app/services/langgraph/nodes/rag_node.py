"""
RAG Node — Truy xuất Context từ VectorDB nội bộ (Hybrid Search).

Vị trí trong Graph:
  [embedding_node] → [rag_node] → [intent_node] hoặc [proceed_rag_search]

Nhiệm vụ:
  1. Nhận query_embeddings[0] (vector của standalone_query) từ Embedding Node.
  2. Gọi Hybrid Retriever: Vector Search + BM25 → RRF → Top 6 Parent.
  3. Ghi kết quả vào state["rag_context"] (chuỗi text) và state["retrieved_chunks"].
  4. CONFIDENCE GATE: Nếu top1_cosine < 0.85 → rag_confidence_failed = True.

Nếu Database chưa sẵn sàng → rag_context = "" (fallback an toàn).
Pipeline tiếp tục bình thường với context rỗng.

Model: KHÔNG gọi LLM — chỉ dùng SQL + Python thuần.
Chi phí: $0 (chỉ tốn DB query time ~50-100ms).
Latency: ~100-300ms tùy dữ liệu.
"""

import time
from app.services.langgraph.state import GraphState
from app.core.config import query_flow_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def rag_node(state: GraphState) -> GraphState:
    """
    🗄️ RAG NODE — Truy xuất ngữ cảnh từ VectorDB nội bộ (Hybrid Search).

    Input:
      - state["standalone_query"]:  Câu hỏi đã reformulate (từ Context Node)
      - state["query_embeddings"]:  [0] = vector 1024D của standalone_query

    Output:
      - state["rag_context"]:       Chuỗi text gộp từ 5 Parent Chunks
      - state["retrieved_chunks"]:  Danh sách RRF ranked chunks (debug/monitoring)

    Logic:
      1. Lấy vector embedding từ state (đã được Embedding Node chuẩn bị)
      2. Gọi hybrid_retrieve() → Vector + BM25 + RRF → 5 Parents
      3. Ghi context vào state
      4. Nếu DB lỗi / chưa sẵn sàng → rag_context = "" (fail-safe)
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    query_embeddings = state.get("query_embeddings", [])
    start_time = time.time()

    # ── Kiểm tra: Có vector embedding nào không? ──
    if not query_embeddings:
        elapsed = time.time() - start_time
        logger.warning("RAG Node [%.3fs] Khong co query_embeddings -> bo qua RAG", elapsed)
        return {
            **state,
            "rag_context": "",
            "retrieved_chunks": [],
        }

    # ── Lấy vector chính (standalone_query) ──
    primary_embedding = query_embeddings[0]

    # ── Gọi Hybrid Retriever ──
    try:
        from app.services.retriever_service import hybrid_retrieve

        # Lấy metadata filter từ Intent Node (nếu có)
        program_level = state.get("program_level_filter")
        program_name = state.get("program_name_filter")

        result = hybrid_retrieve(
            query_text=standalone_query,
            query_embedding=primary_embedding,
            program_level=program_level,
            program_name=program_name,
            query_embeddings=query_embeddings,
        )

        rag_context = result["rag_context"]
        retrieved_chunks = result["retrieved_chunks"]
        top1_cosine = result.get("top1_cosine_score", 0.0)
        elapsed = time.time() - start_time

        logger.info(
            "RAG Node [%.3fs] Hybrid OK: vec=%d, bm25=%d, parents=%d, ctx=%d chars, top1=%.4f",
            elapsed, result['vector_count'], result['bm25_count'],
            len(result['parent_ids']), len(rag_context), top1_cosine
        )

        # ════ CONFIDENCE GATE (chỉ áp dụng cho PROCEED_RAG) ════
        # Kiểm tra ngưỡng tin cậy: nếu top1 cosine < 0.85 → không đủ tin cậy
        retriever_cfg = query_flow_config.retriever
        gate_cfg = retriever_cfg.confidence_gate
        rag_confidence_failed = False

        if gate_cfg.enabled and top1_cosine < gate_cfg.min_top1_cosine:
            rag_confidence_failed = True
            logger.warning(
                "RAG Node CONFIDENCE GATE: top1=%.4f < %.2f -> khong du tin cay",
                top1_cosine, gate_cfg.min_top1_cosine
            )
        else:
            logger.info(
                "RAG Node CONFIDENCE GATE: top1=%.4f >= %.2f -> OK",
                top1_cosine, gate_cfg.min_top1_cosine
            )

        return {
            **state,
            "rag_context": rag_context,
            "retrieved_chunks": retrieved_chunks,
            "rag_confidence_failed": rag_confidence_failed,
            "top1_cosine_score": top1_cosine,
        }

    except ImportError as e:
        elapsed = time.time() - start_time
        logger.warning("RAG Node [%.3fs] psycopg2 chua cai: %s", elapsed, e)
        return {
            **state,
            "rag_context": "",
            "retrieved_chunks": [],
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("RAG Node [%.3fs] Loi truy xuat DB: %s", elapsed, e, exc_info=True)
        return {
            **state,
            "rag_context": "",
            "retrieved_chunks": [],
        }

