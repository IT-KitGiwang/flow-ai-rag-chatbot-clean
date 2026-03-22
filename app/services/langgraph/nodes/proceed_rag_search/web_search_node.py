"""
Web Search Node — Tìm kiếm Web bằng GPT-4o-mini Search Preview (qua OpenRouter).

Vị trí:
  [pr_query_node] → [web_search_node] → [synthesizer_node]

Nhiệm vụ:
  Gọi gpt-4o-mini-search-preview tìm thông tin từ web.
  - Nhánh UFM Search: Chỉ tìm trên 3 domain cốt lõi (ufm.edu.vn, tuyensinh, pdt)
  - Nhánh PR Search: Chỉ tìm trên 3 báo lớn nhất (thanhnien, vnexpress, tuoitre)

TỐI ƯU TỐC ĐỘ:
  - Giới hạn site: operator tối đa 3 domains (thay vì 15-17) để model không bị chậm crawl
  - Prompt ngắn gọn, không nhồi nhét domain list dài
  - max_tokens: 500 (chỉ cần raw data, không cần essay)

Model: openai/gpt-4o-mini-search-preview (via OpenRouter)
Fallback: web_search_results = None (Tiếp tục với RAG nội bộ)
"""

import json
import re
import time
import urllib.request
import urllib.error
from app.services.langgraph.state import GraphState
from app.core.config import query_flow_config
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# DOMAIN MAPPING: Chọn 3 domains phù hợp nhất theo ngữ cảnh
# ══════════════════════════════════════════════════════════

# Các từ khóa → domain ưu tiên (theo thứ tự quan trọng)
_UFM_DOMAIN_MAP = {
    # Tuyển sinh / Điểm chuẩn / Xét tuyển
    "tuyển sinh":      "tuyensinh.ufm.edu.vn",
    "điểm chuẩn":      "tuyensinh.ufm.edu.vn",
    "xét tuyển":        "tuyensinh.ufm.edu.vn",
    "chỉ tiêu":        "tuyensinh.ufm.edu.vn",
    "nguyện vọng":      "tuyensinh.ufm.edu.vn",
    "hồ sơ":           "tuyensinh.ufm.edu.vn",
    # Đào tạo / Học phí / Chương trình
    "đào tạo":         "pdt.ufm.edu.vn",
    "chương trình":    "pdt.ufm.edu.vn",
    "học phí":         "pdt.ufm.edu.vn",
    "tín chỉ":         "pdt.ufm.edu.vn",
    "lịch học":        "pdt.ufm.edu.vn",
    "thời khóa biểu":  "uis.ufm.edu.vn",
    # Nhập học
    "nhập học":        "nhaphoc.ufm.edu.vn",
    "thủ tục":         "nhaphoc.ufm.edu.vn",
    # Sinh viên
    "học bổng":        "ctsv.ufm.edu.vn",
    "rèn luyện":       "ctsv.ufm.edu.vn",
    "ký túc xá":       "ktx.ufm.edu.vn",
}

# Domain mặc định khi không match từ khóa nào
_UFM_DEFAULT_DOMAINS = ["ufm.edu.vn", "tuyensinh.ufm.edu.vn", "pdt.ufm.edu.vn"]

# PR: chỉ dùng 3 báo lớn nhất (thay vì 5) để giảm tải
_PR_TOP_DOMAINS = ["thanhnien.vn", "vnexpress.net", "tuoitre.vn"]


def _select_ufm_domains(query: str, max_domains: int = 3) -> list:
    """
    Chọn tối đa 3 domains UFM phù hợp nhất dựa trên từ khóa trong câu hỏi.
    Luôn bao gồm ufm.edu.vn làm fallback.
    """
    query_lower = query.lower()
    matched = set()

    for keyword, domain in _UFM_DOMAIN_MAP.items():
        if keyword in query_lower:
            matched.add(domain)

    # Luôn có ufm.edu.vn (trang chủ = catch-all)
    matched.add("ufm.edu.vn")

    if len(matched) >= max_domains:
        return list(matched)[:max_domains]

    # Bổ sung domain mặc định nếu chưa đủ
    for d in _UFM_DEFAULT_DOMAINS:
        if d not in matched:
            matched.add(d)
        if len(matched) >= max_domains:
            break

    return list(matched)[:max_domains]


def _build_search_query(
    standalone_query: str, 
    action: str, 
    ufm_queries: list, 
    pr_query: str
) -> str:
    """Xây dựng prompt search cực ngắn gọn + domain giới hạn."""
    
    if action == "PROCEED_RAG_UFM_SEARCH":
        domains = _select_ufm_domains(standalone_query)
        domain_filter = " OR ".join(f"site:{d}" for d in domains)
        
        # Gộp queries thành 1 câu search ngắn
        search_terms = standalone_query
        if ufm_queries:
            search_terms = " | ".join(ufm_queries[:2])
        
        return f"{search_terms} ({domain_filter})"

    else:  # PROCEED_RAG_PR_SEARCH
        domain_filter = " OR ".join(f"site:{d}" for d in _PR_TOP_DOMAINS)
        
        search_terms = pr_query or standalone_query
        return f"UFM {search_terms} ({domain_filter})"


def _extract_citations(text: str) -> list:
    """Trích xuất [text](url) thành list."""
    pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
    matches = re.findall(pattern, text)
    citations = []
    seen_urls = set()
    for text_part, url in matches:
        if url not in seen_urls:
            citations.append({"text": text_part.strip(), "url": url.strip()})
            seen_urls.add(url)
    return citations


def _call_search_model(system_prompt: str, user_content: str, config_section) -> str:
    """Gọi OpenRouter Search API có hỗ trợ Fallback tự động."""
    return _call_gemini_api_with_fallback(
        system_prompt=system_prompt,
        user_content=user_content,
        config_section=config_section,
        node_key="web_search"
    )


def web_search_node(state: GraphState) -> GraphState:
    """🌐 WEB SEARCH NODE (Domain thông minh + Tốc độ tối ưu)."""
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    action = state.get("intent_action", "")
    ufm_queries = state.get("ufm_search_queries") or []
    pr_query = state.get("pr_search_query")
    
    config = query_flow_config.web_search
    start_time = time.time()

    if not config.enabled:
        return {**state, "web_search_results": None, "web_search_citations": None, "next_node": "synthesizer"}

    try:
        search_prompt = _build_search_query(
            standalone_query=standalone_query,
            action=action,
            ufm_queries=ufm_queries,
            pr_query=pr_query,
        )

        # Log prompt gửi đi (debug)
        logger.info("Web Search - Prompt: '%s'", search_prompt[:120])

        raw_result = _call_search_model(
            system_prompt=prompt_manager.get_system("web_search_node"),
            user_content=search_prompt,
            config_section=config,
        )

        citations = _extract_citations(raw_result)
        elapsed = time.time() - start_time
        logger.info("Web Search [%.3fs] (%s) Tim duoc %d trich dan", elapsed, action, len(citations))

        return {
            **state,
            "web_search_results": raw_result,
            "web_search_citations": citations,
            "next_node": "synthesizer",
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Web Search [%.3fs] FALLBACK: %s", elapsed, e, exc_info=True)
        return {
            **state,
            "web_search_results": None,
            "web_search_citations": None,
            "next_node": "synthesizer",
        }
