"""
Web Search Node — Gemini 2.5 Flash + Google Search Tool (Native API).

Vị trí:
  [pr_query_node] → [web_search_node] → [synthesizer_node]

Nhiệm vụ:
  Gọi Gemini 2.5 Flash với google_search tool tìm thông tin từ web.
  - Nhánh UFM Search: Ưu tiên domain ufm.edu.vn và sub-domains
  - Nhánh PR Search: Ưu tiên báo lớn (thanhnien, vnexpress, tuoitre)

CƠ CHẾ:
  1. Gọi Google Gemini API native với tool google_search
  2. Gemini tự quyết định khi nào cần search và search gì
  3. Citations được trích từ groundingMetadata (chính xác, có verify)
  4. Nếu Google API lỗi → Fallback sang OpenRouter search models

Model: gemini-2.5-flash (Google native API + google_search tool)
Fallback: OpenRouter (gpt-4o-search-preview, perplexity/sonar)
"""

import json
import re
import time
import urllib.request
import urllib.error
from datetime import datetime
from app.services.langgraph.state import GraphState
from app.core.config import query_flow_config
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# DOMAIN MAPPING: Chọn domains phù hợp nhất theo ngữ cảnh
# ══════════════════════════════════════════════════════════

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

_UFM_DEFAULT_DOMAINS = ["ufm.edu.vn", "tuyensinh.ufm.edu.vn", "pdt.ufm.edu.vn"]
_PR_TOP_DOMAINS = ["thanhnien.vn", "vnexpress.net", "tuoitre.vn"]


def _select_ufm_domains(query: str, max_domains: int = 3) -> list:
    """Chọn tối đa 3 domains UFM phù hợp nhất dựa trên từ khóa."""
    query_lower = query.lower()
    matched = set()

    for keyword, domain in _UFM_DOMAIN_MAP.items():
        if keyword in query_lower:
            matched.add(domain)

    matched.add("ufm.edu.vn")

    if len(matched) >= max_domains:
        return list(matched)[:max_domains]

    for d in _UFM_DEFAULT_DOMAINS:
        if d not in matched:
            matched.add(d)
        if len(matched) >= max_domains:
            break

    return list(matched)[:max_domains]


def _has_year(text: str) -> bool:
    """Kiểm tra câu query đã chứa năm (2020-2030) chưa."""
    return bool(re.search(r'20[2-3]\d', text))


def _inject_year_anchor(query: str) -> str:
    """Bơm neo thời gian vào query nếu chưa có năm."""
    if _has_year(query):
        return query
    current_year = datetime.now().year
    prev_year = current_year - 1
    return f"{query} năm {prev_year} {current_year}"


def _build_search_query(
    standalone_query: str,
    action: str,
    ufm_queries: list,
    pr_query: str,
) -> str:
    """Xây dựng user prompt cho Gemini + Google Search tool."""

    if action == "PROCEED_RAG_UFM_SEARCH":
        domains = _select_ufm_domains(standalone_query)
        domain_hint = ", ".join(domains)

        search_terms = standalone_query
        if ufm_queries:
            search_terms = " | ".join(ufm_queries[:2])

        search_terms = _inject_year_anchor(search_terms)

        return (
            f"Tìm thông tin chính xác về: {search_terms}\n"
            f"Ưu tiên nguồn: {domain_hint}\n"
            f"Trả lời bằng tiếng Việt, ngắn gọn, có dẫn nguồn."
        )

    else:  # PROCEED_RAG_PR_SEARCH
        domain_hint = ", ".join(_PR_TOP_DOMAINS)
        search_terms = pr_query or standalone_query
        search_terms = _inject_year_anchor(search_terms)

        return (
            f"Tìm bài báo về Trường ĐH Tài chính - Marketing (UFM): {search_terms}\n"
            f"Ưu tiên nguồn: {domain_hint}\n"
            f"Trả lời bằng tiếng Việt, trích dẫn tên bài báo và link."
        )


def _extract_citations_from_text(text: str) -> list:
    """Fallback: Trích xuất [text](url) từ text nếu không có groundingMetadata."""
    pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
    matches = re.findall(pattern, text)
    citations = []
    seen_urls = set()
    for text_part, url in matches:
        if url not in seen_urls:
            citations.append({"text": text_part.strip(), "url": url.strip()})
            seen_urls.add(url)
    return citations


# ══════════════════════════════════════════════════════════
# GOOGLE GEMINI NATIVE API — với google_search tool
# ══════════════════════════════════════════════════════════

def _call_gemini_native_with_search(
    system_prompt: str,
    user_content: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple:
    """
    Gọi Google Gemini API native với google_search tool.

    Returns: (raw_text, citations)
      - raw_text: Nội dung phản hồi từ Gemini
      - citations: List[{text, url}] từ groundingMetadata
    """
    api_key = query_flow_config.api_keys.get_key("google")
    base_url = query_flow_config.api_keys.get_base_url("google")

    if not api_key:
        raise ValueError("Chưa cấu hình GOOGLE_API_KEY trong .env")

    url = f"{base_url.rstrip('/')}/models/{model}:generateContent?key={api_key}"

    body = {
        "contents": [
            {"role": "user", "parts": [{"text": user_content}]}
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    headers = {
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    # ── Parse text từ candidates ──
    raw_text = ""
    candidates = result.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        raw_text = "".join(p.get("text", "") for p in parts if "text" in p)

    # ── Parse citations từ groundingMetadata ──
    citations = []
    if candidates:
        grounding = candidates[0].get("groundingMetadata", {})
        grounding_chunks = grounding.get("groundingChunks", [])
        seen_urls = set()
        for chunk in grounding_chunks:
            web = chunk.get("web", {})
            chunk_url = web.get("uri", "")
            chunk_title = web.get("title", "").strip()
            if chunk_url and chunk_url not in seen_urls:
                citations.append({
                    "text": chunk_title or chunk_url,
                    "url": chunk_url,
                })
                seen_urls.add(chunk_url)

    # Nếu groundingMetadata không có citations, thử regex fallback
    if not citations:
        citations = _extract_citations_from_text(raw_text)

    return raw_text, citations


# ══════════════════════════════════════════════════════════
# WEB SEARCH NODE — Hàm chính cho Graph
# ══════════════════════════════════════════════════════════

def web_search_node(state: GraphState) -> GraphState:
    """
    🌐 WEB SEARCH NODE — Gemini 2.5 Flash + Google Search Tool.

    Luồng:
      1. Gọi Google Gemini native API với google_search tool (PRIMARY)
      2. Nếu lỗi → Fallback sang OpenRouter search models
      3. Trích citations từ groundingMetadata (hoặc regex fallback)
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    action = state.get("intent_action", "")
    ufm_queries = state.get("ufm_search_queries") or []
    pr_query = state.get("pr_search_query")

    config = query_flow_config.web_search
    start_time = time.time()

    if not config.enabled:
        return {
            **state,
            "web_search_results": None,
            "web_search_citations": None,
            "next_node": "synthesizer",
        }

    # ── Xây dựng search prompt ──
    search_prompt = _build_search_query(
        standalone_query=standalone_query,
        action=action,
        ufm_queries=ufm_queries,
        pr_query=pr_query,
    )

    system_prompt = prompt_manager.get_system("web_search_node")
    logger.info("Web Search - Prompt: '%s'", search_prompt[:120])

    # ══════════════════════════════════════════════════════
    # BƯỚC 1: Thử Google Gemini Native + google_search tool
    # ══════════════════════════════════════════════════════
    try:
        raw_result, citations = _call_gemini_native_with_search(
            system_prompt=system_prompt,
            user_content=search_prompt,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
        )

        elapsed = time.time() - start_time
        logger.info(
            "Web Search [%.3fs] GOOGLE OK (%s) %d citations, %d chars",
            elapsed, config.model, len(citations), len(raw_result),
        )

        return {
            **state,
            "web_search_results": raw_result,
            "web_search_citations": citations,
            "next_node": "synthesizer",
        }

    except Exception as google_err:
        elapsed_google = time.time() - start_time
        logger.warning(
            "Web Search [%.3fs] GOOGLE FAIL (%s): %s → Thu fallback OpenRouter",
            elapsed_google, config.model, google_err,
        )

    # ══════════════════════════════════════════════════════
    # BƯỚC 2: Fallback → OpenRouter search models
    # ══════════════════════════════════════════════════════
    try:
        raw_result = _call_gemini_api_with_fallback(
            system_prompt=system_prompt,
            user_content=search_prompt,
            config_section=config,
            node_key="web_search",
        )

        citations = _extract_citations_from_text(raw_result)
        elapsed = time.time() - start_time
        logger.info(
            "Web Search [%.3fs] FALLBACK OK, %d citations, %d chars",
            elapsed, len(citations), len(raw_result),
        )

        return {
            **state,
            "web_search_results": raw_result,
            "web_search_citations": citations,
            "next_node": "synthesizer",
        }

    except Exception as fallback_err:
        elapsed = time.time() - start_time
        logger.error(
            "Web Search [%.3fs] ALL FAILED: Google=%s | Fallback=%s",
            elapsed, google_err, fallback_err, exc_info=True,
        )
        return {
            **state,
            "web_search_results": None,
            "web_search_citations": None,
            "next_node": "synthesizer",
        }
