# app/core/config/rag_search.py
# Cấu hình cho Proceed RAG Search Pipeline (Tách luồng PR và UFM Info)
# Đọc từ: rag_search_config.yaml

from pydantic import BaseModel, Field
from typing import Optional, List
from app.core.config import _load_yaml

_rs_data = _load_yaml("rag_search_config.yaml")

# ============================================================
# PR QUERY GENERATION (Dành cho PR Search)
# ============================================================
_pq = _rs_data.get("pr_query", {})

class PRQueryConfig(BaseModel):
    enabled: bool = _pq.get("enabled", True)
    provider: str = _pq.get("provider", "openrouter")
    model: str = _pq.get("model", "google/gemini-3.1-flash-lite-preview")
    temperature: float = _pq.get("temperature", 0.4)
    max_tokens: int = _pq.get("max_tokens", 150)
    timeout_seconds: int = _pq.get("timeout_seconds", 6)
    system_prompt: str = _pq.get(
        "system_prompt",
        "Sinh 1 câu truy vấn tìm thành tựu UFM liên quan đến chủ đề câu hỏi."
    )

# ============================================================
# UFM DOMAIN MULTI-QUERY (Dành cho Info Search)
# ============================================================
_uq = _rs_data.get("ufm_query", {})

class UFMQueryConfig(BaseModel):
    enabled: bool = _uq.get("enabled", True)
    provider: str = _uq.get("provider", "openrouter")
    model: str = _uq.get("model", "google/gemini-3.1-flash-lite-preview")
    temperature: float = _uq.get("temperature", 0.2)
    max_tokens: int = _uq.get("max_tokens", 150)
    timeout_seconds: int = _uq.get("timeout_seconds", 6)
    system_prompt: str = _uq.get(
        "system_prompt",
        "Sinh 2 câu truy vấn tìm kiếm trên các cổng thông tin UFM."
    )

# ============================================================
# WEB SEARCH AGENT
# ============================================================
_ws = _rs_data.get("web_search", {})

class WebSearchConfig(BaseModel):
    enabled: bool = _ws.get("enabled", True)
    provider: str = _ws.get("provider", "openai")
    model: str = _ws.get("model", "gpt-4o-mini-search-preview")
    temperature: float = _ws.get("temperature", 0.2)
    max_tokens: int = _ws.get("max_tokens", 800)
    timeout_seconds: int = _ws.get("timeout_seconds", 15)
    ufm_domains: List[str] = _ws.get("ufm_domains", [
        "ufm.edu.vn", "tuyensinh.ufm.edu.vn", "nhaphoc.ufm.edu.vn"
    ])
    pr_domains: List[str] = _ws.get("pr_domains", [
        "thanhnien.vn", "vnexpress.net"
    ])
    system_prompt: str = _ws.get(
        "system_prompt",
        "Tìm kiếm thông tin theo domain được chỉ định."
    )

# ============================================================
# SYNTHESIZER 1: INFO SYNTHESIZER (User-Centric)
# ============================================================
_syn_info = _rs_data.get("info_synthesizer", {})

class InfoSynthesizerConfig(BaseModel):
    provider: str = _syn_info.get("provider", "openrouter")
    model: str = _syn_info.get("model", "google/gemini-2.5-flash-preview")
    temperature: float = _syn_info.get("temperature", 0.1)
    max_tokens: int = _syn_info.get("max_tokens", 1500)
    timeout_seconds: int = _syn_info.get("timeout_seconds", 20)
    system_prompt: str = _syn_info.get(
        "system_prompt",
        "Tư vấn tuyển sinh trung lập, trực diện, nghiêm cấm PR."
    )

# ============================================================
# SYNTHESIZER 2: PR SYNTHESIZER
# ============================================================
_syn_pr = _rs_data.get("pr_synthesizer", {})

class PRSynthesizerConfig(BaseModel):
    provider: str = _syn_pr.get("provider", "openrouter")
    model: str = _syn_pr.get("model", "google/gemini-2.5-flash-preview")
    temperature: float = _syn_pr.get("temperature", 0.3)
    max_tokens: int = _syn_pr.get("max_tokens", 1500)
    timeout_seconds: int = _syn_pr.get("timeout_seconds", 20)
    system_prompt: str = _syn_pr.get(
        "system_prompt",
        "Tư vấn tuyển sinh lồng ghép PR, nhưng không chê bôi đối thủ."
    )

# ============================================================
# SANITIZER + VERIFIER
# ============================================================
_san = _rs_data.get("sanitizer", {})

class SanitizerConfig(BaseModel):
    provider: str = _san.get("provider", "openrouter")
    model: str = _san.get("model", "google/gemini-2.5-flash-preview")
    temperature: float = _san.get("temperature", 0.0)
    max_tokens: int = _san.get("max_tokens", 800)
    timeout_seconds: int = _san.get("timeout_seconds", 15)
    max_loops: int = _san.get("max_loops", 2)
    system_prompt: str = _san.get(
        "system_prompt",
        "Kiểm tra draft câu trả lời: trích dẫn, hallucination, tone PR lố."
    )


# ============================================================
# CONTEXT EVALUATOR (Self-RAG Gate)
# ============================================================
_eval = _rs_data.get("context_evaluator", {})

class EvaluatorConfig(BaseModel):
    enabled: bool = _eval.get("enabled", True)
    provider: str = _eval.get("provider", "openrouter")
    model: str = _eval.get("model", "google/gemini-3.0-flash-preview")
    temperature: float = _eval.get("temperature", 0.0)
    max_tokens: int = _eval.get("max_tokens", 10)
    timeout_seconds: int = _eval.get("timeout_seconds", 5)


# ============================================================
# SEMANTIC SEARCH CACHE
# ============================================================
_sc = _rs_data.get("search_cache", {})

class SearchCacheConfig(BaseModel):
    enabled: bool = _sc.get("enabled", True)
    similarity_threshold: float = _sc.get("similarity_threshold", 0.9)
    ttl_hours: int = _sc.get("ttl_hours", 24)
    max_entries: int = _sc.get("max_entries", 200)
