"""
Test Pipeline E2E — Chạy thử toàn bộ LangGraph Pipeline từ input user.

Chạy:
  python tests/test_pipeline_e2e.py
  python tests/test_pipeline_e2e.py --query "Học phí ngành Marketing?"
  python tests/test_pipeline_e2e.py --query "Xin chào" --history '[{"role":"user","content":"Hi"}]'

Script sẽ chạy tuần tự từng Node, in ra state đầy đủ sau mỗi Node.
Kết quả được ghi ra file .txt tại tests/logs/ (đầy đủ output + final answer).
"""

import io
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ══════════════════════════════════════════════════════════
# IMPORTS — Tất cả Node functions
# ══════════════════════════════════════════════════════════
from app.core.config import query_flow_config, models_yaml_data
from app.services.langgraph.nodes.fast_scan_node import fast_scan_node
from app.services.langgraph.nodes.context_node import context_node
from app.services.langgraph.nodes.contextual_guard_node import contextual_guard_node
from app.services.langgraph.nodes.multi_query_node import multi_query_node
from app.services.langgraph.nodes.embedding_node import embedding_node
from app.services.langgraph.nodes.rag_node import rag_node
from app.services.langgraph.nodes.intent_node import intent_node
from app.services.langgraph.nodes.response_node import response_node
from app.services.langgraph.nodes.care_node import care_node

# Sub-graphs (chạy thủ công)
from app.services.langgraph.nodes.proceed_form.graph import form_node
from app.services.langgraph.nodes.proceed_rag_search.graph import proceed_rag_search_pipeline


# ══════════════════════════════════════════════════════════
# FILE LOGGING — Ghi toàn bộ output ra .txt
# ══════════════════════════════════════════════════════════
LOG_DIR = PROJECT_ROOT / "tests" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None  # File handle hiện tại


def _strip_ansi(text: str) -> str:
    """Xóa ANSI escape codes cho file log."""
    return re.sub(r'\033\[[0-9;]*m', '', text)


def _log(text: str = ""):
    """In ra console VÀ ghi vào file log (không có ANSI colors)."""
    print(text)
    if _log_file:
        _log_file.write(_strip_ansi(text) + "\n")


def _open_log(query: str) -> Path:
    """Mở file log mới cho 1 pipeline run."""
    global _log_file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Rút gọn query làm tên file (max 40 ký tự, bỏ ký tự đặc biệt)
    slug = re.sub(r'[^\w\s-]', '', query[:40]).strip().replace(' ', '_')
    log_path = LOG_DIR / f"{ts}_{slug}.txt"
    _log_file = open(log_path, "w", encoding="utf-8")
    return log_path


def _close_log():
    """Đóng file log."""
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None


# ══════════════════════════════════════════════════════════
# DISPLAY UTILS
# ══════════════════════════════════════════════════════════
COLORS = {
    "HEADER":  "\033[95m",
    "BLUE":    "\033[94m",
    "CYAN":    "\033[96m",
    "GREEN":   "\033[92m",
    "YELLOW":  "\033[93m",
    "RED":     "\033[91m",
    "BOLD":    "\033[1m",
    "DIM":     "\033[2m",
    "RESET":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['RESET']}"


def _format_value(val) -> str:
    """Format giá trị state — KHÔNG CẮT, hiển thị toàn bộ để debug token."""
    if val is None:
        return _c("DIM", "null")
    if isinstance(val, bool):
        return _c("GREEN", "✓ True") if val else _c("RED", "✗ False")
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, str):
        if len(val) == 0:
            return _c("DIM", '""')
        return f"{val}\n{'':>55s}({len(val)} chars)"
    if isinstance(val, list):
        if not val:
            return _c("DIM", "[]")
        # Vector embeddings (list of lists of floats) — giữ compact
        if all(isinstance(v, list) for v in val):
            dims = len(val[0]) if val[0] else 0
            return f"[{len(val)} vectors × {dims}D]"
        # List of dicts (retrieved_chunks, web results...)
        if all(isinstance(v, dict) for v in val):
            lines = [f"[{len(val)} items]"]
            for i, item in enumerate(val):
                preview = json.dumps(item, ensure_ascii=False)
                lines.append(f"{'':>55s}  [{i}] {preview}")
            return "\n".join(lines)
        # List of strings (multi_queries, citations...) — FULL
        if all(isinstance(v, str) for v in val):
            lines = [f"[{len(val)} items]"]
            for i, v in enumerate(val):
                lines.append(f"{'':>55s}  [{i}] \"{v}\"")
            return "\n".join(lines)
        return f"[{len(val)} items]"
    return str(val)


def print_banner():
    _log()
    _log(_c("BOLD", "╔══════════════════════════════════════════════════════════════════════════╗"))
    _log(_c("BOLD", "║") + _c("CYAN", "    UFM ADMISSION BOT — Pipeline E2E Test Runner                        ") + _c("BOLD", "║"))
    _log(_c("BOLD", "║") + _c("CYAN", f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                                ") + _c("BOLD", "║"))
    _log(_c("BOLD", "╚══════════════════════════════════════════════════════════════════════════╝"))


def print_active_models():
    """Hiển thị Model đang active cho mỗi node — đọc trực tiếp từ models_config.yaml."""
    _log()
    _log(_c("YELLOW", "  ┌─────────────────────────────────────────────────────────────────┐"))
    _log(_c("YELLOW", "  │                    ACTIVE MODELS (from YAML)                    │"))
    _log(_c("YELLOW", "  ├──────────────────────┬──────────────────────────────────────────┤"))

    qfc = query_flow_config
    rows = [
        ("❷ Summarizer",     qfc.long_query_summarizer.provider, qfc.long_query_summarizer.model),
        ("❹ Guard (Fast)",   qfc.prompt_guard_fast.provider,     qfc.prompt_guard_fast.model),
        ("❺ Reformulate",    qfc.query_reformulation.provider,   qfc.query_reformulation.model),
        ("❻ Guard (Deep)",   qfc.prompt_guard_deep.provider,     qfc.prompt_guard_deep.model),
        ("❼ Multi-Query",    qfc.multi_query.provider,           qfc.multi_query.model),
        ("❽ Embedding",      qfc.embedding.provider,             qfc.embedding.model),
        ("❾a Vector Router", qfc.vector_router.provider,         qfc.vector_router.model),
        ("❾b Semantic Rtr",  qfc.semantic_router.provider,       qfc.semantic_router.model),
        ("⓫a PR Query",      qfc.pr_query.provider,              qfc.pr_query.model),
        ("⓫b UFM Query",     qfc.ufm_query.provider,             qfc.ufm_query.model),
        ("⓫c Web Search",    qfc.web_search.provider,            qfc.web_search.model),
        ("⓫d Info Synth",    qfc.info_synthesizer.provider,      qfc.info_synthesizer.model),
        ("⓫e PR Synth",      qfc.pr_synthesizer.provider,        qfc.pr_synthesizer.model),
        ("⓫f Sanitizer",     qfc.sanitizer.provider,             qfc.sanitizer.model),
        ("⓫g Evaluator",     qfc.context_evaluator.provider,     qfc.context_evaluator.model),
        ("⓬ Care",           models_yaml_data.get("care", {}).get("provider", "?"),
                              models_yaml_data.get("care", {}).get("model", "?")),
        ("⓭ Form Extract",   models_yaml_data.get("form", {}).get("provider", "?"),
                              models_yaml_data.get("form", {}).get("extractor", {}).get("model", "?")),
        ("⓭ Form Draft",     models_yaml_data.get("form", {}).get("provider", "?"),
                              models_yaml_data.get("form", {}).get("drafter", {}).get("model", "?")),
        ("⓮ Main Bot",       qfc.main_bot.provider,              qfc.main_bot.model),
    ]

    for label, provider, model in rows:
        _log(f"  │ {label:<20s} │ {provider}/{model:<36s} │")

    _log(_c("YELLOW", "  └──────────────────────┴──────────────────────────────────────────┘"))


def print_node_header(node_name: str, step: int, emoji: str = "🔹"):
    _log()
    _log(_c("BOLD", f"{'─' * 74}"))
    _log(_c("YELLOW", f"  {emoji} [{step}] {node_name}"))
    _log(_c("BOLD", f"{'─' * 74}"))


def print_state_delta(state: dict, keys_to_show: list):
    """In các key quan trọng từ state."""
    for key in keys_to_show:
        val = state.get(key)
        display = _format_value(val)
        _log(f"    {_c('BLUE', key):>50s}  │  {display}")


def print_status(passed: bool, elapsed: float, msg: str = ""):
    if passed is None:
        status = _c("DIM", "⏭️  SKIPPED")
    elif passed:
        status = _c("GREEN", "✅ PASS")
    else:
        status = _c("RED", "⛔ BLOCKED")

    time_str = _c("CYAN", f"{elapsed:.3f}s")
    _log(f"    {'Status':>43s}  │  {status}  ({time_str})")
    if msg:
        _log(f"    {'Message':>43s}  │  {msg[:200]}")


def print_final_response(state: dict, total_elapsed: float, nodes_executed: int):
    _log()
    _log(_c("BOLD", "╔══════════════════════════════════════════════════════════════════════════╗"))
    _log(_c("BOLD", "║") + _c("GREEN", "                         FINAL RESPONSE                                ") + _c("BOLD", "║"))
    _log(_c("BOLD", "╚══════════════════════════════════════════════════════════════════════════╝"))

    resp = state.get("final_response", "(trống)")
    source = state.get("response_source", "N/A")
    intent = state.get("intent", "N/A")
    intent_action = state.get("intent_action", "N/A")

    _log(f"  Response Source : {_c('YELLOW', source)}")
    _log(f"  Intent          : {_c('CYAN', intent)} → {_c('CYAN', intent_action)}")
    _log(f"  Response Length : {len(resp)} chars")
    _log(f"  Total Time      : {_c('BOLD', f'{total_elapsed:.2f}s')}")
    _log(f"  Nodes Executed  : {nodes_executed}")
    _log(_c("BOLD", "─" * 74))
    _log()
    _log(resp)
    _log()
    _log(_c("BOLD", "═" * 74))


def print_pipeline_summary(timings: list, total_elapsed: float):
    """In bảng tổng kết thời gian từng node."""
    _log()
    _log(_c("YELLOW", "  ┌──────────────────────────────────────────────┬──────────┬──────────┐"))
    _log(_c("YELLOW", "  │ Node                                         │   Time   │  Status  │"))
    _log(_c("YELLOW", "  ├──────────────────────────────────────────────┼──────────┼──────────┤"))

    for entry in timings:
        name = entry["name"][:44]
        elapsed = entry["elapsed"]
        passed = entry["passed"]

        status_str = "PASS" if passed else "BLOCKED" if passed is False else "SKIP"
        status_color = "GREEN" if passed else "RED" if passed is False else "DIM"

        _log(f"  │ {name:<44s} │ {elapsed:>6.3f}s  │ {_c(status_color, f'{status_str:^8s}')} │")

    _log(_c("YELLOW", "  ├──────────────────────────────────────────────┼──────────┼──────────┤"))
    _log(f"  │ {'TOTAL':>44s} │ {_c('BOLD', f'{total_elapsed:>6.2f}s')}  │          │")
    _log(_c("YELLOW", "  └──────────────────────────────────────────────┴──────────┴──────────┘"))


# ══════════════════════════════════════════════════════════
# NODE RUNNER — Chạy 1 node, đo thời gian, in state
# ══════════════════════════════════════════════════════════
def run_node(
    node_fn,
    state: dict,
    node_name: str,
    step: int,
    keys: list,
    emoji: str = "🔹",
    timings: list = None,
) -> dict:
    """Chạy 1 node function, đo thời gian, in kết quả."""
    print_node_header(node_name, step, emoji)
    t0 = time.time()
    try:
        new_state = node_fn(state)
        elapsed = time.time() - t0

        # Detect pass/fail
        passed = True
        msg = ""

        if "FAST SCAN" in node_name.upper():
            if new_state.get("fast_scan_passed") is False:
                passed = False
            msg = new_state.get("fast_scan_message", "")
            # Ghi nhận nếu query bị summarize
            if new_state.get("query_was_summarized"):
                msg += " [Query was summarized by LLM]"

        elif "CONTEXTUAL GUARD" in node_name.upper():
            if new_state.get("contextual_guard_passed") is False:
                passed = False
            msg = new_state.get("contextual_guard_message", "")
            blocked_layer = new_state.get("contextual_guard_blocked_layer")
            if blocked_layer:
                msg += f" [Blocked by Layer {blocked_layer}]"

        elif "CONTEXT" in node_name.upper() and "GUARD" not in node_name.upper():
            # Context Node (Reformulate)
            orig = state.get("user_query", "")
            reformed = new_state.get("standalone_query", "")
            if orig == reformed:
                msg = "Giữ nguyên (no history hoặc skip)"
            else:
                msg = f"Reformulated: '{orig[:40]}' → '{reformed[:60]}'"

        elif "EMBEDDING" in node_name.upper():
            embs = new_state.get("query_embeddings", [])
            if embs:
                msg = f"Nhúng {len(embs)} vectors × {len(embs[0])}D thành công"
            else:
                msg = "Không có embeddings (lỗi hoặc bỏ qua)"

        elif "RAG NODE" in node_name.upper():
            chunks = new_state.get("retrieved_chunks", [])
            top1 = new_state.get("top1_cosine_score", 0.0)
            ctx_len = len(new_state.get("rag_context", ""))
            conf_fail = new_state.get("rag_confidence_failed", False)
            msg = f"Chunks={len(chunks)}, Top1Cosine={top1:.4f}, Context={ctx_len} chars"
            if conf_fail:
                msg += " ⚠️ CONFIDENCE FAILED"

        elif "INTENT" in node_name.upper():
            intent = new_state.get("intent", "")
            action = new_state.get("intent_action", "")
            summary = new_state.get("intent_summary", "")[:60]
            msg = f"intent='{intent}' → action='{action}' | summary='{summary}'"

        elif "RAG SEARCH" in node_name.upper():
            cache_hit = new_state.get("search_cache_hit", False)
            citations = new_state.get("web_search_citations") or []
            source = new_state.get("response_source", "")
            msg = f"source='{source}', cache={'HIT' if cache_hit else 'MISS'}, citations={len(citations)}"

        elif "CARE" in node_name.upper():
            resp_len = len(new_state.get("final_response", ""))
            msg = f"Response={resp_len} chars"

        elif "FORM" in node_name.upper():
            resp_len = len(new_state.get("final_response", ""))
            msg = f"Response={resp_len} chars"

        elif "RESPONSE" in node_name.upper():
            source = new_state.get("response_source", "")
            resp_len = len(new_state.get("final_response", ""))
            msg = f"source='{source}', length={resp_len} chars"

        print_status(passed, elapsed, msg)
        print_state_delta(new_state, keys)

        if timings is not None:
            timings.append({"name": node_name, "elapsed": elapsed, "passed": passed})

        return new_state

    except Exception as e:
        elapsed = time.time() - t0
        print_status(False, elapsed, f"EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        _log()
        _log(_c("RED", traceback.format_exc()))

        if timings is not None:
            timings.append({"name": node_name, "elapsed": elapsed, "passed": False})

        return {**state, "next_node": "end", "final_response": f"Node {node_name} crashed: {e}"}


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════
def run_pipeline(user_query: str, chat_history: list = None):
    """Chạy toàn bộ pipeline tuần tự, in state từng bước, ghi log ra file."""
    log_path = _open_log(user_query)

    print_banner()
    _log(f"\n  📁 Log file: {log_path}")
    print_active_models()

    _log(f"\n  📝 Input Query  : {_c('BOLD', user_query)}")
    if chat_history:
        _log(f"  💬 Chat History : {len(chat_history)} messages")
        for i, msg in enumerate(chat_history[-4:]):  # Hiện 4 messages gần nhất
            role_icon = "👤" if msg.get("role") == "user" else "🤖"
            content_preview = msg.get("content", "")[:80]
            _log(f"     {role_icon} {content_preview}")
    else:
        _log(f"  💬 Chat History : {_c('DIM', 'Trống (lượt đầu tiên)')}")
    _log()

    # ── Khởi tạo state ban đầu ──
    state = {
        "user_query": user_query,
        "chat_history": chat_history or [],
        "normalized_query": "",
        "standalone_query": "",
        "original_query": "",
        "query_was_summarized": False,
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

    step = 1
    timings = []
    pipeline_start = time.time()

    # ═══════════════════════════════════════════════
    # NODE 1: FAST SCAN (Regex + DoS + Keyword + Injection)
    # ═══════════════════════════════════════════════
    state = run_node(fast_scan_node, state,
        "FAST SCAN NODE (Guard Layer 0-1)", step,
        [
            "normalized_query", "original_query", "query_was_summarized",
            "fast_scan_passed", "fast_scan_blocked_layer",
            "fast_scan_message", "next_node",
        ],
        emoji="🛡️", timings=timings,
    )
    step += 1
    if state.get("fast_scan_passed") is False:
        _log(_c("RED", "\n  ⛔ PIPELINE STOPPED — Fast Scan chặn query."))
        total = time.time() - pipeline_start
        print_final_response(state, total, step - 1)
        print_pipeline_summary(timings, total)
        _log(f"\n  📁 Log saved: {log_path}\n")
        _close_log()
        return state

    # ═══════════════════════════════════════════════
    # NODE 2: CONTEXT (Query Reformulation)
    # ═══════════════════════════════════════════════
    state = run_node(context_node, state,
        "CONTEXT NODE (Query Reformulation)", step,
        ["standalone_query", "next_node"],
        emoji="🔄", timings=timings,
    )
    step += 1

    # ═══════════════════════════════════════════════
    # NODE 3: CONTEXTUAL GUARD (LLM Guard Layer 2a+2b)
    # ═══════════════════════════════════════════════
    state = run_node(contextual_guard_node, state,
        "CONTEXTUAL GUARD NODE (Layer 2a+2b)", step,
        [
            "contextual_guard_passed", "contextual_guard_blocked_layer",
            "contextual_guard_message", "next_node",
        ],
        emoji="🔐", timings=timings,
    )
    step += 1
    if state.get("contextual_guard_passed") is False:
        _log(_c("RED", "\n  ⛔ PIPELINE STOPPED — Contextual Guard chặn query."))
        total = time.time() - pipeline_start
        print_final_response(state, total, step - 1)
        print_pipeline_summary(timings, total)
        _log(f"\n  📁 Log saved: {log_path}\n")
        _close_log()
        return state

    # ═══════════════════════════════════════════════
    # NODE 4: MULTI-QUERY (Sinh biến thể tăng Recall)
    # ═══════════════════════════════════════════════
    state = run_node(multi_query_node, state,
        "MULTI-QUERY NODE (Sinh biến thể)", step,
        ["multi_queries", "next_node"],
        emoji="🔀", timings=timings,
    )
    step += 1

    # ═══════════════════════════════════════════════
    # NODE 5: EMBEDDING (BGE-M3 batch nhúng vector)
    # ═══════════════════════════════════════════════
    state = run_node(embedding_node, state,
        "EMBEDDING NODE (BGE-M3 batch)", step,
        ["query_embeddings", "next_node"],
        emoji="📊", timings=timings,
    )
    step += 1

    # ═══════════════════════════════════════════════
    # NODE 6: RAG (Hybrid Search DB — Vector+BM25+RRF)
    # ═══════════════════════════════════════════════
    state = run_node(rag_node, state,
        "RAG NODE (Hybrid Search DB)", step,
        [
            "rag_context", "retrieved_chunks", "rag_confidence_failed",
            "top1_cosine_score", "next_node",
        ],
        emoji="🗄️", timings=timings,
    )
    step += 1

    # ═══════════════════════════════════════════════
    # NODE 7: INTENT (Router trung tâm)
    # ═══════════════════════════════════════════════
    state = run_node(intent_node, state,
        "INTENT NODE (Router Phân Loại Ý Định)", step,
        [
            "intent", "intent_summary", "intent_action",
            "program_level_filter", "program_name_filter",
            "next_node", "response_source",
        ],
        emoji="🧭", timings=timings,
    )
    step += 1

    # ═══════════════════════════════════════════════
    # NODE 8+: AGENT DISPATCH (tùy intent_action)
    # ═══════════════════════════════════════════════
    next_node = state.get("next_node", "response")
    intent_action = state.get("intent_action", "")

    _log()
    _log(_c("HEADER", f"  🚦 Routing Decision: next_node='{next_node}' | intent_action='{intent_action}'"))

    if next_node == "form":
        state = run_node(form_node, state,
            "FORM NODE (Agent Mẫu đơn)", step,
            ["final_response", "response_source", "next_node"],
            emoji="📝", timings=timings,
        )
        step += 1

    elif next_node == "care":
        state = run_node(care_node, state,
            "CARE NODE (Chăm sóc Sinh viên)", step,
            ["final_response", "response_source", "next_node"],
            emoji="💚", timings=timings,
        )
        step += 1

    elif next_node == "rag_search":
        state = run_node(proceed_rag_search_pipeline, state,
            "RAG SEARCH PIPELINE (Evaluator + Web Search)", step,
            [
                "ufm_search_queries", "pr_search_query",
                "web_search_results", "web_search_citations",
                "search_cache_hit", "search_cache_similarity",
                "final_response", "response_source", "next_node",
            ],
            emoji="🌐", timings=timings,
        )
        step += 1

    elif next_node == "response":
        # response_source đã có sẵn (GREET / CLARIFY / BLOCK_FALLBACK)
        _log(_c("CYAN", f"  → Chuyển thẳng sang Response Node (source='{state.get('response_source', '')}')"))

    else:
        _log(_c("RED", f"  ⚠️ next_node không xác định: '{next_node}' → fallback response"))

    # ═══════════════════════════════════════════════
    # NODE FINAL: RESPONSE (LLM Sinh câu trả lời cuối)
    # ═══════════════════════════════════════════════
    state = run_node(response_node, state,
        "RESPONSE NODE (Final Answer)", step,
        ["final_response", "response_source", "next_node"],
        emoji="🎯", timings=timings,
    )

    # ═══════════════════════════════════════════════
    # KẾT QUẢ
    # ═══════════════════════════════════════════════
    total_elapsed = time.time() - pipeline_start
    print_final_response(state, total_elapsed, step)
    print_pipeline_summary(timings, total_elapsed)
    _log(f"\n  📁 Log saved: {log_path}\n")

    _close_log()
    return state


# ══════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ══════════════════════════════════════════════════════════
def interactive_mode():
    """Chế độ tương tác — nhập câu hỏi liên tục."""
    print(_c("BOLD", "\n╔══════════════════════════════════════════════════════════╗"))
    print(_c("CYAN",   "║  UFM Pipeline Tester — Interactive Mode                  ║"))
    print(_c("CYAN",   "║  Nhập câu hỏi để test, 'quit' để thoát                  ║"))
    print(_c("CYAN",   "║  'clear' = xóa history | 'history' = xem history         ║"))
    print(_c("BOLD",   "╚══════════════════════════════════════════════════════════╝\n"))

    chat_history = []

    while True:
        try:
            user_input = input(_c("GREEN", "\n🎤 Nhập câu hỏi: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "clear":
            chat_history = []
            print(_c("YELLOW", "✅ Đã xóa chat history."))
            continue
        if user_input.lower() == "history":
            if not chat_history:
                print(_c("DIM", "(History trống)"))
            else:
                print(json.dumps(chat_history, indent=2, ensure_ascii=False))
            continue

        # Chạy pipeline
        state = run_pipeline(user_input, chat_history)

        # Lưu vào history cho lượt sau
        chat_history.append({"role": "user", "content": user_input})
        final = state.get("final_response", "")
        if final:
            chat_history.append({"role": "assistant", "content": final})

        # Giữ history gọn (10 lượt gần nhất)
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]


# ══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UFM Pipeline E2E Tester")
    parser.add_argument("--query", "-q", help="Câu hỏi test (nếu không có sẽ vào interactive mode)")
    parser.add_argument("--history", help="JSON string chat history")
    args = parser.parse_args()

    if args.query:
        history = []
        if args.history:
            try:
                history = json.loads(args.history)
            except json.JSONDecodeError:
                print("⚠️  --history phải là JSON hợp lệ")
                sys.exit(1)
        run_pipeline(args.query, history)
    else:
        interactive_mode()
