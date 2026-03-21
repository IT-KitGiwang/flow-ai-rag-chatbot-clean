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
from app.core.config import query_flow_config
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
    "RESET":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['RESET']}"

def print_banner():
    _log(_c("BOLD", "=" * 72))
    _log(_c("CYAN", "  UFM ADMISSION BOT — Pipeline E2E Test Runner"))
    _log(_c("CYAN", f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    _log(_c("BOLD", "=" * 72))

def print_active_models():
    """Hiển thị Model đang active cho mỗi node"""
    qfc = query_flow_config
    fb = qfc.fallback_models
    light = fb.get_model_chain("light")
    main = fb.get_model_chain("main")

    _log(_c("YELLOW", "  [ ACTIVE MODELS ]"))
    _log(f"   - Reformulate : {qfc.query_reformulation.provider} / {qfc.query_reformulation.model}")
    _log(f"   - Guard (2a)  : {qfc.prompt_guard_fast.provider} / {qfc.prompt_guard_fast.model}")
    _log(f"   - Guard (2b)  : {qfc.prompt_guard_deep.provider} / {qfc.prompt_guard_deep.model}")
    _log(f"   - Multi-Query : {qfc.multi_query.provider} / {qfc.multi_query.model}")
    _log(f"   - Embedding   : {qfc.embedding.provider} / {qfc.embedding.model}")
    if light:
        _log(f"   - Intent      : {light[0].provider} / {light[0].model} (Primary)")
    if main:
        _log(f"   - Final Resp  : {main[0].provider} / {main[0].model} (Primary)")
    _log(_c("BOLD", "=" * 72))

def print_node_header(node_name: str, step: int):
    _log()
    _log(_c("BOLD", f"{'─' * 72}"))
    _log(_c("YELLOW", f"  [{step}] {node_name}"))
    _log(_c("BOLD", f"{'─' * 72}"))

def print_state_delta(state: dict, keys_to_show: list):
    """In các key quan trọng từ state."""
    for key in keys_to_show:
        val = state.get(key)
        if val is None:
            continue

        # Truncate giá trị dài
        if isinstance(val, str) and len(val) > 300:
            display = val[:300] + f"... ({len(val)} chars)"
        elif isinstance(val, list) and len(val) > 5:
            display = f"[{len(val)} items] " + str(val[:3]) + "..."
        elif isinstance(val, list) and all(isinstance(v, list) for v in val):
            display = f"[{len(val)} vectors, dim={len(val[0]) if val else 0}]"
        else:
            display = str(val)

        _log(f"  {_c('BLUE', key):>45s}  │  {display}")

def print_status(passed: bool, elapsed: float, msg: str = ""):
    status = _c("GREEN", "✅ PASS") if passed else _c("RED", "⛔ BLOCKED")
    time_str = _c("CYAN", f"{elapsed:.3f}s")
    _log(f"  {'Status':>38s}  │  {status} ({time_str})")
    if msg:
        _log(f"  {'Message':>38s}  │  {msg[:200]}")


def print_final_response(state: dict):
    _log()
    _log(_c("BOLD", "═" * 72))
    _log(_c("GREEN", "  FINAL RESPONSE"))
    _log(_c("BOLD", "═" * 72))
    resp = state.get("final_response", "(trống)")
    source = state.get("response_source", "N/A")
    _log(f"  Source: {_c('YELLOW', source)}")
    _log(f"  Length: {len(resp)} chars")
    _log(_c("BOLD", "─" * 72))
    _log(resp)
    _log(_c("BOLD", "═" * 72))


# ══════════════════════════════════════════════════════════
# NODE RUNNER — Chạy 1 node, đo thời gian, in state
# ══════════════════════════════════════════════════════════
def run_node(node_fn, state: dict, node_name: str, step: int, keys: list) -> dict:
    """Chạy 1 node function, đo thời gian, in kết quả."""
    print_node_header(node_name, step)
    t0 = time.time()
    try:
        new_state = node_fn(state)
        elapsed = time.time() - t0

        # Detect pass/fail (chỉ fail khi giá trị CHÍNH XÁC là False)
        passed = True
        if new_state.get("fast_scan_passed") is False:
            passed = False
        if new_state.get("contextual_guard_passed") is False:
            passed = False

        msg = ""
        # Chỉ hiển thị liên quan đến node hiện tại
        if "FAST SCAN" in node_name:
            msg = new_state.get("fast_scan_message", "")
        elif "CONTEXTUAL GUARD" in node_name:
            msg = new_state.get("contextual_guard_message", "")

        print_status(passed, elapsed, msg)
        print_state_delta(new_state, keys)
        return new_state

    except Exception as e:
        elapsed = time.time() - t0
        print_status(False, elapsed, f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return {**state, "next_node": "end", "final_response": f"Node {node_name} crashed: {e}"}


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════
def run_pipeline(user_query: str, chat_history: list = None):
    """Chạy toàn bộ pipeline tuần tự, in state từng bước, ghi log ra file."""
    log_path = _open_log(user_query)
    _log(f"  📁 Log file: {log_path}")
    _log()

    print_banner()
    print_active_models()
    
    _log(f"\n  Input Query: {_c('BOLD', user_query)}")
    if chat_history:
        _log(f"  Chat History: {len(chat_history)} messages")
    _log()

    # ── Khởi tạo state ban đầu ──
    state = {
        "user_query": user_query,
        "chat_history": chat_history or [],
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

    step = 1
    pipeline_start = time.time()

    # ═══════════════════════════════════════════════
    # NODE 1: FAST SCAN (Regex, 0ms, $0)
    # ═══════════════════════════════════════════════
    state = run_node(fast_scan_node, state, "FAST SCAN NODE", step, [
        "normalized_query", "fast_scan_passed", "fast_scan_blocked_layer",
        "fast_scan_message", "next_node",
    ])
    step += 1
    if not state.get("fast_scan_passed"):
        print_final_response(state)
        _close_log()
        return state

    # ═══════════════════════════════════════════════
    # NODE 2: CONTEXT (Gemini Lite Reformulate)
    # ═══════════════════════════════════════════════
    state = run_node(context_node, state, "CONTEXT NODE (Reformulate)", step, [
        "standalone_query", "next_node",
    ])
    step += 1

    # ═══════════════════════════════════════════════
    # NODE 3: CONTEXTUAL GUARD (2 LLM song song)
    # ═══════════════════════════════════════════════
    state = run_node(contextual_guard_node, state, "CONTEXTUAL GUARD NODE", step, [
        "contextual_guard_passed", "contextual_guard_blocked_layer",
        "contextual_guard_message", "next_node",
    ])
    step += 1
    if not state.get("contextual_guard_passed"):
        print_final_response(state)
        _close_log()
        return state

    # ═══════════════════════════════════════════════
    # NODE 4: MULTI-QUERY (Sinh biến thể)
    # ═══════════════════════════════════════════════
    state = run_node(multi_query_node, state, "MULTI-QUERY NODE", step, [
        "multi_queries", "next_node",
    ])
    step += 1

    # ═══════════════════════════════════════════════
    # NODE 5: EMBEDDING (BGE-M3 batch)
    # ═══════════════════════════════════════════════
    state = run_node(embedding_node, state, "EMBEDDING NODE", step, [
        "query_embeddings", "next_node",
    ])
    step += 1

    # ═══════════════════════════════════════════════
    # NODE 6: RAG (Hybrid Search DB)
    # ═══════════════════════════════════════════════
    state = run_node(rag_node, state, "RAG NODE (Hybrid Search)", step, [
        "rag_context", "retrieved_chunks", "rag_confidence_failed",
        "top1_cosine_score", "next_node",
    ])
    step += 1

    # ═══════════════════════════════════════════════
    # NODE 7: INTENT (Router trung tâm)
    # ═══════════════════════════════════════════════
    state = run_node(intent_node, state, "INTENT NODE (Router)", step, [
        "intent", "intent_summary", "intent_action", "next_node",
        "response_source",
    ])
    step += 1

    # ═══════════════════════════════════════════════
    # NODE 8+: AGENT DISPATCH (tùy intent_action)
    # ═══════════════════════════════════════════════
    next_node = state.get("next_node", "response")
    intent_action = state.get("intent_action", "")

    if next_node == "form":
        state = run_node(form_node, state, "FORM NODE (Agent Mẫu đơn)", step, [
            "final_response", "response_source", "next_node",
        ])
        step += 1

    elif next_node == "care":
        state = run_node(care_node, state, "CARE NODE (Chăm sóc SV)", step, [
            "final_response", "response_source", "next_node",
        ])
        step += 1

    elif next_node == "rag_search":
        state = run_node(proceed_rag_search_pipeline, state, "RAG SEARCH PIPELINE (Web + Evaluator)", step, [
            "final_response", "response_source", "web_search_results",
            "web_search_citations", "search_cache_hit", "next_node",
        ])
        step += 1

    # ═══════════════════════════════════════════════
    # NODE FINAL: RESPONSE (LLM Sinh câu trả lời)
    # ═══════════════════════════════════════════════
    state = run_node(response_node, state, "RESPONSE NODE (Final)", step, [
        "final_response", "response_source", "next_node",
    ])

    # ═══════════════════════════════════════════════
    # KẾT QUẢ
    # ═══════════════════════════════════════════════
    total_elapsed = time.time() - pipeline_start
    print_final_response(state)
    _log(f"\n  Total Pipeline Time: {_c('BOLD', f'{total_elapsed:.2f}s')}")
    _log(f"  Nodes Executed: {step}")
    _log(f"  Log saved: {log_path}")
    _log()

    _close_log()
    return state


# ══════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ══════════════════════════════════════════════════════════
def interactive_mode():
    """Chế độ tương tác — nhập câu hỏi liên tục."""
    print(_c("BOLD", "\n╔══════════════════════════════════════════════════════════╗"))
    print(_c("CYAN",   "║  UFM Pipeline Tester — Nhap cau hoi de test             ║"))
    print(_c("CYAN",   "║  Gõ 'quit' hoặc 'exit' để thoát                        ║"))
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
            print(_c("YELLOW", "Đã xóa chat history."))
            continue
        if user_input.lower() == "history":
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
