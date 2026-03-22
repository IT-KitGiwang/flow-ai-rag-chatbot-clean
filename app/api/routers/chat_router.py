"""
Chat Router — FastAPI Endpoint công khai cho Chat API.

Bảo mật:
  - Origin/Referer check: Chỉ cho phép từ domain đã đăng ký
  - Rate Limit per IP: 8 msg/phút
  - Không cần đăng nhập (public chatbot)

Endpoints:
  POST /api/v1/chat/message  → Gửi tin nhắn → LangGraph Pipeline
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.core.config.chat_config import chat_cfg
from app.core.security import chat_rate_limiter, verify_origin
from app.services.chat_workflow import run_chat_pipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Router ──
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


# ══════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════
class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' hoặc 'assistant'")
    content: str = Field(..., description="Nội dung tin nhắn")


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=chat_cfg.history.max_query_length,
        description="Câu hỏi của người dùng (tối đa 2000 ký tự)",
    )
    chat_history: Optional[list[ChatMessage]] = Field(
        default=[],
        description="Lịch sử chat (tối đa 20 tin gần nhất)",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="ID phiên chat (tùy chọn, dùng cho tracking)",
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="Câu trả lời của Bot")
    source: str = Field("", description="Nguồn gốc: rag_direct, form_template, care_agent...")
    intent: str = Field("", description="Intent phân loại")
    blocked: bool = Field(False, description="True nếu tin nhắn bị Guardian chặn")
    elapsed_seconds: float = Field(0.0, description="Thời gian xử lý (giây)")


# ══════════════════════════════════════════════════════════
# MESSAGE — Public Endpoint (Domain-Locked)
# ══════════════════════════════════════════════════════════
@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Gửi tin nhắn Chat (public, domain-locked)",
)
async def send_message(body: ChatRequest, request: Request):
    """
    Gửi tin nhắn và nhận câu trả lời từ Bot.

    **Bảo mật (không cần đăng nhập):**
    - ✅ Chỉ chấp nhận request từ domain đã đăng ký (Origin check)
    - ✅ Rate Limit: 8 tin nhắn/phút/IP
    - ✅ Tin nhắn spam/injection → bị chặn bởi Guardian Nodes bên trong

    **Body (JSON):**
    ```json
    {
      "query": "Học phí ngành Marketing là bao nhiêu?",
      "chat_history": [
        {"role": "user", "content": "Xin chào"},
        {"role": "assistant", "content": "Chào bạn!"}
      ],
      "session_id": "abc123"
    }
    ```
    """
    # ── Lớp 1: Origin Check (chặn domain lạ) ──
    client_ip = verify_origin(request)

    # ── Lớp 2: Rate Limit per IP ──
    chat_rate_limiter.check(client_ip)

    # ── Validate & Trim history ──
    max_hist = chat_cfg.history.max_history_messages
    history = []
    if body.chat_history:
        recent = body.chat_history[-max_hist:]
        history = [{"role": m.role, "content": m.content} for m in recent]

    logger.info(
        "Chat - Message from %s | session=%s | query='%s'",
        client_ip, body.session_id or "none", body.query[:60],
    )

    # ── Lớp 3: Chạy Pipeline (bên trong có Fast Scan + Guard) ──
    try:
        result = run_chat_pipeline(
            query=body.query,
            chat_history=history,
            session_id=body.session_id or client_ip,
        )
    except Exception as e:
        logger.error("Chat - Pipeline crashed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hệ thống gặp lỗi khi xử lý. Vui lòng thử lại.",
        )

    return ChatResponse(
        response=result["response"],
        source=result.get("source", ""),
        intent=result.get("intent", ""),
        blocked=result.get("blocked", False),
        elapsed_seconds=result.get("elapsed_seconds", 0.0),
    )
