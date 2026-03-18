from pydantic import BaseModel, Field
from typing import List, Dict, Literal

# ============================================================
# LỚP 0: Input Validation (Gatekeeper - Chống DoS)
# Chặn ngay lập tức các request khổng lồ trước khi tốn CPU xử lý chuỗi
# ============================================================
class InputValidationConfig(BaseModel):
    max_input_chars: int = Field(
        default=800,
        description="Số ký tự tối đa cho phép. Chặn ngay tại cổng để tránh DoS attack."
    )
    fallback_too_long: str = (
        "Câu hỏi của bạn quá dài (vượt quá 800 ký tự). "
        "Vui lòng tóm tắt ý chính dưới 150 chữ để AI hỗ trợ nhanh nhất nhé."
    )

# ============================================================
# LỚP 1: Keyword Filter & Normalization (Regex & Mapping)
# Tối ưu hiệu năng: Dùng Regex thay vì vòng lặp for cho noise_chars
# ============================================================
class KeywordFilterConfig(BaseModel):
    # Sử dụng Regex List để bắt được cả biến thể bị chèn dấu cách (vd: h a c k)
    banned_regex_patterns: List[str] = [
        r"ch[íi]nh\s*tr[ịi]", r"t[ôo]n\s*gi[áa]o", r"b[ạa]o\s*l[ựu]c", 
        r"h[a@4]ck", r"c[ờo]\s*b[ạa]c", r"ma\s*t[úu]y"
    ]

    teencode_map: Dict[str, str] = {
        "ko": "không", "k": "không", "hk": "không",
        "dc": "được", "đc": "được",
        "trc": "trước", "ns": "nói", "ntn": "như thế nào",
        "lm": "làm", "mk": "mình", "mn": "mọi người",
        "vs": "với", "r": "rồi", "j": "gì"
        # BỎ HOÀN TOÀN việc map số thành chữ (0 -> o) để bảo toàn dữ liệu năm học/điểm số
    }

    # Chuỗi Regex thay thế cho danh sách List tĩnh giúp tăng tốc x20 lần
    noise_regex_pattern: str = r"[\.\*_\-~!#\$%\^&\(\)\+=\|\\/<>\}\{\[\]:;'\"]"

    fallback_message: str = (
        "Xin lỗi, câu hỏi của bạn chứa nội dung không phù hợp hoặc vi phạm tiêu chuẩn cộng đồng. "
        "Vui lòng chỉ đặt câu hỏi liên quan đến tuyển sinh UFM."
    )

# ============================================================
# LỚP 2: Prompt Guard (Security - Injection/Jailbreak Check)
# ============================================================
class PromptGuardConfig(BaseModel):
    provider: str = "groq"
    model: str = "meta-llama/llama-prompt-guard-2-86m"
    
    max_tokens_per_chunk: int = Field(
        default=512,
        description="Băm nhỏ văn bản (Chunking) để tránh lỗi giới hạn 512 tokens."
    )
    safe_threshold: float = Field(
        default=0.75,
        ge=0.0, le=1.0,
        description="Ngưỡng tự tin. >= 0.75 là SAFE."
    )
    fallback_unsafe: str = (
        "Hệ thống phát hiện dấu hiệu bất thường trong câu hỏi của bạn. "
        "Vui lòng diễn đạt lại một cách tự nhiên hơn nhé."
    )

# ============================================================
# LỚP 3: Semantic Router (Intent Classification & Summarization)
# Nhiệm vụ: Xác định xem có cho đi vào RAG không, trả về JSON cứng
# ============================================================
class SemanticRouterConfig(BaseModel):
    provider: str = "groq"
    # Khuyên dùng Gemma-2-9b cho tiếng Việt thay vì Llama-3-8b
    model: str = "qwen/qwen3-32b" 
    temperature: float = 0.0
    
    # Định nghĩa cấu trúc JSON bắt buộc LLM phải tuân theo
    response_format: Literal["json_object"] = "json_object"
    allowed_intents: List[str] = ["tuyen_sinh", "hoc_phi", "ky_tuc_xa", "ngoai_le"]
    
    system_prompt: str = (
        "Bạn là bộ định tuyến ý định cho chatbot tuyển sinh Đại học UFM. "
        "Hãy phân tích câu hỏi và trả về ĐÚNG định dạng JSON sau: "
        '{"intent": "<một_trong_các_intent_cho_phép>", "summary": "<câu_hỏi_đã_làm_sạch>"}'
    )
    
    fallback_out_of_scope: str = (
        "Tôi là trợ lý AI tuyển sinh của UFM. Tôi chỉ có thể giải đáp các thông tin về "
        "ngành học, học phí, quy chế... Các vấn đề khác (như giải bài tập, code), tôi xin phép từ chối."
    )

# ============================================================
# LỚP 4: Main Bot (RAG / LLM Generation)
# ============================================================
class MainBotConfig(BaseModel):
    enabled: bool = True
    provider: str = "groq"
    model: str = "llama-3.1-70b-versatile" # Hoặc Gemini Flash nếu bạn đổi qua Google API
    temperature: float = 0.2

# ============================================================
# CẤU HÌNH TỔNG (Pipeline Orchestrator)
# ============================================================
class QueryFlowConfig(BaseModel):
    input_validation: InputValidationConfig = InputValidationConfig()
    keyword_filter: KeywordFilterConfig = KeywordFilterConfig()
    prompt_guard: PromptGuardConfig = PromptGuardConfig()
    semantic_router: SemanticRouterConfig = SemanticRouterConfig()
    main_bot: MainBotConfig = MainBotConfig()

# Khởi tạo instance config chung để dùng toàn cục
query_flow_config = QueryFlowConfig()