import uuid
from cachetools import TTLCache

# Cache lưu trữ dữ liệu tài liệu mổi session. (TTLCache max 1000 items, TTL 300s = 5 phút)
document_cache = TTLCache(maxsize=1000, ttl=300)

def create_document_session(document_data: dict) -> str:
    """
    Tạo session_id tạm thời cho nội dung tài liệu (tồn tại 5 phút).
    """
    session_id = str(uuid.uuid4())
    # Lưu toàn bộ dữ liệu (bao gồm nội dung 'content' và metadata) vào cache
    document_cache[session_id] = document_data
    return session_id

def get_document_session(session_id: str) -> dict:
    """
    Lấy thông tin tài liệu từ cache bằng session_id.
    """
    return document_cache.get(session_id)
