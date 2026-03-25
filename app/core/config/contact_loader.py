"""
Contact Info Loader — Đọc thông tin liên hệ tĩnh từ file Markdown.

File contact_info.md chứa toàn bộ SĐT, email, địa chỉ phòng ban UFM.
Được cache 1 lần duy nhất khi module load — KHÔNG gọi LLM, KHÔNG tốn token.

Sử dụng:
    from app.core.config.contact_loader import get_contact_block, get_hotline_short

    # Lấy toàn bộ thông tin liên hệ (để gắn vào fallback message dài)
    full_contact = get_contact_block()

    # Lấy dòng hotline ngắn gọn (để gắn vào message block ngắn)
    short = get_hotline_short()
"""

from pathlib import Path

_CONTACT_FILE = Path(__file__).parent / "contact_info.md"

# Cache nội dung khi import module (1 lần duy nhất)
_contact_cache: str | None = None


def _load_contact() -> str:
    """Đọc file contact_info.md, cache lại."""
    global _contact_cache
    if _contact_cache is not None:
        return _contact_cache
    try:
        _contact_cache = _CONTACT_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        _contact_cache = (
            "Hotline Tuyển sinh: (028) 3772 0406 | Email: tuyensinh@ufm.edu.vn\n"
            "Website: https://tuyensinh.ufm.edu.vn"
        )
    return _contact_cache


def get_contact_block() -> str:
    """Trả toàn bộ nội dung contact_info.md (gắn vào fallback dài)."""
    return _load_contact()


def get_hotline_short() -> str:
    """Trả 1 dòng hotline ngắn gọn (gắn vào cuối message block ngắn)."""
    return (
        "💡 **Cần hỗ trợ trực tiếp?**\n"
        "📞 Hotline: (028) 3772 0406 (Cử nhân) | 0967 657 532 (Thạc sĩ/Tiến sĩ)\n"
        "💬 Zalo UFM: [zalo.me/ufmhcm](https://zalo.me/ufmhcm)"
    )
"""Utility to load static contact info from contact_info.md."""
