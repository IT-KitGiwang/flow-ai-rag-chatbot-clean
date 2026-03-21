"""
Form Selector — Xác định loại mẫu đơn phù hợp từ standalone_query.

Dùng keyword matching trước (miễn phí, 0ms).
Nếu không match được → fallback "don_chung".
"""

from app.core.config.form_config import FormTemplateDef
from app.utils.logger import get_logger

logger = get_logger(__name__)


def select_form(standalone_query: str, forms_catalog: list[FormTemplateDef]) -> dict:
    """
    Match standalone_query với danh mục mẫu đơn bằng keyword.

    Args:
        standalone_query: Câu hỏi đã reformulate.
        forms_catalog: List[FormTemplateDef] từ form_cfg.forms.

    Returns:
        dict — entry trong forms_catalog match nhất (dạng dict để các hàm khác dùng .get()).
    """
    query_lower = standalone_query.lower()

    best_match = None
    best_score = 0

    for form_entry in forms_catalog:
        score = 0
        for kw in form_entry.keywords:
            if kw.lower() in query_lower:
                score += 1

        if score > best_score:
            best_score = score
            best_match = form_entry

    if best_match and best_score > 0:
        logger.info(
            "Form Selector - Match: '%s' (score=%d)",
            best_match.id, best_score,
        )
        return best_match.model_dump()

    # Fallback: đơn chung
    fallback = next(
        (f for f in forms_catalog if f.id == "don_chung"),
        forms_catalog[-1] if forms_catalog else None,
    )
    logger.info("Form Selector - Khong match keyword -> fallback don_chung")
    if fallback:
        return fallback.model_dump()
    return {"id": "don_chung", "name": "Đơn hành chính chung", "keywords": [], "template_file": None}
