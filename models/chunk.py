"""Pydantic models for document chunks."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    Metadata for each chunk stored in VectorDB.

    Design principles:
      - Core fields: always present, used for search/filter
      - Lifecycle fields: validity control, version tracking
      - Extensibility: category + extra dict for future use

    ┌──────────────────────────────────────────────────────────┐
    │ CORE — Identity & Structure                             │
    ├──────────────────────────────────────────────────────────┤
    │ source          File nguồn (ThS KDQT.docx)              │
    │ section_path    Breadcrumb (Thạc sĩ KDQT > XÉT TUYỂN)  │
    │ section_name    Tên mục (ĐIỀU KIỆN XÉT TUYỂN)           │
    │ program_name    Tên ngành (KINH DOANH QUỐC TẾ)          │
    │ program_level   Trình độ (thac_si / tien_si)            │
    │ ma_nganh        Mã ngành (8340120)                      │
    │ chunk_index     Vị trí chunk trong section (1, 2, ...)   │
    │ total_chunks    Tổng chunks trong section                │
    ├──────────────────────────────────────────────────────────┤
    │ LIFECYCLE — Validity & Version Control                  │
    ├──────────────────────────────────────────────────────────┤
    │ academic_year   Năm học áp dụng ("2025-2026")           │
    │ valid_from      Hiệu lực từ (date)                      │
    │ valid_until     Hết hạn (date) → auto-exclude outdated  │
    │ is_active       Còn hiệu lực? (True/False)              │
    │ version         Phiên bản nội dung (1, 2, 3...)         │
    │ replaced_by     ID chunk thay thế (nếu bị supersede)    │
    ├──────────────────────────────────────────────────────────┤
    │ RETRIEVAL — Search & Filter Enhancement                 │
    ├──────────────────────────────────────────────────────────┤│
    │ content_hash    SHA256 hash (dedup detection)           │
    │                                                          │
    ├──────────────────────────────────────────────────────────┤
    │ EXTENSIBILITY                                           │
    ├──────────────────────────────────────────────────────────┤
    │ extra           Dict mở rộng bất kỳ                     │
    └──────────────────────────────────────────────────────────┘
    """

    # ═══════════════════════════════════════════════════
    # CORE — Identity & Structure (always present)
    # ═══════════════════════════════════════════════════
    source: str = Field(
        ..., description="Tên file nguồn, VD: ThS KDQT.docx"
    )
    section_path: Optional[str] = Field(
        default=None,
        description="Breadcrumb path, VD: Thạc sĩ KDQT > ĐIỀU KIỆN XÉT TUYỂN",
    )
    section_name: Optional[str] = Field(
        default=None,
        description="Tên section/mục, VD: ĐIỀU KIỆN XÉT TUYỂN",
    )
    program_name: Optional[str] = Field(
        default=None,
        description="Tên chương trình đào tạo, VD: KINH DOANH QUỐC TẾ",
    )
    program_level: Optional[str] = Field(
        default=None,
        description="Trình độ: thac_si | tien_si",
    )
    ma_nganh: Optional[str] = Field(
        default=None,
        description="Mã ngành theo Bộ GD&ĐT, VD: 8340120",
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="Vị trí chunk trong section (1-indexed)",
    )
    total_chunks_in_section: Optional[int] = Field(
        default=None,
        description="Tổng số chunks trong section",
    )

    # ═══════════════════════════════════════════════════
    # LIFECYCLE — Validity & Version Control
    # ═══════════════════════════════════════════════════
    academic_year: Optional[str] = Field(
        default=None,
        description="Năm học áp dụng, VD: 2025-2026. Dùng để filter khi data nhiều năm",
    )
    valid_from: Optional[datetime] = Field(
        default=None,
        description="Thời điểm chunk bắt đầu có hiệu lực",
    )
    valid_until: Optional[datetime] = Field(
        default=None,
        description="Thời điểm chunk hết hạn. Chatbot tự exclude chunk quá hạn khi truy vấn",
    )
    is_active: bool = Field(
        default=True,
        description="True = chunk còn hiệu lực. Set False để soft-delete thay vì xóa",
    )
    version: int = Field(
        default=1,
        description="Phiên bản nội dung. Tăng khi cập nhật thông tin tuyển sinh mới",
    )
    replaced_by: Optional[str] = Field(
        default=None,
        description="chunk_id của chunk thay thế. Khi update, chunk cũ trỏ tới chunk mới",
    )

    # ═══════════════════════════════════════════════════
    # RETRIEVAL — Search & Filter Enhancement
    # ═══════════════════════════════════════════════════
    content_hash: Optional[str] = Field(
        default=None,
        description="SHA256 hash của content. Dùng detect trùng lặp khi re-import",
    )

    # ═══════════════════════════════════════════════════
    # EXTENSIBILITY — Future-proof
    # ═══════════════════════════════════════════════════
    extra: dict = Field(
        default_factory=dict,
        description="Dict mở rộng cho fields chưa dự kiến. VD: {'url': '...', 'keywords': [...]}",
    )


class ProcessedChunk(BaseModel):
    """A processed text chunk ready for embedding."""
    content: str
    metadata: ChunkMetadata
    char_count: int = 0

    def model_post_init(self, __context):
        if not self.char_count:
            self.char_count = len(self.content)


class EmbeddingScore(BaseModel):
    """Quality evaluation of embeddings."""
    avg_score: float
    min_score: float
    max_score: float
    std_dev: float
    distribution: dict = Field(default_factory=dict)
    total_chunks: int = 0
