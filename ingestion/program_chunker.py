"""
Program Chunker — Context-preserving chunking for academic program DOCX files.

Optimized for text-embedding-3-large (8,192 token context).

Strategy (8K-window):
  1. Each DOCX file → merged sections to maximize context per chunk
  2. Title-only sections are MERGED with next content section (no orphan titles)
  3. Consecutive "Giới thiệu chung" sections are merged into one chunk
  4. Each major section (ĐIỀU KIỆN, CƠ HỘI, ...) becomes one rich chunk
  5. Context prefix on every chunk for embedding quality
  6. Only split when section exceeds max_chunk_size (rare with 8K window)
  7. Rich metadata: program_name, program_level, section_name, chunk_index

Token budget guideline (text-embedding-3-large):
  - Max input: 8,192 tokens
  - 1 Vietnamese token ≈ 2-3 chars (BPE tokenizer for Vietnamese is less efficient)
  - Safe max_chunk_size = 6000 chars ≈ 2000-3000 tokens (well within 8K)
  - Leaves room for context prefix overhead
"""
import re
from typing import Optional

from src.core.config import get_settings
from src.core.logger import get_logger
from src.ingestion.docx_reader import DocxDocument, DocxSection
from src.ingestion.preprocessor import preprocess_text, split_sentences_vietnamese
from src.models.chunk import ChunkMetadata, ProcessedChunk

logger = get_logger(__name__)

# Level display names
LEVEL_DISPLAY = {
    "thac_si": "Thạc sĩ",
    "tien_si": "Tiến sĩ",
    "dai_hoc": "Đại học",
    "unknown": "",
}

# ── Mã ngành mapping (level, normalized_name) → code ────────────────
# Lookup uses normalized lowercase name for fuzzy matching
MA_NGANH_MAP = {
    # Tiến sĩ
    ("tien_si", "tài chính – ngân hàng"): "9340201",
    ("tien_si", "tài chính - ngân hàng"): "9340201",
    ("tien_si", "quản trị kinh doanh"): "9340101",
    ("tien_si", "quản lý kinh tế"): "9310110",
    # Thạc sĩ
    ("thac_si", "tài chính – ngân hàng"): "8340201",
    ("thac_si", "tài chính - ngân hàng"): "8340201",
    ("thac_si", "quản trị kinh doanh"): "8340101",
    ("thac_si", "quản lý kinh tế"): "8310110",
    ("thac_si", "marketing"): "8340115",
    ("thac_si", "kinh doanh quốc tế"): "8340120",
    ("thac_si", "kế toán"): "8340301",
    ("thac_si", "toán kinh tế"): "8310108",
    ("thac_si", "kinh tế học"): "8310101",
    ("thac_si", "luật kinh tế"): "8380107",
}


class ProgramChunker:
    """
    Specialized chunker for Vietnamese academic program documents.

    Optimized for text-embedding-3-large (8,192 token window).

    Key features:
      - Merges title-only sections with next content (no orphan chunks)
      - Merges consecutive intro sections into one
      - Context prefix on every chunk
      - Smart splitting at paragraph boundaries with overlap
      - Rich metadata per chunk
    """

    def __init__(
        self,
        max_chunk_size: int = 6000,
        overlap_sentences: int = 2,
        context_prefix: bool = True,
        min_chunk_size: int = 200,
    ):
        settings = get_settings()
        prog_cfg = settings.chunking_config.get("program", {})

        self.max_chunk_size = prog_cfg.get("max_chunk_size", max_chunk_size)
        self.overlap_sentences = prog_cfg.get("overlap_sentences", overlap_sentences)
        self.context_prefix = prog_cfg.get("context_prefix", context_prefix)
        self.min_chunk_size = prog_cfg.get("min_chunk_size", min_chunk_size)

    def chunk(
        self,
        doc: DocxDocument,
        metadata_extra: Optional[dict] = None,
    ) -> list[ProcessedChunk]:
        """
        Chunk a parsed DOCX document into context-enriched chunks.

        Pipeline:
          1. Merge title-only sections
          2. Merge consecutive intro sections
          3. Generate chunks with context prefix
          4. Split oversized sections if needed
        """
        logger.info(
            "program_chunking_start",
            file=doc.filename,
            program=doc.program_name,
            sections=len(doc.sections),
        )

        # Phase 1: Merge title-only and consecutive intro sections
        merged_sections = self._merge_sections(doc.sections)

        # Phase 2: Generate chunks
        all_chunks: list[ProcessedChunk] = []
        for section in merged_sections:
            section_chunks = self._chunk_section(doc, section, metadata_extra)
            all_chunks.extend(section_chunks)

        # Phase 3: Post-process — merge any remaining tiny chunks
        all_chunks = self._merge_tiny_chunks(all_chunks)

        logger.info(
            "program_chunking_done",
            file=doc.filename,
            total_chunks=len(all_chunks),
        )
        return all_chunks

    # ═══════════════════════════════════════════════════════
    # Phase 1: Section Merging
    # ═══════════════════════════════════════════════════════

    def _merge_sections(self, sections: list[DocxSection]) -> list[DocxSection]:
        """
        Merge sections that should not stand alone:
          - Title-only sections (heading with no content) → merge with next
          - Consecutive level-0 sections ("Giới thiệu") → combine into one
        """
        if not sections:
            return []

        merged: list[DocxSection] = []
        i = 0

        while i < len(sections):
            section = sections[i]

            # Case 1: Title-only section (no content body)
            if not section.content.strip() and i + 1 < len(sections):
                next_section = sections[i + 1]

                # Merge: use title heading but next section's content
                combined_heading = section.heading_text
                combined_content = next_section.content
                combined_paragraphs = next_section.paragraphs.copy()

                # If next section also has a heading, prepend it to content
                if next_section.heading_text:
                    combined_content = (
                        next_section.heading_text + "\n\n" + combined_content
                    )
                    combined_paragraphs.insert(0, next_section.heading_text)

                merged.append(DocxSection(
                    heading_level=section.heading_level,
                    heading_text=combined_heading,
                    content=combined_content.strip(),
                    paragraphs=combined_paragraphs,
                ))
                i += 2  # Skip next section (already consumed)
                continue

            # Case 2: Check for consecutive level-0 sections to merge
            if (
                section.heading_level == 0
                and merged
                and merged[-1].heading_level == 0
            ):
                # Merge with previous level-0 section
                prev = merged[-1]
                combined = prev.content
                if section.heading_text:
                    combined += "\n\n" + section.heading_text
                if section.content:
                    combined += "\n\n" + section.content

                merged[-1] = DocxSection(
                    heading_level=0,
                    heading_text=prev.heading_text,
                    content=combined.strip(),
                    paragraphs=prev.paragraphs + (
                        [section.heading_text] if section.heading_text else []
                    ) + section.paragraphs,
                )
                i += 1
                continue

            # Normal section — keep as is
            merged.append(section)
            i += 1

        return merged

    # ═══════════════════════════════════════════════════════
    # Phase 2: Chunk Generation
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _lookup_ma_nganh(program_level: str, program_name: str) -> str:
        """Lookup mã ngành from program level and name."""
        normalized = program_name.strip().lower()
        # Direct lookup
        code = MA_NGANH_MAP.get((program_level, normalized))
        if code:
            return code
        # Fuzzy: try substring matching for cases like 'KDQT' vs full name
        for (level, name), code in MA_NGANH_MAP.items():
            if level == program_level and (
                name in normalized or normalized in name
            ):
                return code
        return ""

    def _build_context_prefix(
        self, doc: DocxDocument, section: DocxSection,
        metadata_extra: Optional[dict] = None,
    ) -> str:
        """
        Build a context prefix string for embedding quality.

        Format:
          [Chương trình: {level} {full_name} ({abbr}) | Mã ngành: {ma_nganh}]
          [Mục: {section_name}]

        Uses metadata_extra (from chunking_settings.yaml program_map) for
        full_name, abbr, and ma_nganh when available — falls back to
        doc.program_name and MA_NGANH_MAP lookup.
        """
        if not self.context_prefix:
            return ""

        extra = metadata_extra or {}
        level_display = LEVEL_DISPLAY.get(doc.program_level, "")

        # Prefer config-sourced data over auto-detected
        full_name = extra.get("full_name") or doc.program_name
        abbr      = extra.get("abbr", "")
        ma_nganh  = extra.get("ma_nganh") or self._lookup_ma_nganh(doc.program_level, doc.program_name)

        parts = []

        # Main identity line
        if level_display and full_name:
            name_part = full_name
            if abbr and abbr.upper() != full_name.upper():
                name_part = f"{full_name} ({abbr})"
            label = f"[Chương trình: {level_display} {name_part}"
            if ma_nganh:
                label += f" | Mã ngành: {ma_nganh}"
            label += "]"
            parts.append(label)
        elif full_name:
            label = f"[Chương trình: {full_name}"
            if ma_nganh:
                label += f" | Mã ngành: {ma_nganh}"
            label += "]"
            parts.append(label)

        # Section line (only for named sections, not intro)
        if section.heading_text and section.heading_level > 0:
            parts.append(f"[Mục: {section.heading_text}]")

        return "\n".join(parts) + "\n\n" if parts else ""

    def _determine_section_name(self, section: DocxSection) -> str:
        """Get a clean section name for metadata."""
        if section.heading_level > 0 and section.heading_text:
            return section.heading_text
        return "Giới thiệu chung"

    def _build_section_path(
        self, doc: DocxDocument, section: DocxSection
    ) -> str:
        """Build breadcrumb-style section path."""
        level_display = LEVEL_DISPLAY.get(doc.program_level, "")
        if level_display:
            path = f"{level_display} {doc.program_name}"
        else:
            path = doc.program_name

        if section.heading_text and section.heading_level > 0:
            path += f" > {section.heading_text}"

        return path

    def _chunk_section(
        self,
        doc: DocxDocument,
        section: DocxSection,
        metadata_extra: Optional[dict],
    ) -> list[ProcessedChunk]:
        """Chunk a single section, splitting if it exceeds max size."""
        # Build context prefix using program info from metadata_extra (abbr, full_name, ma_nganh)
        prefix = self._build_context_prefix(doc, section, metadata_extra)
        content = preprocess_text(section.content, strip_html=True) if section.content else ""
        section_path = self._build_section_path(doc, section)
        section_name = self._determine_section_name(section)

        if not content.strip():
            if not section.heading_text:
                return []
            content = section.heading_text

        effective_max = self.max_chunk_size - len(prefix)
        if effective_max < 500:
            effective_max = 500

        # ma_nganh: prefer from metadata_extra (config lookup), fallback to MA_NGANH_MAP
        extra = metadata_extra or {}
        ma_nganh = extra.get("ma_nganh") or self._lookup_ma_nganh(doc.program_level, doc.program_name)

        # Single chunk (most cases with 8K window)
        if len(content) <= effective_max:
            full_content = prefix + content
            meta = ChunkMetadata(
                source=doc.filename,
                section_path=section_path,
                program_name=doc.program_name,
                program_level=doc.program_level,
                ma_nganh=ma_nganh,
                section_name=section_name,
                chunk_index=1,
                total_chunks_in_section=1,
            )
            return [ProcessedChunk(content=full_content, metadata=meta)]

        # Content too large — split
        return self._split_large_section(
            content=content,
            prefix=prefix,
            effective_max=effective_max,
            doc=doc,
            section_path=section_path,
            section_name=section_name,
            ma_nganh=ma_nganh,
        )

    def _split_large_section(
        self,
        content: str,
        prefix: str,
        effective_max: int,
        doc: DocxDocument,
        section_path: str,
        section_name: str,
        ma_nganh: str = "",
    ) -> list[ProcessedChunk]:
        """Split oversized section at paragraph boundaries with overlap."""
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [content]

        chunk_groups: list[list[str]] = []
        current_group: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)

            # Handle single paragraph exceeding max
            if para_len > effective_max:
                if current_group:
                    chunk_groups.append(current_group)
                    current_group = []
                    current_len = 0

                sentences = split_sentences_vietnamese(para)
                sent_group: list[str] = []
                sent_len = 0
                for sent in sentences:
                    if sent_len + len(sent) > effective_max and sent_group:
                        chunk_groups.append([" ".join(sent_group)])
                        overlap = sent_group[-self.overlap_sentences:] if self.overlap_sentences else []
                        sent_group = overlap.copy()
                        sent_len = sum(len(s) for s in sent_group)
                    sent_group.append(sent)
                    sent_len += len(sent)
                if sent_group:
                    chunk_groups.append([" ".join(sent_group)])
                continue

            if current_len + para_len + 2 > effective_max and current_group:
                chunk_groups.append(current_group)
                if self.overlap_sentences > 0:
                    last_para = current_group[-1]
                    last_sents = split_sentences_vietnamese(last_para)
                    overlap_text = " ".join(last_sents[-self.overlap_sentences:])
                    current_group = [overlap_text]
                    current_len = len(overlap_text)
                else:
                    current_group = []
                    current_len = 0

            current_group.append(para)
            current_len += para_len + 2

        if current_group:
            chunk_groups.append(current_group)

        total = len(chunk_groups)
        chunks: list[ProcessedChunk] = []

        for idx, group in enumerate(chunk_groups, 1):
            group_content = "\n\n".join(group)
            if total > 1:
                part_label = f"[Phần {idx}/{total}]\n"
                full_content = prefix + part_label + group_content
            else:
                full_content = prefix + group_content

            meta = ChunkMetadata(
                source=doc.filename,
                section_path=section_path,
                program_name=doc.program_name,
                program_level=doc.program_level,
                ma_nganh=ma_nganh,
                section_name=section_name,
                chunk_index=idx,
                total_chunks_in_section=total,
                **(metadata_extra or {}),
            )
            chunks.append(ProcessedChunk(content=full_content, metadata=meta))

        return chunks

    # ═══════════════════════════════════════════════════════
    # Phase 3: Post-processing
    # ═══════════════════════════════════════════════════════

    def _merge_tiny_chunks(
        self, chunks: list[ProcessedChunk]
    ) -> list[ProcessedChunk]:
        """Merge any remaining chunks that are too small with neighbors."""
        if len(chunks) <= 1:
            return chunks

        result: list[ProcessedChunk] = []
        for chunk in chunks:
            if (
                chunk.char_count < self.min_chunk_size
                and result
                and result[-1].char_count + chunk.char_count <= self.max_chunk_size
            ):
                # Merge with previous chunk
                prev = result[-1]
                merged_content = prev.content + "\n\n" + chunk.content
                result[-1] = ProcessedChunk(
                    content=merged_content,
                    metadata=prev.metadata,
                )
            else:
                result.append(chunk)

        return result
