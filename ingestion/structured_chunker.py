"""
Structured Chunking for well-organized documents.

Algorithm:
  1. Detect headers (H1–H6, numbered sections, lettered items)
  2. Each section (header → next header) becomes one chunk
  3. Tables are preserved as single chunks
  4. Hierarchical section_path metadata: "Tuyển sinh 2025 > Ngành CNTT > Điểm chuẩn"
"""
import re
from typing import Optional

from src.core.config import get_settings
from src.core.logger import get_logger
from src.ingestion.preprocessor import preprocess_text
from src.models.chunk import ChunkMetadata, ProcessedChunk

logger = get_logger(__name__)

# Default header patterns for Vietnamese admissions documents
DEFAULT_HEADER_PATTERNS = [
    r"^#{1,6}\s+.+",                        # Markdown headers
    r"^[IVXLCDM]+\.\s+.+",                  # Roman numerals
    r"^\d+\.\s+.+",                          # Numbered sections
    r"^[a-zA-Z]\)\s+.+",                     # Lettered items
    r"^(?:Điều|Chương|Mục|Phần)\s+\d+",     # Vietnamese legal structure
    r"^(?:NGÀNH|CHƯƠNG TRÌNH|HỌC PHÍ|ĐIỂM)",  # Admissions-specific
]

TABLE_START = re.compile(r"^\|.*\|", re.MULTILINE)
TABLE_SEPARATOR = re.compile(r"^\|[\s\-:|]+\|", re.MULTILINE)


class StructuredChunker:
    """
    Chunks structured documents based on header hierarchy.
    Ideal for: tables, course catalogs, tuition schedules, structured regulations.
    """

    def __init__(
        self,
        header_patterns: Optional[list[str]] = None,
        preserve_tables: bool = True,
        max_chunk_size: int = 2000,
    ):
        settings = get_settings()
        struct_cfg = settings.chunking_config.get("structured", {})

        raw_patterns = header_patterns or struct_cfg.get("header_patterns", DEFAULT_HEADER_PATTERNS)
        self.header_patterns = [re.compile(p, re.MULTILINE) for p in raw_patterns]
        self.preserve_tables = struct_cfg.get("preserve_tables", preserve_tables)
        self.max_chunk_size = struct_cfg.get("max_chunk_size", max_chunk_size)

    def chunk(
        self,
        text: str,
        source: str,
        metadata_extra: Optional[dict] = None,
    ) -> list[ProcessedChunk]:
        """
        Split structured text into chunks based on headers.

        Args:
            text: Document text
            source: Source file name
            metadata_extra: Additional metadata

        Returns:
            List of ProcessedChunk objects
        """
        text = preprocess_text(text, strip_html=True)
        lines = text.split("\n")

        logger.info("structured_chunking_start", source=source, num_lines=len(lines))

        sections = self._split_by_headers(lines)
        chunks = []

        for section in sections:
            header = section["header"]
            content = section["content"]
            section_path = section["path"]

            # Check if content contains a table
            has_table = bool(TABLE_START.search(content))

            # If content is too large, split further
            if len(content) > self.max_chunk_size and not (self.preserve_tables and has_table):
                sub_chunks = self._split_large_section(content, source, section_path, metadata_extra)
                chunks.extend(sub_chunks)
            else:
                meta = ChunkMetadata(
                    source=source,
                    section_path=section_path,
                    **(metadata_extra or {}),
                )
                chunks.append(ProcessedChunk(content=content, metadata=meta))

        logger.info("structured_chunking_done", source=source, num_chunks=len(chunks))
        return chunks

    def _is_header(self, line: str) -> bool:
        """Check if a line matches any header pattern."""
        stripped = line.strip()
        return any(p.match(stripped) for p in self.header_patterns)

    def _extract_header_text(self, line: str) -> str:
        """Extract clean header text (remove markdown # symbols, etc.)."""
        stripped = line.strip()
        # Remove leading # for markdown headers
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        return stripped

    def _split_by_headers(self, lines: list[str]) -> list[dict]:
        """Split lines into sections based on header detection."""
        sections = []
        current_header = ""
        current_lines = []
        header_stack = []  # For building section_path

        for line in lines:
            if self._is_header(line):
                # Save previous section
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sections.append({
                            "header": current_header,
                            "content": content,
                            "path": " > ".join(header_stack) if header_stack else current_header,
                        })

                current_header = self._extract_header_text(line)
                current_lines = [line]

                # Update header stack (simple: just append)
                header_stack.append(current_header)
                # Keep stack manageable (max depth 4)
                if len(header_stack) > 4:
                    header_stack = header_stack[-4:]
            else:
                current_lines.append(line)

        # Final section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({
                    "header": current_header,
                    "content": content,
                    "path": " > ".join(header_stack) if header_stack else current_header,
                })

        # If no headers were found, treat entire text as one chunk
        if not sections and lines:
            full_text = "\n".join(lines).strip()
            if full_text:
                sections.append({
                    "header": "",
                    "content": full_text,
                    "path": source if 'source' in dir() else "document",
                })

        return sections

    def _split_large_section(
        self,
        content: str,
        source: str,
        section_path: str,
        metadata_extra: Optional[dict],
    ) -> list[ProcessedChunk]:
        """Split an oversized section into smaller chunks at paragraph boundaries."""
        paragraphs = content.split("\n\n")
        chunks = []
        current = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > self.max_chunk_size and current:
                chunk_content = "\n\n".join(current)
                meta = ChunkMetadata(
                    source=source,
                    section_path=section_path,
                    **(metadata_extra or {}),
                )
                chunks.append(ProcessedChunk(content=chunk_content, metadata=meta))
                current = []
                current_len = 0

            current.append(para)
            current_len += len(para)

        if current:
            chunk_content = "\n\n".join(current)
            meta = ChunkMetadata(
                source=source,
                section_path=section_path,
                chunk_type="structured",
                **(metadata_extra or {}),
            )
            chunks.append(ProcessedChunk(content=chunk_content, metadata=meta))

        return chunks
