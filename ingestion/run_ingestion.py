"""
CLI entry point for the ingestion pipeline.

Usage:
  python -m src.ingestion.run_ingestion --dry-run
  python -m src.ingestion.run_ingestion --rebuild
  python -m src.ingestion.run_ingestion --source data/structured/raw/
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.core.config import get_settings, DATA_DIR, PROJECT_ROOT
from src.core.logger import get_logger, setup_logging
from src.ingestion.preprocessor import preprocess_text
from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.structured_chunker import StructuredChunker
from src.ingestion.docx_reader import DocxReader
from src.ingestion.program_chunker import ProgramChunker
from src.ingestion.embedding_generator import EmbeddingGenerator
from src.ingestion.vector_indexer import VectorIndexer

logger = get_logger(__name__)


def read_text_file(path: Path) -> str:
    """Read a text file, trying multiple encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode file: {path}")


def read_pdf_file(path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("pymupdf_not_installed", file=str(path))
        return ""


def read_docx_structured(file_path: Path) -> list:
    """Read and chunk a DOCX file using DocxReader + ProgramChunker."""
    reader = DocxReader()
    chunker = ProgramChunker()
    doc = reader.read(file_path)
    chunks = chunker.chunk(doc)
    return chunks


def collect_files(directory: Path) -> list[Path]:
    """Collect all processable files from a directory."""
    extensions = {".txt", ".md", ".pdf", ".csv", ".json", ".docx"}
    files = []
    if directory.is_file():
        files.append(directory)
    elif directory.is_dir():
        for ext in extensions:
            files.extend(directory.rglob(f"*{ext}"))
    # Filter out Word temp files (start with ~$)
    files = [f for f in files if not f.name.startswith("~$")]
    return sorted(files)


async def run_ingestion(
    dry_run: bool = False,
    rebuild: bool = False,
    source_dir: str | None = None,
):
    """
    Main ingestion pipeline.

    1. Collect files from data/unstructured/ and data/structured/
    2. Chunk using appropriate chunker
    3. Generate dual embeddings (Nvidia + Google)
    4. Store in pgvector
    """
    setup_logging()
    settings = get_settings()

    # Determine source directories
    if source_dir:
        sources = [Path(source_dir)]
    else:
        sources = [
            DATA_DIR / "unstructured" / "raw",
            DATA_DIR / "structured" / "raw",
        ]

    # Collect files
    all_files = []
    for src in sources:
        if src.exists():
            all_files.extend(collect_files(src))

    logger.info("files_collected", count=len(all_files))

    if not all_files:
        logger.warning("no_files_found", sources=[str(s) for s in sources])
        return

    # Initialize chunkers
    semantic_chunker = SemanticChunker()
    structured_chunker = StructuredChunker()

    # Process files
    all_chunks = []
    for file_path in all_files:
        logger.info("processing_file", file=str(file_path))

        # DOCX files get special treatment
        if file_path.suffix == ".docx":
            try:
                chunks = read_docx_structured(file_path)
                all_chunks.extend(chunks)
                logger.info("file_chunked", file=file_path.name, chunks=len(chunks), method="program_chunker")
            except Exception as e:
                logger.error("docx_processing_failed", file=str(file_path), error=str(e))
            continue

        # Read content for non-DOCX files
        if file_path.suffix == ".pdf":
            text = read_pdf_file(file_path)
        else:
            text = read_text_file(file_path)

        if not text.strip():
            logger.warning("empty_file", file=str(file_path))
            continue

        # Choose chunker based on directory
        relative = file_path.relative_to(DATA_DIR)
        is_structured = str(relative).startswith("structured")

        if is_structured:
            chunks = structured_chunker.chunk(text, source=file_path.name)
        else:
            chunks = semantic_chunker.chunk(text, source=file_path.name)

        all_chunks.extend(chunks)
        logger.info("file_chunked", file=file_path.name, chunks=len(chunks))

    logger.info("total_chunks", count=len(all_chunks))

    # Dry run: just preview chunks
    if dry_run:
        preview = [
            {
                "content": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                "metadata": c.metadata.model_dump(),
                "char_count": c.char_count,
            }
            for c in all_chunks[:20]  # Preview first 20
        ]
        preview_path = PROJECT_ROOT / "chunks_preview.json"
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(preview, f, ensure_ascii=False, indent=2)
        logger.info("dry_run_complete", preview_path=str(preview_path), total=len(all_chunks))
        return

    # Generate dual embeddings
    embed_gen = EmbeddingGenerator()
    texts = [c.content for c in all_chunks]

    logger.info("generating_embeddings", count=len(texts))

    # Process in batches (API rate limits)
    batch_size = 20
    all_nvidia_embeds = []
    all_google_embeds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        nv, gg = await embed_gen.embed_dual(batch)
        all_nvidia_embeds.extend(nv)
        all_google_embeds.extend(gg)
        logger.info("batch_embedded", batch=i // batch_size + 1, total_batches=(len(texts) + batch_size - 1) // batch_size)

    await embed_gen.close()

    # Index into pgvector
    indexer = VectorIndexer()

    if rebuild:
        indexer.clear_all()
        logger.info("vectorstore_cleared_for_rebuild")

    indexed = indexer.index_chunks(all_chunks, all_nvidia_embeds, all_google_embeds)
    logger.info("ingestion_complete", indexed=indexed)


def main():
    parser = argparse.ArgumentParser(description="UFM Chatbot - Data Ingestion Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Preview chunks without embedding/indexing")
    parser.add_argument("--rebuild", action="store_true", help="Clear existing data and rebuild")
    parser.add_argument("--source", type=str, help="Specific source directory to process")

    args = parser.parse_args()
    asyncio.run(run_ingestion(dry_run=args.dry_run, rebuild=args.rebuild, source_dir=args.source))


if __name__ == "__main__":
    main()
