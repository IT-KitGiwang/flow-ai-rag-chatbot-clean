"""
chunk_markdown.py
=================
Chunk tất cả file .md trong data/unstructured/markdown/thongtinchung/
→ Lưu JSON + TXT ra data/unstructured/processed/

Cấu trúc mỗi file:
  Dòng 1       : Ngày hiệu lực: dd/mm/yyyy
  Dòng 2..N-1  : Ngữ cảnh (tiêu đề, số văn bản, mô tả...) — dùng làm doc_title
  Dòng marker  : -start-
  Sau -start-  : Nội dung cần chunk (heading-based, giữ nguyên bảng)

Metadata mỗi chunk khớp CHÍNH XÁC với DocumentChunk + ChunkMetadata:
  DocumentChunk : content, source_file, chunk_type, section_path,
                  metadata_ (JSONB), char_count
  ChunkMetadata : source, section_path, section_name, program_name,
                  program_level, ma_nganh, chunk_index,
                  total_chunks_in_section, academic_year, valid_from,
                  valid_until, is_active, version, replaced_by,
                  category, content_hash, priority, extra

Usage:
    python data/Script/chunk_markdown.py
"""

import hashlib
import json
import re
import sys
import io
from datetime import datetime, date
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MD_DIR  = PROJECT_ROOT / "data" / "unstructured" / "markdown" / "thongtinchung"
OUT_DIR = PROJECT_ROOT / "data" / "unstructured" / "processed"/ "thongtinchung"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 1500

CATEGORY_RULES: list[tuple[str, str]] = [
    ("điều kiện xét tuyển",     "dieu_kien_xet_tuyen"),
    ("điều kiện dự tuyển",      "dieu_kien_xet_tuyen"),
    ("điều kiện tốt nghiệp",    "dieu_kien_xet_tuyen"),
    ("yêu cầu đối với người",   "dieu_kien_xet_tuyen"),
    ("phương thức tuyển sinh",  "dieu_kien_xet_tuyen"),
    ("đối tượng",               "dieu_kien_xet_tuyen"),
    ("điều kiện",               "dieu_kien_xet_tuyen"),
    ("hồ sơ đăng ký",           "dieu_kien_xet_tuyen"),
    ("hồ sơ dự tuyển",          "dieu_kien_xet_tuyen"),
    ("nộp hồ sơ",               "dieu_kien_xet_tuyen"),
    ("học phí",                 "hoc_phi"),
    ("lệ phí",                  "hoc_phi"),
    ("học bổ sung kiến thức",   "bo_sung_kien_thuc"),
    ("bổ sung kiến thức",       "bo_sung_kien_thuc"),
    ("học bổng",                "hoc_bong_uu_dai"),
    ("ưu đãi",                  "hoc_bong_uu_dai"),
    ("chính sách ưu",           "hoc_bong_uu_dai"),
    ("ngành đào tạo",           "chuong_trinh_dao_tao"),
    ("chương trình",            "chuong_trinh_dao_tao"),
    ("hình thức và thời gian",  "chuong_trinh_dao_tao"),
    ("địa chỉ",                 "thong_tin"),
    ("liên hệ",                 "thong_tin"),
    ("đăng ký",                 "thong_tin"),
    ("quy trình",               "thong_tin"),
    ("phụ lục",                 "thong_tin"),
    ("mục đích",                "thong_tin"),
    ("phạm vi",                 "thong_tin"),
    ("quy định chung",          "thong_tin"),
    ("điều khoản",              "thong_tin"),
    ("xử lý vi phạm",           "thong_tin"),
]

PRIORITY_MAP: dict[str, int] = {
    "dieu_kien_xet_tuyen": 10,
    "hoc_phi":              9,
    "hoc_bong_uu_dai":      8,
    "co_hoi_nghe_nghiep":   7,
    "bo_sung_kien_thuc":    6,
    "chuong_trinh_dao_tao": 5,
    "thong_tin":            4,
    "gioi_thieu":           3,
}


# ── Header parsing ─────────────────────────────────────────────────────────────
_DATE_LINE = re.compile(r"^Ngày\s+hiệu\s+lực\s*:\s*(\d{1,2}/\d{1,2}/\d{2,4})\s*$", re.IGNORECASE)
_START_MARKER = "-start-"


def parse_file_header(lines: list[str]) -> tuple[date | None, str, int]:
    """
    Phân tích header file theo cấu trúc:
      Dòng 1         : Ngày hiệu lực: dd/mm/yyyy
      Dòng 2..start  : ngữ cảnh / tiêu đề
      Dòng -start-   : marker

    Trả về: (valid_from_date, doc_title, start_index)
    start_index = index dòng đầu tiên SAU -start-
    """
    valid_from: date | None = None
    context_lines: list[str] = []
    start_idx = len(lines)  # fallback: toàn bộ file là content

    for i, line in enumerate(lines):
        stripped = line.strip()
        if i == 0:
            m = _DATE_LINE.match(stripped)
            if m:
                try:
                    valid_from = datetime.strptime(m.group(1).strip(), "%d/%m/%Y").date()
                except ValueError:
                    try:
                        valid_from = datetime.strptime(m.group(1).strip(), "%d/%m/%y").date()
                    except ValueError:
                        pass
                continue  # dòng ngày không vào context

        if stripped == _START_MARKER:
            start_idx = i + 1
            break

        if stripped:
            # Bỏ ký hiệu markdown # và ** khi lấy tiêu đề ngữ cảnh
            ctx = re.sub(r"^#+\s*", "", stripped)
            ctx = re.sub(r"^\*\*|\*\*$", "", ctx).strip()
            ctx = re.sub(r"^\*|\*$", "", ctx).strip()
            if ctx:
                context_lines.append(ctx)

    doc_title = " – ".join(context_lines) if context_lines else "Tài liệu UFM"
    return valid_from, doc_title, start_idx


# ── Program level & academic year ─────────────────────────────────────────────
def detect_program_level(text: str, filename: str) -> str | None:
    combined = (filename + " " + text[:500]).lower()
    if "tiến sĩ" in combined or "tien si" in combined:
        return "tien_si"
    if "thạc sĩ" in combined or "thac si" in combined:
        return "thac_si"
    return None


_YEAR_RANGE_RE = re.compile(r"\b(20\d{2})[–\-](20\d{2})\b")
_YEAR_SINGLE_RE = re.compile(r"\b(20\d{2})\b")


def extract_academic_year(text: str) -> str | None:
    m = _YEAR_RANGE_RE.search(text[:600])
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = _YEAR_SINGLE_RE.search(text[:600])
    if m:
        yr = int(m.group(1))
        return f"{yr}-{yr + 1}"
    return None


# ── Category ───────────────────────────────────────────────────────────────────
def detect_category(section_name: str) -> str:
    lower = section_name.lower()
    for kw, cat in CATEGORY_RULES:
        if kw in lower:
            return cat
    return "thong_tin"


# ── Heading detection ──────────────────────────────────────────────────────────
_HEADING_PATS = [
    re.compile(r"^#{1,4}\s+(.+)$"),                          # # ## ### ####
    re.compile(r"^#{1,4}\s*\*\*(.+?)\*\*\s*$"),              # ## **Heading**
    re.compile(r"^\*\*(Điều\s+\d+[\.\:].*?)\*\*\s*$"),       # **Điều N. ...**
    re.compile(r"^\*\*(\d+[\.\:]\s+[^\*\n]{3,80})\*\*\s*$"), # **N. Tiêu đề**
]


def is_heading(line: str) -> bool:
    s = line.strip()
    return any(p.match(s) for p in _HEADING_PATS)


def heading_text(line: str) -> str:
    s = line.strip()
    for p in _HEADING_PATS:
        m = p.match(s)
        if m:
            txt = m.group(m.lastindex)
            txt = re.sub(r"^\*\*|\*\*$", "", txt).strip()
            txt = re.sub(r"^#+\s*", "", txt).strip()
            return txt
    return s


# ── HTML inline clean (tables are kept raw) ───────────────────────────────────
_HTML_INLINE = re.compile(r"<br\s*/?>", re.IGNORECASE)  # only clean <br/> inside cells


def clean_line(line: str) -> str:
    """Nhẹ nhàng: chỉ xóa <br/> entity trong line bình thường; giữ bảng nguyên."""
    if line.strip().startswith("|"):
        return line   # bảng giữ nguyên hoàn toàn
    return _HTML_INLINE.sub(" ", line)


# ── Section splitting ──────────────────────────────────────────────────────────
def split_sections(content_lines: list[str]) -> list[tuple[str, list[str]]]:
    """
    Chia nội dung (sau -start-) thành [(section_name, body_lines)].
    Heading trở thành tên section.
    """
    sections: list[tuple[str, list[str]]] = []
    cur_name = ""
    cur_lines: list[str] = []

    def flush():
        body = "\n".join(cur_lines).strip()
        if body:
            sections.append((cur_name, cur_lines[:]))
        cur_lines.clear()

    for raw in content_lines:
        line = clean_line(raw.rstrip())
        if is_heading(line):
            flush()
            cur_name = heading_text(line)
            cur_lines = [line]   # giữ heading ở đầu chunk
        else:
            cur_lines.append(line)

    flush()
    return sections


# ── Table-safe paragraph splitting ────────────────────────────────────────────
def split_section_into_chunks(body_lines: list[str], max_chars: int) -> list[str]:
    """
    Ghép body_lines thành các chunk không vượt quá max_chars.
    Bảng (các dòng bắt đầu |) không bị cắt giữa chừng.
    """
    full = "\n".join(body_lines).strip()
    if len(full) <= max_chars:
        return [full] if full else []

    # Tách thành "blocks" — mỗi block là bảng liên tục HOẶC đoạn văn
    blocks: list[str] = []
    buf: list[str] = []
    in_table = False

    for raw_line in body_lines:
        line = raw_line.rstrip()
        if line.strip().startswith("|"):
            if not in_table and buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            in_table = True
            buf.append(line)
        else:
            if in_table:
                blocks.append("\n".join(buf).strip())
                buf = []
                in_table = False
            if not line.strip() and buf:
                buf.append(line)
                if len("\n".join(buf)) >= max_chars // 2:
                    blocks.append("\n".join(buf).strip())
                    buf = []
            elif line.strip():
                buf.append(line)

    if buf:
        blocks.append("\n".join(buf).strip())

    # Gộp / tách blocks
    chunks: list[str] = []
    current = ""

    for block in blocks:
        if not block.strip():
            continue
        candidate = (current + "\n\n" + block).strip() if current else block
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(block) > max_chars:
                # Bảng quá dài → giữ nguyên (không cắt giữa bảng)
                chunks.append(block)
                current = ""
            else:
                current = block

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ── Main file processor ────────────────────────────────────────────────────────
def process_file(md_path: Path) -> list[dict]:
    """
    Đọc và chunk một file .md theo cấu trúc chuẩn.
    Trả về list của chunk dicts — mỗi dict có đúng các fields:
      DocumentChunk  : content, source_file, chunk_type, section_path,
                       char_count, metadata_
      ChunkMetadata  : tất cả fields của class ChunkMetadata trong models/chunk.py
    """
    raw_text = md_path.read_text(encoding="utf-8", errors="replace")
    lines = raw_text.splitlines()
    filename = md_path.name

    # 1. Parse header
    valid_from, doc_title, start_idx = parse_file_header(lines)

    content_lines = lines[start_idx:]

    # 2. Detect doc-level info từ header + vài dòng đầu content
    sample = "\n".join(content_lines[:30])
    program_level = detect_program_level(doc_title + "\n" + sample, filename)
    academic_year = extract_academic_year(doc_title + "\n" + sample)

    # 3. Chia sections
    sections = split_sections(content_lines)

    # 4. Merge sections quá ngắn
    merged: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(sections):
        name, body = sections[i]
        body_text = "\n".join(body).strip()
        while len(body_text) < MIN_CHUNK_CHARS and i + 1 < len(sections):
            i += 1
            nxt_name, nxt_body = sections[i]
            body = body + [""] + nxt_body
            body_text = "\n".join(body).strip()
            if not name:
                name = nxt_name
        merged.append((name, body))
        i += 1

    # 5. Tạo chunks
    chunks: list[dict] = []
    global_idx = 0

    for sec_name, body_lines in merged:
        sub = split_section_into_chunks(body_lines, MAX_CHUNK_CHARS)
        total_in_sec = len(sub)

        for pos, chunk_text in enumerate(sub, 1):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            global_idx += 1
            category     = detect_category(sec_name)
            priority     = PRIORITY_MAP.get(category, 4)
            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
            sec_path     = f"{doc_title} > {sec_name}" if sec_name else doc_title

            # valid_from là date → chuyển sang ISO string cho JSON
            valid_from_str: str | None = valid_from.isoformat() if valid_from else None

            chunk: dict = {
                # ── DocumentChunk columns ──────────────────────────────
                "content":      chunk_text,
                "source_file":  filename,
                "chunk_type":   "structured",
                "section_path": sec_path[:1000],
                "char_count":   len(chunk_text),
                # ── metadata_ JSONB → ChunkMetadata fields ─────────────
                "metadata": {
                    # CORE
                    "source":                   filename,
                    "section_path":             sec_path[:1000],
                    "section_name":             sec_name or None,
                    "program_name":             None,
                    "program_level":            program_level,
                    "ma_nganh":                 None,
                    "chunk_index":              pos,
                    "total_chunks_in_section":  total_in_sec,
                    # LIFECYCLE
                    "academic_year":  academic_year,
                    "valid_from":     valid_from_str,
                    "valid_until":    None,
                    "is_active":      True,
                    "version":        1,
                    "replaced_by":    None,
                    # RETRIEVAL
                    "category":      category,
                    "content_hash":  content_hash,
                    "priority":      priority,
                    # EXTENSIBILITY
                    "extra": {
                        "source_type": "pdf_parsed",
                        "doc_title":   doc_title,
                    },
                },
            }
            chunks.append(chunk)

    return chunks


# ── Quality checks ─────────────────────────────────────────────────────────────
def quality_check(all_chunks: list[dict]) -> list[tuple[str, bool]]:
    required_meta = [
        "source", "section_path", "section_name", "program_name",
        "program_level", "ma_nganh", "chunk_index", "total_chunks_in_section",
        "academic_year", "valid_from", "valid_until", "is_active",
        "version", "replaced_by", "category", "content_hash", "priority", "extra",
    ]
    checks: list[tuple[str, bool]] = []

    checks.append(("Không có chunk rỗng",
                   all(c["content"].strip() for c in all_chunks)))
    checks.append((f"Chunk <= {MAX_CHUNK_CHARS} chars (±10%)",
                   all(c["char_count"] <= MAX_CHUNK_CHARS * 1.1 for c in all_chunks)))
    checks.append(("source_file có giá trị",
                   all(c["source_file"] for c in all_chunks)))
    checks.append(("chunk_type = 'structured'",
                   all(c["chunk_type"] == "structured" for c in all_chunks)))
    checks.append(("section_path có giá trị",
                   all(c["section_path"] for c in all_chunks)))
    checks.append(("content_hash có giá trị",
                   all(c["metadata"]["content_hash"] for c in all_chunks)))
    checks.append(("category có giá trị",
                   all(c["metadata"]["category"] for c in all_chunks)))
    checks.append(("priority > 0",
                   all(c["metadata"]["priority"] > 0 for c in all_chunks)))
    checks.append(("is_active = True",
                   all(c["metadata"]["is_active"] is True for c in all_chunks)))
    checks.append((f"Đủ {len(required_meta)} metadata fields",
                   all(set(c["metadata"].keys()) == set(required_meta) for c in all_chunks)))

    return checks


# ── Export ─────────────────────────────────────────────────────────────────────
def export_json(all_chunks: list[dict], files_info: list[dict], path: Path) -> None:
    data = {
        "summary": {
            "generated_at": datetime.now().isoformat(),
            "source_dir":   str(MD_DIR),
            "total_files":  len(files_info),
            "total_chunks": len(all_chunks),
        },
        "files": files_info,
        "chunks": [
            {
                "chunk_id":     f"{c['source_file']}__chunk_{i+1:03d}",
                "chunk_number": i + 1,
                "content":      c["content"],
                "source_file":  c["source_file"],
                "chunk_type":   c["chunk_type"],
                "section_path": c["section_path"],
                "char_count":   c["char_count"],
                "metadata":     c["metadata"],
            }
            for i, c in enumerate(all_chunks)
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def export_txt(all_chunks: list[dict], path: Path) -> None:
    SEP = "=" * 70
    SEP2 = "-" * 70
    lines = [
        "MARKDOWN CHUNKS — UFM thongtinchung",
        f"Generated : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        f"Total     : {len(all_chunks)} chunks",
        SEP, "",
    ]
    for i, c in enumerate(all_chunks, 1):
        m = c["metadata"]
        lines += [
            SEP,
            f"CHUNK #{i:03d}  [{c['source_file']}]",
            f"Section   : {m['section_name'] or '(đầu tài liệu)'}",
            f"Path      : {c['section_path']}",
            f"Category  : {m['category']}  |  Priority: {m['priority']}  |  "
            f"Chars: {c['char_count']}",
            f"Level     : {m['program_level'] or 'N/A'}  |  "
            f"Year      : {m['academic_year'] or 'N/A'}  |  "
            f"Valid from: {m['valid_from'] or 'N/A'}",
            SEP2,
            c["content"],
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    SEP = "=" * 68

    md_files = sorted(MD_DIR.glob("*.md"))
    md_files = [f for f in md_files if not f.name.startswith(".")]

    if not md_files:
        print(f"⚠  Không tìm thấy file .md trong: {MD_DIR}")
        return

    print(SEP)
    print("  UFM Markdown Chunker — thongtinchung")
    print(f"  Source  : {MD_DIR}")
    print(f"  Output  : {OUT_DIR}")
    print(f"  Files   : {len(md_files)}")
    print(f"  MaxChars: {MAX_CHUNK_CHARS}")
    print(SEP)

    all_chunks: list[dict] = []
    files_info: list[dict] = []
    errors: list[str] = []

    for md_path in md_files:
        print(f"\n  [FILE] {md_path.name}")
        try:
            chunks = process_file(md_path)
            if not chunks:
                print(f"         → 0 chunks  (bỏ qua — không có nội dung sau -start-)")
                continue

            all_chunks.extend(chunks)
            char_counts = [c["char_count"] for c in chunks]
            files_info.append({
                "filename":     md_path.name,
                "chunks_count": len(chunks),
                "min_chars":    min(char_counts),
                "max_chars":    max(char_counts),
                "avg_chars":    round(sum(char_counts) / len(char_counts), 1),
                "doc_title":    chunks[0]["metadata"]["extra"]["doc_title"],
                "valid_from":   chunks[0]["metadata"]["valid_from"],
                "program_level": chunks[0]["metadata"]["program_level"],
                "academic_year": chunks[0]["metadata"]["academic_year"],
            })
            print(f"         → {len(chunks)} chunks | "
                  f"chars min={min(char_counts)} max={max(char_counts)} "
                  f"avg={sum(char_counts)/len(char_counts):.0f}")

        except Exception as e:
            import traceback
            print(f"  [ERROR] {md_path.name}: {e}")
            traceback.print_exc()
            errors.append(f"{md_path.name}: {e}")

    if not all_chunks:
        print("\n✗ Không có chunk nào được tạo.")
        return

    # Quality checks
    print(f"\n{SEP}")
    print(f"  TỔNG KẾT: {len(md_files)} files → {len(all_chunks)} chunks")
    print(SEP)
    print("\n  [QUALITY CHECKS]")
    checks = quality_check(all_chunks)
    all_pass = True
    for name, passed in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"    [{icon}] {name}")
        if not passed:
            all_pass = False

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    ✗ {e}")

    # Export
    json_path = OUT_DIR / "chunks_markdown.json"
    txt_path  = OUT_DIR / "chunks_markdown.txt"
    export_json(all_chunks, files_info, json_path)
    export_txt(all_chunks, txt_path)

    print(f"\n{SEP}")
    if all_pass and not errors:
        print("  >>> ALL CHECKS PASSED — READY FOR EMBEDDING! <<<")
    else:
        print("  >>> SOME CHECKS FAILED — PLEASE REVIEW <<<")
    print(f"\n  Output:")
    print(f"    JSON : {json_path}  ({json_path.stat().st_size // 1024} KB)")
    print(f"    TXT  : {txt_path}  ({txt_path.stat().st_size // 1024} KB)")
    print(SEP)


if __name__ == "__main__":
    main()
