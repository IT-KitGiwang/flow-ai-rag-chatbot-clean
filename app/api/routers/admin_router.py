"""
Admin Router — FastAPI Endpoints cho Admin Ingestion API.

Endpoints:
  POST /api/v1/admin/login          → Lấy JWT token
  POST /api/v1/admin/ingest         → Upload .md files → Background Task
  GET  /api/v1/admin/tasks          → Liệt kê tất cả tasks
  GET  /api/v1/admin/tasks/{id}     → Poll trạng thái 1 task
  DELETE /api/v1/admin/documents    → Soft-delete chunks theo file
"""

import os
from typing import Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form,
    HTTPException, Request, UploadFile, status,
)
from fastapi.responses import JSONResponse

from app.core.config.admin_config import admin_cfg
from app.core.security import (
    admin_rate_limiter,
    create_access_token,
    get_current_admin,
)
from app.services.admin.task_store import task_store
from app.services.admin.ingestion_worker import process_ingestion
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Router ──
router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])

# Allowed extensions
_ALLOWED_EXT = set(admin_cfg.rate_limit.allowed_extensions)
_MAX_FILE_BYTES = admin_cfg.rate_limit.max_file_size_mb * 1024 * 1024


# ══════════════════════════════════════════════════════════
# LOGIN — Lấy JWT Token
# ══════════════════════════════════════════════════════════
@router.post("/login", summary="Đăng nhập Admin → JWT Token")
async def admin_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """
    Xác thực Admin và trả về JWT token.

    Body (form-data):
        - username: Tên đăng nhập
        - password: Mật khẩu
    """
    # Rate limit
    client_ip = request.client.host if request.client else "unknown"
    admin_rate_limiter.check(client_ip)

    # Validate credentials
    cred = admin_cfg.credentials
    if username != cred.username or password != cred.password:
        logger.warning("Admin Login FAILED: username='%s' from %s", username, client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai tên đăng nhập hoặc mật khẩu.",
        )

    token = create_access_token(subject=username)
    logger.info("Admin Login OK: username='%s' from %s", username, client_ip)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_minutes": admin_cfg.jwt.access_token_expire_minutes,
    }


# ══════════════════════════════════════════════════════════
# INGEST — Upload files → Background Processing
# ══════════════════════════════════════════════════════════
@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Nạp file Markdown vào VectorDB",
)
async def ingest_files(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Danh sách file .md"),
    program_level: Optional[str] = Form(None, description="Bậc học: thac_si/tien_si/dai_hoc"),
    program_name: Optional[str] = Form(None, description="Ngành: VD: Marketing"),
    academic_year: Optional[str] = Form(None, description="Năm học: VD: 2025-2026"),
    admin: str = Depends(get_current_admin),
):
    """
    Upload 1 hoặc nhiều file Markdown → Nạp vào VectorDB.

    **Metadata fields (tùy chọn, fallback null nếu không điền):**
    - program_level: Bậc đào tạo (thac_si / tien_si / dai_hoc)
    - program_name: Tên ngành (VD: "Marketing", "QTKD")
    - academic_year: Năm học (VD: "2025-2026")

    Nếu Admin không điền → hệ thống sẽ cố trích xuất từ front-matter file.
    Nếu file cũng không có → giá trị = null (vẫn nạp bình thường).
    """
    # Rate limit
    admin_rate_limiter.check(admin)

    if not files:
        raise HTTPException(status_code=400, detail="Chưa có file nào được upload.")

    # ── Validate & Fallback null cho metadata ──
    # Empty string → None (front-end có thể gửi "" nếu user không điền)
    _VALID_LEVELS = {"thac_si", "tien_si", "dai_hoc"}
    clean_level = program_level.strip() if program_level else None
    if clean_level and clean_level not in _VALID_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"program_level không hợp lệ: '{clean_level}'. "
                   f"Chỉ chấp nhận: {', '.join(_VALID_LEVELS)} hoặc để trống.",
        )
    clean_program = program_name.strip() if program_name else None
    clean_year = academic_year.strip() if academic_year else None

    # Log metadata admin gửi lên
    logger.info(
        "Ingest - Metadata override: level=%s, program=%s, year=%s (by %s)",
        clean_level, clean_program, clean_year, admin,
    )

    tasks_created = []

    for upload_file in files:
        filename = upload_file.filename or "unknown.md"

        # ── Validate extension ──
        ext = os.path.splitext(filename)[1].lower()
        if ext not in _ALLOWED_EXT:
            tasks_created.append({
                "file_name": filename,
                "status": "rejected",
                "reason": f"Chỉ chấp nhận file {', '.join(_ALLOWED_EXT)}",
            })
            continue

        # ── Validate size ──
        content_bytes = await upload_file.read()
        if len(content_bytes) > _MAX_FILE_BYTES:
            tasks_created.append({
                "file_name": filename,
                "status": "rejected",
                "reason": f"File quá lớn ({len(content_bytes) / 1024 / 1024:.1f}MB > {admin_cfg.rate_limit.max_file_size_mb}MB)",
            })
            continue

        # ── Decode content ──
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            tasks_created.append({
                "file_name": filename,
                "status": "rejected",
                "reason": "File không phải UTF-8 encoding",
            })
            continue

        # ── Create task + schedule background worker ──
        task = task_store.create(file_name=filename)

        background_tasks.add_task(
            process_ingestion,
            file_name=filename,
            file_content=content,
            task=task,
            override_level=clean_level,
            override_program=clean_program,
            override_year=clean_year,
        )

        tasks_created.append({
            "file_name": filename,
            "task_id": task.task_id,
            "status": "accepted",
        })
        logger.info("Ingest - Queued: '%s' → task_id=%s (by %s)", filename, task.task_id, admin)

    return {
        "total_files": len(files),
        "accepted": sum(1 for t in tasks_created if t.get("status") == "accepted"),
        "rejected": sum(1 for t in tasks_created if t.get("status") == "rejected"),
        "tasks": tasks_created,
    }


# ══════════════════════════════════════════════════════════
# TASK STATUS — Poll trạng thái
# ══════════════════════════════════════════════════════════
@router.get("/tasks", summary="Liệt kê tất cả tasks")
async def list_tasks(admin: str = Depends(get_current_admin)):
    """Trả về danh sách tất cả ingestion tasks (mới nhất trước)."""
    return {"tasks": task_store.list_all()}


@router.get("/tasks/{task_id}", summary="Xem trạng thái 1 task")
async def get_task_status(
    task_id: str,
    admin: str = Depends(get_current_admin),
):
    """Poll trạng thái chi tiết của 1 ingestion task."""
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' không tồn tại.",
        )
    return task.to_dict()


# ══════════════════════════════════════════════════════════
# DELETE — Soft-delete document
# ══════════════════════════════════════════════════════════
@router.delete("/documents", summary="Soft-delete chunks theo tên file")
async def delete_document(
    file_name: str,
    admin: str = Depends(get_current_admin),
):
    """
    Soft-delete tất cả chunks của 1 file (is_active = FALSE).

    Query params:
        - file_name: Tên file cần xóa (VD: "phuluc1.md")
    """
    from app.services.admin.dedup_service import DedupService
    from app.services.admin.ingestion_worker import _get_db_connection

    conn = None
    try:
        conn = _get_db_connection()
        conn.autocommit = False
        dedup = DedupService(conn)
        count = dedup.soft_delete_old_chunks(file_name)
        dedup.remove_old_log(file_name)

        logger.info("Admin DELETE: '%s' → %d chunks soft-deleted (by %s)", file_name, count, admin)

        return {
            "file_name": file_name,
            "chunks_deleted": count,
            "message": f"Đã soft-delete {count} chunks. Có thể rollback bằng UPDATE is_active = TRUE.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa: {str(e)}",
        )
    finally:
        if conn:
            conn.close()
