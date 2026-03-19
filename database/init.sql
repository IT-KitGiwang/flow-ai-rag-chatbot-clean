-- database/init.sql
-- Script này sẽ tự động chạy MỘT LẦN DUY NHẤT khi container Postgres khởi tạo lần đầu tiên.

-- 1. Bật Extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==============================================================================
-- 2. TẠO BẢNG INTENT EMBEDDINGS (Dành cho Layer 3.1: Vector Router)
-- Lưu trữ các câu hỏi mẫu để so khớp nhanh Intent
-- ==============================================================================
CREATE TABLE IF NOT EXISTS intent_examples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    intent_name VARCHAR(100) NOT NULL,            -- Ví dụ: 'HOC_PHI_HOC_BONG'
    example_text TEXT NOT NULL,                   -- Câu hỏi mẫu: "Học phí trường mình bao nhiêu"
    embedding VECTOR(1024),                       -- Vector 1024 chiều (BAAI/bge-m3)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tạo Index HNSW để search cực nhanh bằng Cosine Similarity
CREATE INDEX ON intent_examples USING hnsw (embedding vector_cosine_ops);


-- ==============================================================================
-- 3. TẠO BẢNG KNOWLEDGE CHUNKS (Dành cho Layer 4: RAG Pipeline)
-- Lưu trữ kiến thức tuyển sinh (PDF, Word, Website) đã được băm nhỏ (chunking)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_title VARCHAR(255),                  -- Tên tài liệu gốc (Ví dụ: 'Đề án tuyển sinh 2025.pdf')
    chunk_content TEXT NOT NULL,                  -- Nội dung văn bản của chunk
    metadata JSONB DEFAULT '{}'::jsonb,           -- Chứa metadata: số trang, URL, chuyên mục...
    embedding VECTOR(1024),                       -- Vector 1024 chiều (BAAI/bge-m3)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tạo Index HNSW cho RAG Search
CREATE INDEX ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);


-- ==============================================================================
-- 4. INSERT DỮ LIỆU MẪU (Mock Data) CHO INTENT ROUTER
-- ==============================================================================
-- Lưu ý: Bạn cần dùng script Python để sinh embedding cho các câu này và update lại, 
-- ở đây chỉ tạo sẵn text mẫu.
INSERT INTO intent_examples (intent_name, example_text) VALUES 
('THONG_TIN_TUYEN_SINH', 'Cho em hỏi điểm chuẩn ngành Marketing năm 2024 là bao nhiêu ạ?'),
('THONG_TIN_TUYEN_SINH', 'Trường tuyển sinh khối nào? Có xét học bạ không?'),
('HOC_PHI_HOC_BONG', 'Học phí 1 kỳ của trường Tài chính Marketing là bao nhiêu?'),
('HOC_PHI_HOC_BONG', 'Trường có chính sách giảm học phí hay học bổng cho sinh viên nghèo không?'),
('DOI_SONG_SINH_VIEN', 'Ký túc xá của trường nằm ở đâu? Có điều hòa không ạ?'),
('THU_TUC_HANH_CHINH', 'Thủ tục làm hồ sơ nhập học cần mang theo những giấy tờ gì?');
