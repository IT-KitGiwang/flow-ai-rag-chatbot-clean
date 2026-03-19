"""
Semantic Chunker sử dụng BAAI/bge-m3 Embedding.

Thuật toán:
  1. Tiền xử lý văn bản → Tách thành Base Blocks (~100 tokens/block)
  2. Gọi API BGE-M3 (qua OpenRouter) để sinh Embedding cho mỗi block
     ⚠️ BATCH theo giới hạn 8192 tokens/request để tránh bị reject
  3. Tính Cosine Similarity giữa các block liền kề
  4. Similarity ≥ 0.6 (60%) → GỘP blocks thành 1 chunk
     Similarity < 0.6 (60%) → CẮT chunk tại đây (ranh giới ngữ nghĩa)
  5. Áp dụng Overlap 100 tokens giữa các chunk liền kề
  6. Gắn Metadata đầy đủ (chunk_id, content_hash, token_count...)

Thiết kế cho:
  - Model: BAAI/bge-m3 (1024 dims, multilingual, hỗ trợ tiếng Việt)
  - Provider: OpenRouter
  - Token limit: 8192 tokens/request
  - Overlap: 100 tokens (văn cảnh liên tục, không bị đứt gãy)
"""

import json
import math
import re
import time
import unicodedata
import urllib.request
import urllib.error
from typing import List, Optional, Tuple

import numpy as np

from models.chunk import ChunkMetadata, ProcessedChunk


# ================================================================
# CẤU HÌNH MẶC ĐỊNH
# ================================================================
DEFAULT_CONFIG = {
    "provider": "openrouter",
    "model": "baai/bge-m3",
    "dimensions": 1024,
    "similarity_threshold": 0.6,        # 60% — ngưỡng cắt chunk
    "overlap_tokens": 100,              # 100 tokens overlap giữa 2 chunk liền kề
    "base_block_tokens": 100,           # Kích thước mỗi Base Block (~100 tokens)
    "min_chunk_tokens": 50,             # Chunk nhỏ nhất (dưới ngưỡng này → gộp)
    "max_chunk_tokens": 1500,           # Chunk lớn nhất (~6000 chars)
    "max_tokens_per_api_call": 7500,    # Giới hạn an toàn cho 1 lần gọi API (< 8192)
    "api_batch_size": 50,               # Số blocks tối đa gửi 1 batch
    "api_timeout": 30,                  # Timeout API (giây)
}


# ================================================================
# HÀM TIỆN ÍCH
# ================================================================
def _normalize_vietnamese(text: str) -> str:
    """Chuẩn hóa Unicode NFC cho tiếng Việt."""
    return unicodedata.normalize("NFC", text)


def _clean_whitespace(text: str) -> str:
    """Xử lý khoảng trắng thừa."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """
    Ước tính số token cho tiếng Việt.
    BPE tokenizer xử lý tiếng Việt kém hiệu quả hơn tiếng Anh:
      ~3.5–4 chars/token (so với ~4–5 chars/token tiếng Anh)
    Dùng hệ số 3.5 để an toàn (không vượt limit).
    """
    return max(1, int(len(text) / 3.5))


def _split_sentences_vietnamese(text: str) -> List[str]:
    """
    Tách câu tiếng Việt, xử lý dấu câu và xuống dòng.
    Giữ nguyên logic từ preprocessor.py nhưng không phụ thuộc import.
    """
    sentences = re.split(r'(?<=[.!?;])\s+(?=[A-ZÀ-Ỹa-zà-ỹ\d])', text)
    result = []
    for sent in sentences:
        parts = sent.split("\n\n")
        result.extend(p.strip() for p in parts if p.strip())
    return result


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Tính Cosine Similarity giữa 2 vector."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ================================================================
# LỚP CHÍNH: SemanticChunkerBGE
# ================================================================
class SemanticChunkerBGE:
    """
    Semantic Chunker sử dụng BAAI/bge-m3 Embedding qua OpenRouter.

    Luồng xử lý:
      Text → Base Blocks → Embedding → Cosine Similarity → Merge/Split → Overlap → Chunks

    Ví dụ sử dụng:
        chunker = SemanticChunkerBGE(api_key="sk-or-v1-...")
        chunks = chunker.chunk(text="...", source="tuyensinh2025.docx")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        config: Optional[dict] = None,
    ):
        """
        Args:
            api_key: OpenRouter API key
            base_url: OpenRouter base URL
            config: Dict ghi đè cấu hình mặc định (xem DEFAULT_CONFIG)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

        # Merge config
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}

        # Thống kê runtime
        self.stats = {
            "total_api_calls": 0,
            "total_tokens_sent": 0,
            "total_time_embedding": 0.0,
        }

    # ================================================================
    # BƯỚC 1: TÁCH TEXT THÀNH BASE BLOCKS
    # ================================================================
    def _split_into_base_blocks(self, text: str) -> List[str]:
        """
        Tách văn bản thành các Base Blocks nhỏ (~100 tokens/block).

        Ưu tiên cắt theo ranh giới câu để giữ nguyên ý nghĩa.
        Nếu 1 câu quá dài → cắt theo ký tự.
        """
        sentences = _split_sentences_vietnamese(text)
        if not sentences:
            return [text] if text.strip() else []

        target_chars = int(self.cfg["base_block_tokens"] * 3.5)  # ~350 chars/block
        blocks = []
        current_block = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)

            # Câu quá dài → cắt nhỏ theo ký tự
            if sent_len > target_chars * 2:
                # Flush block hiện tại trước
                if current_block:
                    blocks.append(" ".join(current_block))
                    current_block = []
                    current_len = 0

                # Cắt câu dài thành nhiều phần
                for i in range(0, sent_len, target_chars):
                    part = sent[i:i + target_chars].strip()
                    if part:
                        blocks.append(part)
                continue

            # Block hiện tại đã đủ lớn → flush
            if current_len + sent_len > target_chars and current_block:
                blocks.append(" ".join(current_block))
                current_block = []
                current_len = 0

            current_block.append(sent)
            current_len += sent_len

        # Block cuối cùng
        if current_block:
            blocks.append(" ".join(current_block))

        return blocks

    # ================================================================
    # BƯỚC 2: GỌI API BGE-M3 ĐỂ SINH EMBEDDING
    # ================================================================
    def _call_embedding_api(self, texts: List[str]) -> List[np.ndarray]:
        """
        Gọi API OpenRouter Embedding cho BAAI/bge-m3.
        Tự động chia batch nếu tổng token vượt 8192.

        Returns:
            List các vector 1024 chiều (np.ndarray)
        """
        all_embeddings = []
        current_batch = []
        current_batch_tokens = 0
        max_tokens = self.cfg["max_tokens_per_api_call"]
        max_batch = self.cfg["api_batch_size"]

        for text in texts:
            text_tokens = _estimate_tokens(text)

            # Nếu thêm text này sẽ vượt limit → gửi batch hiện tại trước
            if (current_batch_tokens + text_tokens > max_tokens
                    or len(current_batch) >= max_batch) and current_batch:
                embeddings = self._send_embedding_batch(current_batch)
                all_embeddings.extend(embeddings)
                current_batch = []
                current_batch_tokens = 0

            # Text đơn lẻ vượt limit → cắt bớt (cực hiếm với base_block ~100 tokens)
            if text_tokens > max_tokens:
                safe_len = int(max_tokens * 3.5)
                text = text[:safe_len]
                text_tokens = _estimate_tokens(text)

            current_batch.append(text)
            current_batch_tokens += text_tokens

        # Gửi batch cuối cùng
        if current_batch:
            embeddings = self._send_embedding_batch(current_batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _send_embedding_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Gửi 1 batch texts tới API Embedding và trả về list vectors."""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "UFM-Admission-Bot/1.0",
        }
        data = {
            "model": self.cfg["model"],
            "input": texts,
            "dimensions": self.cfg["dimensions"],
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.cfg["api_timeout"]) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")[:500]
            raise RuntimeError(
                f"Embedding API Error ({e.code}): {error_body}"
            ) from e

        elapsed = time.time() - start_time

        # Cập nhật stats
        self.stats["total_api_calls"] += 1
        self.stats["total_tokens_sent"] += sum(_estimate_tokens(t) for t in texts)
        self.stats["total_time_embedding"] += elapsed

        # Parse embeddings (API trả về sorted by index)
        raw_data = sorted(result["data"], key=lambda x: x["index"])
        embeddings = [np.array(item["embedding"], dtype=np.float32) for item in raw_data]

        return embeddings

    # ================================================================
    # BƯỚC 3: TÌM RANH GIỚI CẮT CHUNK (Similarity Drop)
    # ================================================================
    def _find_chunk_boundaries(
        self, embeddings: List[np.ndarray]
    ) -> List[Tuple[int, float]]:
        """
        Tìm các điểm cắt chunk dựa trên sự sụt giảm Cosine Similarity.

        Returns:
            List of (block_index, similarity_score) tại mỗi ranh giới.
            Block_index = vị trí bắt đầu chunk MỚI.
        """
        if len(embeddings) <= 1:
            return [(0, 1.0)]

        boundaries = [(0, 1.0)]  # Chunk đầu tiên luôn bắt đầu ở index 0
        threshold = self.cfg["similarity_threshold"]

        for i in range(1, len(embeddings)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < threshold:
                boundaries.append((i, sim))

        return boundaries

    # ================================================================
    # BƯỚC 4: GỘP BLOCKS THÀNH CHUNKS + OVERLAP
    # ================================================================
    def _merge_blocks_to_chunks(
        self,
        blocks: List[str],
        boundaries: List[Tuple[int, float]],
    ) -> List[dict]:
        """
        Gộp các Base Blocks thành Chunks dựa trên ranh giới đã tìm.
        Áp dụng thuật toán Ovelap Block nguyên vẹn để không bao giờ cắt gãy câu.
        """
        if not blocks:
            return []

        # Cấu hình gối đầu bằng số lượng Block 
        # Ví dụ: overlap 100 tokens, 1 block = 100 tokens -> cần overlap 1 block liên tục
        overlap_block_count = max(1, int(self.cfg["overlap_tokens"] / max(1, self.cfg["base_block_tokens"])))
        
        min_chunk_tokens = self.cfg["min_chunk_tokens"]
        max_chunk_tokens = self.cfg["max_chunk_tokens"]
        max_chunk_chars = int(max_chunk_tokens * 3.5)

        boundary_starts = [b[0] for b in boundaries]
        if 0 not in boundary_starts:
            boundary_starts.insert(0, 0)

        chunks = []
        
        for idx, start in enumerate(boundary_starts):
            end = boundary_starts[idx + 1] if idx + 1 < len(boundary_starts) else len(blocks)
            
            # LẤY NỘI DUNG CHÍNH CỦA CHUNK HIỆN TẠI
            current_chunk_blocks = blocks[start:end]
            
            # TẠO OVERLAP TỪ CHUNK TRƯỚC (NẾU CÓ)
            overlap_content = ""
            actual_overlap_tokens = 0
            
            if idx > 0:
                # Lấy các block cuối của chunk TRƯỚC ĐÓ để làm overlap
                prev_start = boundary_starts[idx-1]
                prev_end = start
                # Lấy tối đa 'overlap_block_count' blocks dư cuối của đoạn trước đưa vào
                overlap_source_blocks = blocks[max(prev_start, prev_end - overlap_block_count):prev_end]
                if overlap_source_blocks:
                    overlap_content = " ".join(overlap_source_blocks)
                    actual_overlap_tokens = _estimate_tokens(overlap_content)

            # Hợp nhất Overlap + Current Content
            content = (overlap_content + " " + " ".join(current_chunk_blocks)).strip()

            # Giới hạn tổng dung lượng (phòng hờ mốc chót an toàn)
            if len(content) > max_chunk_chars:
                content = content[:max_chunk_chars]

            # Kiểm tra gộp chunk nếu kích thước hiện tại quá nhỏ nhắn
            if (_estimate_tokens(content) < min_chunk_tokens 
                    and chunks 
                    and _estimate_tokens(chunks[-1]["content"]) + _estimate_tokens(content) <= max_chunk_tokens):
                chunks[-1]["content"] += " " + content
                chunks[-1]["block_range"] = (chunks[-1]["block_range"][0], end)
                continue

            chunks.append({
                "content": content,
                "overlap_tokens": actual_overlap_tokens,
                "block_range": (start, end),
            })

        return chunks

    # ================================================================
    # BƯỚC 5: LUỒNG TỔNG — chunk()
    # ================================================================
    def chunk(
        self,
        text: str,
        source: str,
        metadata_extra: Optional[dict] = None,
    ) -> List[ProcessedChunk]:
        """
        Luồng xử lý hoàn chỉnh Semantic Chunking.

        Args:
            text: Văn bản thô cần chunking
            source: Tên file nguồn (VD: "tuyensinh2025.docx")
            metadata_extra: Dict metadata bổ sung (program_name, academic_year...)

        Returns:
            List[ProcessedChunk] - Các chunk đã xử lý, sẵn sàng Embedding & Insert DB
        """
        # 1. Tiền xử lý
        text = _normalize_vietnamese(text)
        text = _clean_whitespace(text)

        if not text.strip():
            return []

        # 2. Tách thành Base Blocks (~100 tokens/block)
        blocks = self._split_into_base_blocks(text)

        if not blocks:
            return []

        # 3. Sinh Embedding cho các blocks
        # Nếu chỉ có 1 block → không cần tính similarity
        if len(blocks) == 1:
            extra = metadata_extra or {}
            meta = ChunkMetadata(source=source, chunk_index=1, total_chunks_in_section=1, **extra)
            return [ProcessedChunk(content=blocks[0], metadata=meta)]

        embeddings = self._call_embedding_api(blocks)

        # 4. Tìm ranh giới cắt chunk (Cosine Similarity < 60%)
        boundaries = self._find_chunk_boundaries(embeddings)

        # 5. Gộp blocks thành chunks + overlap 100 tokens
        raw_chunks = self._merge_blocks_to_chunks(blocks, boundaries)

        # 6. Tạo ProcessedChunk objects với metadata đầy đủ
        total = len(raw_chunks)
        processed_chunks = []
        extra = metadata_extra or {}

        for idx, raw in enumerate(raw_chunks, 1):
            meta = ChunkMetadata(
                source=source,
                chunk_index=idx,
                total_chunks_in_section=total,
                overlap_tokens=raw["overlap_tokens"],
                **extra,
            )
            processed_chunks.append(
                ProcessedChunk(content=raw["content"], metadata=meta)
            )

        return processed_chunks

    # ================================================================
    # FALLBACK: Chunk không cần Embedding (Khi API lỗi)
    # ================================================================
    def chunk_fallback(
        self,
        text: str,
        source: str,
        metadata_extra: Optional[dict] = None,
    ) -> List[ProcessedChunk]:
        """
        Fallback chunking (không cần API Embedding).
        Cắt theo kích thước cố định + overlap câu.
        Dùng khi API BGE-M3 bị lỗi hoặc không có kết nối mạng.
        """
        text = _normalize_vietnamese(text)
        text = _clean_whitespace(text)

        if not text.strip():
            return []

        sentences = _split_sentences_vietnamese(text)
        target_chars = int(self.cfg["max_chunk_tokens"] * 3.5)
        overlap_sents = 2  # Overlap 2 câu cuối

        chunks = []
        current = []
        current_len = 0
        extra = metadata_extra or {}

        for sent in sentences:
            if current_len + len(sent) > target_chars and current:
                content = " ".join(current)
                meta = ChunkMetadata(source=source, **extra)
                chunks.append(ProcessedChunk(content=content, metadata=meta))

                # Overlap: giữ lại 2 câu cuối
                current = current[-overlap_sents:]
                current_len = sum(len(s) for s in current)

            current.append(sent)
            current_len += len(sent)

        # Chunk cuối cùng
        if current:
            content = " ".join(current)
            meta = ChunkMetadata(source=source, **extra)
            chunks.append(ProcessedChunk(content=content, metadata=meta))

        # Cập nhật chunk_index
        for i, c in enumerate(chunks, 1):
            c.metadata.chunk_index = i
            c.metadata.total_chunks_in_section = len(chunks)

        return chunks

    # ================================================================
    # THỐNG KÊ
    # ================================================================
    def get_stats(self) -> dict:
        """Trả về thống kê runtime (số API calls, tokens, thời gian)."""
        return {**self.stats}

    def reset_stats(self):
        """Reset thống kê."""
        self.stats = {
            "total_api_calls": 0,
            "total_tokens_sent": 0,
            "total_time_embedding": 0.0,
        }
