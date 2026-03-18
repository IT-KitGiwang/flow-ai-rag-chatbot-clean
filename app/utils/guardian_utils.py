# app/utils/guardian_utils.py
# Chứa logic xử lý thực tế cho các lớp bảo vệ (0, 1, 2)

import re
from typing import Tuple, Dict
from app.core.config import query_flow_config

class GuardianService:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Chuẩn hóa văn bản: Chuyển chữ thường, thay thế Teencode/Leetspeak."""
        # Chuyển về chữ thường
        text = text.lower()
        
        # Mapping Teencode / Leetspeak
        teencode_map = query_flow_config.keyword_filter.teencode_map
        # Sắp xếp các key dài hơn trước để tránh thay thế nhầm (VD: 'ko' trước 'k')
        sorted_keys = sorted(teencode_map.keys(), key=len, reverse=True)
        
        for key in sorted_keys:
            # Chỉ thay thế nếu nó đứng độc lập (word boundary) để tránh sai sót
            text = re.sub(rf'\b{re.escape(key)}\b', teencode_map[key], text)
            
        return text

    @staticmethod
    def check_layer_0_input_validation(text: str) -> Tuple[bool, str]:
        """LỚP 0: Kiểm tra độ dài ký tự."""
        config = query_flow_config.input_validation
        if len(text) > config.max_input_chars:
            return False, config.fallback_too_long
        return True, ""

    @staticmethod
    def check_layer_1_keyword_filter(text: str) -> Tuple[bool, str]:
        """LỚP 1a: Kiểm tra từ khóa cấm (nội dung nhạy cảm) bằng Regex."""
        config = query_flow_config.keyword_filter
        normalized_text = GuardianService.normalize_text(text)
        
        for pattern in config.banned_regex_patterns:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                return False, config.fallback_message
        return True, ""

    @staticmethod
    def check_layer_1b_injection_filter(text: str) -> Tuple[bool, str]:
        """LỚP 1b: Regex chống Injection/Jailbreak sơ đẳng (0ms)."""
        config = query_flow_config.keyword_filter
        normalized_text = GuardianService.normalize_text(text)
        
        for pattern in config.injection_regex_patterns:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                return False, config.fallback_injection
        return True, ""

    @staticmethod
    def _call_groq_api(provider: str, model: str, messages: list, temperature: float = 0.0, max_tokens: int = 10) -> str:
        """Hàm gọi API Groq dùng chung cho cả 2a và 2b."""
        import json
        import urllib.request
        import urllib.error
        
        api_key = query_flow_config.api_keys.get_key(provider)
        base_url = query_flow_config.api_keys.get_base_url(provider)
        
        if not api_key:
            raise ValueError(f"Chưa cấu hình API Key cho provider '{provider}'")
        
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "UFM-Admission-Bot/1.0"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()

    @staticmethod
    def check_layer_2a_prompt_guard_fast(text: str) -> Tuple[bool, str]:
        """LỚP 2a: Llama 86M quét nhanh bằng điểm số."""
        import urllib.error
        
        config = query_flow_config.prompt_guard_fast
        
        try:
            output = GuardianService._call_groq_api(
                provider=config.provider,
                model=config.model,
                messages=[{"role": "user", "content": text}],
                max_tokens=config.max_tokens_per_chunk
            )
            print(f"   [Debug 2a] Llama Guard Score: '{output}'")
            
            try:
                score = float(output)
            except ValueError:
                if "unsafe" in output.lower():
                    return False, config.fallback_unsafe
                return True, ""
            
            # Score >= 0.9 → Tấn công rõ ràng → CHẶN ngay
            if score >= config.score_threshold:
                return False, f"{config.fallback_unsafe} (Score: {score:.2%})"
            return True, ""
            
        except ValueError as e:
            return True, f"Bỏ qua 2a ({str(e)})"
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            return True, f"Bỏ qua 2a (API Error: {error_body[:200]})"
        except Exception as e:
            return True, f"Bỏ qua 2a (Lỗi: {str(e)})"

    @staticmethod
    def check_layer_2b_prompt_guard_deep(text: str) -> Tuple[bool, str]:
        """LỚP 2b: Qwen 7B quét sâu bằng tiếng Việt (SAFE/UNSAFE)."""
        import urllib.error
        
        config = query_flow_config.prompt_guard_deep
        
        try:
            messages = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": text}
            ]
            output = GuardianService._call_groq_api(
                provider=config.provider,
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=10
            )
            print(f"   [Debug 2b] Qwen Guard trả về: '{output}'")
            
            # Qwen trả về SAFE hoặc UNSAFE
            if "unsafe" in output.lower():
                return False, config.fallback_unsafe
            return True, ""
            
        except ValueError as e:
            return True, f"Bỏ qua 2b ({str(e)})"
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            return True, f"Bỏ qua 2b (API Error: {error_body[:200]})"
        except Exception as e:
            return True, f"Bỏ qua 2b (Lỗi: {str(e)})"

    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, str, int]:
        """Chạy toàn bộ luồng Guardian."""
        # LỚP 0
        is_l0_ok, msg_l0 = cls.check_layer_0_input_validation(query)
        if not is_l0_ok:
            return False, msg_l0, 0

        # LỚP 1a: Từ khóa cấm (nội dung)
        is_l1_ok, msg_l1 = cls.check_layer_1_keyword_filter(query)
        if not is_l1_ok:
            return False, msg_l1, 1

        # LỚP 1b: Regex chống Injection (0ms)
        is_l1b_ok, msg_l1b = cls.check_layer_1b_injection_filter(query)
        if not is_l1b_ok:
            return False, msg_l1b, 1

        # LỚP 2a: Llama 86M quét nhanh (Score)
        is_l2a_ok, msg_l2a = cls.check_layer_2a_prompt_guard_fast(query)
        if not is_l2a_ok:
            return False, msg_l2a, 2

        # LỚP 2b: Qwen 7B quét sâu tiếng Việt (SAFE/UNSAFE)
        is_l2b_ok, msg_l2b = cls.check_layer_2b_prompt_guard_deep(query)
        if not is_l2b_ok:
            return False, msg_l2b, 2

        return True, "SAFE", 2

