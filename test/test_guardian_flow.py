# test/test_guardian_flow.py
# Chạy script này để nhập liệu và kiểm tra Luồng Guardian (0, 1, 2)
import sys
import os
from pathlib import Path

# Thêm đường dẫn gốc (root) vào PYTHONPATH để import được các module trong app
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.guardian_utils import GuardianService
from app.core.config import query_flow_config

def run_test():
    """Vòng lặp tương tác kiểm tra luồng bảo vệ."""
    print("="*60)
    print("   UFM ADMISSION BOT - GUARDIAN PIPELINE TESTER   ")
    print("="*60)
    print("- Nhập 'exit' để thoát.")
    print("- Thử gõ: 'ko chính trị', 'h@ck', hoặc câu chat cực dài.")
    
    while True:
        try:
            user_input = input("\n[USER_PROMPT] >> ")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Đã thoát trình kiểm tra. Chào bạn!")
                break
            
            if not user_input.strip():
                continue
            
            # 1. Chẩn đoán LỚP 0 (Input Validation)
            print("-" * 30)
            print("Đang phân tích LỚP 0 (Độ dài)...")
            is_l0, msg_l0 = GuardianService.check_layer_0_input_validation(user_input)
            if not is_l0:
                print(f"❌ CHẶN TẠI LỚP 0: {msg_l0}")
                continue
            else:
                print(f"✅ LỚP 0 - OK (Độ dài: {len(user_input)} chars)")

            # 2. Chẩn đoán LỚP 1 (Keyword Filter)
            print("Đang phân tích LỚP 1 (Từ khóa cấm & Regex)...")
            # Tự in ra chuỗi đã được chuẩn hóa để admin dễ phân tích
            normalized_in = GuardianService.normalize_text(user_input)
            print(f"   [Normalization: {normalized_in}]")
            
            is_l1, msg_l1 = GuardianService.check_layer_1_keyword_filter(user_input)
            if not is_l1:
                print(f"❌ CHẶN TẠI LỚP 1: {msg_l1}")
                continue
            else:
                print("✅ LỚP 1a - OK (Không phát hiện từ khóa bẩn)")

            # 2b. Chẩn đoán LỚP 1b (Regex chống Injection - 0ms)
            print("Đang phân tích LỚP 1b (Regex chống Injection)...")
            is_l1b, msg_l1b = GuardianService.check_layer_1b_injection_filter(user_input)
            if not is_l1b:
                print(f"❌ CHẶN TẠI LỚP 1b: {msg_l1b}")
                continue
            else:
                print("✅ LỚP 1b - OK (Không phát hiện từ khóa tấn công)")

            # 3a. Chẩn đoán LỚP 2a (Llama 86M - Quét nhanh Score)
            print("Đang quét LỚP 2a (Llama Guard - Score)...")
            is_l2a, msg_l2a = GuardianService.check_layer_2a_prompt_guard_fast(user_input)
            if not is_l2a:
                print(f"❌ CHẶN TẠI LỚP 2a: {msg_l2a}")
                continue
            else:
                if "Bỏ qua" in msg_l2a:
                    print(f"⚠️ LỚP 2a - CẢNH BÁO: {msg_l2a}")
                else:
                    print("✅ LỚP 2a - OK (Score thấp, không phải tấn công rõ ràng)")

            # 3b. Chẩn đoán LỚP 2b (Qwen 7B - Quét sâu tiếng Việt)
            print("Đang quét LỚP 2b (Qwen Guard - Tiếng Việt)...")
            is_l2b, msg_l2b = GuardianService.check_layer_2b_prompt_guard_deep(user_input)
            if not is_l2b:
                print(f"❌ CHẶN TẠI LỚP 2b: {msg_l2b}")
                continue
            else:
                if "Bỏ qua" in msg_l2b:
                    print(f"⚠️ LỚP 2b - CẢNH BÁO: {msg_l2b}")
                else:
                    print("✅ LỚP 2b - OK (Qwen xác nhận SAFE)")

            print("=" * 60)
            print("🎯 KẾT QUẢ CUỐI CÙNG: CÂU HỎI AN TOÀN - Pass tới Lớp 3 (Intent Router)!")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\nĐã thoát.")
            break
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    run_test()
