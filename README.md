# Fine-tune-Deepseek-OCR-with-Vietnamese-Dataset

Đồ án môn học: **Nhập môn Xử lý ngôn ngữ tự nhiên**  
**Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM**

**Sinh viên thực hiện:** Phạm Quang Thịnh - [23127485]  
**Giảng viên hướng dẫn:** TS. Nguyễn Hồng Bửu Long

---

## Giới thiệu
Dự án này thực hiện tinh chỉnh (Fine-tuning) mô hình Vision Language Model (sử dụng thư viện **Unsloth**) trên bộ dữ liệu chữ viết tay tiếng Việt. Mục tiêu là cải thiện khả năng nhận diện tiếng Việt (dấu thanh, chữ viết tháu) và chuẩn hóa định dạng đầu ra.

## Kết quả (Results)
Sau 150 bước huấn luyện với kỹ thuật **QLoRA**, mô hình đạt được sự cải thiện vượt bậc trên tập Test độc lập (400 mẫu):

| Metric | Baseline (Gốc) | Fine-tuned (Sau khi train) | Cải thiện |
|:---|:---:|:---:|:---:|
| **CER** (Lỗi ký tự) | 31.92% | **14.05%** | ⬇️ 17.87% |
| **WER** (Lỗi từ) | 66.69% | **33.31%** | ⬇️ 33.39% |

### So sánh trực quan
Dưới đây là kết quả thực tế trên các mẫu chữ viết tay khó:
![Visual Comparison](images/comparison_ocr.png)
*(Mô hình Fine-tuned cải thiện độ chính xác dấu thanh và sửa lỗi từ vô nghĩa khá tốt so với Baseline)*

---

## Dataset

Dự án sử dụng bộ dữ liệu **UIT-HWDB-line** bao gồm ảnh chữ viết tay tiếng Việt.

- **Nguồn dữ liệu gốc:** https://github.com/nghiangh/UIT-HWDB-dataset
- **Dữ liệu đã sử dụng trong đồ án:** https://drive.google.com/file/d/1KRla0siXCDxv9nRs-9XYdEqRbViKH0jE/view?usp=drive_link

## Model Checkpoint
Do giới hạn dung lượng GitHub, trọng số mô hình (LoRA Adapters) được lưu trữ tại Google Drive.  
**[TẢI MODEL TẠI ĐÂY](https://drive.google.com/drive/folders/1ESQruMMXlkr5KTK7rKzQa5gOJVEMYkVY?usp=drive_link)**

---

## 🚀 Hướng dẫn chạy (Usage)

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt

from unsloth import FastVisionModel

# Load model & tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    "đường/dẫn/đến/folder/checkpoint",
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

# Chạy thử
image_path = "test_image.jpg"
instruction = "<image>\nFree OCR."
# ... (Code inference chi tiết xem trong notebook)
```