import os
from pathlib import Path
import pandas as pd


class DataPreprocessor:
    """Lớp xử lý và làm sạch dữ liệu đơn hàng."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cleaned_df = None
        self.audit_df = None

    def clean_orders(self) -> pd.DataFrame:
        """Làm sạch dữ liệu đơn hàng và ghi nhật ký Row Audit."""

        # 1. Kiểm tra cột bắt buộc
        required_cols = [
            "order_id",
            "customer_unique_id",
            "order_status",
            "order_purchase_timestamp",
            "payment_value",
        ]
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise KeyError(f"Thiếu các cột bắt buộc: {missing_cols}")

        audit = []

        def log(step, before, after, note=""):
            audit.append(
                {
                    "step": step,
                    "rows_before": before,
                    "rows_after": after,
                    "rows_removed": before - after,
                    "note": note,
                }
            )

        data = self.df.copy()
        log("0. Dữ liệu thô", len(data), len(data), "Điểm xuất phát")

        # Bước 1: Ép kiểu dữ liệu
        n = len(data)
        data["order_purchase_timestamp"] = pd.to_datetime(
            data["order_purchase_timestamp"], errors="coerce"
        )
        data["payment_value"] = pd.to_numeric(data["payment_value"], errors="coerce")
        log("1. Ép kiểu dữ liệu", n, len(data), "Đã chuyển timestamp & payment_value")

        # Bước 2: Chỉ giữ đơn delivered
        n = len(data)
        data = data[data["order_status"] == "delivered"]
        log(
            "2. Lọc order_status == 'delivered'",
            n,
            len(data),
            "Chỉ giữ đơn giao thành công",
        )

        # Bước 3: Xóa dòng thiếu thông tin cốt lõi
        n = len(data)
        core_cols = [
            "order_id",
            "customer_unique_id",
            "order_purchase_timestamp",
            "payment_value",
        ]
        data = data.dropna(subset=core_cols)
        log("3. Xóa missing value cốt lõi", n, len(data), f"Cột kiểm tra: {core_cols}")

        # Bước 4: Lọc payment_value > 0
        n = len(data)
        data = data[data["payment_value"] > 0]
        log("4. Lọc payment_value > 0", n, len(data), "Loại bỏ giao dịch <= 0")

        # Bước 5: Điền "Unknown" cho cột danh mục sản phẩm nếu dính missing
        category_cols = [
            c
            for c in [
                "product_category_name",
                "product_category_name_english",
                "category",
            ]
            if c in data.columns
        ]
        for col in category_cols:
            data[col] = data[col].fillna("Unknown")

        # Bước 6: Khử trùng lặp do phép join
        n = len(data)
        if "payment_sequential" in data.columns:
            data = data.drop_duplicates(subset=["order_id", "payment_sequential"])
            note = "Khử trùng lặp theo (order_id, payment_sequential)"
        else:
            data = data.drop_duplicates(subset=["order_id"])
            note = "Khử trùng lặp theo order_id"
        log("5. Khử trùng lặp dòng", n, len(data), note)

        # Lưu lại kết quả vào thuộc tính của Class
        self.audit_df = pd.DataFrame(audit)
        self.cleaned_df = data.reset_index(drop=True)

        # In nhật ký số dòng
        print("=" * 70)
        print("ROW AUDIT – NHẬT KÝ SỐ DÒNG LÀM SẠCH")
        print("=" * 70)
        print(self.audit_df.to_string(index=False))
        print("=" * 70)

        return self.cleaned_df

    def save_processed_data(
        self,
        output_dir: str = "data/processed",
        output_filename: str = "olist_cleaned_dataset.csv",
    ) -> None:
        """Lưu dữ liệu đã làm sạch vào thư mục chỉ định."""
        if self.cleaned_df is None:
            raise ValueError(
                "Vui lòng chạy hàm `.clean_orders()` trước khi lưu dữ liệu!"
            )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Lưu file dữ liệu làm sạch
        cleaned_file_path = out_path / output_filename
        self.cleaned_df.to_csv(
            cleaned_file_path, index=False, encoding="utf-8-sig"
        )
        print(
            f"✅ Đã lưu file dữ liệu làm sạch tại: {cleaned_file_path.resolve()}"
        )

# if __name__ == "__main__":
#     # Chạy trực tiếp từ Terminal (đứng tại thư mục gốc PROJECT REPO)
#     joined_data_path = Path("data/joined_data/olist_joined_dataset.csv")

#     if joined_data_path.exists():
#         print(f"Đang đọc dữ liệu từ: {joined_data_path.resolve()}")
#         df_raw = pd.read_csv(joined_data_path)

#         # Khởi tạo class trực tiếp (vì đã ở trong cùng file)
#         processor = DataPreprocessor(df_raw)
#         processor.clean_orders()
#         processor.save_processed_data()
#     else:
#         print(
#             f"❌ Không tìm thấy file {joined_data_path}. Vui lòng kiểm tra lại đường dẫn."
#         )