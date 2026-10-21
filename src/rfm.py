from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class RFMProcessor:
    """Class tính toán và tiền xử lý đặc trưng RFM cho khách hàng."""

    def __init__(self, df_clean: pd.DataFrame):
        self.df_clean = df_clean
        self.rfm_df = None
        self.df_final = None

    def compute_rfm(self) -> pd.DataFrame:
        """Tổng hợp dữ liệu thành 1 dòng / 1 customer_unique_id (tính R, F, M thô)."""
        snapshot_date = self.df_clean[
            "order_purchase_timestamp"
        ].max() + pd.Timedelta(days=1)

        orders = (
            self.df_clean.groupby(
                ["customer_unique_id", "order_id"], as_index=False
            )
            .agg(
                payment_value=("payment_value", "sum"),
                order_date=("order_purchase_timestamp", "first"),
            )
        )

        g = orders.groupby("customer_unique_id")
        self.rfm_df = pd.DataFrame(
            {
                "Recency": (snapshot_date - g["order_date"].max()).dt.days,
                "Frequency": g["order_id"].nunique(),
                "Monetary": g["payment_value"].sum().round(2),
            }
        ).reset_index()

        print(f" Đã tính RFM thô cho {len(self.rfm_df):,} khách hàng duy nhất.")
        return self.rfm_df

    def handle_rfm_outliers_capping(
        self,
        df: pd.DataFrame,
        columns=["Recency", "Frequency", "Monetary"],
        lower_quantile=0.01,
        upper_quantile=0.99,
    ) -> pd.DataFrame:
        """Xử lý ngoại lai bằng phương pháp Capping (Winsorization)."""
        df_capped = df.copy()

        for col in columns:
            lower_bound = df_capped[col].quantile(lower_quantile)
            upper_bound = df_capped[col].quantile(upper_quantile)
            df_capped[col] = np.clip(df_capped[col], lower_bound, upper_bound)

        print(f"[OK] Đã xử lý ngoại lai bằng Capping ({lower_quantile*100}% - {upper_quantile*100}%).")
        return df_capped

    def preprocess_features(
        self, feature_cols=["Recency", "Frequency", "Monetary"], upper_quantile=0.99
    ) -> pd.DataFrame:
        """Thực hiện chuỗi tiền xử lý: Capping Outliers -> Log Transform -> Scaling."""
        if self.rfm_df is None:
            self.compute_rfm()

        print("\n" + "=" * 70)
        print("GIAI ĐOẠN TIỀN XỬ LÝ ĐẶC TRƯNG (FEATURE PREPROCESSING)")
        print("=" * 70)

        # Bước 1: Xử lý ngoại lai bằng Capping
        df_processed = self.handle_rfm_outliers_capping(
            self.rfm_df, columns=feature_cols, upper_quantile=upper_quantile
        )

        print("\nĐộ lệch (Skewness) trước khi Log Transformation (đã cap):")
        print(df_processed[feature_cols].skew())

        # Bước 2: Log Transformation
        for col in feature_cols:
            df_processed[f"{col}_log"] = np.log1p(df_processed[col])

        # Bước 3: Chuẩn hóa đặc trưng (StandardScaler)
        scaled_cols = [f"{col}_log" for col in feature_cols]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_processed[scaled_cols])

        df_scaled = pd.DataFrame(
            scaled_data, columns=[f"{col}_scaled" for col in scaled_cols]
        )
        self.df_final = pd.concat(
            [
                df_processed[["customer_unique_id"]].reset_index(drop=True),
                df_scaled,
            ],
            axis=1,
        )

        print("\n[OK] Đã hoàn tất xử lý Capping -> Log Transform -> StandardScaler.")
        return self.df_final

    def save_processed_data(
        self,
        output_dir: str = "data/processed",
        output_filename: str = "rfm_features_scaled.csv",
    ) -> None:
        """Hàm lưu bảng RFM đã xử lý vào thư mục data/processed/."""
        if self.df_final is None:
            raise ValueError(
                "Vui lòng chạy hàm `.preprocess_features()` trước khi lưu dữ liệu!"
            )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        out_processed_path = out_path / output_filename
        self.df_final.to_csv(out_processed_path, index=False, encoding="utf-8-sig")
        print(f"Đã lưu bộ dữ liệu RFM scaled tại: {out_processed_path.resolve()}")


# if __name__ == "__main__":
#     # Ví dụ khi chạy trực tiếp file rfm.py từ Terminal
#     cleaned_data_path = Path("data/processed/olist_cleaned_dataset.csv")

#     if cleaned_data_path.exists():
#         print(f"Đang đọc dữ liệu đã làm sạch từ: {cleaned_data_path.resolve()}")
#         df_clean = pd.read_csv(cleaned_data_path)

#         # Chuyển kiểu timestamp về dạng datetime
#         df_clean["order_purchase_timestamp"] = pd.to_datetime(
#             df_clean["order_purchase_timestamp"]
#         )

#         # Thực thi RFM Pipeline
#         rfm_processor = RFMProcessor(df_clean)
#         rfm_processor.compute_rfm()
#         rfm_processor.preprocess_features()
#         rfm_processor.save_processed_data()
#     else:
#         print(
#             f" Không tìm thấy file {cleaned_data_path}. Vui lòng chạy preprocessing.py trước."
#         )