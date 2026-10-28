from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class RFMProcessor:
    """Lớp tính toán và xử lý tiền dữ liệu RFM:

    Tính RFM thô -> Capping -> Log Transform -> Scaling -> Trực quan hóa & Lưu.
    """

    def __init__(
        self,
        df_clean: pd.DataFrame,
        features: list = None,
        monetary_quantile: float = 0.99,
        output_dir: str = "reports/figures",
    ):
        """Khởi tạo RFMProcessor với DataFrame sạch và các cấu hình."""
        self.df_clean = df_clean.copy()
        self.df_clean["order_purchase_timestamp"] = pd.to_datetime(self.df_clean["order_purchase_timestamp"])
        self.features = features if features is not None else ["Recency", "Frequency", "Monetary"]
        self.monetary_quantile = monetary_quantile
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lưu trữ các dữ liệu sau từng bước
        self.rfm_df = None       # RFM thô
        self.monetary_upper = None
        self.df_capped = None   # Sau Capping
        self.df_logged = None   # Sau Log Transform
        self.df_final = None    # Sau Scaling

        # Scaler
        self.scaler = StandardScaler()

    def _save_show(self, fig: plt.Figure, filename: str) -> None:
        """Hàm trợ lý để lưu và hiển thị biểu đồ."""
        save_path = self.output_dir / filename
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ tại: {save_path.resolve()}")
        plt.show()
        plt.close(fig)

    def compute_rfm(self, output_dir: str = None) -> pd.DataFrame:
        """Tổng hợp dữ liệu thành 1 dòng / 1 customer_unique_id (tính R, F, M thô) và lưu file."""
        snapshot_date = self.df_clean["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

        orders = (
            self.df_clean.groupby(["customer_unique_id", "order_id"], as_index=False)
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

        print(f"Đã tính RFM thô cho {len(self.rfm_df):,} khách hàng duy nhất.")

        # Tự động tạo thư mục và xuất file rfm_raw.csv nếu có truyền output_dir
        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True) # Tạo thư mục data/processed nếu chưa có
            save_file_path = out_path / "rfm_raw.csv"
            self.rfm_df.to_csv(save_file_path, index=False, encoding="utf-8-sig")
            print(f"Đã lưu dữ liệu RFM thô tại: {save_file_path.resolve()}")

        return self.rfm_df

    def cap_monetary(self, plot: bool = True) -> pd.DataFrame:
        """Bước 1: Chặn trần (Capping) giá trị Monetary theo phân vị (Quantile)."""
        if self.rfm_df is None:
            self.compute_rfm()

        self.df_capped = self.rfm_df.copy()
        self.monetary_upper = self.df_capped["Monetary"].quantile(self.monetary_quantile)
        self.df_capped["Monetary"] = self.df_capped["Monetary"].clip(upper=self.monetary_upper)

        changed = self.rfm_df["Monetary"] > self.monetary_upper
        print(f"Ngưỡng P{int(self.monetary_quantile * 100)} của Monetary: {self.monetary_upper:,.2f}")
        print(f"Số khách hàng bị chặn trần: {changed.sum():,}")
        print(f"Tỷ lệ bị chặn trần: {changed.mean():.2%}")

        if plot:
            self.plot_capping()

        return self.df_capped

    def plot_capping(self) -> None:
        """Vẽ và lưu biểu đồ so sánh trước và sau Capping."""
        if self.df_capped is None:
            raise ValueError("Vui lòng thực hiện `.cap_monetary()` trước khi vẽ biểu đồ!")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

        plots = [
            (axes[0], self.rfm_df["Monetary"], "Trước capping", "skyblue"),
            (axes[1], self.df_capped["Monetary"], f"Sau capping P{int(self.monetary_quantile * 100)}", "sandybrown"),
        ]

        for ax, values, title, color in plots:
            sns.boxplot(y=values, color=color, ax=ax)
            ax.set_title(title)
            ax.set_ylabel("Monetary")
            ax.axhline(
                self.monetary_upper,
                color="red",
                linestyle="--",
                linewidth=1,
                label=f"P99 = {self.monetary_upper:,.2f}",
            )
            ax.legend()

        fig.suptitle("Monetary trước và sau capping", fontsize=15, fontweight="bold")
        fig.tight_layout()

        self._save_show(fig, "01_monetary_boxplot_capping.png")

    def log_transform(self, plot: bool = True) -> pd.DataFrame:
        """Bước 2: Biến đổi Log cho các đặc trưng Frequency (log1p) và Monetary (log)."""
        if self.df_capped is None:
            self.cap_monetary(plot=False)

        self.df_logged = self.df_capped.copy()
        self.df_logged["Frequency"] = np.log1p(self.df_capped["Frequency"])
        self.df_logged["Monetary"] = np.log(self.df_capped["Monetary"])

        print("Đã biến đổi Frequency bằng log1p(F).")
        print("Đã biến đổi Monetary bằng log(M).")

        if plot:
            self.plot_log_transform()

        return self.df_logged

    def plot_log_transform(self) -> None:
        """Vẽ và lưu biểu đồ so sánh trước và sau Log Transformation."""
        if self.df_logged is None:
            raise ValueError("Vui lòng thực hiện `.log_transform()` trước khi vẽ biểu đồ!")

        log_features = ["Frequency", "Monetary"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        for j, col in enumerate(log_features):
            rows_config = [
                (0, self.df_capped, "Trước log", "skyblue"),
                (1, self.df_logged, "Sau log", "sandybrown"),
            ]

            for row, table, label, color in rows_config:
                values = table[col]
                ax = axes[row, j]

                sns.histplot(
                    x=values,
                    discrete=(True if (col == "Frequency" and row == 0) else False),
                    bins=40,
                    color=color,
                    ax=ax,
                )

                ax.set_title(f"{col} — {label}\nĐộ lệch = {values.skew():.3f}")
                ax.set_ylabel("Số khách hàng")

                if row == 0:
                    ax.set_xlabel(col)
                elif col == "Frequency":
                    ax.set_xlabel("log1p(Frequency)")
                else:
                    ax.set_xlabel("log(Monetary)")

            ymax = max(axes[0, j].get_ylim()[1], axes[1, j].get_ylim()[1])
            axes[0, j].set_ylim(0, ymax)
            axes[1, j].set_ylim(0, ymax)

        fig.suptitle("Frequency và Monetary trước và sau log transformation", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        self._save_show(fig, "02_frequency_monetary_histogram_log.png")

    def scale_features(self, plot: bool = True) -> pd.DataFrame:
        """Bước 3: Chuẩn hóa dữ liệu bằng StandardScaler."""
        if self.df_logged is None:
            self.log_transform(plot=False)

        scaled_array = self.scaler.fit_transform(self.df_logged[self.features])

        # Giữ lại customer_unique_id trong DataFrame kết quả
        self.df_final = pd.DataFrame(
            scaled_array,
            columns=self.features,
            index=self.df_logged.index,
        )
        if "customer_unique_id" in self.df_logged.columns:
            self.df_final.insert(0, "customer_unique_id", self.df_logged["customer_unique_id"])

        print(f"Đã chuẩn hóa toàn bộ {len(self.df_final):,} khách hàng với {len(self.features)} đặc trưng.")

        if plot:
            self.plot_scaling()

        return self.df_final

    def plot_scaling(self) -> None:
        """Vẽ và lưu biểu đồ so sánh trước và sau StandardScaler."""
        if self.df_final is None:
            raise ValueError("Vui lòng thực hiện `.scale_features()` trước khi vẽ biểu đồ!")

        fig, axes = plt.subplots(2, 3, figsize=(18, 9))

        for j, col in enumerate(self.features):
            rows_config = [
                (0, self.df_logged, "Trước chuẩn hóa", "skyblue"),
                (1, self.df_final, "Sau chuẩn hóa", "sandybrown"),
            ]

            for row, table, label, color in rows_config:
                ax = axes[row, j]

                sns.histplot(x=table[col], bins=40, color=color, ax=ax)
                ax.set_title(f"{col} — {label}")
                ax.set_ylabel("Số khách hàng")

                if row == 1:
                    ax.set_xlabel(f"{col} đã chuẩn hóa")
                elif col == "Frequency":
                    ax.set_xlabel("log1p(Frequency)")
                elif col == "Monetary":
                    ax.set_xlabel("log(Monetary)")
                else:
                    ax.set_xlabel("Recency")

            ymax = max(axes[0, j].get_ylim()[1], axes[1, j].get_ylim()[1])
            axes[0, j].set_ylim(0, ymax)
            axes[1, j].set_ylim(0, ymax)

        fig.suptitle("R, F, M trước và sau StandardScaler", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        self._save_show(fig, "03_rfm_histogram_standardscaler.png")

    def preprocess_features(self, plot: bool = True) -> pd.DataFrame:
        """Hàm pipeline chạy toàn bộ từ tính RFM thô -> Capping -> Log -> Scaling."""
        self.compute_rfm()
        self.cap_monetary(plot=plot)
        self.log_transform(plot=plot)
        self.scale_features(plot=plot)
        return self.df_final

    def save_processed_data(
        self,
        output_dir: str = "data/processed",
        output_filename: str = "rfm_features_scaled.csv",
    ) -> None:
        """Hàm lưu bảng RFM đã scaled vào thư mục được chỉ định."""
        if self.df_final is None:
            raise ValueError(
                "Vui lòng chạy `.preprocess_features()` hoặc `.scale_features()` trước khi lưu dữ liệu!"
            )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        out_processed_path = out_path / output_filename
        self.df_final.to_csv(out_processed_path, index=False, encoding="utf-8-sig")
        print(f"Đã lưu bộ dữ liệu RFM scaled tại: {out_processed_path.resolve()}")