import os
from pathlib import Path
import pandas as pd


def build_joined_dataset(
    raw_dir: str = "data/raw_data",
    output_dir: str = "data/joined_data",
    output_filename: str = "olist_joined_dataset.csv",
) -> pd.DataFrame:
    """Đọc các bảng dữ liệu thô từ raw_dir, tiến hành merge/join và gom nhóm,

    sau đó lưu kết quả vào output_dir/output_filename.

    Parameters:
    -----------
    raw_dir : str
        Đường dẫn tới thư mục chứa các file dữ liệu thô (.csv).
    output_dir : str
        Đường dẫn tới thư mục lưu trữ dữ liệu đã joined.
    output_filename : str
        Tên file CSV đầu ra.

    Returns:
    --------
    pd.DataFrame
        Bảng dữ liệu hoàn chỉnh sau khi joined.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    print(f"1. Đang đọc dữ liệu thô từ: {raw_path.resolve()}")

    # 1. Đọc các file dữ liệu cần thiết từ data/raw_data
    customers = pd.read_csv(raw_path / "olist_customers_dataset.csv")
    orders = pd.read_csv(raw_path / "olist_orders_dataset.csv")
    order_items = pd.read_csv(raw_path / "olist_order_items_dataset.csv")
    order_payments = pd.read_csv(raw_path / "olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv(raw_path / "olist_order_reviews_dataset.csv")
    products = pd.read_csv(raw_path / "olist_products_dataset.csv")
    translation = pd.read_csv(
        raw_path / "product_category_name_translation.csv"
    )

    print("2. Đang tiến hành ghép và gom nhóm dữ liệu...")

    # 2. Đổi tên cột ngành hàng tiếng Anh sang 'category'
    products_trans = products.merge(
        translation, on="product_category_name", how="left"
    ).rename(columns={"product_category_name_english": "category"})

    # Ghép ngành hàng vào bảng order_items
    items_with_cat = order_items.merge(
        products_trans[["product_id", "category"]],
        on="product_id",
        how="left",
    )

    # 3. Gom nhóm các bảng có quan hệ 1-Nhiều về cấp độ đơn hàng (order_id)
    # (a) Bảng items: gom tổng phí ship và lấy tên ngành hàng của đơn
    items_agg = (
        items_with_cat.groupby("order_id")
        .agg(
            freight_value=("freight_value", "sum"),
            category=("category", "first"),
        )
        .reset_index()
    )

    # (b) Bảng payments: gom tổng tiền thanh toán và lấy loại giao dịch
    payments_agg = (
        order_payments.groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_type=("payment_type", "first"),
        )
        .reset_index()
    )

    # (c) Bảng reviews: lấy điểm đánh giá của đơn hàng
    reviews_agg = (
        order_reviews.groupby("order_id")
        .agg(review_score=("review_score", "first"))
        .reset_index()
    )

    # 4. Nối tất cả các bảng vào bảng orders ban đầu (giữ nguyên toàn bộ đơn hàng)
    df_merged = orders.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
                "customer_state",
                "customer_city",
            ]
        ],
        on="customer_id",
        how="left",
    )
    df_merged = df_merged.merge(items_agg, on="order_id", how="left")
    df_merged = df_merged.merge(payments_agg, on="order_id", how="left")
    df_merged = df_merged.merge(reviews_agg, on="order_id", how="left")

    # 5. Giữ lại các cột cần thiết
    columns_to_keep = [
        "customer_unique_id",
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "payment_type",
        "payment_value",
        "freight_value",
        "category",
        "customer_state",
        "customer_city",
        "review_score",
    ]

    df_final = df_merged[columns_to_keep]

    # 6. Kiểm tra & Tạo thư mục đầu ra nếu chưa tồn tại
    out_path.mkdir(parents=True, exist_ok=True)

    # 7. Lưu kết quả
    file_save_path = out_path / output_filename
    df_final.to_csv(file_save_path, index=False, encoding="utf-8-sig")
    print(f"Đã nối các bảng thành công! Đã lưu file tại: {file_save_path.resolve()}")

    return df_final
#if __name__ == "__main__":
#   build_joined_dataset()
