import os
from pathlib import Path

# ==========================================
# 1. PATH CONFIGURATION (Đường dẫn dự án)
# ==========================================
# Xác định thư mục gốc của dự án (Directory chứa file run_pipeline.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# Thư mục dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Thư mục lưu Model & Reports
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# ==========================================
# 2. FILE PATHS (Đường dẫn các file cụ thể)
# ==========================================
# File dữ liệu thô (Raw Data)
RAW_CUSTOMERS_PATH = RAW_DATA_DIR / "olist_customers_dataset.csv"
RAW_ORDERS_PATH = RAW_DATA_DIR / "olist_orders_dataset.csv"
RAW_ORDER_PAYMENTS_PATH = RAW_DATA_DIR / "olist_order_payments_dataset.csv"
RAW_ORDER_ITEMS_PATH = RAW_DATA_DIR / "olist_order_items_dataset.csv"
RAW_ORDER_REVIEWS_PATH = RAW_DATA_DIR / "olist_order_reviews_dataset.csv"
RAW_PRODUCTS_PATH = RAW_DATA_DIR / "olist_products_dataset.csv"
RAW_TRANSLATION_PATH = RAW_DATA_DIR / "product_category_name_translation.csv"

# File dữ liệu đã xử lý (Processed Data)
JOINED_DATA_PATH = PROCESSED_DATA_DIR / "olist_joined_dataset.csv"
RFM_FEATURES_PATH = PROCESSED_DATA_DIR / "rfm_features.csv"
FINAL_SEGMENTS_PATH = PROCESSED_DATA_DIR / "final_customer_segments.csv"

# File lưu mô hình (Saved Models & Scalers)
SCALER_PATH = MODELS_DIR / "scaler_rfm.pkl"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_rfm_model.pkl"