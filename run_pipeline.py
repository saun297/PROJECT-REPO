"""
OLIST Customer Segmentation - Pipeline Chinh

Chay lenh sau tu thu muc goc du an:
    python run_pipeline.py

Hoac chi chay tung buoc:
    python run_pipeline.py --step join
    python run_pipeline.py --step preprocess
    python run_pipeline.py --step rfm
    python run_pipeline.py --step cluster
    python run_pipeline.py --step visualize

Pipeline gom 5 buoc:
    Buoc 1 - Join       : Ghep 7 bang raw CSV thanh 1 bang tong hop
    Buoc 2 - Preprocess : Lam sach du lieu (loc, xoa null, khu trung lap)
    Buoc 3 - RFM        : Tinh chi so RFM + log transform + StandardScaler
    Buoc 4 - Cluster    : Train 3 mo hinh (K-Means, K-Medoids, Fuzzy C-Means)
    Buoc 5 - Visualize  : Ve bieu do va xuat bao cao bang CSV
"""

import argparse
import sys
import time
from pathlib import Path

# Dam bao thu muc goc nam trong Python path
ROOT_DIR = Path(_file_).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Import cac module trong src/
from src.olist_joined_dataset import build_joined_dataset
from src.preprocessing import DataPreprocessor
from src.rfm import RFMProcessor
from src.clustering import train_and_save_kmeans, train_and_save_kmedoids, train_and_save_fcm
from src.visualization import merge_and_report_kmeans_fast, ClusterVisualizer
import pandas as pd

# Duong dan tap trung
RAW_DIR              = ROOT_DIR / "data" / "raw_data"
JOINED_DIR           = ROOT_DIR / "data" / "joined_data"
PROCESSED_DIR        = ROOT_DIR / "data" / "processed"
MODELS_DIR           = ROOT_DIR / "models"
REPORTS_DIR          = ROOT_DIR / "reports"

JOINED_CSV           = JOINED_DIR  / "olist_joined_dataset.csv"
CLEANED_CSV          = PROCESSED_DIR / "olist_cleaned_dataset.csv"
RFM_SCALED_CSV       = PROCESSED_DIR / "rfm_features_scaled.csv"
KMEANS_CLUSTERED_CSV = PROCESSED_DIR / "rfm_kmeans_clustered.csv"

KMEANS_MODEL_PATH    = MODELS_DIR / "kmeans_rfm_model.pkl"
KMEDOIDS_MODEL_PATH  = MODELS_DIR / "kmedoids_rfm_model.pkl"
FCM_MODEL_PATH       = MODELS_DIR / "fcm_rfm_model.pkl"


# ===================================================================
# Buoc 1 - JOIN RAW DATA
# ===================================================================
def step_join():
    print("\n" + "=" * 60)
    print("  BUOC 1 - GHEP DU LIEU THO (JOIN RAW DATA)")
    print("=" * 60)
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Khong tim thay thu muc raw data: {RAW_DIR}\n"
            "-> Hay dat 7 file CSV vao thu muc data/raw_data/"
        )
    df = build_joined_dataset(
        raw_dir=str(RAW_DIR),
        output_dir=str(JOINED_DIR),
        output_filename="olist_joined_dataset.csv",
    )
    print(f"[OK] Buoc 1 hoan tat - {len(df):,} dong da duoc ghep.")
    return df


# ===================================================================
# Buoc 2 - PREPROCESS
# ===================================================================
def step_preprocess():
    print("\n" + "=" * 60)
    print("  BUOC 2 - LAM SACH DU LIEU (PREPROCESS)")
    print("=" * 60)
    if not JOINED_CSV.exists():
        print("[!] Chua co file joined. Tu dong chay Buoc 1...")
        step_join()
    df = pd.read_csv(JOINED_CSV)
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )
    preprocessor = DataPreprocessor(df)
    df_clean = preprocessor.clean_orders()
    preprocessor.save_processed_data(
        output_dir=str(PROCESSED_DIR),
        output_filename="olist_cleaned_dataset.csv",
    )
    print(f"[OK] Buoc 2 hoan tat - {len(df_clean):,} dong sach.")
    return df_clean


# ===================================================================
# Buoc 3 - TINH RFM
# ===================================================================
def step_rfm():
    print("\n" + "=" * 60)
    print("  BUOC 3 - TINH CHI SO RFM & TIEN XU LY DAC TRUNG")
    print("=" * 60)
    if not CLEANED_CSV.exists():
        print("[!] Chua co file cleaned. Tu dong chay Buoc 2...")
        step_preprocess()
    df_clean = pd.read_csv(CLEANED_CSV)
    df_clean["order_purchase_timestamp"] = pd.to_datetime(
        df_clean["order_purchase_timestamp"], errors="coerce"
    )
    rfm_processor = RFMProcessor(df_clean)
    rfm_processor.compute_rfm()
    rfm_processor.preprocess_features()
    rfm_processor.save_processed_data(
        output_dir=str(PROCESSED_DIR),
        output_filename="rfm_features_scaled.csv",
    )
    print("[OK] Buoc 3 hoan tat - File RFM scaled da luu.")
    return rfm_processor.df_final


# ===================================================================
# Buoc 4 - TRAIN CAC MO HINH PHAN CUM
# ===================================================================
def step_cluster():
    print("\n" + "=" * 60)
    print("  BUOC 4 - TRAIN MO HINH PHAN CUM (K-MEANS / K-MEDOIDS / FCM)")
    print("=" * 60)
    if not RFM_SCALED_CSV.exists():
        print("[!] Chua co file RFM. Tu dong chay Buoc 3...")
        step_rfm()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data_path = str(RFM_SCALED_CSV)

    # 4a. K-Means
    print("\n[4a] Dang train K-Means...")
    t0 = time.time()
    _, df_kmeans = train_and_save_kmeans(
        data_path=data_path,
        model_save_path=str(KMEANS_MODEL_PATH),
        output_csv_path=str(KMEANS_CLUSTERED_CSV),
        n_clusters=4,
    )
    print(f"     Thoi gian: {time.time() - t0:.2f}s")

    # 4b. K-Medoids
    print("\n[4b] Dang train K-Medoids (co the mat vai phut)...")
    t0 = time.time()
    _, df_kmedoids = train_and_save_kmedoids(
        data_path=data_path,
        model_save_path=str(KMEDOIDS_MODEL_PATH),
        k=4,
    )
    print(f"     Thoi gian: {time.time() - t0:.2f}s")

    # 4c. Fuzzy C-Means
    print("\n[4c] Dang train Fuzzy C-Means...")
    t0 = time.time()
    _, df_fcm = train_and_save_fcm(
        data_path=data_path,
        model_save_path=str(FCM_MODEL_PATH),
        output_csv_path=str(PROCESSED_DIR / "rfm_fcm_clustered.csv"),
        c=4,
    )
    print(f"     Thoi gian: {time.time() - t0:.2f}s")

    print("\n[OK] Buoc 4 hoan tat - Da train va luu 3 mo hinh vao models/")
    return df_kmeans, df_kmedoids, df_fcm


# ===================================================================
# Buoc 5 - VISUALIZATION
# ===================================================================
def step_visualize():
    print("\n" + "=" * 60)
    print("  BUOC 5 - TRUC QUAN HOA KET QUA PHAN CUM")
    print("=" * 60)
    if not KMEANS_CLUSTERED_CSV.exists() or not CLEANED_CSV.exists():
        print("[!] Chua co file ket qua. Tu dong chay Buoc 4...")
        step_cluster()

    rfm_summary, business_summary, df_merged = merge_and_report_kmeans_fast(
        clustered_path=str(KMEANS_CLUSTERED_CSV),
        olist_cleaned_path=str(CLEANED_CSV),
        output_merged_csv=str(PROCESSED_DIR / "rfm_kmeans_merged_detail.csv"),
        tables_report_dir=str(REPORTS_DIR / "tables"),
    )
    print("\nBang RFM Summary theo Cum:")
    print(rfm_summary.to_string())
    print("\nBang Business Summary theo Cum:")
    print(business_summary.to_string())

    viz = ClusterVisualizer(df_merged, cluster_col="Cluster")
    print("\n[5b] Ve bieu do PCA, Snake Plot, Heatmap, Boxplot...")
    viz.plot_rfm_overview()
    print("\n[5c] Ve Radar Chart va phan bo dia ly...")
    viz.plot_business_insights()
    print("\n[5d] Ve Top 3 danh muc san pham theo cum...")
    viz.plot_top_categories(top_n=3)
    print("\n[OK] Buoc 5 hoan tat - Bao cao luu tai reports/tables/")


# ===================================================================
# ENTRY POINT
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="OLIST Customer Segmentation Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--step",
        choices=["join", "preprocess", "rfm", "cluster", "visualize", "all"],
        default="all",
        help=(
            "Chon buoc muon chay:\n"
            "  join       - Ghep 7 bang raw CSV\n"
            "  preprocess - Lam sach du lieu\n"
            "  rfm        - Tinh RFM + chuan hoa\n"
            "  cluster    - Train K-Means / K-Medoids / FCM\n"
            "  visualize  - Ve bieu do va xuat bao cao\n"
            "  all        - Chay toan bo pipeline (mac dinh)\n"
        ),
    )
    args = parser.parse_args()
    start_total = time.time()

    print("\n" + "=" * 60)
    print("  OLIST CUSTOMER SEGMENTATION - PIPELINE BAT DAU")
    print("=" * 60)

    STEP_MAP = {
        "join":       step_join,
        "preprocess": step_preprocess,
        "rfm":        step_rfm,
        "cluster":    step_cluster,
        "visualize":  step_visualize,
    }

    if args.step == "all":
        for fn in STEP_MAP.values():
            fn()
    else:
        STEP_MAP[args.step]()

    elapsed = time.time() - start_total
    print("\n" + "=" * 60)
    print(f"  PIPELINE HOAN TAT - Tong thoi gian: {elapsed:.1f}s")
    print("=" * 60 + "\n")


if _name_ == "_main_":
    main()