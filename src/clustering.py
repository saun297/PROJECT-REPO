#K-Means
import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans

def train_and_save_kmeans(
    data_path="../data/processed/rfm_features_scaled.csv",
    model_save_path="../models/kmeans_rfm_model.pkl",
    output_csv_path="../data/processed/rfm_kmeans_clustered.csv",
    n_clusters=4,
    random_state=42
):
    """
    Train mô hình K-Means cho dữ liệu RFM đã scale, gán nhãn cụm vào DataFrame,
    lưu mô hình (.pkl) và xuất dữ liệu phân cụm ra file CSV.
    """
    # 1. Đọc dữ liệu
    df = pd.read_csv(data_path)
    feature_cols = ['Recency_log_scaled', 'Frequency_log_scaled', 'Monetary_log_scaled']
    X = df[feature_cols].values

    # 2. Train mô hình K-Means và gán nhãn cụm trực tiếp vào DataFrame
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)

    # 3. Tạo thư mục và lưu mô hình (.pkl)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(kmeans, model_save_path)

    # 4. Lưu DataFrame đã có nhãn Cluster ra file CSV để dùng phân tích sau này
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"✅ Train K-Means (K={n_clusters}) thành công!")
    print(f"💾 Đã lưu mô hình tại: {model_save_path}")
    print(f"📊 Đã lưu DataFrame chứa nhãn cụm tại: {output_csv_path}")

    return kmeans, df

# ----------------------------------------------------
# Cách gọi hàm khi bạn cần lấy nhãn để phân tích:
# ----------------------------------------------------
# model, df_clustered = train_and_save_kmeans()

#K-Medoids
import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def train_and_save_kmedoids(
    data_path="../data/processed/rfm_features_scaled.csv",
    model_save_path="../models/kmedoids_rfm_model.pkl",
    k=4,
    max_iter=20,
    random_state=42,
    batch_size=5000,
):
    """Train mô hình K-Medoids tối ưu bộ nhớ (Batched PAM) cho dữ liệu RFM lớn

    và lưu thông tin mô hình ra file .pkl.

    Parameters:
    ----------
    data_path : str
        Đường dẫn tới file CSV chứa dữ liệu RFM đã scale.
    model_save_path : str
        Đường dẫn xuất file mô hình (.pkl).
    k : int
        Số lượng cụm.
    max_iter : int
        Số vòng lặp tối đa.
    random_state : int
        Hạt giống ngẫu nhiên.
    batch_size : int
        Kích thước batch khi tính khoảng cách để tránh tràn bộ nhớ.

    Returns:
    -------
    model_dict : dict
        Dictionary chứa các thông tin chính của mô hình (medoids,
        medoid_indices, k).
    df : DataFrame
        Dữ liệu RFM đã bổ sung cột 'Cluster_KMedoids'.
    """
    # 1. Đọc dữ liệu
    df = pd.read_csv(data_path)
    feature_cols = [
        'Recency_log_scaled',
        'Frequency_log_scaled',
        'Monetary_log_scaled',
    ]
    X = df[feature_cols].values
    n_samples = len(X)

    print(f"Kích thước tập dữ liệu đầu vào: {n_samples} dòng, {X.shape[1]} cột")
    print(f"Đang train K-Medoids (K={k}) trên toàn bộ dữ liệu...")

    # 2. Khởi tạo K-Medoids
    np.random.seed(random_state)
    medoid_indices = np.random.choice(n_samples, size=k, replace=False)
    medoids = X[medoid_indices]

    t0 = time.time()
    for it in range(max_iter):
        # Tính khoảng cách từ toàn bộ điểm tới K tâm Medoids
        distances = cdist(X, medoids, metric='euclidean')
        labels = np.argmin(distances, axis=1)

        new_medoid_indices = np.copy(medoid_indices)

        # Cập nhật Medoids mới cho từng cụm
        for i in range(k):
            cluster_indices = np.where(labels == i)[0]
            cluster_size = len(cluster_indices)

            if cluster_size > 0:
                cluster_points = X[cluster_indices]
                sum_dists = np.zeros(cluster_size)

                # Chia nhỏ cụm thành các batch để tính tổng khoảng cách
                for start in range(0, cluster_size, batch_size):
                    end = min(start + batch_size, cluster_size)
                    batch_pts = cluster_points[start:end]
                    d_batch = cdist(
                        batch_pts, cluster_points, metric='euclidean'
                    )
                    sum_dists[start:end] = d_batch.sum(axis=1)

                best_idx_in_cluster = np.argmin(sum_dists)
                new_medoid_indices[i] = cluster_indices[best_idx_in_cluster]

        # Kiểm tra điều kiện hội tụ
        if np.array_equal(medoid_indices, new_medoid_indices):
            print(f"✅ Thuật toán đã hội tụ hoàn toàn tại vòng lặp thứ {it+1}!")
            break

        medoid_indices = new_medoid_indices
        medoids = X[medoid_indices]
        print(
            f"   Hoàn thành vòng lặp {it+1} - Thời gian đã chạy:"
            f" {time.time() - t0:.2f}s"
        )

    # Gán nhãn cụm cuối cùng
    distances = cdist(X, medoids, metric='euclidean')
    labels = np.argmin(distances, axis=1)
    t1 = time.time()

    # Gán kết quả vào DataFrame
    df['Cluster_KMedoids'] = labels

    # 3. Đóng gói mô hình thành Dictionary
    model_dict = {
        'medoids': medoids,
        'medoid_indices': medoid_indices,
        'k': k,
        'feature_cols': feature_cols,
    }

    # 4. Tạo thư mục đích nếu chưa tồn tại và lưu file .pkl
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model_dict, model_save_path)

    print(f"⚡ Train hoàn tất trong: {t1 - t0:.2f} giây")
    print(f"💾 Đã lưu mô hình K-Medoids thành công tại: {model_save_path}")

    return model_dict, df


# ----------------------------------------------------
# Cách gọi hàm:
# ----------------------------------------------------
# model_kmedoids, df_clustered = train_and_save_kmedoids()
import os
import joblib
import numpy as np
import pandas as pd


def train_and_save_fcm(
    data_path="../data/processed/rfm_features_scaled.csv",
    model_save_path="../models/fcm_rfm_model.pkl",
    output_csv_path="../data/processed/rfm_fcm_clustered.csv",
    c=4,
    m=2.0,
    max_iter=100,
    error=1e-5,
    random_state=42,
):
    """Train thuật toán Fuzzy C-Means (FCM) cho dữ liệu RFM, gán thông tin độ thuộc vào DataFrame,

    xuất file CSV và lưu file mô hình .pkl.

    Parameters:
    ----------
    data_path : str
        Đường dẫn tới file CSV chứa dữ liệu RFM đã scale.
    model_save_path : str
        Đường dẫn xuất file mô hình (.pkl).
    output_csv_path : str
        Đường dẫn xuất file CSV chứa kết quả phân cụm và xác suất độ thuộc.
    c : int
        Số lượng cụm mờ.
    m : float
        Hệ số mờ (fuzziness coefficient).
    max_iter : int
        Số vòng lặp tối đa.
    error : int/float
        N ngưỡng hội tụ.
    random_state : int
        Hạt giống ngẫu nhiên.

    Returns:
    -------
    model_dict : dict
        Dictionary chứa thông tin mô hình (centroids, u_matrix, c, m,
        feature_cols).
    df : DataFrame
        Dữ liệu RFM đã bổ sung nhãn cụm và xác suất độ thuộc.
    """
    # 1. Đọc dữ liệu
    df = pd.read_csv(data_path)
    feature_cols = [
        'Recency_log_scaled',
        'Frequency_log_scaled',
        'Monetary_log_scaled',
    ]
    X = df[feature_cols].values
    n_samples, n_features = X.shape

    print(
        f"Kích thước tập dữ liệu đầu vào: {n_samples} dòng, {n_features} cột"
    )
    print(f"Đang train mô hình Fuzzy C-Means (C={c}, m={m})...")

    # 2. Xây dựng và chạy thuật toán FCM
    np.random.seed(random_state)
    U = np.random.rand(c, n_samples)
    U = U / np.sum(U, axis=0, keepdims=True)

    for iteration in range(max_iter):
        U_old = U.copy()

        # A. Tính toán tâm cụm mờ (centroids)
        Um = U**m
        centroids = (Um @ X) / np.sum(Um, axis=1, keepdims=True)

        # B. Tính khoảng cách Euclidean
        dists = np.zeros((c, n_samples))
        for i in range(c):
            dists[i, :] = np.linalg.norm(X - centroids[i], axis=1)
        dists = np.fmax(dists, 1e-10)  # Tránh lỗi chia 0

        # C. Cập nhật ma trận độ thuộc U
        power = 2.0 / (m - 1.0)
        inv_dists = (1.0 / dists) ** power
        U = inv_dists / np.sum(inv_dists, axis=0, keepdims=True)

        # Kiểm tra điều kiện hội tụ
        if np.linalg.norm(U - U_old) < error:
            print(
                f"✅ Thuật toán FCM đã hội tụ hoàn toàn tại vòng lặp thứ"
                f" {iteration + 1}!"
            )
            break

    labels = np.argmax(U, axis=0)

    # 3. Gán nhãn cụm và độ thuộc vào DataFrame
    df['Cluster_FCM'] = labels
    df['Max_Membership_Score'] = np.max(U, axis=0)

    for i in range(c):
        df[f'Prob_Cluster_{i}'] = U[i, :]

    # 4. Đóng gói thông tin mô hình FCM
    model_dict = {
        'centroids': centroids,
        'u_matrix': U,
        'c': c,
        'm': m,
        'feature_cols': feature_cols,
    }

    # 5. Lưu file .pkl và xuất CSV
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model_dict, model_save_path)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"💾 Đã lưu mô hình FCM tại: {model_save_path}")
    print(f"📊 Đã lưu DataFrame kết quả phân cụm mờ tại: {output_csv_path}")

    return model_dict, df


# ----------------------------------------------------
# Cách gọi hàm:
# ----------------------------------------------------
# model_fcm, df_fcm = train_and_save_fcm()