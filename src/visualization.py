import os
import pandas as pd


import os
import pandas as pd


def merge_and_report_kmeans_fast(
    clustered_path="../data/processed/rfm_kmeans_clustered.csv",
    rfm_raw_path="../data/processed/rfm_raw.csv",
    olist_cleaned_path="../data/processed/olist_cleaned_dataset.csv",
    output_merged_csv="../data/processed/rfm_kmeans_merged_detail.csv",
    tables_report_dir="../reports/tables",
):
    """Gom nhóm Olist theo customer_unique_id (chuẩn khóa chính 1-1),

    merge với nhãn cụm RFM K-Means và thông tin RFM thô (raw),
    lưu báo cáo tự động siêu tốc.
    """
    # 1. Kiểm tra sự tồn tại của các file đầu vào
    for path in [clustered_path, rfm_raw_path, olist_cleaned_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vui lòng kiểm tra lại đường dẫn file: {path}")

    # 2. Đọc dữ liệu RFM clustered và RFM raw (1 dòng / customer_unique_id)
    df_clustered = pd.read_csv(clustered_path)
    df_raw = pd.read_csv(rfm_raw_path)

    # Lấy nhãn Cluster từ file clustered và merge với các chỉ số RFM thô
    raw_cols = ['Recency', 'Frequency', 'Monetary']
    cols_to_merge = ['customer_unique_id'] + [c for c in raw_cols if c in df_raw.columns]
    
    # Kết hợp thông tin Cluster và giá trị RFM thô
    df_rfm = pd.merge(
        df_clustered[['customer_unique_id', 'Cluster']], 
        df_raw[cols_to_merge], 
        on='customer_unique_id', 
        how='left'
    )

    # 3. Bảng thống kê RFM thô theo Cluster
    cluster_rfm_summary = df_rfm.groupby('Cluster')[raw_cols].mean()
    cluster_rfm_summary['So_luong'] = df_rfm['Cluster'].value_counts()
    cluster_rfm_summary['Ty_le_%'] = (df_rfm['Cluster'].value_counts(normalize=True) * 100).round(2)

    # 4. Đọc dữ liệu Olist cleaned
    df_olist = pd.read_csv(olist_cleaned_path)
    cat_col = 'category' if 'category' in df_olist.columns else 'product_category_name'

    # 5. Tối ưu xử lý bảng Olist (Chuẩn 1-1)
    num_cols = [c for c in ['freight_value', 'review_score'] if c in df_olist.columns]
    num_agg = df_olist.groupby('customer_unique_id')[num_cols].mean().reset_index()

    cat_cols_to_keep = ['customer_unique_id', 'customer_state', 'customer_city']
    if cat_col in df_olist.columns:
        cat_cols_to_keep.append(cat_col)
        
    cat_df = df_olist[cat_cols_to_keep].drop_duplicates(subset=['customer_unique_id'])
    df_olist_grouped = pd.merge(num_agg, cat_df, on='customer_unique_id', how='left')

    # 6. Merge RFM (chứa chỉ số thô + Cluster) với Olist cleaned
    df_merged = pd.merge(df_rfm, df_olist_grouped, on='customer_unique_id', how='left')

    # 7. Bảng tổng hợp kinh doanh theo Cluster (dùng RFM thô)
    agg_dict = {col: 'mean' for col in raw_cols + ['freight_value', 'review_score'] if col in df_merged.columns}
    cluster_business_summary = df_merged.groupby('Cluster').agg(agg_dict).reset_index()

    # Lấy Mode cấp Cluster cho thông tin định danh
    for col in ['customer_state', 'customer_city', cat_col]:
        if col in df_merged.columns:
            top_per_cluster = df_merged.groupby('Cluster')[col].agg(
                lambda x: x.value_counts().index[0] if not x.empty else None
            ).reset_index()
            cluster_business_summary = pd.merge(cluster_business_summary, top_per_cluster, on='Cluster')

    cluster_business_summary = cluster_business_summary.round(2)

    # 8. Lưu file ra ổ đĩa
    os.makedirs(os.path.dirname(output_merged_csv), exist_ok=True)
    os.makedirs(tables_report_dir, exist_ok=True)

    df_merged.to_csv(output_merged_csv, index=False)
    cluster_rfm_summary.to_csv(os.path.join(tables_report_dir, "kmeans_rfm_summary.csv"))
    cluster_business_summary.to_csv(os.path.join(tables_report_dir, "kmeans_business_summary.csv"), index=False)

    print(f"⚡ Đã xử lý xong siêu tốc! Tổng: {len(df_merged)} khách hàng (1-1) với chỉ số RFM thô.")
    return cluster_rfm_summary, cluster_business_summary, df_merged


# ----------------------------------------------------
# Cách gọi hàm:
# ----------------------------------------------------
# rfm_summary, business_summary, df_merged = load_kmeans_and_merge_rfm()

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def merge_and_report_kmeans_fast(
    clustered_path="../data/processed/rfm_kmeans_clustered.csv",
    rfm_raw_path="../data/processed/rfm_raw.csv",
    olist_cleaned_path="../data/processed/olist_cleaned_dataset.csv",
    output_merged_csv="../data/processed/rfm_kmeans_merged_detail.csv",
    tables_report_dir="../reports/tables",
):
    """Gom nhóm Olist theo customer_unique_id (chuẩn khóa chính 1-1),
    merge với nhãn cụm RFM K-Means và thông tin RFM thô (raw),
    lưu báo cáo tự động siêu tốc.
    """
    if not os.path.exists(clustered_path) or not os.path.exists(rfm_raw_path) or not os.path.exists(olist_cleaned_path):
        raise FileNotFoundError("Vui lòng kiểm tra lại đường dẫn các file CSV đầu vào!")

    df_clustered = pd.read_csv(clustered_path)
    df_raw = pd.read_csv(rfm_raw_path)

    raw_cols = ['Recency', 'Frequency', 'Monetary']
    cols_to_merge = ['customer_unique_id'] + [c for c in raw_cols if c in df_raw.columns]
    
    df_rfm = pd.merge(
        df_clustered[['customer_unique_id', 'Cluster']], 
        df_raw[cols_to_merge], 
        on='customer_unique_id', 
        how='left'
    )

    cluster_rfm_summary = df_rfm.groupby('Cluster')[raw_cols].mean()
    cluster_rfm_summary['So_luong'] = df_rfm['Cluster'].value_counts()
    cluster_rfm_summary['Ty_le_%'] = (df_rfm['Cluster'].value_counts(normalize=True) * 100).round(2)

    df_olist = pd.read_csv(olist_cleaned_path)
    cat_col = 'category' if 'category' in df_olist.columns else 'product_category_name'

    num_cols = [c for c in ['freight_value', 'review_score'] if c in df_olist.columns]
    num_agg = df_olist.groupby('customer_unique_id')[num_cols].mean().reset_index()

    cat_cols_to_keep = ['customer_unique_id', 'customer_state', 'customer_city']
    if cat_col in df_olist.columns:
        cat_cols_to_keep.append(cat_col)
        
    cat_df = df_olist[cat_cols_to_keep].drop_duplicates(subset=['customer_unique_id'])
    df_olist_grouped = pd.merge(num_agg, cat_df, on='customer_unique_id', how='left')

    df_merged = pd.merge(df_rfm, df_olist_grouped, on='customer_unique_id', how='left')

    agg_dict = {col: 'mean' for col in raw_cols + ['freight_value', 'review_score'] if col in df_merged.columns}
    cluster_business_summary = df_merged.groupby('Cluster').agg(agg_dict).reset_index()

    for col in ['customer_state', 'customer_city', cat_col]:
        if col in df_merged.columns:
            top_per_cluster = df_merged.groupby('Cluster')[col].agg(
                lambda x: x.value_counts().index[0] if not x.empty else None
            ).reset_index()
            cluster_business_summary = pd.merge(cluster_business_summary, top_per_cluster, on='Cluster')

    cluster_business_summary = cluster_business_summary.round(2)

    os.makedirs(os.path.dirname(output_merged_csv), exist_ok=True)
    os.makedirs(tables_report_dir, exist_ok=True)

    df_merged.to_csv(output_merged_csv, index=False)
    cluster_rfm_summary.to_csv(os.path.join(tables_report_dir, "kmeans_rfm_summary.csv"))
    cluster_business_summary.to_csv(os.path.join(tables_report_dir, "kmeans_business_summary.csv"), index=False)

    print(f"⚡ Đã xử lý xong siêu tốc! Tổng: {len(df_merged)} khách hàng (1-1) với chỉ số RFM thô.")
    return cluster_rfm_summary, cluster_business_summary, df_merged


class ClusterVisualizer:
    """Class hỗ trợ trực quan hóa và tự động lưu biểu đồ báo cáo."""

    def __init__(
        self,
        df,
        cluster_col='Cluster',
        feature_cols=None,
        palette='tab10',
        figures_dir='../reports/figures',
    ):
        self.df = df.copy()
        self.cluster_col = cluster_col
        self.feature_cols = feature_cols or ['Recency', 'Frequency', 'Monetary']
        self.palette = palette
        self.num_clusters = self.df[self.cluster_col].nunique()
        self.figures_dir = figures_dir
        
        # Tự động tạo thư mục reports/figures nếu chưa có
        if self.figures_dir:
            os.makedirs(self.figures_dir, exist_ok=True)

    def _save_fig(self, filename):
        """Hàm bổ trợ lưu biểu đồ ra thư mục figures."""
        if self.figures_dir:
            save_path = os.path.join(self.figures_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🖼️ Đã lưu biểu đồ tại: {save_path}")

    def plot_rfm_overview(self, centers=None, filename='rfm_overview.png'):
        """Vẽ và tự động lưu bộ 4 biểu đồ tổng quan RFM."""
        X_raw = self.df[self.feature_cols].values
        X_log = np.log1p(np.maximum(0, X_raw))
        X_scaled = StandardScaler().fit_transform(X_log)

        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(X_scaled)
        self.df['PCA1'] = pca_features[:, 0]
        self.df['PCA2'] = pca_features[:, 1]

        cluster_summary = self.df.groupby(self.cluster_col)[self.feature_cols].mean()

        fig = plt.figure(figsize=(18, 12))

        # 1. PCA 2D
        plt.subplot(2, 2, 1)
        sns.scatterplot(
            data=self.df, x='PCA1', y='PCA2', hue=self.cluster_col,
            palette=self.palette, alpha=0.3, s=15, legend='full'
        )
        if centers is not None:
            centers_pca = pca.transform(centers)
            plt.scatter(
                centers_pca[:, 0], centers_pca[:, 1], s=200, c='black',
                marker='X', edgecolor='white', linewidth=1.5, label='Centroids'
            )
        plt.title(f'1. Biểu đồ PCA - Trực quan hóa 2D ({self.cluster_col})', fontsize=13, fontweight='bold')
        plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        # 2. Snake Plot
        plt.subplot(2, 2, 2)
        scaler = MinMaxScaler()
        df_rfm_scaled = pd.DataFrame(scaler.fit_transform(self.df[self.feature_cols]), columns=self.feature_cols)
        df_rfm_scaled[self.cluster_col] = self.df[self.cluster_col].values
        df_melt = pd.melt(df_rfm_scaled, id_vars=[self.cluster_col], value_vars=self.feature_cols, var_name='Metric', value_name='Value_Scaled')
        sns.lineplot(data=df_melt, x='Metric', y='Value_Scaled', hue=self.cluster_col, palette=self.palette, marker='o', linewidth=2.5)
        plt.title('2. Snake Plot - Đường đặc trưng RFM (Min-Max Scaled)', fontsize=13, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)

        # 3. Heatmap
        plt.subplot(2, 2, 3)
        sns.heatmap(cluster_summary, annot=True, cmap='YlGnBu', fmt='.2f', cbar=True, annot_kws={'size': 11})
        plt.title('3. Heatmap - Giá trị Trung bình RFM Thô', fontsize=13, fontweight='bold')

        # 4. Boxplot
        plt.subplot(2, 2, 4)
        df_melt_raw = pd.melt(self.df, id_vars=[self.cluster_col], value_vars=self.feature_cols, var_name='Metric', value_name='Value_Raw')
        sns.boxplot(data=df_melt_raw, x='Metric', y='Value_Raw', hue=self.cluster_col, palette=self.palette, showfliers=False)
        plt.title('4. Boxplot - Phân bố chi tiết RFM Thô', fontsize=13, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        self._save_fig(filename)
        plt.show()

    def plot_business_insights(
        self,
        freight_col='freight_value',
        review_col='review_score',
        state_col='customer_state',
        filename='business_insights.png',
    ):
        """Vẽ và tự động lưu Radar Chart & Phân bố địa lý."""
        num_cols = [c for c in self.feature_cols + [freight_col, review_col] if c in self.df.columns]

        radar_df = self.df.groupby(self.cluster_col)[num_cols].mean().reset_index()
        scaler = MinMaxScaler()
        radar_scaled = pd.DataFrame(scaler.fit_transform(radar_df[num_cols]), columns=num_cols)
        radar_scaled[self.cluster_col] = radar_df[self.cluster_col]

        categories = num_cols
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig = plt.figure(figsize=(18, 8))

        ax1 = fig.add_subplot(1, 2, 1, polar=True)
        ax1.set_theta_offset(np.pi / 2)
        ax1.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories, size=11, fontweight='bold')
        ax1.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ['0.25', '0.50', '0.75', '1.00'], color='grey', size=8)
        plt.ylim(0, 1.05)

        colors = sns.color_palette(self.palette, self.num_clusters)
        for i, row in radar_scaled.iterrows():
            values = row[categories].values.flatten().tolist()
            values += values[:1]
            cluster_id = int(row[self.cluster_col])
            ax1.plot(angles, values, linewidth=2, label=f'Cụm {cluster_id}', color=colors[i])
            ax1.fill(angles, values, color=colors[i], alpha=0.15)

        plt.title('1. Radar Chart - Chân dung đa chiều', fontsize=13, fontweight='bold', pad=25)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        ax2 = fig.add_subplot(1, 2, 2)
        top_states = self.df[state_col].value_counts().head(5).index
        df_top_states = self.df[self.df[state_col].isin(top_states)]
        state_cluster_dist = (pd.crosstab(df_top_states[self.cluster_col], df_top_states[state_col], normalize='index') * 100)
        state_cluster_dist.plot(kind='bar', stacked=True, ax=ax2, colormap='Set3', edgecolor='black', alpha=0.85)
        plt.title('2. Phân bố Top 5 Bang địa lý (%) theo từng Cụm', fontsize=13, fontweight='bold')
        plt.xlabel('Cụm (Cluster)')
        plt.ylabel('Tỷ lệ (%)')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.legend(title='Customer State', bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        self._save_fig(filename)
        plt.show()

    def plot_top_categories(self, category_col='category', top_n=3, filename='top_categories.png'):
        """Vẽ và tự động lưu biểu đồ Top N danh mục sản phẩm."""
        cat_cluster = self.df.groupby([self.cluster_col, category_col]).size().reset_index(name='Count')
        top_cats = (
            cat_cluster.sort_values('Count', ascending=False)
            .groupby(self.cluster_col)
            .head(top_n)
            .copy()
        )

        cluster_totals = self.df.groupby(self.cluster_col).size().to_dict()
        top_cats['Percentage'] = top_cats.apply(
            lambda row: (row['Count'] / cluster_totals[row[self.cluster_col]]) * 100,
            axis=1,
        )

        fig, axes = plt.subplots(
            nrows=(self.num_clusters + 1) // 2,
            ncols=2,
            figsize=(16, 4 * ((self.num_clusters + 1) // 2)),
        )
        axes = axes.flatten()
        colors = sns.color_palette(self.palette, self.num_clusters)

        for cluster_id in range(self.num_clusters):
            ax = axes[cluster_id]
            cluster_data = top_cats[top_cats[self.cluster_col] == cluster_id].sort_values(by='Count', ascending=True)

            bars = ax.barh(
                cluster_data[category_col],
                cluster_data['Percentage'],
                color=colors[cluster_id],
                edgecolor='black',
                alpha=0.85,
            )

            ax.set_title(f'Cụm {cluster_id}: Top {top_n} Danh mục phổ biến nhất', fontsize=12, fontweight='bold')
            ax.set_xlabel('Tỷ lệ chiếm lĩnh (%)')
            ax.set_xlim(0, max(top_cats['Percentage']) + 5)
            ax.grid(axis='x', linestyle='--', alpha=0.6)

            for bar in bars:
                xval = bar.get_width()
                ax.text(xval + 0.5, bar.get_y() + bar.get_height() / 2.0, f'{xval:.1f}%', ha='left', va='center', fontweight='bold')

        for j in range(self.num_clusters, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        self._save_fig(filename)
        plt.show()
# ====================================================
# CÁCH SỬ DỤNG TRONG FILE NOTEBOOK:
# ====================================================
# # 1. Khởi tạo class với DataFrame df_merged
# viz = ClusterVisualizer(df_merged, cluster_col='Cluster')

# # 2. Vẽ bộ biểu đồ RFM cơ bản (Truyền tâm cụm nếu có)
# viz.plot_rfm_overview(centers=model.cluster_centers_)

# # 3. Vẽ Radar Chart & Phân bố địa lý
# viz.plot_business_insights()

# # 4. Vẽ Top 3 Danh mục sản phẩm phổ biến nhất
# viz.plot_top_categories(top_n=3)