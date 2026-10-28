# 🛒 DỰ ÁN PHÂN CỤM KHÁCH HÀNG THƯƠNG MẠI ĐIỆN TỬ OLIST
### *Customer Segmentation using RFM Analysis & Unsupervised Machine Learning*

> **Đơn vị thực hiện**: Nhóm sinh viên lớp 51K29.1 - Trường Đại học Kinh tế, Đại học Đà Nẵng (DUE)  
> **Ngôn ngữ & Thư viện chính**: Python 3.9+, Scikit-Learn, Scipy, Pandas, Seaborn, Matplotlib

---

## 📌 1. Giới thiệu tổng quan về dự án

Dự án áp dụng phương pháp phân tích **RFM (Recency - Frequency - Monetary)** kết hợp các thuật toán **Học máy không giám sát (Unsupervised Learning)** trên tập dữ liệu thương mại điện tử **Brazilian E-Commerce Public Dataset by Olist** (~100.000 đơn hàng thực tế).

### Mục tiêu chính:
1. **Làm sạch & Gộp dữ liệu**: Hợp nhất 7 bảng dữ liệu thô (khách hàng, đơn hàng, thanh toán, đánh giá, sản phẩm...) thành bảng tổng hợp theo từng khách hàng duy nhất (`customer_unique_id`).
2. **Kỹ thuật trích xuất đặc trưng RFM**: Tính toán Recency (Độ mới), Frequency (Tần suất mua), Monetary (Tổng chi tiêu); xử lý ngoại lai (Capping), biến đổi Log và chuẩn hóa với `StandardScaler`.
3. **Thực nghiệm 3 mô hình phân cụm**:
   - **K-Means Clustering**: Phân cụm truyền thống theo khoảng cách Euclidean.
   - **K-Medoids Clustering**: Tối ưu bộ nhớ với thuật toán Batched PAM, kháng nhiễu và ngoại lai.
   - **Fuzzy C-Means (FCM)**: Phân cụm mờ xác suất, giúp nhận biết mức độ thuộc về từng cụm của mỗi khách hàng.
4. **Trực quan hóa & Định hình chân dung khách hàng**: Xuất các biểu đồ PCA 2D, Snake plot, Radar Chart đa chiều, phân bố địa lý bang và cơ cấu ngành hàng.
5. **Ứng dụng thực tế cho A/B Testing**: Tận dụng ma trận xác suất của FCM để nhận diện nhóm khách hàng ở vùng ranh giới phân vân ($\text{Diff} < 0.15$), từ đó chia mẫu phân tầng 50/50 (**Control vs Treatment**) phục vụ triển khai chiến dịch tiếp thị thử nghiệm.

---

## ⚙️ 2. Hướng dẫn cài đặt dự án

Thực hiện theo các bước sau để thiết lập môi trường chạy dự án:

### Bước 1: Tải mã nguồn về máy
Mở Terminal / PowerShell và clone repo (hoặc tải file ZIP về giải nén):
```bash
git clone <URL_REPO_CUA_BAN>
cd PROJECT-REPO-main
```

### Bước 2: Chuẩn bị dữ liệu thô (Raw Data)
Đảm bảo 7 file CSV của bộ dữ liệu Olist đã được đặt trong thư mục `data/raw_data/`:
- `olist_customers_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_products_dataset.csv`
- `product_category_name_translation.csv`

### Bước 3: Tạo môi trường ảo & Cài đặt thư viện
```bash
# 1. Khởi tạo môi trường ảo (Khuyến nghị dùng Python >= 3.9)
python -m venv venv

# 2. Kích hoạt môi trường ảo:
# Trên Windows:
venv\Scripts\activate
# Trên Linux / macOS:
source venv/bin/activate

# 3. Cài đặt các gói phụ thuộc:
pip install -r requirements.txt
```

---

## 🚀 3. Hướng dẫn cách chạy dự án

Dự án cung cấp **2 cách chạy linh hoạt**: chạy tự động toàn bộ qua dòng lệnh CLI hoặc chạy tương tác từng bước trên Jupyter Notebook.

### Cách 1: Chạy toàn bộ tự động qua Pipeline (Khuyên dùng)
Chỉ cần thực thi **1 lệnh duy nhất** từ thư mục gốc của dự án:
```bash
python run_pipeline.py
```
> Lệnh này sẽ tự động chạy toàn bộ 5 bước từ đầu đến cuối: ghép dữ liệu thô $\to$ làm sạch $\to$ tính RFM $\to$ huấn luyện cả 3 mô hình $\to$ vẽ biểu đồ và xuất báo cáo bảng biểu.

#### Chạy riêng từng bước cụ thể (nếu cần gỡ lỗi hoặc xem chi tiết):
```bash
# Bước 1: Nối 7 bảng raw thành 1 bảng tổng hợp
python run_pipeline.py --step join

# Bước 2: Làm sạch dữ liệu (lọc đơn hủy, xử lý null, khử trùng lặp)
python run_pipeline.py --step preprocess

# Bước 3: Tính toán chỉ số RFM, Capping, Log transform và StandardScaler
python run_pipeline.py --step rfm

# Bước 4: Train và lưu 3 mô hình (K-Means, K-Medoids, Fuzzy C-Means)
python run_pipeline.py --step cluster

# Bước 5: Trực quan hóa và xuất báo cáo CSV ra reports/
python run_pipeline.py --step visualize
```

---

### Cách 2: Khám phá tương tác bằng Jupyter Notebook

Khởi động Jupyter Notebook từ terminal:
```bash
jupyter notebook
```
Sau đó truy cập vào thư mục `notebooks/` và chạy lần lượt 5 notebook theo thứ tự:

1. **`01_data_understanding.ipynb`**: Khám phá dữ liệu thô, phân tích khóa ngoại và ghép 7 bảng Olist.
2. **`02_rfm_eda.ipynb`**: Phân tích khám phá chỉ số RFM cơ sở, trực quan hóa phân phối lệch, phát hiện ngoại lai và tương quan Spearman.
3. **`03_clustering_experiments.ipynb`**: Quét số cụm tối ưu ($K=2 \to 10$) qua 4 chỉ số (Elbow, Silhouette, Calinski-Harabasz, Davies-Bouldin); Benchmark chi tiết 3 thuật toán K-Means, K-Medoids và Fuzzy C-Means.
4. **`04_cluster_interpretation.ipynb`**: Kết hợp nhãn cụm với thông tin giá trị đơn hàng, phí vận chuyển, điểm đánh giá, bang địa lý và danh mục sản phẩm để xây dựng chân dung 4 nhóm khách hàng.
5. **`05_ab_testing_preparation.ipynb`**: Lọc nhóm khách hàng vùng ranh giới dựa vào xác suất FCM, chia ngẫu nhiên phân tầng 50/50 thành nhóm **Control** và **Treatment** để sẵn sàng cho chiến dịch A/B Testing.

---

## 📂 4. Giải thích chi tiết cấu trúc dự án

```text
PROJECT-REPO-main/
│
├── README.md                         
├── requirements.txt                   
├── run_pipeline.py                     
├── .gitignore                         
│
├── data/                            
│   ├── raw_data/                     
│   ├── joined_data/                   
│   └── processed/                      
│
├── models/                           
│   ├── kmeans_rfm_model.pkl           
│   ├── kmedoids_rfm_model.pkl        
│   └── fcm_rfm_model.pkl              
│
├── notebooks/                         
│   ├── 01_data_understanding.ipynb    
│   ├── 02_rfm_eda.ipynb               
│   ├── 03_clustering_experiments.ipynb
│   ├── 04_cluster_interpretation.ipynb
│   └── 05_ab_testing_preparation.ipynb 
│
├── reports/                            
│   ├── figures/                        
│   └── tables/                        
│
└── src/                              
    ├── __init__.py                    
    ├── config.py                     
    ├── olist_joined_dataset.py        
    ├── preprocessing.py               
    ├── rfm.py                          
    ├── clustering.py                  
    └── visualization.py              
```

---

## 📊 5. Tóm tắt kết quả phân cụm ($K=4$) & Ứng dụng kinh doanh

Dựa trên kết quả thực nghiệm chọn $K=4$ tối ưu, các cụm khách hàng được định hình như sau:

| Cụm | Tên chân dung | Đặc điểm hành vi chính | Đề xuất giải pháp Marketing |
|:---:|:---|:---|:---|
| **0** | **Nguy cơ rời bỏ** *(At Risk)* | Đã rất lâu không mua sắm (Recency cao), tần suất và chi tiêu đều thấp. | Gửi email kích hoạt lại (Re-activation), tặng voucher giảm giá có thời hạn ngắn. |
| **1** | **Khách mới / Tiềm năng** *(Promising)* | Vừa mua hàng gần đây, chi tiêu trung bình, tần suất mua 1 lần. | Nuôi dưỡng qua chuỗi Onboarding, gợi ý sản phẩm bán chéo (Cross-selling). |
| **2** | **Khách VIP** *(Champions)* | Chi tiêu số tiền rất lớn (Monetary cao vượt trội), đánh giá sao cao. | Chế độ chăm sóc khách hàng VIP độc quyền, quà tri ân, tích điểm đổi quà. |
| **3** | **Khách phổ thông** *(Regulars)* | Mua sắm định kỳ, giá trị đơn hàng phổ biến, ổn định. | Khuyến mãi kích thích gia tăng kích thước giỏ hàng (Upselling, Freeship khi đạt ngưỡng). |

### Ứng dụng thực tế cho A/B Testing:
- Từ mô hình **Fuzzy C-Means**, hệ thống trích xuất độ chênh lệch xác suất giữa Cụm Top 1 và Cụm Top 2 ($\text{Diff} = P_{\text{Top 1}} - P_{\text{Top 2}}$).
- Các khách hàng có $\text{Diff} < 0.15$ được xác định là khách hàng thuộc **vùng ranh giới phân vân**.
- File `danh_sach_AB_Testing_chia_nhom.csv` chia các khách hàng này thành 2 nhóm ngẫu nhiên phân tầng 50/50:
  - **Nhóm Control (50%)**: Áp dụng ưu đãi/thông điệp mặc định của Cụm Top 1.
  - **Nhóm Treatment (50%)**: Áp dụng ưu đãi/thông điệp thử nghiệm của Cụm Top 2 nhằm đánh giá mức độ dịch chuyển hành vi mua sắm.

---

## 👥 6. Thông tin nhóm thực hiện

Dự án được thực hiện bởi nhóm sinh viên lớp **51K29.1** - Trường **Đại học Kinh tế, Đại học Đà Nẵng (DUE)**:

1. **Ngô Phương Thảo** - 51K29.1 - [ngphth.297@gmail.com](mailto:ngphth.297@gmail.com)
2. **Lương Duy Khôi** - 51K29.1 - [luongduykhoi141007@gmail.com](mailto:luongduykhoi141007@gmail.com)
3. **Thân Đức Vũ Khanh** - 51K29.1 - [vukhanhdn2007@gmail.com](mailto:vukhanhdn2007@gmail.com)
4. **Phạm Mai Thùy Trâm** - 51K29.1 - [thuytramphammai805@gmail.com](mailto:thuytramphammai805@gmail.com)
