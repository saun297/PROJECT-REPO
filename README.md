<h1 align="center"> DỰ ÁN PHÂN CỤM KHÁCH HÀNG THƯƠNG MẠI ĐIỆN TỬ OLIST</h1>

Ứng dụng kỹ thuật phân tích RFM và các mô hình học máy không giám sát để nhận diện, phân cụm tập khách hàng trên sàn Olist; từ đó cung cấp căn cứ dữ liệu phục vụ việc thiết kế và đề xuất các chính sách khuyến mãi, tối ưu hóa hiệu quả kinh doanh. 

## Cài đặt

```python
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

Write your content here...

## Cấu trúc dự án
```text
olist-customer-segmentation/
│
├── README.md
├── requirements.txt
├── .gitignore
├── run_pipeline.py
│
├── data/
│   ├── raw/
│   │   ├── olist_customers_dataset.csv
│   │   ├── olist_orders_dataset.csv
│   │   └── olist_order_payments_dataset.csv
│   │   └── olist_order_items_dataset.csv 
│   │   └── olist_order_reviews_dataset.csv 
│   │   └── olist_products_dataset.csv 
│   │   └── product_category_name_translation.csv 
│   │   └── olist_joined_dataset.csv 
│   ├── processed/                                
│   │   └── .gitkeep
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_rfm_eda.ipynb
│   ├── 03_clustering_experiments.ipynb
│   └── 04_cluster_interpretation.ipynb     
│
├── src/
│   ├── __init__.py      
│   ├── config.py                 
│   ├── olist_joined_dataset.py      
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── rfm.py
│   ├── visualization.py
│   ├── clustering.py
│   ├── evaluation.py
│
├── models/
│   └── .gitkeep 
│
├── reports/
      ├── figures/
      │       └── .gitkeep  
      ├── tables/
      │      └── .gitkeep 
      └── final_report.docx
```
## Thông tin tác giả

1. Ngô Phương Thảo - 51K29.1 - ngphth.297@gmail.com
2. Lương Duy Khôi - 51K29.1 - luongduykhoi141007@gmail.com
3. Thân Đức Vũ Khanh - 51K29.1 - vukhanhdn2007@gmail.com
4. Phạm Mai Thùy Trâm - 51K29.1 - thuytramphammai805@gmail.com
