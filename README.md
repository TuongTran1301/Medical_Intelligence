# MedialIntelligence
medialIntelligence

# Kích hoạt môi trường ảo
  ```bash
  # Tạo môi trường ảo
  python -m venv venv

  # Kích hoạt môi trường ảo
  # Windows:
  venv\Scripts\activate
  # Linux/Mac:
  source venv/bin/activate

  # Cài đặt thư viện
  pip install -r requirements.txt

  #random data
  python generate_sample_data.py
  
  ```
  

1. Huấn luyện mô hình:
   ```bash
   python src/train_model.py
   ```

2. Dự đoán:
   ```bash
   python main.py
   ```
