import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

class DataProcessor:
    def __init__(self, raw_data_path):
        """
        Khởi tạo lớp DataProcessor với đường dẫn đến tệp dữ liệu.
        """
        self.raw_data_path = raw_data_path
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Tải dữ liệu từ tệp CSV."""
        return pd.read_csv(self.raw_data_path)
    
    def preprocess_data(self, df, target_col='target'):
        """
        Tiền xử lý dữ liệu: chuẩn bị đặc trưng và mục tiêu.
        """
        # Danh sách các cột đặc trưng
        feature_columns = [
            'SGOT', 'SGPT', 'HDL-C', 'LDL-C', 'GGT', 
            'Cre', 'Uric', 'HCT', 'MCV', 'LYM', 'BachCauMono'
        ]
        
        # Xác minh dữ liệu chứa đủ các cột cần thiết
        required_columns = feature_columns + [target_col]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Các cột bị thiếu: {missing_columns}")

        # Xử lý giá trị thiếu
        df = df.dropna()

        # Chuẩn bị dữ liệu đầu vào và mục tiêu
        X = df[feature_columns]
        y = df[target_col]

        return X, y

    def split_and_scale_data(self, X, y):
        """
        Chia dữ liệu thành tập huấn luyện và kiểm tra, sau đó chuẩn hóa.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_scaler(self, output_path):
        """
        Lưu bộ chia tỷ lệ đã được huấn luyện.
        """
        with open(output_path, 'wb') as f:
            pickle.dump(self.scaler, f)
