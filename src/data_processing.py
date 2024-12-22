import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class DataProcessor:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Tải dữ liệu gốc từ tệp CSV"""
        return pd.read_csv(self.raw_data_path)
    
    def preprocess_data(self, df):
        """Tiền xử lý dữ liệu"""
        # Chia features và target
        X = df[['Tuoi', 'GioiTinh', 'BMI', 'DuongHuyet', 'HbA1c', 'Insulin']]
        y = df['BiTieuDuong']
        
        # Chuyển đổi giới tính
        X['GioiTinh'] = X['GioiTinh'].map({'Nam': 0, 'Nu': 1})
        
        return X, y
    
    def split_and_scale_data(self, X, y):
        """Chia dữ liệu và tỷ lệ các đặc trưng"""
        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Tính tỷ lệ các đặc trưng
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_scaler(self, output_path):
        """Lưu bộ tỷ lệ đã được điều chỉnh"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.scaler, f)
