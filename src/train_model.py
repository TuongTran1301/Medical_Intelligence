from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from data_processing import DataProcessor
from evaluation import evaluate_model

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train(self, X_train, y_train):
        """Huấn luyện mô hình"""
        self.model.fit(X_train, y_train)
    
    def save_model(self, output_path):
        """Lưu mô hình đã huấn luyện"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @staticmethod
    def load_model(model_path):
        """Tải một mô hình đã được huấn luyện"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

def main():
    # Khởi tạo bộ xử lý
    processor = DataProcessor('data/raw/diabetes_data.csv')
    
    # Tải và xử lý dữ liệu
    df = processor.load_data()
    X, y = processor.preprocess_data(df)
    X_train, X_test, y_train, y_test = processor.split_and_scale_data(X, y)
    
    # Huấn luyện mô hình
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    
    # Đánh giá mô hình
    evaluate_model(trainer.model, X_test, y_test)
    
    # Lưu mô hình và bộ chia tỷ lệ
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/random_forest_model.pkl')
    processor.save_scaler('models/scaler.pkl')

if __name__ == '__main__':
    main()
