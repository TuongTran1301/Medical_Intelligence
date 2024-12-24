import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

class DataProcessor:
    def __init__(self, raw_data_path, feature_columns):
        self.raw_data_path = raw_data_path
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()

    def load_data(self):
        try:
            return pd.read_csv(self.raw_data_path, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(self.raw_data_path, encoding='latin1')

    def preprocess_data(self, df, target_col='target'):
        required_columns = self.feature_columns + [target_col]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing columns detected: {missing_columns}")
            for col in missing_columns:
                if col == 'Gender':
                    df[col] = 'Unknown'
                elif col in self.feature_columns:
                    df[col] = 0
                else:
                    raise ValueError(f"Cannot process missing column: {col}")

        # Chuẩn hóa và mã hóa cột Gender
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].fillna('Unknown').str.strip().str.capitalize()
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'Unknown': -1})

        # Xử lý dữ liệu thiếu: thay thế giá trị thiếu trong các cột số bằng giá trị trung bình
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Xóa các hàng còn thiếu dữ liệu (nếu vẫn còn)
        df.dropna(inplace=True)

        # Kiểm tra lại dữ liệu sau xử lý
        if df.isnull().values.any():
            print("Dữ liệu vẫn còn giá trị thiếu sau khi xử lý:")
            print(df.isnull().sum())
            raise ValueError("Dữ liệu vẫn còn giá trị thiếu sau khi xử lý.")

        X = df[self.feature_columns]
        y = df[target_col]
        return X, y


    def split_and_scale_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_scaler(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.scaler, f)
