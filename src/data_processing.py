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
        """Load dataset from file with UTF-8 or fallback to Latin-1 encoding."""
        try:
            return pd.read_csv(self.raw_data_path, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(self.raw_data_path, encoding='latin1')

    def preprocess_data(self, df, target_col='target'):
        """Preprocess dataset: handle missing columns, encode gender, and fill missing values."""
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

        # Encode Gender column
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].fillna('Unknown').str.strip().str.capitalize()
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'Unknown': -1})

        # Fill missing values in numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # Drop rows with any remaining missing values
        df.dropna(inplace=True)

        # Map target column if necessary
        if target_col == 'Result':
            df[target_col] = df[target_col].map({2: 0, 1: 1})  # 0: Bình thường, 1: Bị bệnh

        X = df[self.feature_columns]
        y = df[target_col]
        return X, y

    def split_and_scale_data(self, X, y):
        """Split and scale dataset for training and testing."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_scaler(self, output_path):
        """Save the scaler to a file."""
        with open(output_path, 'wb') as f:
            pickle.dump(self.scaler, f)