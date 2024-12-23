from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from data_processing import DataProcessor
from evaluation import evaluate_model

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=1000, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def save_model(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_model(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def get_feature_importance(self, feature_names):
        importance = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

def process_and_train(raw_data_path, feature_columns, target_col, model_path, scaler_path):
    """Helper function to process data and train a model."""
    processor = DataProcessor(raw_data_path, feature_columns)
    df = processor.load_data()
    X, y = processor.preprocess_data(df, target_col=target_col)
    X_train, X_test, y_train, y_test = processor.split_and_scale_data(X, y)

    trainer = ModelTrainer()
    trainer.train(X_train, y_train)

    evaluate_model(trainer.model, X_test, y_test)

    trainer.save_model(model_path)
    processor.save_scaler(scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

def main():
    os.makedirs('models', exist_ok=True)

    # Dataset A
    process_and_train(
        raw_data_path='data/raw/extended_health_data.csv',
        feature_columns=['SGOT', 'SGPT', 'HDL-C', 'LDL-C', 'GGT', 'Cre', 'Uric', 'HCT', 'MCV', 'LYM', 'BachCauMono'],
        target_col='target',
        model_path='models/random_forest_model_a.pkl',
        scaler_path='models/scaler_a.pkl'
    )

    # Dataset B
    process_and_train(
        raw_data_path='data/raw/data2.csv',
        feature_columns=['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin', 'Alkphos', 'SGPT', 'SGOT', 'Total Proteins', 'ALB', 'A/G Ratio'],
        target_col='Result',
        model_path='models/random_forest_model_b.pkl',
        scaler_path='models/scaler_b.pkl'
    )

if __name__ == '__main__':
    main()
