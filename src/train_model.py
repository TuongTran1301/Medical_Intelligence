from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from data_processing import DataProcessor
from evaluation import evaluate_model

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

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

def main():
    processor = DataProcessor('data/raw/extended_health_data.csv')
    df = processor.load_data()
    X, y = processor.preprocess_data(df)
    X_train, X_test, y_train, y_test = processor.split_and_scale_data(X, y)

    trainer = ModelTrainer()
    trainer.train(X_train, y_train)

    evaluate_model(trainer.model, X_test, y_test)

    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/random_forest_model.pkl')
    processor.save_scaler('models/scaler.pkl')

if __name__ == '__main__':
    main()
