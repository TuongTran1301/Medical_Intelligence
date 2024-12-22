from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Tạo dự đoán
    y_pred = model.predict(X_test)
    
    # In báo cáo phân loại
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Vẽ ma trận
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues') 
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
