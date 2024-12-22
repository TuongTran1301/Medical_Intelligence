import pickle
import pandas as pd

def load_model_and_scaler():
    """Tải mô hình và bộ chia tỷ lệ đã được huấn luyện"""
    with open('models/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_diabetes(tuoi, gioi_tinh, bmi, duong_huyet, hba1c, insulin):
    """Dự đoán xác suất mắc bệnh tiểu đường cho bệnh nhân mới"""
    inputVal = {
        'tuoi': tuoi, 'gioi_tinh': gioi_tinh, 'bmi': bmi, 'duong_huyet': duong_huyet, 'hba1c': hba1c, 'insulin': insulin
    }
    print("Dữ liệu đầu vào: ", inputVal)

    # Tải mô hình và bộ chia tỷ lệ
    model, scaler = load_model_and_scaler()
    
    # Chuẩn bị dữ liệu đầu vào
    new_data = pd.DataFrame([[tuoi, gioi_tinh, bmi, duong_huyet, hba1c, insulin]], 
                          columns=['Tuoi', 'GioiTinh', 'BMI', 'DuongHuyet', 'HbA1c', 'Insulin'])
    
    # Chuyển đổi giới tính
    new_data['GioiTinh'] = new_data['GioiTinh'].map({'Nam': 0, 'Nu': 1})
    
    new_data_scaled = scaler.transform(new_data)
    # Chia tỷ lệ dữ liệu
    new_data_scaled = scaler.transform(new_data)
    
    # Dự đoán xác suất
    probability = model.predict_proba(new_data_scaled)[0][1]
    
    return probability

if __name__ == '__main__':
    # Ví dụ sử dụng
    xac_suat = predict_diabetes(
        tuoi=19,
        gioi_tinh='Nam',
        bmi=127.5,
        duong_huyet=200,
        hba1c=6.5,
        insulin=10.2
    )
    print(f"Xác suất mắc bệnh tiểu đường: {xac_suat:.2%}")
