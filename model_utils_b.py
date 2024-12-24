import pickle
import pandas as pd

def load_model_and_scaler_b():
    with open('models/random_forest_model_b.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler_b.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_health_b(input_data):
    feature_columns = ['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin', 'Alkphos', 'SGPT', 'SGOT', 
                       'Total Proteins', 'ALB', 'A/G Ratio']
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    
    model, scaler = load_model_and_scaler_b()
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]
    advice = []

    # Phân tích từng chỉ số
    if input_data['Total Bilirubin'] > 1.2:
        advice.append("Chỉ số Bilirubin toàn phần cao, có thể là dấu hiệu của tổn thương gan hoặc bệnh vàng da. "
                      "Hãy kiểm tra chức năng gan và tránh các tác nhân gây hại như rượu bia.")
    if input_data['Direct Bilirubin'] > 0.3:
        advice.append("Chỉ số Bilirubin trực tiếp cao, cần theo dõi thêm về tình trạng ống mật hoặc gan.")
    if input_data['Alkphos'] > 120:
        advice.append("Chỉ số Alkphos cao, có thể liên quan đến bệnh lý xương hoặc tắc nghẽn mật.")
    if input_data['SGPT'] > 40:
        advice.append("Chỉ số SGPT cao, đây là dấu hiệu cảnh báo tổn thương tế bào gan. Tránh thực phẩm nhiều dầu mỡ và rượu bia.")
    if input_data['SGOT'] > 40:
        advice.append("Chỉ số SGOT cao, cần kiểm tra nguy cơ tổn thương gan hoặc cơ tim.")
    if input_data['Total Proteins'] < 6.0 or input_data['Total Proteins'] > 8.3:
        advice.append("Chỉ số Protein toàn phần bất thường, cần đánh giá thêm về dinh dưỡng hoặc chức năng gan thận.")
    if input_data['ALB'] < 3.5:
        advice.append("Chỉ số Albumin thấp, có thể là dấu hiệu của suy dinh dưỡng hoặc bệnh gan mạn tính.")
    if input_data['A/G Ratio'] < 1.1:
        advice.append("Tỷ lệ A/G thấp, cần theo dõi chức năng gan hoặc nguy cơ nhiễm trùng.")
    
    # Dựa trên xác suất tổng thể
    if probability < 0.3:
        advice.append("Sức khỏe tổng thể tốt, không có dấu hiệu bất thường nghiêm trọng. Tiếp tục duy trì chế độ sống lành mạnh.")
    elif 0.3 <= probability < 0.7:
        advice.append("Có một số dấu hiệu bất thường cần chú ý. Nên kiểm tra sức khỏe định kỳ để theo dõi thêm.")
    else:
        advice.append("Nguy cơ cao về bệnh gan. Hãy tham khảo ý kiến bác sĩ ngay lập tức để được chẩn đoán và điều trị kịp thời.")
    
    return probability, advice