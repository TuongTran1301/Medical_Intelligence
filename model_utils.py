import pickle
import pandas as pd


import pickle
import pandas as pd

def load_model_and_scaler_a():
    with open('models/random_forest_model_a.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler_a.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_health_a(input_data):
    feature_columns = ['SGOT', 'SGPT', 'HDL-C', 'LDL-C', 'GGT', 'Cre', 'Uric', 'HCT', 'MCV', 'LYM', 'BachCauMono']
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    
    model, scaler = load_model_and_scaler_a()
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]
    advice = generate_advice(input_data, probability)
    
    return probability, advice

def generate_advice(input_data, probability):
    """
    Tạo lời khuyên dựa trên các chỉ số đầu vào và xác suất dự đoán.

    Chỉ trả về tối đa 5 lời khuyên quan trọng nhất dựa trên độ lệch so với giá trị bình thường,
    tầm quan trọng y tế và độ ảnh hưởng đến xác suất dự đoán.
    """
    advice = []

    # Phân loại nguy cơ tổng quát
    if probability < 0.1:
        advice.append("Bạn có nguy cơ rất thấp. Duy trì lối sống lành mạnh và kiểm tra sức khỏe định kỳ.")
    elif probability < 0.3:
        advice.append("Bạn có nguy cơ thấp. Chú ý lối sống lành mạnh và theo dõi sức khỏe thường xuyên.")
    elif probability < 0.5:
        advice.append("Nguy cơ trung bình. Cần thay đổi lối sống và cân nhắc gặp bác sĩ.")
    elif probability < 0.7:
        advice.append("Nguy cơ cao. Hãy tham khảo ý kiến bác sĩ và thay đổi chế độ ăn uống ngay.")
    else:
        advice.append("Nguy cơ rất cao! Cần gặp bác sĩ chuyên khoa và tuân thủ các khuyến nghị y tế ngay lập tức.")

    # Giá trị tham chiếu cho các chỉ số
    reference_ranges = {
        'SGOT': (10, 40),
        'SGPT': (10, 40),
        'HDL-C': (40, 60),
        'LDL-C': (70, 130),
        'GGT': (10, 60),
        'Cre': (0.6, 1.2),
        'Uric': (3, 7),
        'HCT': (35, 50),
        'MCV': (80, 100),
        'LYM': (20, 50),
        'BachCauMono': (2, 10)
    }

    # Trọng số tầm quan trọng y tế
    health_importance_weights = {
        'SGOT': 2.0,
        'SGPT': 2.0,
        'HDL-C': 1.5,
        'LDL-C': 1.5,
        'GGT': 2.0,
        'Cre': 1.0,
        'Uric': 1.0,
        'HCT': 1.2,
        'MCV': 1.2,
        'LYM': 1.0,
        'BachCauMono': 1.0
    }

    # Độ quan trọng của mô hình (giả sử bạn đã lấy từ model.feature_importances_)
    model_importances = {
        'SGOT': 0.15,
        'SGPT': 0.12,
        'HDL-C': 0.10,
        'LDL-C': 0.10,
        'GGT': 0.08,
        'Cre': 0.05,
        'Uric': 0.07,
        'HCT': 0.09,
        'MCV': 0.08,
        'LYM': 0.05,
        'BachCauMono': 0.05
    }

    # Tính toán độ lệch và tổng ảnh hưởng
    detailed_advice = []
    deviations = []

    for key, (low, high) in reference_ranges.items():
        value = input_data[key]
        health_weight = health_importance_weights.get(key, 1.0)
        model_importance = model_importances.get(key, 0.1)

        if value < low:
            deviation = (low - value) * health_weight * model_importance
            deviations.append((key, deviation))
            if key == 'SGOT':
                detailed_advice.append("SGOT thấp. Có thể do thiếu protein, hãy bổ sung thịt, cá, hoặc các nguồn protein thực vật.")
            elif key == 'SGPT':
                detailed_advice.append("SGPT thấp. Có thể do thiếu năng lượng, nên ăn đầy đủ các nhóm chất dinh dưỡng.")
            elif key == 'HDL-C':
                detailed_advice.append("HDL-C thấp. Tăng cường vận động thể chất và ăn thực phẩm giàu omega-3.")
            elif key == 'LDL-C':
                detailed_advice.append("LDL-C thấp. Điều này tốt, nhưng cần đảm bảo đủ dinh dưỡng để duy trì sức khỏe.")
            elif key == 'Cre':
                detailed_advice.append("Creatinine thấp. Điều này có thể do suy dinh dưỡng, cần bổ sung protein.")
            elif key == 'Uric':
                detailed_advice.append("Uric acid thấp. Điều này không phổ biến, nhưng cần kiểm tra dinh dưỡng.")
            elif key == 'HCT':
                detailed_advice.append("HCT thấp. Cần kiểm tra thiếu máu và bổ sung thực phẩm giàu sắt.")
            elif key == 'MCV':
                detailed_advice.append("MCV thấp. Kiểm tra nguy cơ thiếu máu thiếu sắt.")
            elif key == 'LYM':
                detailed_advice.append("LYM thấp. Có thể do nhiễm trùng hoặc suy giảm miễn dịch.")
            elif key == 'BachCauMono':
                detailed_advice.append("Bạch cầu Mono thấp. Cần kiểm tra nguy cơ suy giảm miễn dịch.")
        elif value > high:
            deviation = (value - high) * health_weight * model_importance
            deviations.append((key, deviation))
            if key == 'SGOT':
                detailed_advice.append("SGOT cao. Tránh rượu bia, thực phẩm chiên xào, và ăn thêm trái cây giàu vitamin C.")
            elif key == 'SGPT':
                detailed_advice.append("SGPT cao. Tránh thực phẩm nhiều dầu mỡ và tăng cường thực phẩm bảo vệ gan.")
            elif key == 'HDL-C':
                detailed_advice.append("HDL-C cao. Đây là dấu hiệu tốt, tiếp tục duy trì lối sống lành mạnh.")
            elif key == 'LDL-C':
                detailed_advice.append("LDL-C cao. Giảm thực phẩm chứa chất béo bão hòa và tăng cường rau xanh.")
            elif key == 'GGT':
                detailed_advice.append("GGT cao. Kiểm tra chức năng gan và hạn chế tiêu thụ rượu bia.")
            elif key == 'Cre':
                detailed_advice.append("Creatinine cao. Kiểm tra chức năng thận và giảm tiêu thụ protein động vật.")
            elif key == 'Uric':
                detailed_advice.append("Uric acid cao. Tránh thực phẩm giàu purine như hải sản và nội tạng động vật.")
            elif key == 'HCT':
                detailed_advice.append("HCT cao. Tăng cường uống nước và giảm muối trong chế độ ăn.")
            elif key == 'MCV':
                detailed_advice.append("MCV cao. Có thể do thiếu vitamin B12, cần bổ sung qua thực phẩm hoặc thực phẩm chức năng.")
            elif key == 'LYM':
                detailed_advice.append("LYM cao. Kiểm tra nhiễm trùng hoặc bệnh lý liên quan đến hệ bạch huyết.")
            elif key == 'BachCauMono':
                detailed_advice.append("Bạch cầu Mono cao. Có thể do viêm nhiễm, cần gặp bác sĩ để kiểm tra.")

    # Sắp xếp theo tổng ảnh hưởng và chọn tối đa 4 khuyến nghị
    deviations.sort(key=lambda x: x[1], reverse=True)
    selected_advice = [detailed_advice[i] for i, (key, _) in enumerate(deviations[:4])]

    # Kết hợp với lời khuyên tổng quát
    important_advice = advice + selected_advice

    # Khuyến nghị lối sống chung (thêm nếu cần)
    if len(important_advice) < 5:
        important_advice.append("Tăng cường thực phẩm giàu chất xơ như rau xanh, hạt chia, và yến mạch.")

    return important_advice