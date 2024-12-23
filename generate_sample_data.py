import pandas as pd
import numpy as np

np.random.seed(42)

# Số lượng mẫu dữ liệu
n_samples = 1000

# Tạo dữ liệu cho các chỉ số
columns = {
    'Tuoi': np.random.randint(18, 80, n_samples),
    'GioiTinh': np.random.choice(['Nam', 'Nu'], n_samples),
    'BMI': np.random.normal(25, 4, n_samples).round(1),
    'Glu': np.random.normal(100, 15, n_samples).round(1),
    'SGOT': np.random.normal(20, 10, n_samples).round(1),
    'SGPT': np.random.normal(25, 15, n_samples).round(1),
    'HDL-C': np.random.normal(50, 10, n_samples).round(1),
    'LDL-C': np.random.normal(130, 30, n_samples).round(1),
    'GGT': np.random.normal(30, 20, n_samples).round(1),
    'Cre': np.random.normal(1, 0.2, n_samples).round(2),
    'Uric': np.random.normal(5.5, 1.2, n_samples).round(1),
    'HCT': np.random.normal(45, 5, n_samples).round(1),
    'MCV': np.random.normal(90, 5, n_samples).round(1),
    'LYM': np.random.normal(30, 10, n_samples).round(1),
    'BachCauMono': np.random.normal(6, 2, n_samples).round(1)
}

# Tạo DataFrame
df = pd.DataFrame(columns)

# Điều chỉnh các giá trị vượt ngưỡng
adjustments = {
    'BMI': (16, 35),
    'Glu': (70, 180),
    'SGOT': (10, 40),
    'SGPT': (10, 50),
    'HDL_C': (20, 70),
    'LDL_C': (70, 190),
    'GGT': (5, 80),
    'Cre': (0.5, 1.5),
    'Uric': (3.0, 8.0),
    'HCT': (35, 50),
    'MCV': (80, 100),
    'LYM': (15, 45),
    'BachCauMono': (2, 10)
}
for col, (min_val, max_val) in adjustments.items():
    df.loc[df[col] < min_val, col] = min_val
    df.loc[df[col] > max_val, col] = max_val

# Tạo cột target dựa trên các điều kiện nguy cơ
conditions = (
    (df['BMI'] > 30) | 
    (df['Glu'] > 126) | 
    (df['LDL_C'] > 160) | 
    (df['HDL_C'] < 40) | 
    (df['Uric'] > 7.0) | 
    (df['SGOT'] > 35) | 
    (df['SGPT'] > 40) | 
    (df['GGT'] > 50)
)
df['target'] = conditions.astype(int)

# Thêm cột lời khuyên
advices = []
for _, row in df.iterrows():
    advice = []
    if row['BMI'] > 30:
        advice.append("Giảm cân và duy trì chế độ ăn uống lành mạnh.")
        advice.append("Tham khảo bác sĩ dinh dưỡng để xây dựng thực đơn phù hợp.")
        advice.append("Tăng cường vận động ít nhất 30 phút mỗi ngày.")
    if row['Glu'] > 126:
        advice.append("Kiểm tra đường huyết thường xuyên và hạn chế đồ ngọt.")
        advice.append("Tham khảo bác sĩ nội tiết để điều chỉnh chế độ ăn và thuốc.")
        advice.append("Duy trì chế độ ăn ít tinh bột và đường.")
    if row['LDL_C'] > 160:
        advice.append("Tránh thực phẩm giàu cholesterol.")
        advice.append("Tăng cường ăn rau xanh và các loại hạt tốt cho tim mạch.")
        advice.append("Tham khảo bác sĩ để xem xét dùng thuốc giảm mỡ máu nếu cần.")
    if row['HDL_C'] < 40:
        advice.append("Tăng cường vận động thể chất.")
        advice.append("Ăn các thực phẩm giàu omega-3 như cá hồi, cá thu.")
    if row['Uric'] > 7.0:
        advice.append("Hạn chế thực phẩm giàu purine như thịt đỏ.")
        advice.append("Uống nhiều nước để hỗ trợ đào thải axit uric.")
        advice.append("Hạn chế rượu bia, đặc biệt là bia.")
    if row['SGOT'] > 35 or row['SGPT'] > 40:
        advice.append("Theo dõi chức năng gan định kỳ.")
        advice.append("Hạn chế đồ ăn dầu mỡ và rượu bia.")
        advice.append("Tham khảo bác sĩ để kiểm tra thêm về gan.")
    if row['GGT'] > 50:
        advice.append("Hạn chế rượu bia.")
        advice.append("Kiểm tra chức năng gan và tham khảo ý kiến bác sĩ.")
    if row['Cre'] > 1.2:
        advice.append("Theo dõi chức năng thận thường xuyên.")
        advice.append("Hạn chế tiêu thụ muối và protein quá mức.")
    if row['HCT'] > 50:
        advice.append("Kiểm tra máu để loại trừ các vấn đề về tuần hoàn.")
        advice.append("Uống nhiều nước để duy trì thể tích máu ổn định.")
    if not advice:
        advice.append("Không có nguy cơ đáng kể. Duy trì lối sống lành mạnh.")
    advices.append(" ".join(advice))

df['Advice'] = advices

# Lưu dữ liệu ra file CSV
df.to_csv('data/raw/extended_health_data.csv', index=False)

# Hiển thị thông tin mẫu
def preview_data():
    print("\n5 dòng dữ liệu đầu tiên:\n", df.head())
    print("\nThống kê cơ bản:\n", df.describe())
    print("\nPhân bố target:")
    print(df['target'].value_counts())

preview_data()
