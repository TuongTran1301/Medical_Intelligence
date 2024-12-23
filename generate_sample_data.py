import pandas as pd
import numpy as np

np.random.seed(42)

# Số lượng mẫu dữ liệu
n_samples = 1000

# Tạo dữ liệu cho các chỉ số
columns = {
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
    'HDL-C': (20, 70),
    'LDL-C': (70, 190),
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
    (df['LDL-C'] > 160) | 
    (df['HDL-C'] < 40) | 
    (df['Uric'] > 7.0) | 
    (df['SGOT'] > 35) | 
    (df['SGPT'] > 40) | 
    (df['GGT'] > 50)
)
df['target'] = conditions.astype(int)


# Lưu dữ liệu ra file CSV
df.to_csv('data/raw/extended_health_data.csv', index=False)

# Hiển thị thông tin mẫu
def preview_data():
    print("\n5 dòng dữ liệu đầu tiên:\n", df.head())
    print("\nThống kê cơ bản:\n", df.describe())
    print("\nPhân bố target:")
    print(df['target'].value_counts())

preview_data()
