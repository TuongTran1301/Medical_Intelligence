import pandas as pd
import numpy as np
np.random.seed(42)

# Tạo 1000 mẫu dữ liệu
n_samples = 10000

# Tạo dữ liệu cho từng cột
data = {
    'Tuoi': np.random.randint(18, 80, n_samples),
    'GioiTinh': np.random.choice(['Nam', 'Nu'], n_samples),
    'BMI': np.random.normal(25, 4, n_samples).round(1),  # BMI từ 17-33
    'DuongHuyet': np.random.normal(110, 30, n_samples).round(1),  # 50-170 mg/dL
    'HbA1c': np.random.normal(6, 1, n_samples).round(1),  # 4-8%
    'Insulin': np.random.normal(15, 5, n_samples).round(1),  # 5-25 µU/mL
}

# Tạo DataFrame
df = pd.DataFrame(data)

# Điều chỉnh giá trị cho hợp lý
df.loc[df['BMI'] < 16, 'BMI'] = 16
df.loc[df['BMI'] > 35, 'BMI'] = 35
df.loc[df['DuongHuyet'] < 70, 'DuongHuyet'] = 70
df.loc[df['DuongHuyet'] > 180, 'DuongHuyet'] = 180
df.loc[df['HbA1c'] < 4, 'HbA1c'] = 4
df.loc[df['HbA1c'] > 9, 'HbA1c'] = 9
df.loc[df['Insulin'] < 5, 'Insulin'] = 5
df.loc[df['Insulin'] > 25, 'Insulin'] = 25

# Tạo cột BiTieuDuong dựa trên các điều kiện thực tế
conditions = (
    (df['DuongHuyet'] > 126) & 
    (df['HbA1c'] > 6.5) & 
    (df['BMI'] > 27)
)
df['BiTieuDuong'] = conditions.astype(int)

# Lưu file
df.to_csv('data/raw/diabetes_data.csv', index=False)

# Hiển thị 5 dòng đầu tiên
print(df.head())
print("\nThống kê cơ bản:")
print(df.describe())
print("\nSố lượng bệnh nhân tiểu đường:", df['BiTieuDuong'].sum())
