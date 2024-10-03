import pandas as pd
import numpy as np

# ダミーデータの作成
np.random.seed(42)  # 再現性のためにシードを設定

num_records = 1000
person_ids = np.arange(1, num_records + 1)
ages = np.random.randint(18, 80, size=num_records)
genders = np.random.choice(['Male', 'Female', 'Other'], size=num_records, p=[0.49, 0.49, 0.02])
occupations = np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Farmer', 'Artist', 'Unemployed'], size=num_records)
incomes = np.random.normal(50000, 15000, size=num_records).astype(int)
regions = np.random.choice(['North', 'South', 'East', 'West'], size=num_records)

# データフレームの作成
df = pd.DataFrame({
    'person_id': person_ids,
    'age': ages,
    'gender': genders,
    'occupation': occupations,
    'income': incomes,
    'region': regions
})

# データをCSVファイルに保存（BigQueryにロードするため）
df.to_csv('census_data.csv', index=False)
