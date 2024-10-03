import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'IPAexGothic'

# サービスアカウントキーのパスを正しく設定
client = bigquery.Client.from_service_account_json('blissful-acumen-403212-abed564ad043.json')

def get_aggregated_data(epsilon, delta, attribute):
    if attribute == 'income':
        query = f"""
        SELECT WITH DIFFERENTIAL_PRIVACY
        OPTIONS (
            epsilon = {epsilon}, 
            delta = {delta}, 
            privacy_unit_column = person_id
        )
            CASE
                WHEN income < 20000 THEN '0 - 20,000'
                WHEN income BETWEEN 20000 AND 40000 THEN '20,001 - 40,000'
                WHEN income BETWEEN 40001 AND 60000 THEN '40,001 - 60,000'
                WHEN income BETWEEN 60001 AND 80000 THEN '60,001 - 80,000'
                WHEN income BETWEEN 80001 AND 100000 THEN '80,001 - 100,000'
                ELSE '100,001以上'
            END AS income_range,
            COUNT(*) AS count
        FROM
            `blissful-acumen-403212.my_dataset.census_data`
        GROUP BY
            income_range
        ORDER BY
            income_range
        """
    else:
        query = f"""
        SELECT WITH DIFFERENTIAL_PRIVACY
        OPTIONS (
            epsilon = {epsilon}, 
            delta = {delta}, 
            privacy_unit_column = person_id
        )
            {attribute},
            COUNT(*) AS count
        FROM
            `blissful-acumen-403212.my_dataset.census_data`
        GROUP BY
            {attribute}
        ORDER BY
            {attribute}
        """
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    return df
def get_average_income(epsilon, delta, group_by_attribute):
    # プライバシー予算をSUMとCOUNTに分割
    epsilon_sum = epsilon / 2
    epsilon_count = epsilon / 2
    delta_sum = delta / 2
    delta_count = delta / 2

    query = f"""
    WITH sum_data AS (
        SELECT WITH DIFFERENTIAL_PRIVACY
        OPTIONS (
            epsilon = {epsilon_sum}, 
            delta = {delta_sum}, 
            privacy_unit_column = person_id
        )
            {group_by_attribute},
            SUM(income) AS noisy_sum_income
        FROM
            `blissful-acumen-403212.my_dataset.census_data`
        GROUP BY
            {group_by_attribute}
    ),
    count_data AS (
        SELECT WITH DIFFERENTIAL_PRIVACY
        OPTIONS (
            epsilon = {epsilon_count}, 
            delta = {delta_count}, 
            privacy_unit_column = person_id
        )
            {group_by_attribute},
            COUNT(*) AS noisy_count
        FROM
            `blissful-acumen-403212.my_dataset.census_data`
        GROUP BY
            {group_by_attribute}
    )
    SELECT
        sum_data.{group_by_attribute},
        SAFE_DIVIDE(noisy_sum_income, noisy_count) AS average_income
    FROM
        sum_data
    JOIN
        count_data
    ON
        sum_data.{group_by_attribute} = count_data.{group_by_attribute}
    """
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    return df
def get_cross_tabulation(epsilon, delta, attribute1, attribute2):
    query = f"""
    SELECT WITH DIFFERENTIAL_PRIVACY
    OPTIONS (
        epsilon = {epsilon}, 
        delta = {delta}, 
        privacy_unit_column = person_id
    )
        {attribute1},
        {attribute2},
        COUNT(*) AS count
    FROM
        `blissful-acumen-403212.my_dataset.census_data`
    GROUP BY
        {attribute1}, {attribute2}
    """
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    return df
def get_age_distribution(epsilon, delta):
    query = f"""
    SELECT WITH DIFFERENTIAL_PRIVACY
    OPTIONS (
        epsilon = {epsilon}, 
        delta = {delta}, 
        privacy_unit_column = person_id
    )
        age,
        COUNT(*) AS count
    FROM
        `blissful-acumen-403212.my_dataset.census_data`
    GROUP BY
        age
    ORDER BY
        age
    """
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    return df
def get_age_group_occupation(epsilon, delta):
    query = f"""
    WITH age_groups AS (
        SELECT
            CASE
                WHEN age BETWEEN 18 AND 29 THEN '18-29'
                WHEN age BETWEEN 30 AND 39 THEN '30-39'
                WHEN age BETWEEN 40 AND 49 THEN '40-49'
                WHEN age BETWEEN 50 AND 59 THEN '50-59'
                ELSE '60+'
            END AS age_group,
            occupation,
            person_id
        FROM
            `blissful-acumen-403212.my_dataset.census_data`
    )
    SELECT WITH DIFFERENTIAL_PRIVACY
    OPTIONS (
        epsilon = {epsilon}, 
        delta = {delta}, 
        privacy_unit_column = person_id
    )
        age_group,
        occupation,
        COUNT(*) AS count
    FROM
        age_groups
    GROUP BY
        age_group, occupation
    """
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    return df

def main():
    st.title("国勢調査データの差分プライバシー集計アプリ")

    st.write("このアプリでは、国勢調査データを差分プライバシーを適用して集計し、結果を表示します。")

    # ユーザーが集計したい属性を選択
    attribute = st.selectbox(
        "集計したい属性を選択してください",
        ('age', 'gender', 'occupation', 'income', 'region')
    )

    # プライバシー係数を選択
    epsilon = st.slider("プライバシー係数 ε を選択してください（値が大きいほどノイズが少なくなります）", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    delta = 1e-5  # deltaは固定値

    # BigQueryからデータを取得
    data = get_aggregated_data(epsilon, delta, attribute)

    if data.empty:
        st.warning("結果が抑制されました。プライバシー係数を増やすか、データのサイズを確認してください。")
        return

    st.write(f"差分プライバシーを適用した {attribute} 別の集計結果:")
    st.write(data)

    # グラフで表示
    plt.figure(figsize=(10, 6))
    if attribute == 'income':
        x_labels = data['income_range']
        plt.bar(x_labels, data['count'])
        plt.xlabel('Income Range')
    else:
        x_labels = data[attribute].astype(str)
        plt.bar(x_labels, data['count'])
        plt.xlabel(attribute)
    plt.ylabel('人数（ノイズ追加後）')
    plt.title(f'ノイズが加えられた {attribute} 別の人数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    # ダウンロード用のCSVファイルを生成
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="集計結果をCSVでダウンロード",
        data=csv,
        file_name='aggregated_data.csv',
        mime='text/csv',
    )

if __name__ == "__main__":
    main()
