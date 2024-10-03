import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'IPAexGothic'

# サービスアカウントキーのパスを正しく設定
# client = bigquery.Client.from_service_account_json('cryptic-pipe-435706-n0-abed564ad043.json')
client = bigquery.Client.from_service_account_json('cryptic-pipe-435706-n0-e11d64723857.json')

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
            `cryptic-pipe-435706-n0.my_dataset.census_data`
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
            `cryptic-pipe-435706-n0.my_dataset.census_data`
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
            `cryptic-pipe-435706-n0.my_dataset.census_data`
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
            `cryptic-pipe-435706-n0.my_dataset.census_data`
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
        `cryptic-pipe-435706-n0.my_dataset.census_data`
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
        `cryptic-pipe-435706-n0.my_dataset.census_data`
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
            `cryptic-pipe-435706-n0.my_dataset.census_data`
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

    # 集計タイプの選択
    aggregation_type = st.selectbox(
        "集計タイプを選択してください",
        ('カウント', '平均所得', 'クロス集計', '年齢分布', '年齢層別職業分布')
    )

    # プライバシー係数の選択
    epsilon = st.slider("プライバシー係数 ε を選択してください", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    delta = 1e-5

    if aggregation_type == 'カウント':
        # ユーザーが集計したい属性を選択
        attribute = st.selectbox(
            "集計したい属性を選択してください",
            ('age', 'gender', 'occupation', 'income', 'region')
        )

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

    elif aggregation_type == '平均所得':
        group_by_attribute = st.selectbox(
            "グループ化する属性を選択してください",
            ('age', 'gender', 'occupation', 'region')
        )

        data = get_average_income(epsilon, delta, group_by_attribute)

        if data.empty or data['average_income'].isnull().all():
            st.warning("結果が抑制されました。プライバシー係数を増やすか、データのサイズを確認してください。")
            return

        st.write(f"差分プライバシーを適用した {group_by_attribute} 別の平均所得:")
        st.write(data)

        # グラフで表示
        plt.figure(figsize=(10, 6))
        x_labels = data[group_by_attribute].astype(str)
        plt.bar(x_labels, data['average_income'])
        plt.xlabel(group_by_attribute)
        plt.ylabel('平均所得（ノイズ追加後）')
        plt.title(f'ノイズが加えられた {group_by_attribute} 別の平均所得')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

    elif aggregation_type == 'クロス集計':
        attribute1 = st.selectbox(
            "1つ目の属性を選択してください",
            ('age', 'gender', 'occupation', 'region')
        )
        attribute2 = st.selectbox(
            "2つ目の属性を選択してください",
            ('age', 'gender', 'occupation', 'region')
        )
        if attribute1 == attribute2:
            st.warning("異なる属性を選択してください。")
            return
        data = get_cross_tabulation(epsilon, delta, attribute1, attribute2)

        if data.empty:
            st.warning("結果が抑制されました。プライバシー係数を増やすか、データのサイズを確認してください。")
            return

        st.write(f"差分プライバシーを適用した {attribute1} と {attribute2} のクロス集計結果:")
        st.write(data)

        # ピボットテーブルの作成
        pivot_table = data.pivot(index=attribute1, columns=attribute2, values='count').fillna(0)
        pivot_table = pivot_table.astype(float)  # ここで数値型に変換

        st.write("ピボットテーブル:")
        st.write(pivot_table)

        # ヒートマップの作成
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='Blues')
        plt.title(f'{attribute1} と {attribute2} のクロス集計ヒートマップ')
        plt.xlabel(attribute2)
        plt.ylabel(attribute1)
        plt.tight_layout()
        st.pyplot(plt)

    elif aggregation_type == '年齢分布':
        data = get_age_distribution(epsilon, delta)

        if data.empty:
            st.warning("結果が抑制されました。プライバシー係数を増やすか、データのサイズを確認してください。")
            return

        st.write("差分プライバシーを適用した年齢分布:")
        st.write(data)

        # ヒストグラムの作成
        plt.figure(figsize=(10, 6))
        plt.bar(data['age'], data['count'])
        plt.xlabel('年齢')
        plt.ylabel('人数（ノイズ追加後）')
        plt.title('ノイズが加えられた年齢分布')
        plt.tight_layout()
        st.pyplot(plt)

    elif aggregation_type == '年齢層別職業分布':
        data = get_age_group_occupation(epsilon, delta)

        if data.empty:
            st.warning("結果が抑制されました。プライバシー係数を増やすか、データのサイズを確認してください。")
            return

        st.write("差分プライバシーを適用した年齢層別職業分布:")
        st.write(data)

        # 可視化
        plt.figure(figsize=(12, 8))
        sns.barplot(x='age_group', y='count', hue='occupation', data=data)
        plt.title('年齢層別職業分布（ノイズ追加後）')
        plt.xlabel('年齢層')
        plt.ylabel('人数')
        plt.legend(title='職業', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(plt)

    else:
        st.warning("無効な集計タイプが選択されました。")

if __name__ == "__main__":
    main()