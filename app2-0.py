import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sns
import pipeline_dp

plt.rcParams['font.family'] = 'IPAexGothic'

# サービスアカウントキーのパスを設定
client = bigquery.Client.from_service_account_json('cryptic-pipe-435706-n0-e11d64723857.json')

def get_data(attributes):
    # 必要な属性を選択してデータを取得
    query = f"""
    SELECT person_id, {', '.join(attributes)}
    FROM   `cryptic-pipe-435706-n0.my_dataset.census_data`
    """
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    return df

def get_aggregated_data_pipeline_dp(epsilon, delta, attribute):
    df = get_data([attribute, 'person_id'])

    if attribute == 'income':
        # Incomeを範囲に分割
        bins = [0, 20000, 40000, 60000, 80000, 100000, float('inf')]
        labels = ['0 - 20,000', '20,001 - 40,000', '40,001 - 60,000',
                  '60,001 - 80,000', '80,001 - 100,000', '100,001以上']
        df['income_range'] = pd.cut(df['income'], bins=bins, labels=labels, right=True)
        attribute = 'income_range'

    data = df.to_dict('records')

    # プライバシー予算の設定
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=epsilon, total_delta=delta)

    # バックエンドの選択
    backend = pipeline_dp.LocalBackend()

    # DPEngineの初期化
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    # 集計パラメータの設定
    params = pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        metrics=[pipeline_dp.Metrics.COUNT],
        max_partitions_contributed=1,
        max_contributions_per_partition=1
    )

    # データエクストラクターの設定
    data_extractors = pipeline_dp.DataExtractors(
        privacy_id_extractor=lambda x: x['person_id'],
        partition_extractor=lambda x: x[attribute],
        value_extractor=lambda x: None
    )

    # 差分プライバシー集計の実行
    dp_result = dp_engine.aggregate(data, params, data_extractors)
    budget_accountant.compute_budgets()

    # 結果をDataFrameに変換
    dp_df = pd.DataFrame(dp_result, columns=[attribute, 'count'])

    # 'count'列から実際の数値を抽出
    dp_df['count'] = dp_df['count'].apply(lambda x: x.count)

    return dp_df

def get_average_income_pipeline_dp(epsilon, delta, group_by_attribute):
    df = get_data([group_by_attribute, 'income', 'person_id'])
    data = df.to_dict('records')

    # プライバシー予算の分割
    epsilon_sum = epsilon / 2
    epsilon_count = epsilon / 2
    delta_sum = delta / 2
    delta_count = delta / 2

    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=epsilon, total_delta=delta)

    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    # SUMの計算
    sum_params = pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        metrics=[pipeline_dp.Metrics.SUM],
        max_partitions_contributed=1,
        max_contributions_per_partition=1,
        min_value=0,
        max_value=100000  # 収入の最大値に合わせて調整
    )

    data_extractors_sum = pipeline_dp.DataExtractors(
        privacy_id_extractor=lambda x: x['person_id'],
        partition_extractor=lambda x: x[group_by_attribute],
        value_extractor=lambda x: x['income'] if x['income'] is not None else 0
    )

    dp_sum_result = dp_engine.aggregate(data, sum_params, data_extractors_sum)

    # COUNTの計算
    count_params = pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        metrics=[pipeline_dp.Metrics.COUNT],
        max_partitions_contributed=1,
        max_contributions_per_partition=1
    )

    data_extractors_count = pipeline_dp.DataExtractors(
        privacy_id_extractor=lambda x: x['person_id'],
        partition_extractor=lambda x: x[group_by_attribute],
        value_extractor=lambda x: None
    )

    dp_count_result = dp_engine.aggregate(data, count_params, data_extractors_count)

    budget_accountant.compute_budgets()

    # 結果をマージして平均を計算
    sum_df = pd.DataFrame(dp_sum_result, columns=[group_by_attribute, 'sum_income'])
    count_df = pd.DataFrame(dp_count_result, columns=[group_by_attribute, 'count'])

    # 'sum_income'と'count'から実際の数値を抽出
    sum_df['sum_income'] = sum_df['sum_income'].apply(lambda x: x.sum)
    count_df['count'] = count_df['count'].apply(lambda x: x.count)

    merged_df = pd.merge(sum_df, count_df, on=group_by_attribute)
    merged_df['average_income'] = merged_df['sum_income'] / merged_df['count']
    return merged_df[[group_by_attribute, 'average_income']]

def get_cross_tabulation_pipeline_dp(epsilon, delta, attribute1, attribute2):
    df = get_data([attribute1, attribute2, 'person_id'])
    data = df.to_dict('records')

    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=epsilon, total_delta=delta)
    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    params = pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        metrics=[pipeline_dp.Metrics.COUNT],
        max_partitions_contributed=1,
        max_contributions_per_partition=1
    )

    data_extractors = pipeline_dp.DataExtractors(
        privacy_id_extractor=lambda x: x['person_id'],
        partition_extractor=lambda x: (x[attribute1], x[attribute2]),
        value_extractor=lambda x: None
    )

    dp_result = dp_engine.aggregate(data, params, data_extractors)
    budget_accountant.compute_budgets()

    # タプルを個別の列に展開
    dp_df = pd.DataFrame(dp_result, columns=['attributes', 'count'])
    dp_df[attribute1] = dp_df['attributes'].apply(lambda x: x[0])
    dp_df[attribute2] = dp_df['attributes'].apply(lambda x: x[1])
    dp_df = dp_df.drop(columns=['attributes'])

    # 'count'列から実際の数値を抽出
    dp_df['count'] = dp_df['count'].apply(lambda x: x.count)

    return dp_df

def get_age_distribution_pipeline_dp(epsilon, delta):
    df = get_data(['age', 'person_id'])
    data = df.to_dict('records')

    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=epsilon, total_delta=delta)
    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    params = pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        metrics=[pipeline_dp.Metrics.COUNT],
        max_partitions_contributed=1,
        max_contributions_per_partition=1
    )

    data_extractors = pipeline_dp.DataExtractors(
        privacy_id_extractor=lambda x: x['person_id'],
        partition_extractor=lambda x: x['age'],
        value_extractor=lambda x: None
    )

    dp_result = dp_engine.aggregate(data, params, data_extractors)
    budget_accountant.compute_budgets()

    dp_df = pd.DataFrame(dp_result, columns=['age', 'count'])

    # 'count'列から実際の数値を抽出
    dp_df['count'] = dp_df['count'].apply(lambda x: x.count)

    return dp_df

def get_age_group_occupation_pipeline_dp(epsilon, delta):
    df = get_data(['age', 'occupation', 'person_id'])
    df['age_group'] = df['age'].apply(lambda x: '18-29' if 18 <= x <= 29 else
                                                  '30-39' if 30 <= x <= 39 else
                                                  '40-49' if 40 <= x <= 49 else
                                                  '50-59' if 50 <= x <= 59 else
                                                  '60+')
    data = df.to_dict('records')

    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=epsilon, total_delta=delta)
    backend = pipeline_dp.LocalBackend()
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    params = pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        metrics=[pipeline_dp.Metrics.COUNT],
        max_partitions_contributed=1,
        max_contributions_per_partition=1
    )

    data_extractors = pipeline_dp.DataExtractors(
        privacy_id_extractor=lambda x: x['person_id'],
        partition_extractor=lambda x: (x['age_group'], x['occupation']),
        value_extractor=lambda x: None
    )

    dp_result = dp_engine.aggregate(data, params, data_extractors)
    budget_accountant.compute_budgets()

    # タプルを個別の列に展開
    dp_df = pd.DataFrame(dp_result, columns=['attributes', 'count'])
    dp_df['age_group'] = dp_df['attributes'].apply(lambda x: x[0])
    dp_df['occupation'] = dp_df['attributes'].apply(lambda x: x[1])
    dp_df = dp_df.drop(columns=['attributes'])

    # 'count'列から実際の数値を抽出
    dp_df['count'] = dp_df['count'].apply(lambda x: x.count)

    return dp_df

def main():
    st.title("国勢調査データの差分プライバシー集計アプリ（pipeline_dp版）")

    st.write("このアプリでは、国勢調査データを差分プライバシーを適用して集計し、結果を表示します。pipeline_dpライブラリを使用しています。")

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

        # データの取得と差分プライバシー集計
        data = get_aggregated_data_pipeline_dp(epsilon, delta, attribute)

        if data.empty:
            st.warning("結果が抑制されました。プライバシー係数を増やすか、データのサイズを確認してください。")
            return

        st.write(f"差分プライバシーを適用した {attribute} 別の集計結果:")
        st.write(data)

        # グラフで表示
        plt.figure(figsize=(10, 6))
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

        data = get_average_income_pipeline_dp(epsilon, delta, group_by_attribute)

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
        data = get_cross_tabulation_pipeline_dp(epsilon, delta, attribute1, attribute2)

        if data.empty:
            st.warning("結果が抑制されました。プライバシー係数を増やすか、データのサイズを確認してください。")
            return

        st.write(f"差分プライバシーを適用した {attribute1} と {attribute2} のクロス集計結果:")
        st.write(data)

        # ピボットテーブルの作成
        pivot_table = data.pivot(index=attribute1, columns=attribute2, values='count').fillna(0)

        # 'count'が数値であることを確認
        pivot_table = pivot_table.astype(float)

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
        data = get_age_distribution_pipeline_dp(epsilon, delta)

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
        data = get_age_group_occupation_pipeline_dp(epsilon, delta)

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
