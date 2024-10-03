import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sns
import pipeline_dp
from pandas_gbq import to_gbq

plt.rcParams['font.family'] = 'IPAexGothic'

# サービスアカウントキーのパスを設定
client = bigquery.Client.from_service_account_json('your-service-account-key.json')

def main():
    st.title("国勢調査データの差分プライバシー集計アプリ（アップロード機能付き）")

    st.write("このアプリでは、CSVファイルをアップロードして、差分プライバシーを適用した集計を行います。")

    # ファイルアップロード
    uploaded_file = st.file_uploader("データファイルをアップロードしてください（CSV形式）", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("アップロードされたデータ:")
        st.write(df.head())

        # BigQueryにアップロード
        project_id = 'your-project-id'
        dataset_id = 'your_dataset'
        table_id = 'uploaded_data'  # テーブル名は任意

        # データセットの存在を確認し、存在しない場合は作成
        dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
        try:
            client.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"  # データセットのロケーションを指定
            client.create_dataset(dataset)
            st.success(f"データセット {dataset_id} を作成しました。")

        # テーブルにデータをアップロード
        to_gbq(df, f'{dataset_id}.{table_id}', project_id=project_id, if_exists='replace')
        st.success("データがBigQueryにアップロードされました。")

        # プライバシー係数の選択
        epsilon = st.slider("プライバシー係数 ε を選択してください", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        delta = 1e-5

        # 集計タイプの選択
        aggregation_type = st.selectbox(
            "集計タイプを選択してください",
            ('カウント', '平均所得', 'クロス集計', '年齢分布', '年齢層別職業分布')
        )

        # 以下、既存の集計処理を実行
        # get_data関数を修正して、アップロードしたデータを使用するようにします
        def get_data(attributes):
            query = f"""
            SELECT person_id, {', '.join(attributes)}
            FROM `{project_id}.{dataset_id}.{table_id}`
            """
            query_job = client.query(query)
            result = query_job.result()
            df = result.to_dataframe()
            return df

        # 以降、get_data関数を使用してデータを取得し、差分プライバシーの集計を実行

        # ...（既存の集計関数と可視化処理をここに追加）

    else:
        st.info("データファイルをアップロードしてください。")

if __name__ == "__main__":
    main()
