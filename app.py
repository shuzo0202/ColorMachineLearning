import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import plotly.graph_objects as go
import os
import datetime
import shutil

# モデルとその学習データを保存する関数
def save_model_with_data(model, feature_cols, training_data, filename="model_with_data.pkl"):
    """
    モデル、特徴量列、および学習に使用したデータを一緒に保存する
    
    Parameters:
    model: 学習済みモデル
    feature_cols: 特徴量の列名リスト
    training_data: 学習に使用したデータフレーム
    filename: 保存先ファイル名
    """
    # アーカイブディレクトリの確認と作成
    archive_dir = "model_archives"
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    
    # 既存のモデルファイルがある場合はアーカイブフォルダに移動
    if os.path.exists(filename):
        # タイムスタンプを付けてファイル名を作成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_filename = f"{archive_dir}/model_data_{timestamp}.pkl"
        
        # ファイルをコピー
        shutil.copy2(filename, archive_filename)
        st.info(f"既存のモデルと学習データをアーカイブしました: {archive_filename}")
    
    # 新しいモデルとデータを保存
    with open(filename, "wb") as f:
        pickle.dump((model, feature_cols, training_data), f)
    st.success(f"新しいモデルと学習データを {filename} として保存しました。")
    
    # セッション状態にも保存
    st.session_state['model'] = model
    st.session_state['feature_columns'] = feature_cols
    st.session_state['training_data'] = training_data

# モデルを更新する強化学習版関数
def update_model_with_reinforcement(new_data, add_noise=False, noise_count=5):
    """
    既存のモデルと学習データに新しいデータを追加して再学習する強化学習バージョン
    
    Parameters:
    new_data: 新しく追加する学習データ
    add_noise: ノイズデータを追加するかどうか
    noise_count: 追加するノイズデータの個数
    
    Returns:
    model: 更新されたモデル
    feature_cols: 特徴量列名リスト
    """
    # 古いモデルと学習データを読み込む
    try:
        with open("model_with_data.pkl", "rb") as f:
            _, _, old_training_data = pickle.load(f)
        st.info(f"既存の学習データ {len(old_training_data)} 件を読み込みました。")
    except:
        st.warning("既存の学習データが見つからないため、新しいデータだけで学習します。")
        old_training_data = pd.DataFrame()
    
    # 新しいデータを整形
    # 新しいDataFrameを作成（必要な列だけを抽出）
    train_data = {
        'L': new_data['L'].values,
        'a': new_data['a'].values,
        'b': new_data['b'].values,
        '光沢感': new_data['光沢感'].values,
        '主観の色分類': new_data['予測色分類'].values  # 予測色分類の値を主観の色分類として使用
    }
    new_training_data = pd.DataFrame(train_data)
    
    # 古いデータと新しいデータを結合
    if not old_training_data.empty:
        # 古いデータと新しいデータに同じ列があることを確認
        new_training_data = new_training_data[['L', 'a', 'b', '光沢感', '主観の色分類']]
        old_training_data = old_training_data[['L', 'a', 'b', '光沢感', '主観の色分類']]
        
        # データを結合
        combined_data = pd.concat([old_training_data, new_training_data], ignore_index=True)
    else:
        combined_data = new_training_data
    
    st.info(f"古いデータ {len(old_training_data)} 件と新しいデータ {len(new_training_data)} 件を結合し、合計 {len(combined_data)} 件のデータで学習します。")
    
    # 欠損値の処理
    combined_data['光沢感'] = combined_data['光沢感'].fillna("None")
    
    # 数値列の欠損値をチェックして処理
    numerical_cols = ['L', 'a', 'b']
    if combined_data[numerical_cols].isna().any().any():
        st.warning("Lab値に欠損値があります。平均値で補完します。")
        for col in numerical_cols:
            combined_data[col] = combined_data[col].fillna(combined_data[col].mean())
    
    # ノイズデータ生成（オプション）
    if add_noise:
        st.info(f"データ拡張中: 各データポイントに対して {noise_count} 個のノイズ付きデータを生成しています...")
        
        # ノイズデータを作成する前に、インデックスをリセット
        combined_data = combined_data.reset_index(drop=True)
        
        augmented_list = []
        for idx, row in combined_data.iterrows():
            for i in range(int(noise_count)):
                new_row = row.copy()
                new_row['L'] = row['L'] + np.random.uniform(-0.5, 0.5)
                new_row['a'] = row['a'] + np.random.uniform(-0.5, 0.5)
                new_row['b'] = row['b'] + np.random.uniform(-0.5, 0.5)
                augmented_list.append(new_row)
        
        # 拡張データを結合
        if augmented_list:  # リストが空でないことを確認
            df_aug = pd.DataFrame(augmented_list)
            combined_data = pd.concat([combined_data, df_aug], ignore_index=True)
            st.success(f"データ拡張完了: {len(combined_data)} 行のデータでモデルを更新します")
    
    # モデル学習用のデータを準備
    X = combined_data[['L', 'a', 'b', '光沢感']]
    X = pd.get_dummies(X, columns=['光沢感'])
    y = combined_data['主観の色分類']
    
    # 特徴量カラムを保存
    feature_cols = X.columns.tolist()
    
    # モデル学習
    new_model = RandomForestClassifier(random_state=42)
    new_model.fit(X, y)
    
    # モデルと学習データを保存
    save_model_with_data(new_model, feature_cols, combined_data)
    
    return new_model, feature_cols

# モデルと学習データを読み込む関数
def load_model_with_data():
    """
    保存済みのモデルと学習データを読み込む
    
    Returns:
    bool: 読み込みが成功したかどうか
    """
    try:
        with open("model_with_data.pkl", "rb") as f:
            model, feature_cols, training_data = pickle.load(f)
        st.session_state['model'] = model
        st.session_state['feature_columns'] = feature_cols
        st.session_state['training_data'] = training_data
        st.info("保存済みのモデルと学習データを読み込みました。")
        return True
    except Exception as e:
        st.error("学習済みモデルが見つかりません。まずは『モデル学習』から実行してください。")
        return False

st.set_page_config(page_title="色分類モデルアプリ", layout="wide")
st.title("色分類モデル作成・予測アプリ")

# セッション状態の初期化（必要な変数を保持するため）
if 'current_menu' not in st.session_state:
    st.session_state['current_menu'] = "モデル学習"
if 'train_df' not in st.session_state:
    st.session_state['train_df'] = None
if 'test_df' not in st.session_state:
    st.session_state['test_df'] = None
if 'edited_df' not in st.session_state:
    st.session_state['edited_df'] = None
if 'corr_df' not in st.session_state:
    st.session_state['corr_df'] = None
if 'lab_df' not in st.session_state:
    st.session_state['lab_df'] = None
if 'training_data' not in st.session_state:
    st.session_state['training_data'] = None
if 'focused_classes_list' not in st.session_state:
    # すべての色分類に初期化
    st.session_state['focused_classes_list'] = [
        "高白色", "白", "ナチュラル", "黒", "グレー", 
        "赤", "オレンジ", "茶", "黄色", "緑", 
        "青", "紫", "ピンク", "金", "銀"
    ]

# サイドバーで機能を選択し、セッション状態に保存
menu_option = st.sidebar.radio("メニュー", ["モデル学習", "色分類予測", "修正・モデル更新", "Lab色空間プロット"],
                              index=["モデル学習", "色分類予測", "修正・モデル更新", "Lab色空間プロット"].index(st.session_state['current_menu']))
st.session_state['current_menu'] = menu_option

# ----- 1. モデル学習 -----
if menu_option == "モデル学習":
    st.header("モデル学習")
    
    st.markdown("### 学習データの読み込み")
    st.write("※学習データCSVをアップロードしない場合は、内蔵サンプルデータを使用します。")
    
    train_file = st.file_uploader("学習データCSVをアップロード（任意）", type=["csv"], key="train_upload")
    if train_file is not None:
        try:
            df_train = pd.read_csv(train_file)
            st.session_state['train_df'] = df_train  # セッションに保存
        except Exception as e:
            st.error("CSVの読み込みエラー: " + str(e))
    elif st.session_state['train_df'] is not None:
        df_train = st.session_state['train_df']  # セッションから復元
    else:
        # サンプルデータ（例1・例2）
        sample_data = {
            "商品コード": [123456, 542134],
            "上位銘柄コード": [1234567, 3204578],
            "銘柄名": ["NTラシャ", "オフメタル"],
            "色名": ["無垢", "金"],
            "主観の色分類": ["白", "金"],
            "L": [99.24, 95.75],
            "a": [0.36, 12.47],
            "b": [-10.64, 37.55],
            "光沢感": [np.nan, "メタリック"]
        }
        df_train = pd.DataFrame(sample_data)
        st.session_state['train_df'] = df_train  # セッションに保存
    
    st.subheader("学習データ")
    st.dataframe(df_train)
    
    # 学習データ拡張のチェックボックス状態を保持
    if 'noise_checkbox' not in st.session_state:
        st.session_state['noise_checkbox'] = False
    if 'noise_count' not in st.session_state:
        st.session_state['noise_count'] = 5
    
    st.markdown("### データ拡張（Lab値にノイズ追加）")
    
    # チェックボックスの状態をセッションに保存
    noise_checkbox = st.checkbox("Lab値にノイズを追加（データ拡張）", value=st.session_state['noise_checkbox'])
    st.session_state['noise_checkbox'] = noise_checkbox
    
    if noise_checkbox:
        noise_count = st.number_input("追加するノイズの個数", min_value=1, max_value=10, value=st.session_state['noise_count'], step=1)
        st.session_state['noise_count'] = noise_count
        
        # ノイズ追加処理
        if st.button("ノイズ追加を実行"):
            augmented_list = []
            for idx, row in df_train.iterrows():
                for i in range(int(noise_count)):
                    new_row = row.copy()
                    new_row['L'] = row['L'] + np.random.uniform(-0.5, 0.5)
                    new_row['a'] = row['a'] + np.random.uniform(-0.5, 0.5)
                    new_row['b'] = row['b'] + np.random.uniform(-0.5, 0.5)
                    augmented_list.append(new_row)
            df_aug = pd.DataFrame(augmented_list)
            df_train = pd.concat([df_train, df_aug], ignore_index=True)
            st.session_state['train_df'] = df_train  # 更新されたデータをセッションに保存
            st.write("ノイズ追加後の学習データ:")
            st.dataframe(df_train)
    
    st.markdown("### モデル学習")
    if st.button("モデルを学習する"):
        # 既存のモデルと学習データを読み込む (あれば)
        existing_data = pd.DataFrame()
        try:
            with open("model_with_data.pkl", "rb") as f:
                _, _, existing_data = pickle.load(f)
            st.info(f"既存の学習データ {len(existing_data)} 件を読み込みました。")
        except:
            st.info("新規モデルを作成します。")
        
        # 現在の学習データを処理
        df_train['光沢感'] = df_train['光沢感'].fillna("None")
        
        # 既存データがあれば結合
        if not existing_data.empty:
            combine_data = st.checkbox("既存の学習データと結合する", value=True)
            if combine_data:
                # データの整形と結合
                current_data = df_train[['L', 'a', 'b', '光沢感', '主観の色分類']]
                combined_data = pd.concat([existing_data, current_data], ignore_index=True)
                st.success(f"既存データと結合し、合計 {len(combined_data)} 件のデータで学習します。")
                df_train = combined_data
        
        # 特徴量：Lab値と光沢感（光沢感はダミー変数化）
        X = df_train[['L', 'a', 'b', '光沢感']]
        X = pd.get_dummies(X, columns=['光沢感'])
        # 目的変数：主観の色分類
        y = df_train['主観の色分類']
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        feature_cols = X.columns.tolist()
        
        # モデルとデータを保存
        save_model_with_data(model, feature_cols, df_train)
        
# ----- 2. 色分類予測 -----
elif menu_option == "色分類予測":
    st.header("色分類予測")
    
    # モデルがセッションに未登録なら、model_with_data.pkl から読み込み
    if 'model' not in st.session_state:
        load_model_with_data()

    st.markdown("### 色分類予測用商品のデータアップロード")
    test_file = st.file_uploader("色分類をしたい商品のCSVをアップロード", type=["csv"], key="test_upload")

    # テストデータの処理
    df_test = None
    if test_file is not None:
        try:
            df_test = pd.read_csv(test_file)
            st.session_state['test_df'] = df_test  # セッションに保存
        except Exception as e:
            st.error("CSVの読み込みエラー: " + str(e))
    elif st.session_state.get('test_df') is not None:
        df_test = st.session_state['test_df']  # セッションから復元
        st.info("前回アップロードされたデータを使用しています。")

    if df_test is not None:
        st.subheader("アップロードされたデータ")
        st.dataframe(df_test)

        # 予測処理
        if 'predicted_test_df' not in st.session_state or st.button("予測を実行"):
            if 'model' in st.session_state:
                # 光沢感の欠損値を "None" で補完
                df_test['光沢感'] = df_test['光沢感'].fillna("None")

                # 特徴量をダミー変数化して整形
                X_test = df_test[['L', 'a', 'b', '光沢感']]
                X_test = pd.get_dummies(X_test, columns=['光沢感'])
                X_test = X_test.reindex(columns=st.session_state['feature_columns'], fill_value=0)

                # 予測確率を取得
                model = st.session_state['model']
                predictions_proba = model.predict_proba(X_test)
                predictions_classes = model.classes_

                # 1番目の予測を追加
                df_test['予測色分類'] = predictions_classes[predictions_proba.argmax(axis=1)]
                df_test['予測確率'] = predictions_proba.max(axis=1).round(4) * 100  # パーセント表示

                # 2番目の予測を追加
                second_pred_indices = np.argsort(-predictions_proba, axis=1)[:, 1]  # 2番目に高い確率のインデックス
                df_test['予測色分類2'] = [predictions_classes[idx] for idx in second_pred_indices]
                df_test['予測確率2'] = [predictions_proba[i, idx].round(4) * 100 for i, idx in enumerate(second_pred_indices)]

                st.session_state['predicted_test_df'] = df_test
                st.session_state['predictions_proba'] = predictions_proba
                st.session_state['predictions_classes'] = predictions_classes
            else:
                st.error("モデルが読み込まれていません。「モデル学習」ページで学習を実行してください。")
        else:
            # 既にセッションにある場合はそれを利用
            df_test = st.session_state['predicted_test_df']

        # 過去に編集した結果があればそれを反映
        if st.session_state.get('edited_df') is not None:
            df_test = st.session_state['edited_df']

        # 色の選択肢リスト
        colors = [
            '高白色', '白', 'ナチュラル', '黒', 'グレー', 
            '赤', 'オレンジ', '茶', '黄色', '緑', 
            '青', '紫', 'ピンク', '金', '銀'
        ]

        st.subheader("予測結果（編集可能）")
        st.markdown("※上位2つの予測色分類とその確率を表示しています。「予測色分類」列を編集して修正できます。")
        
        edited_df = st.data_editor(
            data=df_test,
            column_config={
                "予測色分類": st.column_config.SelectboxColumn(
                    "予測色分類",
                    help="予測色分類を必要に応じて修正してください",
                    options=colors,
                    default=None
                ),
                "予測確率": st.column_config.ProgressColumn(
                    "予測確率 (%)",
                    help="第1予測の確率",
                    format="%.1f",
                    min_value=0,
                    max_value=100
                ),
                "予測色分類2": st.column_config.TextColumn(
                    "予測色分類2", 
                    help="第2候補の予測色分類"
                ),
                "予測確率2": st.column_config.ProgressColumn(
                    "予測確率2 (%)",
                    help="第2予測の確率",
                    format="%.1f",
                    min_value=0,
                    max_value=100
                )
            },
            disabled=[col for col in df_test.columns if col != "予測色分類"],
            use_container_width=True,
            key="prediction_editor"
        )
        st.session_state['edited_df'] = edited_df  # 編集結果をセッションに保存
        st.session_state['predicted_test_df'] = edited_df  # 編集内容も保持

        
        # 結果のダウンロードボタン
        csv_data = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button("予測結果をCSVでダウンロード", data=csv_data,
                          file_name="predicted_colors.csv", mime="text/csv")
        
        # モデル更新機能
        st.markdown("### 修正を反映してモデルを更新")
        
        # データ拡張（ノイズ追加）オプションの状態保持
        add_noise = st.checkbox("Lab値にノイズを追加（データ拡張）", key="add_noise_prediction")
        
        noise_count = 5  # デフォルト値
        if add_noise:
            noise_count = st.number_input("追加するノイズの個数", min_value=1, max_value=10, 
                                         value=5, step=1, key="noise_count_prediction")
        
        # 確率の閾値設定（情報表示用）
        confidence_threshold = st.slider(
            "再学習に使用する予測の信頼度閾値 (%)", 
            min_value=0, 
            max_value=100, 
            value=50, 
            step=5,
            help="※ここで設定した閾値は情報表示のみに使用されます。修正した予測は閾値に関わらず全て使用されます。"
        )
        
        if st.button("修正を反映してモデル更新"):
            try:
                df_update = edited_df.copy()
                
                # 確率が閾値以上のデータを表示（情報表示のみ）
                high_conf_mask = df_update['予測確率'] >= confidence_threshold
                high_conf_count = high_conf_mask.sum()
                total_count = len(df_update)
                
                st.info(f"全 {total_count} 件のうち、信頼度が{confidence_threshold}%以上の予測は {high_conf_count} 件です。")
                
                # 予測色分類に欠損値があるデータを除外
                has_prediction = df_update['予測色分類'].notna()
                if not has_prediction.all():
                    num_missing = (~has_prediction).sum()
                    st.warning(f"予測色分類が欠損している {num_missing} 件のデータは除外します。")
                    df_update = df_update[has_prediction]
                    
                    if len(df_update) == 0:
                        st.error("有効なデータがありません。処理を中止します。")
                        proceed_with_update = False
                    else:
                        proceed_with_update = True
                else:
                    proceed_with_update = True
                
                if proceed_with_update:
                    # 強化学習モードでモデルを更新
                    new_model, feature_cols = update_model_with_reinforcement(
                        df_update, 
                        add_noise=add_noise,
                        noise_count=noise_count
                    )
                    
                    # 予測結果をリセットして次回予測時に最新モデルを使用させる
                    if 'predicted_test_df' in st.session_state:
                        del st.session_state['predicted_test_df']
                        
                    st.success("モデルが強化学習的に更新されました。次回の予測実行時に最新モデルが使用されます。")
            except Exception as e:
                st.error(f"モデル更新中にエラーが発生しました: {str(e)}")
                import traceback
                st.error(traceback.format_exc())  # デバッグ用に詳細なエラー情報を表示

# ----- 3. 修正・モデル更新 -----
elif menu_option == "修正・モデル更新":
    st.header("修正・モデル更新")
    
    st.markdown("### 予測結果の手動修正")
    st.markdown("※アップロードされた予測結果CSVの「主観の色分類」欄のみ編集できます。")
    correction_file = st.file_uploader("修正する予測結果CSVをアップロード", type=["csv"], key="correction_upload")
    if correction_file is not None:
        try:
            df_corr = pd.read_csv(correction_file)
        except Exception as e:
            st.error("CSVの読み込みエラー: " + str(e))
        else:
            # 色の選択肢リスト
            colors = [
                '高白色', '白', 'ナチュラル', '黒', 'グレー', 
                '赤', 'オレンジ', '茶', '黄色', '緑', 
                '青', '紫', 'ピンク', '金', '銀'
            ]
            
            st.subheader("予測結果（編集前）")
            st.dataframe(df_corr)
            
            # 編集不可の列リストを作成（主観の色分類以外のすべての列）
            disabled_columns = [col for col in df_corr.columns if col != "主観の色分類"]
            
            # カラム設定で主観の色分類をセレクトボックスにし、他の列は編集不可に
            st.subheader("編集テーブル")
            edited_df = st.data_editor(
                data=df_corr,
                column_config={
                    "主観の色分類": st.column_config.SelectboxColumn(
                        "主観の色分類",
                        help="正しい色分類を選択してください",
                        options=colors,
                        default=None
                    )
                },
                disabled=disabled_columns,
                use_container_width=True,
                key="prediction_editor"
            )
            
            
            st.markdown("### 修正を反映してモデルを更新")
            if st.button("修正を反映してモデル更新"):
                try:
                    df_update = edited_df.copy()
                    df_update = df_update.rename(columns={"予測色分類": "主観の色分類"})
                    df_update = df_update[['L', 'a', 'b', '光沢感', '主観の色分類']]
                    df_update['光沢感'] = df_update['光沢感'].fillna("None")
                    X_new = df_update[['L', 'a', 'b', '光沢感']]
                    X_new = pd.get_dummies(X_new, columns=['光沢感'])
                    X_new = X_new.reindex(columns=st.session_state['feature_columns'], fill_value=0)
                    y_new = df_update['主観の色分類']
                    
                    new_model = RandomForestClassifier(random_state=42)
                    new_model.fit(X_new, y_new)
                    
                    st.success("モデルが更新されました。")
                    st.session_state['model'] = new_model
                    
                    with open("model_updated.pkl", "wb") as f:
                        pickle.dump((new_model, st.session_state['feature_columns']), f)
                    st.info("更新されたモデルが model_updated.pkl として保存されました。")
                except Exception as e:
                    st.error("モデル更新中にエラーが発生しました: " + str(e))
# ----- 4. Lab色空間プロット -----
elif menu_option == "Lab色空間プロット":
    st.header("Lab色空間上に色分類をプロット")
    
    # サイドバー：データ入力
    st.sidebar.subheader("データ入力")
    upload_option = st.sidebar.radio(
        "データソースの選択",
        ["サンプルデータを使用", "CSVファイルをアップロード"],
        key="lab_data_source"
    )
    
    def create_sample_data_lab():
        data = {
            "商品コード": ["123456", "542134", "354812"],
            "上位銘柄コード": ["1234567", "3204578", "6401548"],
            "銘柄名": ["NTラシャ", "オフメタル", "タント"],
            "色名": ["無垢", "金", "N-5"],
            "主観の色分類": ["白", "金", "ナチュラル"],
            "L": [99.24, 95.75, 75.21],
            "a": [0.36, 12.47, 4.18],
            "b": [-10.64, 37.55, 1.02],
            "光沢感": [None, "メタリック", None]
        }
        return pd.DataFrame(data)
    
    if upload_option == "サンプルデータを使用":
        df_lab = create_sample_data_lab()
    else:
        uploaded_file_lab = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"], key="lab_csv")
        if uploaded_file_lab is not None:
            try:
                df_lab = pd.read_csv(uploaded_file_lab)
            except Exception as e:
                st.error("CSV読み込みエラー: " + str(e))
                df_lab = create_sample_data_lab()
        else:
            df_lab = create_sample_data_lab()
    
    st.subheader("データプレビュー")
    st.dataframe(df_lab)
    
    # 使用する色分類の列を決定（予測結果があれば "予測色分類"、なければ "主観の色分類"）
    if "予測色分類" in df_lab.columns:
        color_col = "予測色分類"
    else:
        color_col = "主観の色分類"
    st.write("使用する色分類の列: ", color_col)
    
    # 色分類 → マーカー色 のマッピング辞書（15種類）
    color_mapping = {
        "高白色": "rgb(255,255,255)",
        "白": "rgb(230,230,200)",
        "ナチュラル": "rgb(222,184,135)",
        "黒": "rgb(0,0,0)",
        "グレー": "rgb(128,128,128)",
        "赤": "rgb(255,0,0)",
        "オレンジ": "rgb(255,165,0)",
        "茶": "rgb(165,42,42)",
        "黄色": "rgb(255,255,100)",
        "緑": "rgb(0,128,0)",
        "青": "rgb(0,0,255)",
        "紫": "rgb(128,0,128)",
        "ピンク": "rgb(255,192,203)",
        "金": "rgb(255,215,0)",
        "銀": "rgb(192,192,192)"
    }
    
    # 各行の色分類に応じたマーカー色を設定（定義がない場合はグレー）
    df_lab["marker_color"] = df_lab[color_col].apply(lambda x: color_mapping.get(x, "rgb(128,128,128)"))
    
    # サイドバー：注目する色分類の選択（これ以外は透明度を0.1で表示）
    focused_classes = st.sidebar.multiselect(
        "注目する色分類",
        options=list(color_mapping.keys()),
        default=list(color_mapping.keys())
    )
    
    # サイドバー：Lab値フィルターの設定
    st.sidebar.subheader("Labフィルター")
    L_range = st.sidebar.slider("L (明度)", 0.0, 110.0, (0.0, 110.0), key="L_range_lab")
    a_range = st.sidebar.slider("a (-方向: 緑 / +方向: 赤)", -110.0, 110.0, (-110.0, 110.0), key="a_range_lab")
    b_range = st.sidebar.slider("b (-方向: 青 / +方向: 黄)", -110.0, 110.0, (-110.0, 110.0), key="b_range_lab")
    
    # フィルター後のデータ抽出
    df_filtered = df_lab[
        df_lab["L"].between(L_range[0], L_range[1]) &
        df_lab["a"].between(a_range[0], a_range[1]) &
        df_lab["b"].between(b_range[0], b_range[1])
    ]
    st.sidebar.markdown("**フィルター後のデータ件数: **" + str(len(df_filtered)))
    
    # 3Dプロットの作成
    fig = go.Figure()
    
    # 色分類ごとにグループ化してプロット
    for cls, group in df_filtered.groupby(color_col):
        # 注目している色分類であれば不透明度1.0、そうでなければ0.1
        opacity_value = 1.0 if cls in focused_classes else 0.1
        fig.add_trace(go.Scatter3d(
            x=group["a"],
            y=group["b"],
            z=group["L"],
            mode="markers",
            marker=dict(
                size=8,
                color=group["marker_color"].tolist(),
                opacity=opacity_value,
                line=dict(width=1, color='gray')
            ),
            name=str(cls),
            text=[f"{row['銘柄名']} {row['色名']} ({row[color_col]})" for idx, row in group.iterrows()],
            hovertemplate="銘柄名: %{text}<br>L: %{z:.2f}<br>a: %{x:.2f}<br>b: %{y:.2f}<extra></extra>"
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="a*", range=[-110, 110]),
            yaxis=dict(title="b*", range=[-110, 110]),
            zaxis=dict(title="L*", range=[0, 110])
        ),
        width=800,
        height=800,
        margin=dict(l=10, r=10, b=10, t=10)
    )
    
    st.subheader("Lab色空間の3Dプロット")
    st.plotly_chart(fig, use_container_width=True)
