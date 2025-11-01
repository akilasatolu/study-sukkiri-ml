# 5-1
import pandas as pd
df = pd.read_csv('sukkiri-ml2-codes/datafiles/iris.csv')
# print(df.head(3))  # 先頭3行表示

# 5-2
# print(df['種類'].unique())  # 種類の列のユニークな値を取得

# 5-3
# syurui = df['種類'].unique()
# print(syurui[0])  # 1つ目の種類を表示

# 5-4
# print(df['種類'].value_counts()) # 各種類の出現回数を表示

# 5-5
# print(df.tail(3))  # 末尾3行表示

# 5-6
# print(df.isnull())  # 欠損値の確認
# print(df.isnull().sum())  # 各列の欠損値の合計を表示

# 5-7
# print(df.isnull().any(axis = 0))  # 各列に欠損値があるか確認

# 5-8
# print(df.sum())  # 各列の合計を表示

# 5-9
# print(df.isnull().sum())  # 各列の欠損値の合計を表示

# 5-10
# df2 = df.dropna(how = 'any', axis = 0)  # 欠損値を含む行を削除
# print(df2.tail(3))  # 末尾3行表示

# 5-11
# print(df.isnull().any(axis = 0))  # 各列に欠損値があるか確認

# 5-12
# df['花弁長さ'] = df['花弁長さ'].fillna(0)  # 花弁長さの欠損値を0で埋める
# print(df.tail(3))  # 末尾3行表示

# 5-13
# print(df.mean(numeric_only=True))  # 各列の平均を表示

# 5-14
# print(df['がく片長さ'].mean())  # がく片長さの平均を表示

# 5-15
# print(df.select_dtypes(include='number').std())  # 各列の標準偏差を表示

# 5-16
colmean = df.mean(numeric_only=True)  # 各列の平均を取得
df2 = df.fillna(colmean)  # 各列の欠損値を列の平均で埋める
# print(df2.isnull().any(axis = 0))  # 各列に欠損値があるか確認

# 5-17
xcol = ['がく片長さ', 'がく片幅', '花弁長さ', '花弁幅']  # 説明変数の列名リスト
x = df2[xcol]
t = df2['種類']  # 目的変数の列

# 5-18
from sklearn import tree
model = tree.DecisionTreeClassifier(random_state=0, max_depth=2) # 決定木モデルの作成

# 5-19
# model.fit(x, t)  # モデルの学習
# print(model.score(x, t))  # 学習データに対する正解率を表示

# 5-20
from sklearn.model_selection import train_test_split
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)  # 訓練データとテストデータに分割

# 5-21
# print(x_train.shape)  # 訓練データの説明変数の形状を表示
# print(x_test.shape)   # テストデータの説明変数の形状を

# 5-22
model.fit(x_train, t_train)  # モデルの学習
# print(model.score(x_test, t_test))  # テストデータに対する正解率を表示

# 5-23
# import pickle
# with open('models/5-23.pkl', 'wb') as f:
#     pickle.dump(model, f)

# 5-24
# print(model.tree_.feature) # 各ノードで使用された特徴量のインデックスを表示

# 5-25
# print(model.tree_.threshold) # 各ノードでの閾値を表示

# 5-26
# print(model.tree_.value[1]) # ノード1のクラスごとのサンプル数を表示
# print(model.tree_.value[3]) # ノード3のクラスごとのサンプル数を表示
# print(model.tree_.value[4]) # ノード4のクラスごとのサンプル数を表示

# 5-27
# print(model.classes_)  # クラスの一覧を表示

# 5-28
# sklearn version0.21以降
x_train.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  # 説明変数の列名を英語に変更
from sklearn import plot_tree
plot_tree(model, feature_names=x_train.columns, filled=True)
