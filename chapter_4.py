# 4-1
# import pandas as pd


# 4-2
# data={
#     '身長': [170, 180, 160, 155, 175],
#     '体重': [70, 80, 60, 50, 75],
#     '年齢': [30, 40, 20, 25, 35],
# }
# df = pd.DataFrame(data)
# print(df)


# 4-3
# print(type(df))
# print(df.shape)


# 4-4
# df.index = ['Aさん', 'Bさん', 'Cさん', 'Dさん', 'Eさん']
# print(df)


# 4-5
# df.columns = ['height', 'weight', 'age']
# print(df)


# 4-6
# print(df.index)
# print(df.columns)
# print(df['height'])


# 4-7
# data2 = [
#     [165, 65, 28],
#     [185, 85, 45],
# ]
# df2 = pd.DataFrame(data2, index=['Fさん', 'Gさん'], columns=['height', 'weight', 'age'])
# print(df2)


# 4-8
# df_KvsT = pd.read_csv('sukkiri-ml2-codes/datafiles/KvsT.csv')
# print(df_KvsT)
# print(df_KvsT.head())  # 先頭5行表示
# print(df_KvsT.tail())  # 末尾5行表示
# print(df_KvsT.head(3))  # 先頭3行表示


# 4-9
# print(df_KvsT['派閥'])  # 体重の列を取得


# 4-10
# col_list = ['年代', '派閥']
# print(df_KvsT[col_list])  # 複数列の取得


# 4-11
# print(type(df_KvsT['年代']))  # Series型
# print(type(df_KvsT[col_list]))  # DataFrame型


# 4-12
# print(df_KvsT['派閥'])


# 4-13
# x_cols = ['身長', '体重', '年代']
# x = df_KvsT[x_cols]
# print(x)


# 4-14
# t = df_KvsT['派閥']
# print(t)


# 4-15
# from sklearn import tree


#4-16
# モデルの準備
# random_state=0で乱数のシード値を固定
# model = tree.DecisionTreeClassifier(random_state=0)
# 学習
# model.fit(x, t)


# 4-17
# 予測
# new_data = [[170, 70, 20]]
# new_df = pd.DataFrame(new_data, columns=x_cols)
# predicted = model.predict(new_df)
# print(predicted)  # 予測結果の表示


# 4-18
# new_data = [[170, 70, 20],[158,48,20]]
# new_df = pd.DataFrame(new_data, columns=x_cols)
# predicted = model.predict(new_df)
# print(predicted)  # 予測結果の表示


# 4-19
# 正解率の表示
# model_score = model.score(x, t)
# print(model_score)


# 4-20
# import pickle
# モデルの保存
# with open('models/4-20.pkl', 'wb') as f:
#     pickle.dump(model, f)


# 4-21
# with open('models/4-20.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)
#     new_df2 = pd.DataFrame([[175, 75, 30]], columns=x_cols)
#     predicted = loaded_model.predict(new_df2)
#     print(predicted)  # 予測結果の表示


# 4-22
import pandas as pd
from sklearn import tree
# データの読み込み
df_KvsT = pd.read_csv('sukkiri-ml2-codes/datafiles/KvsT.csv')
# 説明変数と目的変数の設定
x_cols = ['身長', '体重', '年代']
x = df_KvsT[x_cols]
t = df_KvsT['派閥']
# モデルの準備と学習
model = tree.DecisionTreeClassifier(random_state=0)
model.fit(x, t)
# 正解率の計算
model_score = model.score(x, t)
import pickle
# モデルの保存
with open('models/4-20.pkl', 'wb') as f:
    pickle.dump(model, f)
print(model_score)