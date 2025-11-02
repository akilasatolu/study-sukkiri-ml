# 6-1
import pandas as pd
df = pd.read_csv('sukkiri-ml2-codes/datafiles/cinema.csv')
# print(df.head(3))  # 先頭3行表示

# 6-2
# print(df.isnull().any(axis=0))  # 列ごとに欠損値があるか確認

# 6-3
df2 = df.fillna(df.mean())
# print(df2.isnull().any(axis=0))

# 6-4
# print(df2.plot(kind = 'scatter', x = 'SNS2', y = 'sales', color = 'blue'))

# 6-5
# print(df2.plot(kind = 'scatter', x = 'SNS1', y = 'sales'))
# print(df2.plot(kind = 'scatter', x = 'SNS2', y = 'sales'))
# print(df2.plot(kind = 'scatter', x = 'actor', y = 'sales'))
# print(df2.plot(kind = 'scatter', x = 'original', y = 'sales'))

# 6-6
# for name in df2.columns:
#     if name == 'cinema' or name == 'sales':
#         continue
#     print(df2.plot(kind = 'scatter', x = name, y = 'sales'))

# 6-7
# no = df2[(df2["SNS2"]>1000) & (df2["sales"]<8500)].index
# df3 = df2.drop(no, axis=0)

# 6-8
# test = pd.DataFrame(
#     {
#         "A": [1, 2, 3, 4, 5],
#         "B": [10, 20, 30, 40, 50]
#     }
# )

# 6-9
# print(test[test["A"]<2])

# 6-10
# print(test["A"]<2)

# 6-11
# print(df[(df["SNS2"] > 1000)&(df["sales"] < 8500)])

# 6-12
no = df2[(df2["SNS2"]>1000) & (df2["sales"]<8500)].index
# print(no)

# 6-13
# print(test.drop(3, axis=0))

# 6-14
# print(test.drop('B', axis=1))

# 6-15
df3 = df2.drop(no, axis=0)
# print(df3.shape)

# 6-16
# col = ['SNS1', 'SNS2', 'actor', 'original']
# x = df3[col]
# t = df3['sales']

# 6-17
# print(df3.loc[2, 'SNS1'])

# 6-18
# index = [2,4,6]
# col = ['SNS1', 'actor']
# print(df3.loc[index, col])

# 6-19
# list = [0,1,2,3,4,5]
# print(list[1:4])  # インデックス1〜3までを抽出

# 6-20
# print(df3.loc[0:3, :'actor'])  # インデックス0〜3まで、'actor'列までを抽出

# 6-21
x = df3.loc[:, 'SNS1':'original']  # 全行、'SNS1'列から'original'列までを抽出
t = df3['sales']  # 目的変数を抽出

# 6-22
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)

# 6-23
from sklearn.linear_model import LinearRegression

# 6-24
model = LinearRegression()

# 6-25
model.fit(x_train, y_train)  # 学習用データで学習

# 6-26
# new = pd.DataFrame([[150, 700, 300, 0]], columns=x_train.columns)
# print(model.predict(new))  # 新しいデータで予測

# 6-27
# print(model.score(x_test, y_test))  # テスト用データで評価

# 6-28
# from sklearn.metrics import mean_absolute_error
# pred = model.predict(x_test)
# print(mean_absolute_error(y_pred=pred, y_true=y_test))  # 平均絶対誤差を計算

# 6-29
# print(model.score(x_test, y_test))  # 決定係数を計算

# 6-30
# import pickle
# with open('models/6-30.pkl', 'wb') as f:
#     pickle.dump(model, f)

# 6-31
# print(model.coef_)  # 回帰係数を表示
# print(model.intercept_)  # 切片を表示

# 6-32
# tmp = pd.DataFrame(model.coef_)
# tmp.index = x_train.columns
# print(tmp)