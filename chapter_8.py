# 8-1
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 8-2
df = pd.read_csv('sukkiri-ml2-codes/datafiles/Boston.csv')
# print(df.head(2))

# 8-3
# print(df['CRIME'].value_counts())

# 8-4
crime = pd.get_dummies(df['CRIME'], drop_first=True, dtype=int)
df2 = pd.concat([df, crime], axis=1)
df2 = df2.drop('CRIME', axis=1)
# print(df2.head())

# 8-5
train_val, test = train_test_split(df2, test_size=0.2, random_state=0)

# 8-6
# print(train_val.isnull().sum())

# 8-7
train_val_mean = train_val.mean()
train_val2 = train_val.fillna(train_val_mean)
# print(train_val2)
# print(train_val2.isnull().sum())

# 8-8
# colname = train_val2.columns
# for name in colname:
#     train_val2.plot(kind='scatter', x=name, y='PRICE')

# 8-9
# out_line1 = train_val2[(train_val2['RM'] < 6) & (train_val2['PRICE'] > 40)].index
# out_line2 = train_val2[(train_val2['PTRATIO'] > 18) & (train_val2['PRICE'] > 40)].index
# print(out_line1, out_line2)

# 8-10
train_val3 = train_val2.drop([76], axis=0)
# print(train_val3)

# 8-11
col = ['INDUS', 'NOX', 'RM', 'PTRATIO', 'LSTAT', 'PRICE']
train_val4 = train_val3[col]
# print(train_val4.head())

# 8-12
# print(train_val4.corr())

# 8-13
train_col = train_val4.corr()['PRICE']
# print(train_col)

# 8-14
# print(abs(2))
# print(abs(-2))

# 8-15
# se = pd.Series([1, -2, 3, -4])
# print(se.map(abs))

# 8-16
# abs_col = train_col.map(abs)
# print(abs_col)

# 8-17
# print(abs_col.sort_values(ascending=False))

# 8-18
col = ['RM', 'LSTAT', 'PTRATIO']
x = train_val4[col]
t = train_val4[['PRICE']]
x_train, x_val, y_train, y_val = train_test_split(x, t, test_size=0.2, random_state=0)

# 8-19
from sklearn.preprocessing import StandardScaler
sc_model_x = StandardScaler()
sc_model_x.fit(x_train)
sc_x = sc_model_x.transform(x_train)
# print(sc_x)

# 8-20
tmp_df = pd.DataFrame(sc_x, columns=x_train.columns)
# print(tmp_df.mean())

# 8-21
# print(tmp_df.std())

# 8-22
sc_model_y = StandardScaler()
sc_model_y.fit(y_train)
sc_y = sc_model_y.transform(y_train)
# print(sc_y)

# 8-23
model = LinearRegression()
model.fit(sc_x, sc_y)

# 8-24
# print(model.score(x_val, y_val))

# 8-25
sc_x_val = sc_model_x.transform(x_val)
sc_y_val = sc_model_y.transform(y_val)
# print(model.score(sc_x_val, sc_y_val))

# 8-26
# 検証データでの平均値と標準偏差を調べて、その値で標準化をしてはいけない
# sc_model_x2.fit(x_val)
# sc_model_y2.fit(y_val)

# 8-27
def learn(x, t):
    # 訓練データと検証データに分割
    x_train, x_val, y_train, y_val = train_test_split(x, t, test_size=0.2, random_state = 0)
    # 訓練データを標準化
    sc_model_x = StandardScaler()
    sc_model_y = StandardScaler()
    sc_model_x.fit(x_train)
    sc_model_y.fit(y_train)
    sc_x_train = sc_model_x.transform(x_train)
    sc_y_train = sc_model_y.transform(y_train)
    # 学習
    model = LinearRegression()
    model.fit(sc_x_train, sc_y_train)
    # 検証データを標準化
    sc_x_val = sc_model_x.transform(x_val)
    sc_y_val = sc_model_y.transform(y_val)
    # 訓練データと検証データの決定係数計算
    train_score = model.score(sc_x_train, sc_y_train)
    val_score = model.score(sc_x_val, sc_y_val)
    return train_score, val_score

# 8-28
# x = train_val3.loc[ :,['RM', 'LSTAT', 'PTRATIO']]
# t = train_val3[ ['PRICE']]
# s1, s2 = learn(x, t)
# print(s1, s2)

# 8-29
x = train_val3.loc[:,['RM', 'LSTAT', 'PTRATIO', 'INDUS']]
t = train_val3[['PRICE']]
# s1, s2 = learn(x, t)
# print(s1, s2)

# 8-30
# print(x['RM'] ** 2)

# 8-31
x['RM2'] = x['RM'] ** 2
x = x.drop('INDUS', axis = 1)
# print(x.head())

# 8-32
# x.loc[2000] = [10, 7, 8, 100]

# 8-33
# s1, s2 = learn(x, t)
# print(s1, s2)

# 8-34
x['LSTAT2'] = x['LSTAT'] ** 2
x['PTRATIO2'] = x['PTRATIO'] ** 2
# s1, s2 = learn(x, t)
# print(s1, s2)

# 8-35
# se1 = pd.Series([1,2,3])
# se2 = pd.Series([10,20,30])
# se1 + se2

# 8-36
x['RM * LSTAT'] = x['RM'] * x['LSTAT']

# 8-37
# s1, s2 = learn(x, t)
# print(s1, s2)

# 8-38
sc_model_x2 = StandardScaler()
sc_model_y2 = StandardScaler()
sc_model_x2.fit(x)
sc_model_y2.fit(t)
sc_x = sc_model_x2.transform(x)
sc_y = sc_model_y2.transform(t)
model = LinearRegression()
model.fit(sc_x, sc_y)

# 8-39
# テストデータの前処理
test2 = test.fillna(train_val.mean())
x_test = test2.loc[:,['RM', 'LSTAT', 'PTRATIO']]
y_test = test2[['PRICE']]
x_test['RM2'] = x_test['RM'] ** 2
x_test['LSTAT2'] = x_test['LSTAT'] ** 2
x_test['PTRATIO2'] = x_test['PTRATIO'] ** 2
x_test['RM * LSTAT'] = x_test['RM'] * x_test['LSTAT']
sc_x_test = sc_model_x2.transform(x_test)
sc_y_test = sc_model_y2.transform(y_test)

# 8-40
# print(model.score(sc_x_test, sc_y_test))

# 8-41
# import pickle
# with open('models/8-41-1.pkl', 'wb') as f:
#     pickle.dump(model, f)
# with open('models/8-41-2.pkl', 'wb') as f:
#     pickle.dump(sc_model_x2, f)
# with open('models/8-41-3.pkl', 'wb') as f:
#     pickle.dump(sc_model_y2, f)
