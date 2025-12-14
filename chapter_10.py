# 10-1
import pandas as pd
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/bike.tsv')
# print(df.head(3))

# 10-2
df = pd.read_csv('sukkiri-ml2-codes/datafiles/bike.tsv', sep='\t')
# print(df.head(3))

# 10-3
# df2 = pd.read_csv('sukkiri-ml2-codes/datafiles/weather.csv')

# 10-4
weather = pd.read_csv('sukkiri-ml2-codes/datafiles/weather.csv', encoding='shift_jis')
# print(weather.head(3))

# 10-5
temp = pd.read_json('sukkiri-ml2-codes/datafiles/temp.json')
# print(temp.head(3))

# 10-6
# temp.T
# print(temp.T.head(3))

# 10-7
df2 = df.merge(weather, how='inner', on='weather_id')
df2 = df2.sort_values(by='dteday')
# print(df2.head(3))

# 10-8
# print(df2.groupby('weather')['cnt'].mean())

# 10-9
temp = temp.T
# print(temp.loc[199:201])

# 10-10
# print(df2[df2['dteday']=='2011-07-20'])

# 10-11
df3 = df2.merge(temp, how='left', on='dteday')
# print(df3[df3['dteday']=='2011-07-20'])

# 10-12
import matplotlib
matplotlib.use("Agg")  # フリーズ回避
import matplotlib.pyplot as plt
# df3['temp'].plot(kind='line')

# 10-13
# df3[['temp', 'hum']].plot(kind='line')
# plt.savefig("10-13.png")

# 10-14
# df3['temp'].plot(kind='hist')
# df3['hum'].plot(kind='hist', alpha=0.5)
# plt.savefig("10-14.png")

# 10-15
# df3['atemp'].loc[695:705].plot(kind='line')
# plt.savefig("10-15.png")

# 10-16
# df3['atemp'] = df3['atemp'].astype(float)
# df3['atemp'] = df3['atemp'].interpolate()
# df3.loc[695:705, 'atemp'].plot()
# plt.savefig("10-16.png")

# 10-17
# iris_df = pd.read_csv('sukkiri-ml2-codes/datafiles/iris.csv')
# non_df = iris_df.dropna()
# x = non_df.loc[:, 'がく片幅':'花弁幅']
# t = non_df['がく片長さ']
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(x, t)

# 10-18
# condition = iris_df['がく片長さ'].isnull()
# non_data = iris_df.loc[condition]
# x = non_data.loc[:, 'がく片幅':'花弁幅']
# pred = model.predict(x)
# iris_df.loc[condition, 'がく片長さ'] = pred

# 10-19
from sklearn.covariance import MinCovDet
df4 = df3.loc[:, 'atemp':'windspeed']
df4 = df4.dropna()
mcd = MinCovDet(random_state=0, support_fraction=0.7)
mcd.fit(df4)
distance = mcd.mahalanobis(df4)
# print(distance)

# 10-20
distance = pd.Series(distance)
# distance.plot(kind = 'box')
# plt.savefig("10-20.png")

# 10-21
tmp = distance.describe()
# print(tmp)

# 10-22
iqr = tmp['75%'] - tmp['25%']
jougen = 1.5 * iqr + tmp['75%']
kagen = tmp['25%'] - 1.5 * iqr
outliner = distance[(distance > jougen) | (distance < kagen)]
print(outliner)

