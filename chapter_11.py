# 11-1
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# 11-2
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/Boston.csv')
# df = df.drop('CRIME', axis=1)
# df = df.fillna(df.mean())
# df = df.drop([76], axis = 0)
# t = df[['PRICE']]
# x = df.loc[:,['RM','PTRATIO','LSTAT']]
# sc = StandardScaler()
# sc_x = sc.fit_transform(x)
# sc2 = StandardScaler()
# sc_t = sc2.fit_transform(t)

# 11-3
# from sklearn.preprocessing import PolynomialFeatures
# pf = PolynomialFeatures(degree=2, include_bias=False)
# pf_x = pf.fit_transform(sc_x)
# print(pf_x.shape)

# 11-4
# print(pf.get_feature_names_out())

# 11-5
# from sklearn.linear_model import LinearRegression
# x_train, x_test, t_train, t_test = train_test_split(pf_x, sc_t, test_size=0.3, random_state=0)
# model = LinearRegression()
# model.fit(x_train, t_train)
# print(model.score(x_train, t_train))
# print(model.score(x_test, t_test))

# 11-6
# from sklearn.linear_model import Ridge
# ridgeModel = Ridge(alpha=10)
# ridgeModel.fit(x_train, t_train)
# print(ridgeModel.score(x_train, t_train))
# print(ridgeModel.score(x_test, t_test))

# 11-7
# maxScore = 0
# maxIndex = 0
# for i in range(1, 2001):
#     num = i / 100
#     ridgeModel = Ridge(random_state=0, alpha=num)
#     ridgeModel.fit(x_train, t_train)
#     result = ridgeModel.score(x_test, t_test)
#     if result > maxScore:
#         maxScore = result
#         maxIndex = num
# print(maxIndex, maxScore)
# 17.62 0.852875480149763

# 11-8
# from sklearn.linear_model import Ridge
# ridgeModel = Ridge(alpha=17.62)
# ridgeModel.fit(x_train, t_train)
# print(sum(abs(model.coef_)[0]))
# print(abs(ridgeModel.coef_)[0])

# 11-9
# from sklearn.linear_model import Lasso
# x_train, x_test, t_train, t_test = train_test_split(pf_x, sc_t, test_size=0.3, random_state=0)
# model = Lasso(alpha=0.1)
# model.fit(x_train, t_train)
# print(model.score(x_train, t_train))
# print(model.score(x_test, t_test))

# 11-10
# weight = model.coef_
# print(pd.Series(weight, index=pf.get_feature_names_out()))

# 11-11
# import pandas as pd
# from sklearn.model_selection import train_test_split
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/Boston.csv')
# df = df.fillna(df.mean(numeric_only=True))
# x = df.loc[:, 'ZN': 'LSTAT']
# t = df['PRICE']
# x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

# 11-12
# from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor(max_depth=10, random_state=0)
# model.fit(x_train, t_train)
# print(model.score(x_test, t_test))

# 11-13
# print(pd.Series(model.feature_importances_, index=x.columns))
