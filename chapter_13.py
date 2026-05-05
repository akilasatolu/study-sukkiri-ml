# 13-1
# import pandas as pd
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/cinema.csv')
# df = df.fillna(df.mean())
# x = df.loc[:, 'SNS1':'original']
# t = df['sales']
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(x, t)

# 13-2
# from sklearn.metrics import mean_squared_error
# pred = model.predict(x)
# mse = mean_squared_error(pred, t)
# print(mse)

# 13-3
# import math
# print(math.sqrt(mse))

# 13-4
# from sklearn.metrics import mean_absolute_error
# yosoku = [2,3,5,7,11,13]
# target = [3,5,8,11,16,19]
# mse = mean_squared_error(yosoku, target)
# print("rmse:{}".format(math.sqrt(mse)))
# print("mae:{}".format(mean_absolute_error(yosoku, target)))
# print('外れ値の混入')
# yosoku = [2,3,5,7,11,13,46]
# target = [3,5,8,11,16,19,23]
# mse = mean_squared_error(yosoku, target)
# print("rmse:{}".format(math.sqrt(mse)))
# print("mae:{}".format(mean_absolute_error(yosoku, target)))

# 13-5
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/Survived.csv')
# df = df.fillna(df.mean(numeric_only=True))
# x = df[['Pclass', 'Age']]
# t = df['Survived']

# 13-6
# from sklearn import tree
# model = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
# model.fit(x, t)

# 13-7
# from sklearn.metrics import classification_report
# pred = model.predict(x)
# out_put = classification_report(y_pred = pred, y_true = t)
# print(out_put)

# 13-8
# out_put = classification_report(y_pred = pred, y_true = t, output_dict=True)
# print(pd.DataFrame(out_put))

# 13-9
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/cinema.csv')
# df = df.fillna(df.mean())
# x = df.loc[:, 'SNS1':'original']
# t = df['sales']

# 13-10
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=3, shuffle=True, random_state=0)

# 13-11
# from sklearn.model_selection import cross_validate
# model = LinearRegression()
# result = cross_validate(model, x, t, cv=kf, scoring='r2', return_train_score=True)
# print(result)

# 13-12
# print(sum(result['test_score']) / len(result['test_score']))

# 13-13
# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)