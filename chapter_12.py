# 12-1
# import pandas as pd
# from sklearn import tree
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/KvsT.csv')
# x = df.loc[:, '体重':'年代']
# t = df['派閥']
# model = tree.DecisionTreeClassifier(max_depth=1, random_state=0)
# model.fit(x, t)
# data = [[65,20]]
# print(model.predict(data))
# print(model.predict_proba(data))

# 12-2
# import pandas as pd
# from sklearn.model_selection import train_test_split
# df =pd.read_csv('sukkiri-ml2-codes/datafiles/iris.csv')
# print(df.head(2))

# 12-3
# df_mean = df.mean(numeric_only=True)
# train2 = df.fillna(df_mean)
# x = train2.loc[:, :'花弁幅']
# t = train2['種類']
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# new = sc.fit_transform(x)

# 12-4
# x_train, x_val, y_train, y_val = train_test_split(new, t, test_size=0.2, random_state=0)

# # 12-5
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(random_state=0, C=0.1, multi_class='auto', solver='lbfgs')

# 12-6
# model.fit(x_train, y_train)
# print(model.score(x_train, y_train))
# print(model.score(x_val, y_val))

# 12-7
# print(model.coef_)

# 12-8
# x_new = [[1,2,3,4]]
# print(model.predict(x_new))

# 12-9
# print(model.predict_proba(x_new))

# 12-10
# import pandas as pd
# from sklearn.model_selection import train_test_split

#12-11
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/Survived.csv')
# print(df.head(2))

#12-12
# def fill_age(df, pclass, survived, value):
#     joken1 = df['Pclass'] == pclass
#     joken2 = df['Survived'] == survived
#     joken3 = df['Age'].isnull()
#     df.loc[(joken1) & (joken2) & (joken3), 'Age'] = value
#     return df

# print(df.pivot_table(index='Pclass', columns='Survived', values='Age'))

# df = fill_age(df, 1, 0, 43)
# df = fill_age(df, 1, 1, 35)
# df = fill_age(df, 2, 0, 33)
# df = fill_age(df, 2, 1, 25)
# df = fill_age(df, 3, 0, 26)
# df = fill_age(df, 3, 1, 20)

#12-13
# col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
# x = df[col]
# t = df['Survived']
# dummy = pd.get_dummies(df['Sex'], drop_first=True, dtype=int)
# x = pd.concat([x, dummy], axis=1)
# print(x.head(2))

#12-14
# from sklearn.ensemble import RandomForestClassifier
# x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)
# model = RandomForestClassifier(random_state=0, n_estimators=100)

#12-15
# model.fit(x_train, y_train)
# print(model.score(x_train, y_train))
# print(model.score(x_test, y_test))

#12-16
# from sklearn import tree
# model2 = tree.DecisionTreeClassifier(random_state=0)
# model2.fit(x_train, y_train)
# print(model2.score(x_train, y_train))
# print(model2.score(x_test, y_test))

#12-17
# importance = model.feature_importances_
# print(pd.Series(importance, index=x.columns))

#12-18
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)
# base_model = DecisionTreeClassifier(random_state=0, max_depth=5)
# model = AdaBoostClassifier(n_estimators=500, random_state=0, estimator=base_model, algorithm='SAMME')
# model.fit(x_train, y_train)
# print(model.score(x_train, y_train))
# print(model.score(x_test, y_test))

#12-19
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/cinema.csv')
# df = df.fillna(df.mean())
# x = df.loc[:, 'SNS1':'original']
# t = df['sales']
# x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(random_state=0, n_estimators=100)
# model.fit(x_train, y_train)
# print(model.score(x_test, y_test))

#12-20
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.tree import DecisionTreeRegressor
# base = DecisionTreeRegressor(random_state=0, max_depth=3)
# model = AdaBoostRegressor(random_state=0, n_estimators=500, estimator=base)
# model.fit(x_train, y_train)
# print(model.score(x_test, y_test))