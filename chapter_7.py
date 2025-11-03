# 7-1
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/Survived.csv')
# print(df.head(2))  # 先頭2行表示

# 7-2
# print(df['Survived'].value_counts()) # 生存者と死亡者の数を表示

# 7-3
# print(df.isnull().sum())  # 列ごとの欠損値の数を表示

# 7-4
# print(df.shape)

# 7-5
# df['Age'] = df['Age'].fillna(df['Age'].mean()) # Age列の欠損値を平均値で補完
# df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) # Embarked列の欠損値を最頻値で補完

# 7-6
# col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
# x = df[col]
# t = df['Survived']

# 7-7
# x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)
# print(x_train.shape)

# 7-8
# model = tree.DecisionTreeClassifier(max_depth=5, random_state=0, class_weight='balanced')
# model.fit(x_train, y_train)

# 7-9
# print(model.score(x_test, y_test))

# 7-10
def learn(x, t, depth=3):
    x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)
    model = tree.DecisionTreeClassifier(max_depth=depth, random_state=0, class_weight='balanced')
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    score2 = model.score(x_test, y_test)
    return round(score, 3), round(score2, 3), model

# 7-11
# for j in range(1,15):
#     train_score, test_score, model = learn(x, t, depth = j)
#     sentence = "訓練データの正解率:{}"
#     sentence2 = "テストデータの正解率:{}"
#     total_sentence='深さ{}:'+sentence + sentence2
#     print(total_sentence.format(j, train_score, test_score))

# 7-12
df2 = pd.read_csv('sukkiri-ml2-codes/datafiles/Survived.csv')
# print(df2['Age'].mean()) # Age列の平均値を表示
# print(df2['Age'].median()) # Age列の中央値を表示

# 7-13
# print(df2.groupby('Survived')['Age'].mean())  # 生存者と死亡者のAge列の平均値を表示

# 7-14
# print(df2.groupby('Pclass')['Age'].mean())  # 各PclassごとのAge列の平均値を表示

# 7-15
# print(pd.pivot_table(df2, index = 'Survived', columns = 'Pclass', values = 'Age')) # ピボットテーブルで各PclassごとのAge列の平均値を表示

# 7-16
# print(pd.pivot_table(df2, index = 'Survived', columns = 'Pclass', values = 'Age', aggfunc='max')) # ピボットテーブルで各PclassごとのAge列の最大値を表示

# 7-17
is_null = df2['Age'].isnull()
df2.loc[(df2['Pclass'] == 1) & (df2['Survived'] == 0) & (is_null), 'Age'] = 43
df2.loc[(df2['Pclass'] == 1) & (df2['Survived'] == 1) & (is_null), 'Age'] = 35
df2.loc[(df2['Pclass'] == 2) & (df2['Survived'] == 0) & (is_null), 'Age'] = 33
df2.loc[(df2['Pclass'] == 2) & (df2['Survived'] == 1) & (is_null), 'Age'] = 25
df2.loc[(df2['Pclass'] == 3) & (df2['Survived'] == 0) & (is_null), 'Age'] = 26
df2.loc[(df2['Pclass'] == 3) & (df2['Survived'] == 1) & (is_null), 'Age'] = 20

# 7-18
# col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
# x = df2[col]
# t = df2['Survived']
# for j in range(1,15):
#     train_score, test_score, model = learn(x, t, depth = j)
#     sentence = "訓練データの正解率:{}"
#     sentence2 = "テストデータの正解率:{}"
#     total_sentence='深さ{}:'+sentence + sentence2
#     print(total_sentence.format(j, train_score, test_score))

# 7-19
# sex = df2.groupby('Sex')['Survived'].mean()
# print(sex)

# 7-20
# sex.plot(kind = 'bar')

# 7-21
col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']
x = df2[col]
t = df2['Survived']
# train_score, test_score, model = learn(x, t, depth = 5) # 文字列は特徴量にできないためエラーになる

# 7-22
male = pd.get_dummies(df2['Sex'], drop_first = True, dtype=int)
# print(male)

# 7-23
# male = pd.get_dummies(df2['Sex'], dtype=int)
# print(male)

# 7-24
# print(pd.get_dummies(df2['Embarked'], drop_first = True, dtype=int))

# 7-25
# print(pd.get_dummies(df2['Embarked'], dtype=int))

# 7-26
x_temp = pd.concat([x, male], axis=1)
# print(x_temp.head(2))

# 7-27
# tmp = pd.concat([x,x], axis=0) # xを行方向に連結
# print(tmp.shape)

# 7-28
x_new = x_temp.drop('Sex', axis=1)
# for j in range(1, 6):
#     s1, s2, m = learn(x_new, t, depth=j)
#     s = '深さ{}:訓練データの正解率:{} テストデータの正解率:{}'
#     print(s.format(j, s1, s2))

# 7-29
s1, s2, model = learn(x_new, t, depth=5)

# import pickle
# with open('models/7-29.pkl', 'wb') as f:
#     pickle.dump(model, f)

# 7-30
# print(model.feature_importances_)

# 7-31
# print(pd.DataFrame(model.feature_importances_, index=x_new.columns))
      