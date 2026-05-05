# 14-1
# import pandas as pd
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/Boston.csv')
# print(df.head(2))

# 14-2
# df2 = df.fillna(df.mean(numeric_only=True))

# 14-3
# dummu = pd.get_dummies(df2['CRIME'], drop_first=True, dtype=int)
# df3 = df2.join(dummu)
# df3 = df3.drop(["CRIME"], axis=1)
# print(df3.head(2))

# 14-4
# from sklearn.preprocessing import StandardScaler
# df4 = df3.astype("float")
# sc = StandardScaler()
# sc_df = sc.fit_transform(df4)

# 14-5
# from sklearn.decomposition import PCA

# 14-6
# model = PCA(n_components=2, whiten=True)

# 14-7
# model.fit(sc_df)

# 14-8
# print(model.components_[0])
# print(model.components_[1])

# 14-9
# new = model.transform(sc_df)
# new_df = pd.DataFrame(new)
# print(new_df.head(3))

# 14-10
# new_df.columns = ["PC1", "PC2"]
# df5 = pd.DataFrame(sc_df, columns = df4.columns)
# df6 = pd.concat([df5, new_df], axis=1)

# 14-11
# df_corr = df6.corr()
# print(df_corr.loc[: 'very_low', 'PC1':])

# 14-12
# pc_corr = df_corr.loc[:'very_low', 'PC1':]
# print(pc_corr['PC1'].sort_values(ascending=False))

# 14-13
# print(pc_corr['PC2'].sort_values(ascending=False))

# 14-14
# col = ['Countryside', 'Exclusive residential']
# new_df.columns = col
# new_df.plot.scatter(x='Countryside', y='Exclusive residential')

# 14-15
# model = PCA(whiten=True)
# tmp = model.fit_transform(sc_df)
# print(tmp.shape)

# 14-16
# print(model.explained_variance_ratio_)

# 14-17
# ratio = model.explained_variance_ratio_
# array = []
# for i in range(len(ratio)):
#     ruiseki = sum(ratio[0:i+1])
#     array.append(ruiseki)
# pd.Series(array).plot(kind='line')

# 14-18
# thred = 0.8
# for i in range(len(array)):
#     if array[i] >= thred:
#         print(i+1)
#         break

# 14-19
# model = PCA(n_components=5, whiten=True)
# model.fit(sc_df)
# new = model.transform(sc_df)

# 14-20
# col = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
# new_df2 = pd.DataFrame(new, columns=col)
# new_df2.to_csv('sukkiri-ml2-codes/datafiles/Boston_pca.csv', index=False)
