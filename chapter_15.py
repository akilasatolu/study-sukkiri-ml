# 15-1
# import pandas as pd
# df = pd.read_csv('sukkiri-ml2-codes/datafiles/Wholesale.csv')
# print(df.head(3))

# 15-2
# print(df.isnull().sum())

# 15-3
# df = df.drop(["Channel", "Region"], axis=1)

# 15-4
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc_df = sc.fit_transform(df)
# sc_df = pd.DataFrame(sc_df, columns=df.columns)

# 15-5
# from sklearn.cluster import KMeans

# 15-6
# model = KMeans(n_clusters=3, random_state=0)

# 15-7
# model.fit(sc_df)

# 15-8
# print(model.labels_)

# 15-9
# sc_df["cluster"] = model.labels_
# print(sc_df.head(2))

# 15-10
# print(sc_df.groupby("cluster").mean())

# 15-11
# cluster_mean = sc_df.groupby("cluster").mean()
# cluster_mean.plot(kind="bar")

# 15-12
# sse_list = []
# for n in range(2, 31):
#     model = KMeans(n_clusters=n, random_state=0)
#     model.fit(sc_df)
#     sse = model.inertia_
#     sse_list.append(sse)
# print(sse_list)

# 15-13
# se = pd.Series(sse_list)
# num = range(2, 31)
# se.index = num
# se.plot(kind="line")

# 15-14
# model = KMeans(n_clusters=5, random_state=0)
# model.fit(sc_df)
# sc_df["cluster"] = model.labels_
# sc_df.to_csv("clustered_Wholesale.csv", index=False)