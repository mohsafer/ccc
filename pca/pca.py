
import pandas as pd
import os

df_mal=pd.DataFrame()
for i in range(414):
	file="../dataset/malware/"+str(i+1)+".csv"
	df=pd.read_csv(file, delimiter=",", header = 0, skipinitialspace=True)
	df["target"]=pd.Series(["malign" for x in range(len(df))])
	df_mal=df_mal.append(df)

df_ben=pd.DataFrame()
for i in range(288):
	file="../dataset/safe/"+str(i+1)+".csv"
	df=pd.read_csv(file, delimiter=",", header = 0, skipinitialspace=True)
	df["target"]=pd.Series(["benign" for x in range(len(df))])
	df_ben=df_ben.append(df)

print(df_mal.shape)
print(df_ben.shape)

print(df_mal.head(5))
print(df_ben.head(5))

df=df_mal.append(df_ben)
print(df.shape)

features=df.columns.values.tolist()[:-1]
print(features)

#####

from sklearn.preprocessing import StandardScaler
# Separating out the features
x = df.loc[:,features].values

# Separating out the target
y = df.loc[:,['target']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

#####

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
#pca = PCA(.95)

principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

principalDf.reset_index(inplace=True, drop=True)
df.reset_index(inplace=True, drop=True)

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
finalDf.reset_index(inplace=True, drop=True)

print(pca.explained_variance_ratio_)

#####

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['malign', 'benign']
colors = ['r', 'g']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
ax.legend(targets)
ax.grid()

plt.savefig("pca.png", bbox_inches="tight", paper="a0", orientation="landscape", dpi=200)
plt.savefig("pca.eps", bbox_inches="tight", paper="a0", orientation="landscape", dpi=300)

plt.clf
plt.cla
plt.figure(clear=True)
plt.close("all")


