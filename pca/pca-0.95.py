
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

#print(df_mal.shape)
#print(df_ben.shape)

#print(df_mal.head(5))
#print(df_ben.head(5))

df=df_mal.append(df_ben)
print(df.shape)

features=df.columns.values.tolist()[:-1]
#print(features)

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
pca = PCA(0.95)

principalComponents = pca.fit_transform(x)

print(principalComponents.shape)

#principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

#principalDf.reset_index(inplace=True, drop=True)
#df.reset_index(inplace=True, drop=True)

#finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
#finalDf.reset_index(inplace=True, drop=True)

print(pca.explained_variance_ratio_)



