import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import cluster as Kcluster, metrics as SK_Metrics
import random
import processing as pr


# df=pd.DataFrame()
# cols=[]
# lamdas=np.random.poisson(lam=[50],size=10)
# for l in lamdas:
# 	for i in range(0,5):
# 		cols.append(l)
# # print lamdas
# i=1
# for lamd in cols:
# 		df["col"+str(i)]=np.random.poisson(lam=lamd,size=10000)
# 		i+=1
		 
# df["Class"]=np.ones(10000);
# df["Class"][0:5000]=0
# # print df
# df=df.reindex(np.random.permutation(df.index))
# df.to_csv('rand_data2.csv', index=False);
# print df
# print "Data frame before Sampling: "+str(df.shape)

# def stratified_sampling(mData,fraction,clusters=3):
# 	df=mData
# 	kmeans=KMeans(n_clusters=clusters).fit(df)
# 	df['cluster']=kmeans.labels_
# 	cl_rows=[]
# 	for i in range(clusters):
# 		cl_rows.append(df.ix[random.sample(df[df['cluster']==i].index,(int)(len(df[df['cluster']==i])*fraction))])
# 	df=pd.concat(cl_rows)
# 	del df['cluster']
# # 	return df

def dim_reduction(mData):
	std= StandardScaler().fit_transform(mData)
	mean_vec = np.mean(std, axis=0)
	cov_mat = (std - mean_vec).T.dot((std - mean_vec))/(std.shape[0]-1)
	cov_mat = np.cov(std.T)
	eig_vals, eig_vecs = np.linalg.eig(cov_mat)
	eig_vals=sorted(eig_vals, reverse=True)
	#print eig_vals
	comp=0
	for vr in eig_vals:
		if(vr>0.5):
			comp+=1
	pca=PCA( n_components=comp)	
	return {'pca':pca,'eigen_values':eig_vals,'data':pd.DataFrame(pca.fit_transform(mData)),'n_indim':comp};

# df = pd.read_csv('rand_data.csv',sep=',')
# # sdf=stratified_sampling(df,0.5,5)
# pcaData=dim_reduction(df.ix[:,0:50])
# df1=pcaData['data']
# print df1
# df1['Class']=df['Class']
# pr.processData(df)
# print "Data frame after Sampling: "+str(sdf.shape)

