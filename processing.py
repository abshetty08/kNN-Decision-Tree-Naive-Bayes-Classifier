#implementing naive bayes
import numpy as np 
import pandas as pd 
from sklearn import model_selection
import math
from scipy.stats import norm

def classProb(mean,sd,keys,inp):
	probs={}
	for i in range(len(keys)):
		probs[keys[i]]=1
		for j in range(len(mean[i])):
			x=inp[j]
			probs[keys[i]]*= norm.pdf(x,mean[i][j],sd[i][j])
	return probs

def predict(mean,sd,keys,inp):
	probs=classProb(mean,sd,keys,inp)
	bLabel,bProb=None,-1
	for classValue,prob in probs.iteritems():
		if bLabel is None or bProb < prob:
			bProb=prob
			bLabel=classValue
	return bLabel

def getPredictions(test_data,mean_vals,sd_vals,keys):
	predictions=[]
	test=test_data.as_matrix()
	for i in range(len(test)):
		result=predict(mean_vals,sd_vals,keys,test[i])
	 	predictions.append(result)
	return predictions
def accuracy(pred,expval):
	count=0
	for i in range(len(expval)):
		if(pred[i]==expval[i]):
			count+=1
	return (float(count)/len(expval))*100


def processData(df):
	# df=df.apply(pd.to_numeric,errors='coerce')
	names=df.columns
	train_data,test_data = model_selection.train_test_split(df,test_size=0.005)
	grpData=train_data.groupby('Letter')
	mean_vals = grpData[names[1:len(names)]].mean().as_matrix()
	sd_vals = grpData[names[1:len(names)]].std().as_matrix()
	keys= grpData.groups.keys()
	expval = np.array(test_data['Letter'])
	test_data= test_data.drop(['Letter'],1)
	pred=getPredictions(test_data,mean_vals,sd_vals,keys)
	print accuracy(pred,expval)

# df = pd.read_csv('breast-cancer-wisconsin.data.txt',sep=',',header=None)
# names=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitosis','Class']
# df.columns=names
# df=df.drop('Sample code number',1)
# df=df.apply(pd.to_numeric,errors='coerce')
# train_data,test_data = model_selection.train_test_split(df,test_size=0.3)
# print len(test_data)
# grpData=train_data.groupby('Class')
# mean_vals = grpData[names[1:len(names)-1]].mean().as_matrix()
# #print mean_vals
# sd_vals = grpData[names[1:len(names)-1]].std().as_matrix()
# keys= grpData.groups.keys()
# expval = np.array(test_data['Class'])
# test_data= test_data.drop(['Class'],1)
#print predict(mean_vals,sd_vals,keys,[10,7,7,6,4,10,4,1,2])
#print test_data

