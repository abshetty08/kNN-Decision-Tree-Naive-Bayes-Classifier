import numpy as np
import pandas as pd
from collections import Counter
import warnings
from math import sqrt
import random
import csv
from sklearn import model_selection
from sklearn import preprocessing, cross_validation, neighbors
from rand import dim_reduction


# K nearest neighbors function
def k_nearest_neighbors(data, predict, k=3):
	#Check to see if number of features > k (not mandatory)
	# if len(data) >= k:
	# 	warnings.warn('k is set to a value less than total voting groups!') #Throw a warning if k value is less than the number of classes

	distances = []	#List to hold the distances of all the data points from the "predict" data point

	#Calculate euclidean distance of each data point from the "predict" data point
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))	#calculates euclidean distance (its faster and is generic wrt any number of dimensions/features)
			# print euclidean_distance
			distances.append([euclidean_distance, group])

	# print distances

	votes = [i[1] for i in sorted(distances)[:k]]	#Reads the first "k" values with least euclidean distance
	# for i in sorted(distances)[:k]:
	# 	votes = [i[1]]
	
	# print(votes)
	# print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]	#vote_result gives predicted value
	# confidence = float(Counter(votes).most_common(1)[0][1]) / k
	
	# print(vote_result, confidence)
	# print(vote_result)

	return vote_result#, confidence


# main function
if __name__ == "__main__":
	

	df = pd.read_csv('rand_data.csv')
	X = np.array(df.drop(['Class'],1))
	y = np.array(df['Class'])

	# print df
	# print X,y

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.01)
	
	for j in range(1):
		# df = pd.read_csv('rand_data.csv')
		# X = np.array(df.drop(['Class'],1))
		# y = np.array(df['Class'])

		# # print df
		# # print X,y

		# X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.01)

		X_train_list = X_train.tolist()
		train_set = {}

		for i in range(len(X_train)):
			if not y_train[i] in train_set:
				train_set[y_train[i]] = []

			temp1 = y_train[i]
			temp2 = X_train_list[i]
			# print temp1
			# print temp2
			train_set[temp1].append(temp2)
			
		print len(train_set)

		X_test_list = X_test.tolist()
		test_set = {}

		for i in range(len(X_test)):
			if not y_test[i] in test_set:
				test_set[y_test[i]] = []

			temp1 = y_test[i]
			temp2 = X_test_list[i]
			# print temp1
			# print temp2
			test_set[temp1].append(temp2)
			
		print len(test_set)

	# --------------------------------------------------------------------
	# Prediction 
	# --------------------------------------------------------------------
		correct = 0
		total = 0

		for group in test_set:
			for data in test_set[group]:
				vote = k_nearest_neighbors(train_set, data, k=5)
				# print "vote = " + str(vote)
				if group == vote:
					correct += 1
				total += 1

				# print "total = " + str(total)

		print('Accuracy before PCA: ' + str(float(correct)/(total)))
		# print('Confidence:', confidence)

		# for group in test_set:
		# 	print group
		# 	for data in test_set[group]:
		# 		print data


	# print(k_nearest_neighbors(train_set, [12,14,12,8,5,9,10,4,3,5,10,7,10,12,2,6], 5))	#W
	# print(k_nearest_neighbors(train_set, [6,8,9,6,4,6,8,2,9,10,9,8,3,8,4,7], 5))	#X

# ---------------------------------------------------------------------
# After PCA
# ---------------------------------------------------------------------

	pcaData=dim_reduction(df.ix[:,1:17])
	df1=pcaData['data']
	# print df1

	df1['Class'] = df['Class']
	# print df1

	X = np.array(df1.drop(['Class'],1))
	y = np.array(df1['Class'])
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.005)

	pca_accuracies = []
	for j in range(1):
		X_train_list = X_train.tolist()
		train_set = {}

		for i in range(len(X_train)):
			if not y_train[i] in train_set:
				train_set[y_train[i]] = []

			temp1 = y_train[i]
			temp2 = X_train_list[i]
			# print temp1
			# print temp2
			train_set[temp1].append(temp2)
			
		print len(train_set)

		X_test_list = X_test.tolist()
		test_set = {}

		for i in range(len(X_test)):
			if not y_test[i] in test_set:
				test_set[y_test[i]] = []

			temp1 = y_test[i]
			temp2 = X_test_list[i]
			# print temp1
			# print temp2
			test_set[temp1].append(temp2)
			
		print len(test_set)

		correct = 0
		total = 0
		k_val = 5
		for group in test_set:
			for data in test_set[group]:
				# vote = k_nearest_neighbors(train_set, data, k=5+j*20)
				vote = k_nearest_neighbors(train_set, data, k=5)
				# print "vote = " + str(vote)
				if group == vote:
					correct += 1
				total += 1

				# print "total = " + str(total)

		Accuracy = float(correct)/(total)
		print('Accuracy after PCA:' + str(Accuracy))



