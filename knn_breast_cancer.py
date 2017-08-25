import numpy as np
import pandas as pd
from collections import Counter
import warnings
from math import sqrt
import random
from sklearn import model_selection
from rand import dim_reduction
# from sklearn import preprocessing, cross_validation
#from sklearn.model_selection import train_test_split

# K nearest neighbors function
def k_nearest_neighbors(data, predict, k=3):
	#Check to see if number of features > k (not mandatory)
	if len(data) >= k:
		warnings.warn('k is set to a value less than total voting groups!') #Throw a warning if k value is less than the number of classes

	distances = []	#List to hold the distances of all the data points from the "predict" data point

	#Calculate euclidean distance of each data point from the "predict" data point
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))	#calculates euclidean distance (its faster and is generic wrt any number of dimensions/features)
			distances.append([euclidean_distance, group])

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

	df = pd.read_csv('breast-cancer-wisconsin.data.txt')	#read CSV file
	# print(df)
	df.replace('?', -99999, inplace=True)	#plug missing data points with outlier
	df.drop(['id'], 1, inplace=True)	#column id does not contribute to the decision process, ignore this column
	full_data = df.astype(float).values.tolist()	#convert all data points to float type values to maintain uniformity across all values
	
	# print df	
	
	accuracies = []
	for i in range(1):
		
		# random.shuffle(full_data)	#this is done so that the training and test data are selected randomly and not in the order given in the dataset

		test_size = 0.3		#Percentage of date to be considered for test dataset
		train_set = {2:[], 4:[]}	#create a dictionary for the 2 classes in the training dataset (2 - benign, 4 - malignant)
		test_set = {2:[], 4:[]}	#create a dictionary for the 2 classes in the test dataset (2 - benign, 4 - malignant)
		
		# test_set = {}
		train_data = full_data[:-int(test_size*len(full_data))]	#assigns elements indexed from beginning (row wise)
		test_data = full_data[-int(test_size*len(full_data)):]	#assigns elements indexed from middle (row wise)

		#Assigning values to training data
		for i in train_data:
			train_set[i[-1]].append(i[:-1])
		
		#Assigning values to test data
		for i in test_data:
			test_set[i[-1]].append(i[:-1])
		
		# ------------------------------------------------------------
		# Dataset processing
		# ------------------------------------------------------------

		# X_train_list = X_train.tolist()
		# train_set = {}

		# for i in range(len(X_train)):
		# 	if not y_train[i] in train_set:
		# 		train_set[y_train[i]] = []

		# 	temp1 = y_train[i]
		# 	temp2 = X_train_list[i]
		# 	# print temp1
		# 	# print temp2
		# 	train_set[temp1].append(temp2)
			
		# print len(train_set)

		# X_test_list = X_test.tolist()
		# test_set = {}

		# for i in range(len(X_test)):
		# 	if not y_test[i] in test_set:
		# 		test_set[y_test[i]] = []

		# 	temp1 = y_test[i]
		# 	temp2 = X_test_list[i]
		# 	# print temp1
		# 	# print temp2
		# 	test_set[temp1].append(temp2)
			
		# print len(test_set)

		# ------------------------------------------------------------
		# Dataset processing end
		# ------------------------------------------------------------

		# print(test_size)
		# print(len(train_set))
		# print(len(test_set))
		# print(len(train_data))
		# print(len(test_data))
		# print(len(full_data))

		correct = 0
		total = 0


		for group in test_set:
			for data in test_set[group]:
				vote = k_nearest_neighbors(train_set, data, k=5)
				if group == vote:
					correct += 1
				total += 1
		# print('Accuracy:', float(correct)/(total))
		# print('Confidence:', confidence)
		accuracies.append(float(correct)/(total))

	print('Accuracy before PCA:')
	print('Accuracy:' + str(sum(accuracies)/len(accuracies)))



	# print(k_nearest_neighbors(train_set, [4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0]))
	# print(k_nearest_neighbors(train_set, [8,10,10,8,7,10,9,7,1]))

	# --------------------------------------------------------------------
	# PCA
	# --------------------------------------------------------------------

	# print df
	pcaData=dim_reduction(df.ix[:,0:10])
	df1=pcaData['data']
	# print df1

	df1['class'] = df['class']
	full_data = df1.astype(float).values.tolist()	#convert all data points to float type values to maintain uniformity across all values
	# print df1 
	test_size = 0.3		#Percentage of date to be considered for test dataset
	train_set = {2:[], 4:[]}	#create a dictionary for the 2 classes in the training dataset (2 - benign, 4 - malignant)
	test_set = {2:[], 4:[]}	#create a dictionary for the 2 classes in the test dataset (2 - benign, 4 - malignant)
	# test_set = {}
	train_data = full_data[:-int(test_size*len(full_data))]	#assigns elements indexed from beginning (row wise)
	test_data = full_data[-int(test_size*len(full_data)):]	#assigns elements indexed from middle (row wise)

	#Assigning values to training data
	for i in train_data:
		train_set[i[-1]].append(i[:-1])
		
	#Assigning values to test data
	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct = 0
	total = 0


	for group in test_set:
		for data in test_set[group]:
			vote = k_nearest_neighbors(train_set, data, k=5)
			if group == vote:
				correct += 1
			total += 1

	print('Accuracy after PCA:')
	print('Accuracy:' + str(float(correct)/(total)))
	# print('Confidence:', confidence)
	# accuracies.append(float(correct)/(total))

	# print(sum(accuracies)/len(accuracies))

	# --------------------------------------------------------------------
	# PCA ends
	# --------------------------------------------------------------------
