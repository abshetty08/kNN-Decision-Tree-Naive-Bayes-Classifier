import numpy as np
import pandas as pd
from collections import Counter
import warnings
from math import sqrt
import random
from sklearn import preprocessing, cross_validation

# K nearest neighbors function
def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('k is set to a value less than total voting groups!')

	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	# for i in sorted(distances)[:k]:
	# 	votes = [i[1]]
	
	# print(votes)
	# print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = float(Counter(votes).most_common(1)[0][1]) / k
	
	# print(vote_result, confidence)

	return vote_result, confidence

# main function
if __name__ == "__main__":
	
	a = 10
	b = 20
	seq = np.random.poisson(a, (1000,10))
	df = pd.DataFrame(seq)
	df[10] = 1

	accuracies = []
	for i in range(25):
		# df = pd.read_csv('breast-cancer-wisconsin.data.txt')	#read CSV file
		# df.replace('?', -99999, inplace=True)
		# df.drop(['id'], 1, inplace=True)
		full_data = df.astype(float).values.tolist()
		random.shuffle(full_data)

		test_size = 0.2
		train_set = {2:[], 4:[]}
		test_set = {2:[], 4:[]}
		# test_set = {}
		train_data = full_data[:-int(test_size*len(full_data))]
		test_data = full_data[-int(test_size*len(full_data)):]

		for i in train_data:
			train_set[i[-1]].append(i[:-1])

		for i in test_data:
			test_set[i[-1]].append(i[:-1])



		correct = 0
		total = 0

		for group in test_set:
			for data in test_set[group]:
				vote, confidence = k_nearest_neighbors(train_set, data, k=5)
				if group == vote:
					correct += 1
				total += 1
		# print('Accuracy:', float(correct)/(total))
		# print('Confidence:', confidence)
		accuracies.append(float(correct)/(total))

	print(sum(accuracies)/len(accuracies))



