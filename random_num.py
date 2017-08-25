import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from k_nearest2 import k_nearest_neighbors

# mean = [0, 0]
# cov = [[1, 0], [0, 100]]  # diagonal covariance

# x, y = np.random.multivariate_normal(mean, cov, (5,5)).T
# print(x)
# print(y)
# plt.plot(x, y, 'x')
# plt.axis('equal')
# plt.show()

#------------------------------------------------------------------------#
# Data generation function 
#------------------------------------------------------------------------#

def generate_data():
	a = 10
	b = 20
	seq = np.random.poisson(a, (1000,10))
	df = pd.DataFrame(seq)
	df[5] = 1
	# print(seq)
	# print(df)

	# print(df[:][:])

	df.to_csv('generated_data.csv', sep=',')
	return df


#------------------------------------------------------------------------#
# Data generation function ends
#------------------------------------------------------------------------#


#------------------------------------------------------------------------#
# main function 
#------------------------------------------------------------------------#

if __name__ == "__main__":
	df = generate_data()
	print(df)
	# df = pd.read_csv('generated_data.csv')	#read CSV file
	df.replace('?', -99999, inplace=True)
	df.replace(' ', -99999, inplace=True)
	df.drop([5], 1, inplace=True)
	print(df)
	# df.drop(['id'], 1, inplace=True)
	

	



#------------------------------------------------------------------------#
# main function ends
#------------------------------------------------------------------------#





# # with open('test.csv', 'wb', newline='') as fp:
# #     a = csv.writer(fp, delimiter=',')
# #     data = np.random.beta(a, b, size=10)
# #     a.writerows(data)

# lam_val = range(10,110,10)
# # lam_val = np.random.beta(a, b, size=10)
# seq2 = np.random.poisson(lam=(lam_val), size=(100, len(lam_val)))
# print(seq2)
# count, bins, ignored = plt.hist(seq2, 14, normed=True)
# plt.show()
