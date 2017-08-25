import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from k_nearest2 import k_nearest_neighbors

# a = 200
# seq = np.random.poisson(a, ( ))
# df = pd.DataFrame(seq)
# df[50] = 1
# # print(seq)

# # print(df[-5][])

# # test_size = 0.2
# # train_set = {2:[], 4:[]}
# # test_set = {2:[], 4:[]}

# # print(test_size)
# # print(len(train_set))
# # print(test_set)

# val_new = pd.DataFrame(np.random.poisson(250,(10000,50)))
# val_new[50] = 2
# # print(val_new)

# val_new.to_csv('generated_data2.csv', index=False)
# df.to_csv('generated_data.csv', index=False)

# df = pd.read_csv('generated_data.csv')
# val_new = pd.read_csv('generated_data2.csv')

# print(df)
# print(val_new)


lambdas = np.random.poisson(200, 50).to_matrix()
print(lambdas)

