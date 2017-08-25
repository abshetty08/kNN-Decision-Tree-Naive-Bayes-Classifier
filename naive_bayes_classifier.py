
from sklearn.naive_bayes import GaussianNB

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
from sklearn import model_selection
import processing as pr

df = pd.read_csv('letter-recognition.data.csv')
X = np.array(df.drop(['Letter'],1))
y = np.array(df['Letter'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.005)

clf = GaussianNB()
clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[12,14,12,8,5,9,10,4,3,5,10,7,10,12,2,6], [6,8,9,6,4,6,8,2,9,10,9,8,3,8,4,7]])
example_measures = example_measures.reshape(len(example_measures),-1)

# print(k_nearest_neighbors(train_set, [12,14,12,8,5,9,10,4,3,5,10,7,10,12,2,6], 5))	#W
# print(k_nearest_neighbors(train_set, [6,8,9,6,4,6,8,2,9,10,9,8,3,8,4,7], 5))	#X


# prediction = clf.predict(example_measures)
# print(prediction)
pr.processData(df)