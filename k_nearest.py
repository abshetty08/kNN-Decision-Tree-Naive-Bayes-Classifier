import numpy as np
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
print(df)
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

# print(len(X))
# print(len(y))
# print(len(X_train))
# print(len(X_test))
# print(len(y_train))
# print(len(y_test))

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[10,12,9,1,1,7,3,2,1], [4,2,1,1,1,2,1,1,1]])
example_measures = example_measures.reshape(len(example_measures),-1)


prediction = clf.predict(example_measures)
print(prediction)


pca = PCA(n_components=2)
fit = pca.fit(X_train)

# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print("Fit components: ")
print(fit.components_)

# X_transformed = pca.fit_transform(X_train)
# print X_transformed
