import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
import mglearn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()
print("Keys of iris datset: \n {}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + '\n...')
print("Target Names: {}".format(iris_dataset['target_names']))
print("Feature Names: \n{}".format(iris_dataset['feature_names']))

print("Type of data:", type(iris_dataset['data']))
print("Shape of data:", iris_dataset['data'].shape)
print("First five rows of data:\n", iris_dataset['data'][:5])
print("Type of target:", type(iris_dataset['target']))
print("Shape of target:", iris_dataset['target'].shape)
print("Target:\n", iris_dataset['target'])


X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'],random_state=0
)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
df = pd.DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])
scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)
plt.show()