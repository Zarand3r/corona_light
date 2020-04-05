import numpy as np
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys, os

if __name__ == '__main__':
datafile 
data = np.loadtxt(open('data/train_2008.csv', 'rb'), delimiter=",", skiprows=1)
	X = data[:, 1:-1]
	y = data[:, -1]
	train_X, test_X, train_y, test_y=train_test_split(X,y,test_size=0.3)
	train_X=train_X[:100,:]#first 100
	train_y=train_y[:100]# first 100