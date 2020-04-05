import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys, os

if __name__ == '__main__':
	datafile = ''
	data = np.loadtxt(open(datafile, 'rb'), delimiter=",", skiprows=1)
	X = data[:, 1:-1]
	y = data[:, -1]
	train_X, test_X, train_y, test_y=train_test_split(X,y,test_size=0.3)
	# train_X=train_X[:1000,:]#first 100
	# train_y=train_y[:1000]# first 100

	# grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
	grid={"C":[0.001, 0.01, 0.1, 100], "penalty":["l1"]}# l1 lasso l2 ridge
	logistic=LogisticRegression()
	clf=GridSearchCV(logistic, grid, cv=10, scoring='roc_auc')
	clf.fit(train_X,train_y)


	print("tuned hpyerparameters :(best parameters) ",clf.best_params_)
	print("accuracy :",clf.best_score_)