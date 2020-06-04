import itertools
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn import linear_model

import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/models/data_processing')
import loader

death_time = 14

def reject_outliers(data, m = 1.):
	data = np.array(data)
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d/mdev if mdev else 0.
	return data[s<m]

def moving_average(data, window=3):
	moving_data = []
	for i in range(len(data)):
		if i >= window-1:
			average = np.sum(data[i-(window-1):i+1])/window
			moving_data.append(average)
	return moving_data

def find_peak(data, window=3):
	moving_data = moving_average(data, window=window)
	max_index = np.argmax(moving_data)
	return max_index + (window-1)


def visualize_peaks(data, X, Y, window=3, plot=True, savefig=True):
	data.dropna(subset=Y+[X], inplace=True)
	name = X
	x_feature = data[X].values
	x_features = []
	y_features = []
	for y in Y:
		name = name + "_" + y
		y_feature = data[y].values
		death_time = find_peak(x_feature, window=window) - find_peak(y_feature, window=window)
		print(death_time)
		if death_time > 0:
			x_features.append(x_feature[death_time:])
			y_features.append(y_feature[:-1*death_time])
		elif death_time == 0:
			x_features.append(x_feature)
			y_features.append(y_feature)
		else:
			x_features.append(x_feature[:death_time])
			y_features.append(y_feature[-1*death_time:])

	# x_feature = x_feature.rolling(window=3).mean()
	# print(x_feature)
	if plot:
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('First', fontsize = 15)
		ax.set_ylabel('Second', fontsize = 15)
		ax.set_title('Beds', fontsize = 20)
		for index, y_feature in enumerate(y_features):
			scatter = ax.scatter(x_features[index], y_feature, s = 10, label=Y[index])
		ax.grid()
		ax.legend()
		if savefig:
			plt.savefig("figures/"+name)
		plt.show()

def visualize(data, X, Y, moving=False, plot=True, savefig=True):
	data.dropna(subset=Y+[X], inplace=True)
	name = X
	x_feature = data[X].values
	y_features = []
	if moving and type(moving) == int:
		for y in Y:
			name = name + "_" + y
			y_feature = data[y].values
			y_feature = moving_average(y_feature, window=moving)
			y_features.append(y_feature)
		if moving > 0:
			x_feature = x_feature[(moving-1):]
	else:
		for y in Y:
			name = name + "_" + y
			y_features.append(data[y].values)

	if plot:
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('First', fontsize = 15)
		ax.set_ylabel('Second', fontsize = 15)
		ax.set_title('Beds', fontsize = 20)
		for index, y_feature in enumerate(y_features):
			scatter = ax.scatter(x_feature, y_feature, s = 10, label=Y[index])
		ax.grid()
		ax.legend()
		if savefig:
			plt.savefig("figures/"+name)
		plt.show()

def linear_regression(data, X, Y, cutoff=None, window=3, moving=False, plot=True, savefig=True):
	data.dropna(subset=[X,Y], inplace=True)
	name = X + "_" + Y
	x_feature = data[X].values
	y_feature = data[Y].values
	print(y_feature)
	death_time = find_peak(x_feature, window=window) - find_peak(y_feature, window=window)
	print(death_time)
	if death_time > 0:
			x_feature = x_feature[death_time:]
			y_feature = y_feature[:-1*death_time]
	elif death_time < 0:
		x_feature = x_feature[:death_time]
		y_feature = y_feature[-1*death_time:]

	if moving and type(moving) == int:
		if moving > 0:
			x_feature = np.array(moving_average(x_feature, window=moving))
			y_feature = np.array(moving_average(y_feature, window=moving))

	if cutoff is not None:
		x_feature = x_feature[:cutoff]
		y_feature = y_feature[:cutoff]
		# x_feature = x_feature[5:]
		# y_feature = y_feature[5:]
		print(y_feature)
	x_feature = x_feature.reshape(-1, 1)

	lm = linear_model.LinearRegression()
	model = lm.fit(x_feature,y_feature)
	predictions = lm.predict(x_feature)

	if plot:
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('First', fontsize = 15)
		ax.set_ylabel('Second', fontsize = 15)
		ax.set_title('Beds', fontsize = 20)
		ax.scatter(x_feature, y_feature, s = 10, label=Y)
		ax.scatter(x_feature, predictions, s = 10, label="Predicted")
		ax.grid()
		ax.legend()
		if savefig:
			plt.savefig("figures/fit_moving_"+name)
		plt.show()


if __name__ == '__main__':
	state_tests = loader.load_data("/data/us/covid/daily_state_tests.csv")[["date", "state", "hospitalizedCurrently","hospitalizedCumulative","inIcuCurrently","inIcuCumulative","onVentilatorCurrently","onVentilatorCumulative","death","hospitalized","fips","deathIncrease","hospitalizedIncrease"]]
	state_tests["date"]=pd.to_datetime(state_tests["date"], format="%Y%m%d")
	loader.convert_dates(state_tests, "date")
	state_tests = state_tests.sort_values('date_processed', ascending=True)
	# state = loader.query(state_tests, "fips", 36)
	state = loader.query(state_tests, "state", "NY")
	# visualize(state, "date_processed", ["deathIncrease", "hospitalizedCurrently"], moving=7)
	# visualize_peaks(state, "deathIncrease", ["hospitalizedCurrently"], window=7)
	# visualize(state, "date_processed", ["deathIncrease", "hospitalizedIncrease"], moving=7)
	# visualize_peaks(state, "deathIncrease", ["hospitalizedIncrease"], window=7)
	# visualize_peaks(state, "deathIncrease", ["onVentilatorCurrently"], window=7)
	# linear_regression(state, "deathIncrease", "hospitalizedCurrently", window=7, moving=3)
	linear_regression(state, "deathIncrease", "onVentilatorCurrently", cutoff=-10, window=7, moving=7)

