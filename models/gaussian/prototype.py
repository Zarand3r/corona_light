import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from multiprocessing import Pool

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import scipy.integrate
from sklearn.metrics import mean_squared_error
from scipy.linalg import svd
from scipy.optimize import least_squares
import datetime

import bokeh.io
import bokeh.application
import bokeh.application.handlers
import bokeh.models
import holoviews as hv
# bokeh.io.output_notebook()
hv.extension('bokeh')

import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/models/data_processing')
import loader

death_time = 16

def add_active_cases(us, data_active_cases):
	active_cases = loader.load_data(data_active_cases)
	active_cases['FIPS']=active_cases['FIPS'].astype(int)
	loader.convert_dates(active_cases, "Date")
	difference = (pd.to_datetime(active_cases['Date'])[0] - pd.to_datetime(us['date'])[0])/np.timedelta64(1, 'D')

	active_column = []
	end = len(us)-1
	for index, row in us.iterrows():
		print(f"{index}/{end}")
		county = row['fips']
		date = row['date_processed'] 
		if date < difference:
			active_column.append(-1)
		else:
			entry = (active_cases[(active_cases.date_processed==date-difference) & (active_cases.FIPS == county)])["Active"].values
			if len(entry) != 0:
				active_column.append(entry[0])
			else:
				active_column.append(-1)

	us["active_cases"] = active_column
	return us



def process_data(data_covid, data_population, save=True):
	covid = loader.load_data(data_covid)
	loader.convert_dates(covid, "date")
	population = loader.load_data(data_population)
	covid.loc[covid["county"]=='New York City', "fips"]=36061
	covid['Population'] = covid.apply(lambda row: loader.query(population, "FIPS", row.fips)['total_pop'], axis=1)
	covid.dropna(subset=['fips'], inplace=True)
	covid['fips']=covid['fips'].astype(int)
	covid = add_active_cases(covid, "/data/us/covid/JHU_daily_US.csv")
	if save:
		covid.to_csv(f"{homedir}" + "/models/gaussian/production/us_training_data.csv")
	return covid


###########################################################



###########################################################
def test(end, death_metric="deaths"):

	counties_dates = []
	counties_death_errors = []
	counties_fips = []
	nonconvergent = []
	parameters = {}

	us = process_data("/data/us/covid/nyt_us_counties_daily.csv", "/data/us/demographics/county_populations.csv")
	us = loader.load_data("/models/gaussian/production/us_training_data.csv")
	policies = loader.load_data("/data/us/other/policies.csv")
	fips_key = loader.load_data("/data/us/processing_data/fips_key.csv", encoding="latin-1")
	# fips_list = fips_key["FIPS"][0:10]
	fips_list = [36061] #56013,1017
	total = len(fips_list)

	for index, county in enumerate(fips_list):
		print(f"{index+1} / {total}")
		county_data = loader.query(us, "fips", county)
		county_data['avg_deaths'] = county_data.iloc[:,6].rolling(window=3).mean()
		county_data = county_data[2:]
		print(county_data['deaths'])
	
		


def fit_single_county(input_dict):
	us = input_dict["us"]
	policies = input_dict["policies"]
	county = input_dict["county"]
	end = input_dict["end"]
	regime = input_dict["regime"]
	weight = input_dict["weight"]
	guesses = input_dict["guesses"]
	start= input_dict["start"]
	quick = input_dict["quick"]
	fitQ = input_dict["fitQ"]
	getbounds = input_dict["getbounds"]
	death_metric = input_dict["death_metric"]
	nonconvergent = None 
	parameters = []

	




if __name__ == '__main__':
	end = datetime.datetime(2020, 6, 30)
	test(end)














# import numpy as np
# from matplotlib import pyplot as plt

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# np.random.seed(1)


# def f(x):
#     """The function to predict."""
#     return x * np.sin(x)

# # ----------------------------------------------------------------------
# #  First the noiseless case
# X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# # Observations
# y = f(X).ravel()

# # Mesh the input space for evaluations of the real function, the prediction and
# # its MSE
# x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# # Instantiate a Gaussian Process model
# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# # Fit to data using Maximum Likelihood Estimation of the parameters
# gp.fit(X, y)

# # Make the prediction on the meshed x-axis (ask for MSE as well)
# y_pred, sigma = gp.predict(x, return_std=True)

# # Plot the function, the prediction and the 95% confidence interval based on
# # the MSE
# plt.figure()
# plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
# plt.plot(X, y, 'r.', markersize=10, label='Observations')
# plt.plot(x, y_pred, 'b-', label='Prediction')
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([y_pred - 1.9600 * sigma,
#                         (y_pred + 1.9600 * sigma)[::-1]]),
#          alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$x$')
# plt.ylabel('$f(x)$')
# plt.ylim(-10, 20)
# plt.legend(loc='upper left')

# # ----------------------------------------------------------------------
# # now the noisy case
# X = np.linspace(0.1, 9.9, 20)
# X = np.atleast_2d(X).T

# # Observations and noise
# y = f(X).ravel()
# dy = 0.5 + 1.0 * np.random.random(y.shape)
# noise = np.random.normal(0, dy)
# y += noise

# # Instantiate a Gaussian Process model
# gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
#                               n_restarts_optimizer=10)

# # Fit to data using Maximum Likelihood Estimation of the parameters
# gp.fit(X, y)

# # Make the prediction on the meshed x-axis (ask for MSE as well)
# y_pred, sigma = gp.predict(x, return_std=True)

# # Plot the function, the prediction and the 95% confidence interval based on
# # the MSE
# plt.figure()
# plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
# plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
# plt.plot(x, y_pred, 'b-', label='Prediction')
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([y_pred - 1.9600 * sigma,
#                         (y_pred + 1.9600 * sigma)[::-1]]),
#          alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$x$')
# plt.ylabel('$f(x)$')
# plt.ylim(-10, 20)
# plt.legend(loc='upper left')

# plt.show()