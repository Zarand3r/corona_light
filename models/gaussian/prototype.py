import os
import datetime
import itertools
from multiprocessing import Pool

import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.integrate
from sklearn.metrics import mean_squared_error

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

death_time = 14

def process_data(data_covid, data_population, save=True):
	covid = loader.load_data(data_covid)
	loader.convert_dates(covid, "date")
	population = loader.load_data(data_population)
	covid.loc[covid["county"]=='New York City', "fips"]=36061
	covid['Population'] = covid.apply(lambda row: loader.query(population, "FIPS", row.fips)['total_pop'], axis=1)
	covid.dropna(subset=['fips'], inplace=True)
	covid['fips']=covid['fips'].astype(int)
	# covid = add_active_cases(covid, "/data/us/covid/JHU_daily_US.csv")
	if save:
		covid.to_csv(f"{homedir}" + "/models/gaussian/us_training_data.csv")
	return covid

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

###########################################################

def plot_deaths(res, data, extrapolate=14, boundary=None, death_metric="deaths"):   
	# res is results from fitting

	t = np.arange(0, len(data))
	tp = np.arange(0, len(data)+extrapolate)

	p = bokeh.plotting.figure(plot_width=1000,
							  plot_height=600,
							 title = ' PECAIQR Model',
							 x_axis_label = 't (days)',
							 y_axis_label = '# people')

	p.circle(t, data[death_metric], color ='black', legend='Real Death')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[], name="figures/test.png"):
	X = X.ravel()
	mu = mu.ravel()
	uncertainty = 1.96 * np.sqrt(np.diag(cov))
	
	plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.3)
	plt.plot(X, mu, label='Mean')
	for i, sample in enumerate(samples):
		plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
	if X_train is not None:
		plt.scatter(X_train, Y_train, c="black", s=2)
	plt.legend()
	plt.savefig(name)


###########################################################



def kernel(X1, X2, l=1.0, sigma_f=1.0):
	'''
	Isotropic squared exponential kernel. Computes 
	a covariance matrix from points in X1 and X2.
		
	Args:
		X1: Array of m points (m x d).
		X2: Array of n points (n x d).

	Returns:
		Covariance matrix (m x n).
	'''
	sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
	return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)



def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, noise=100):
	'''  
	Computes the suffifient statistics of the GP posterior predictive distribution 
	from m training data X_train and Y_train and n new inputs X_s.
	
	Args:
		X_s: New input locations (n x d).
		X_train: Training locations (m x d).
		Y_train: Training targets (m x 1).
		l: Kernel length parameter.
		sigma_f: Kernel vertical variation parameter.
		sigma_y: Noise parameter.
	
	Returns:
		Posterior mean vector (n x d) and covariance matrix (n x n).
	'''
	K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
	K_s = kernel(X_train, X_s, l, sigma_f)
	K_ss = kernel(X_s, X_s, l, sigma_f) + (noise**2) * np.eye(len(X_s))
	K_inv = inv(K)
	
	# Equation (4)
	mu_s = K_s.T.dot(K_inv).dot(Y_train)

	# Equation (5)
	cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
	
	return mu_s, cov_s


def calculate_noise(county_data, death_metric):
	# find standard deviation away from moving average
	firstnonzero = next((index for index,value in enumerate(county_data[death_metric].values) if value != 0), None)
	actual_deaths = (county_data['deaths'].values)[firstnonzero:]
	moving_deaths = (county_data['avg_deaths'].values)[firstnonzero:]
	residuals = []
	for index in range(1, len(actual_deaths)):
		moving_change = moving_deaths[index] - moving_deaths[index-1]
		if moving_change > 0:
			residue = actual_deaths[index] - moving_deaths[index]
			residue = residue/moving_change
			residuals.append(residue)
	return np.std(residuals)


###########################################################
def test1(end, death_metric="deaths"):
	from sklearn import gaussian_process
	from sklearn.gaussian_process import GaussianProcessRegressor
	from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

	counties_dates = []
	counties_death_errors = []
	counties_fips = []

	# us = process_data("/data/us/covid/nyt_us_counties_daily.csv", "/data/us/demographics/county_populations.csv")
	us = loader.load_data("/models/gaussian/us_training_data.csv")
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

		dates = pd.to_datetime(county_data["date"].values)
		extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')
		print(extrapolate)

		# X = np.arange(0, len(county_data)+extrapolate)
		X_pred = np.arange(0, len(county_data)+extrapolate).reshape(-1,1)
		X_train = np.arange(0, len(county_data)).reshape(-1, 1)
		Y_train = county_data[death_metric].values



		# kernel = ConstantKernel() + Matern(length_scale=1, nu=3/2) + WhiteKernel(noise_level=1)
		# kernel = WhiteKernel(noise_level=1)
		# gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
		# gp.fit(X_train, Y_train)
		# GaussianProcessRegressor(alpha=1e-10, copy_X_train=True,
		# kernel=1**2 + Matern(length_scale=2, nu=1.5) + WhiteKernel(noise_level=1),
		# n_restarts_optimizer=1, normalize_y=False,
		# optimizer='fmin_l_bfgs_b', random_state=None)
		# y_pred, sigma = gp.predict(X_pred, return_std=True)

		clf = GaussianProcessRegressor(random_state=42, alpha=0.1)
		clf.fit(X_train, Y_train)
		y_pred, sigma = clf.predict(X_pred, return_std=True)

		# Plot the function, the prediction and the 95% confidence interval based on
		# the MSE
		plt.figure()
		plt.scatter(X_train, Y_train, c='b', label="Daily Deaths")
		plt.plot(X_pred, y_pred, label="Prediction")
		plt.fill_between(X_pred[:, 0], y_pred - sigma, y_pred + sigma,
				 alpha=0.5, color='blue')

		# plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
		# plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
		# plt.plot(x_pred, y_pred, 'b-', label='Prediction')
		# plt.fill(np.concatenate([x, x[::-1]]),
		#          np.concatenate([y_pred - 1.9600 * sigma,
		#                         (y_pred + 1.9600 * sigma)[::-1]]),
		#          alpha=.5, fc='b', ec='None', label='95% confidence interval')

		plt.legend(loc='upper left')
		plt.show()
		


def test2(end, death_metric="deaths"):
	counties_dates = []
	counties_death_errors = []
	counties_fips = []

	# us = process_data("/data/us/covid/nyt_us_counties_daily.csv", "/data/us/demographics/county_populations.csv")
	us = loader.load_data("/models/gaussian/us_training_data.csv")
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

		dates = pd.to_datetime(county_data["date"].values)
		extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')
		print(extrapolate)

		X_pred = np.arange(0, len(county_data)+extrapolate).reshape(-1,1)
		X_train = np.arange(0, len(county_data)).reshape(-1, 1)
		Y_train = county_data[death_metric].values

		noise = 20
		# Compute mean and covariance of the posterior predictive distribution
		# mu_s, cov_s = posterior_predictive(X_pred, X_train, Y_train, sigma_y=noise)
		# samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
		# plot_gp(mu_s, cov_s, X_pred, X_train=X_train, Y_train=Y_train, samples=samples)

		params = [
			(3.0, 0.2, 0.1),
			(6.0, 0.2, 0.2),
			(6.0, 0.3, 0.2),
			(6.0, 0.1, 0.2),
			(8.0, 0.2, 0.05),
			(10.0, 0.3, 0.1),
		]

		plt.figure(figsize=(12, 8))
		for i, (l, sigma_f, sigma_y) in enumerate(params):
			mu_s, cov_s = posterior_predictive(X_pred, X_train, Y_train, l=l, sigma_f=sigma_f, sigma_y=sigma_y, noise=noise)
			plt.subplot(3, 2, i + 1)
			plt.subplots_adjust()
			plt.title(f'l = {l}, sigma_f = {sigma_f}, sigma_y = {sigma_y}')
			# plot_gp(mu_s, cov_s, X_pred, X_train=X_train, Y_train=Y_train)
			samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
			print(cov_s)
			plot_gp(mu_s, cov_s, X_pred, X_train=X_train, Y_train=Y_train, samples=samples, name=f"figures/{county}_{death_metric}test.png")
		plt.show()



def fit_single_county(input_dict):
	#put the logic to fit a single county here
	#all the data should be in input_dict
	us = input_dict["us"]
	policies = input_dict["policies"]
	county = input_dict["county"]
	end = input_dict["end"]
	death_metric = input_dict["death_metric"]

	county_data = loader.query(us, "fips", county)
	county_data['avg_deaths'] = county_data.iloc[:,6].rolling(window=3).mean()
	county_data = county_data[2:]
	if len(county_data) == 0:
		return None

	dates = pd.to_datetime(county_data["date"].values)
	extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')

	noise = calculate_noise(county_data, death_metric=death_metric)
	print(noise)
	# X_pred = np.arange(0, len(county_data)+extrapolate).reshape(-1,1)
	# X_train = np.arange(0, len(county_data)).reshape(-1, 1)
	# Y_train = county_data[death_metric].values

	# # Try using object oriented design. 
	# # Create an instance for each county, pass the county specific parameters as instance variables, so we dont have to keep passing them between functions
	# # To get results (death_cdf), call its main function below, and return them in the tuple

	# death_cdf = []
	# return (dates, death_cdf, county) 
	return None

def multi_submission(end, death_metric="deaths"):
	counties_dates = []
	counties_death_errors = []
	counties_fips = []

	# This first line regenerates the us_training_data.csv with new data pulled from upstream. It is usually commented out because I dont want to run it every time I test this function.
	# us = process_data("/data/us/covid/nyt_us_counties_daily.csv", "/data/us/demographics/county_populations.csv")
	us = loader.load_data("/models/gaussian/us_training_data.csv")
	policies = loader.load_data("/data/us/other/policies.csv")
	fips_key = loader.load_data("/data/us/processing_data/fips_key.csv", encoding="latin-1")
	fips_list = fips_key["FIPS"][100:110]

	data = []
	for index, county in enumerate(fips_list):
		input_dict = {}
		input_dict["us"] = us
		input_dict["policies"] = policies
		input_dict["county"] = county
		input_dict["end"] = end
		input_dict["death_metric"] = death_metric
		data.append(input_dict)
	
	pool = Pool(os.cpu_count())
	results = pool.map(fit_single_county, data)
	
	for result in results:
		if result is not None:
			(dates, death_cdf, county) = result
			counties_dates.append(dates)
			counties_death_errors.append(death_cdf)
			counties_fips.append(county)

	output_dict = {"counties_dates": np.array(counties_dates), "counties_death_errors": np.array(counties_death_errors), "counties_fips": np.array(counties_fips)}
	return output_dict
	

if __name__ == '__main__':
	end = datetime.datetime(2020, 6, 30)
	# test2(end, death_metric="avg_deaths")
	multi_submission(end)














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