import itertools
import os
from multiprocessing import Pool

import math
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
death_time = 14
from fit_counties4helper import*

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
		covid.to_csv(f"{homedir}" + "/models/epidemiological/production/us_training_data.csv")
	return covid


###########################################################

def get_variables(res, data, index):
	extrapolate = -1
	if index > len(data)-1:
		extrapolate = index + (index-len(data)+1)
	s = model(res.x, data, extrapolate=extrapolate)
	P = s[:,0][index]
	E = s[:,1][index]
	C = s[:,2][index]
	A = s[:,3][index]
	I = s[:,4][index]
	Q = s[:,5][index]
	R = s[:,6][index]

	return (P,E,C,A,I,Q,R)

def get_deaths(res, data, extrapolate=14):   
	s = model(res.x, data, len(data)+extrapolate)
	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]
	tp = np.arange(0, len(data)+extrapolate)
	deaths = list(zip(tp,D))
	return deaths

def get_death_cdf(death_pdf, extrapolate, switch=True):
	death_cdf = []
	for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]: #make this a separate function
		forecast = list(np.nanpercentile(death_pdf, percentile, axis=0))
		death_cdf.append(forecast)

	if switch == False:
		extrapolate = int(extrapolate)
		end = len(death_cdf[-1])
		if extrapolate >= 14:
			end = end - extrapolate + 14
		# max_total = death_cdf[-1][-1*(extrapolate-1):end]
		# max_total_previous = death_cdf[-1][-1*(extrapolate):end-1]
		# min_total = death_cdf[0][-1*(extrapolate-1):end]
		# min_total_previous = death_cdf[0][-1*(extrapolate):end-1]
		# max_daily_change = [i - j for i, j in zip(max_total, max_total_previous)]
		# min_daily_change = [i - j for i, j in zip(min_total, min_total_previous)]
		# expected_total = death_cdf[4][-1*(extrapolate-1):end]
		max_daily_change = death_cdf[-1][-1*(extrapolate-1):end]
		min_daily_change = death_cdf[0][-1*(extrapolate-1):end]
		expected_daily_change = death_cdf[4][-1*(extrapolate-1):end]

		expected = np.mean(expected_daily_change)
		diff = np.mean(np.array(max_daily_change)-np.array(min_daily_change))
		# ratio = np.mean(np.array(max_daily_change)/np.array(min_daily_change))
		ratio = diff/expected
		if ratio > 1.5:
			print("recalculate error bounds")
			# See how general these parameter variances are
			# [1.16498627e-05 2.06999186e-05 5.41782152e-04 6.49380289e-06
			#  4.84675662e-05 3.57516920e-05 1.98097480e-05 8.96749155e-06
			#  3.90853805e-06 3.22475887e-06 4.37489771e-06 3.47350497e-05
			#  1.22894548e-06 3.21246366e-05 1.15024316e-04 3.08582517e-04
			#  1.02787854e-04 2.77456475e-05 4.87059431e-05 8.25090225e-04
			#  5.81252202e-04 1.02128167e-03 3.15389632e-05 0.00000000e+00
			#  5.93277957e-05]
			death_cdf = None

	return death_cdf

def reject_outliers(data, m = 1.):
	data = np.array(data)
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d/mdev if mdev else 0.
	return data[s<m]
	
# def reject_outliers(data, m=2):
# 	data = np.array(data)
#     return data[abs(data - np.mean(data)) < m * np.std(data)]

# returns standard deviation of fitted parameters
def get_param_errors(res, residuals):
	pfit = res.x
	pcov = res.jac
	pcov = np.dot(pcov.T, pcov)
	pcov = np.linalg.pinv(pcov) #uses svd
	pcov = np.diag(pcov)
	# residuals = (residuals - residuals.min()) / (residuals.max() - residuals.min())
	rcov = np.cov(residuals)
	perr = pcov * rcov
	perr = np.sqrt(perr)
	return perr



def model_beyond(fit, params, data, guess_bounds, extrapolate=-1, error_start=-1):
	offset = len(data)+error_start
	N = data['Population'].values[0] # total population
	P0 = fit[:,0][offset]
	E0 = fit[:,1][offset]
	C0 = fit[:,2][offset]
	A0 = fit[:,3][offset]
	I0 = fit[:,4][offset]
	Q0 = fit[:,5][offset]
	# Q0 = data['active_cases'].values[0] #fit to active cases instead
	R0 = fit[:,6][offset]

	# D0 = data[death_metric].values[error_start]
	D0 = fit[:,7][offset]
	yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	
	n = len(data)+extrapolate
	args = (params, N, n)
	try:
		s = scipy.integrate.odeint(pecaiqr, yz_0, np.arange(offset, n), args=args)
		# s = scipy.integrate.solve_ivp(fun=lambda t, y: pecaiqr(y, t, params, N, n), t_span=[offset, n], y0=yz_0)
	except RuntimeError:
		print('RuntimeError', params)
		death_series = quickie(fit, data, guess_bounds, error_start=error_start)
		return death_series
		# return np.zeros((n, len(yz_0)))

	D = s[:,7]
	daily_D = []
	for index, death in enumerate(D):
		if index == 0:
			daily_D.append(0)
		else:
			daily_D.append(death-D[index-1])
	return daily_D


def estimate_bounds(res, data, fit, tail=False):
	mean = 1
	deviation = 0.2
	end = len(data)
	if tail and type(tail) == int:
		firstnonzero = next((i for i,value in enumerate(data["avg_deaths"].values) if value != 0), None)
		tail = max(firstnonzero, tail)
		actual_current = (data["deaths"].values)[tail:end]
		actual_previous = (data["deaths"].values)[-1+tail:-1]
		actual_slope = [i - j for i, j in zip(actual_current, actual_previous)]
		fit_current = fit[:,7][tail:end]
		fit_previous = fit[:,7][-1+tail:-1]
		fit_slope = [i - j for i, j in zip(fit_current, fit_previous)]
		slope_ratio = np.array(actual_slope)/np.array(fit_slope)
		if len(slope_ratio) > 0:
			# mean = 0
			# deviation = np.std(abs(slope_ratio))
			mean = np.mean(slope_ratio)
			deviation = np.std(slope_ratio)
			if deviation > 0.4:
				deviation = 0.4
			if mean < 1-deviation/2:
				mean = 1-deviation/2
			elif mean > 1 + deviation/2:
				mean = 1 + deviation/2

	else:
		firstnonzero = next((i for i,value in enumerate(data["avg_deaths"].values) if value != 0), None)
		actual_current = (data["avg_deaths"].values)[firstnonzero+1:end]
		actual_previous = (data["avg_deaths"].values)[firstnonzero:-1]
		actual_slope = [i - j for i, j in zip(actual_current, actual_previous)]
		fit_current = fit[:,7][firstnonzero+1:end]
		fit_previous = fit[:,7][firstnonzero:-1]
		fit_slope = [i - j for i, j in zip(fit_current, fit_previous)]
		slope_ratio = np.array(actual_slope)/np.array(fit_slope)
		slope_ratio = reject_outliers(slope_ratio, m=3)
		if len(slope_ratio) > 0:
			# mean = 0
			# deviation = np.std(abs(slope_ratio))
			mean = np.mean(slope_ratio)
			deviation = np.std(slope_ratio)
			if deviation > 0.2:
				deviation = 0.2
			if mean < 1-deviation/2:
				mean = 1-deviation/2
			elif mean > 1 + deviation/2:
				mean = 1 + deviation/2

	return (mean,deviation)

def quickie(fit, data, guess_bounds, error_start=-1):
	if error_start is None:
		error_start = -1*len(data)+1
	offset = len(data)+error_start
	# bound = []
	change_bound = []
	predictions = fit
	scaler = 1
	if guess_bounds is not None:
		bound_mean, bound_deviation = guess_bounds
		scaler = np.random.normal(loc=bound_mean, scale=bound_deviation)
		predictions = fit[:,7][(offset-1):]
	# previous = predictions[0]
	for index, point in enumerate(predictions):
		if index > 0:
			current = predictions[index]
			change = current-predictions[index-1]
			change = change*(scaler)
			change_bound.append(change)
			# bound_point = previous + change
			# bound.append(bound_point)
			# previous = bound_point
	# bound = np.array(bound)
	# return bound
	return change_bound

# returns uncertainty of the fit for all variables
def get_fit_errors(res, p0_params, data, extrapolate=14, error_start=-1, quick=False, tail=False, death_metric="deaths"):
	if error_start is None:
		error_start = -1*len(data)+1

	fit = model(res.x, data, extrapolate)
	if type(tail) == int and error_start is not None:
		if error_start < tail:
			tail = error_start
		tail = len(data) + tail
		if tail <= 1:
			tail = False

	guess_bounds = estimate_bounds(res,data,fit,tail=tail)
	if guess_bounds == (None, None):
		return np.zeros((1,int(len(data)+extrapolate)))

	uncertainty = []
	samples = 100

	if extrapolate > 0 :
		if quick:
			for i in range(samples):
				death_series = quickie(fit, data, guess_bounds, error_start=error_start)
				latest_D = (data[death_metric].values)[-1]
				# death_series = np.concatenate((data[death_metric].values[0:len(data)], death_series[-1*error_start:]))
				death_series = np.concatenate((data["daily_deaths"].values[0:len(data)+error_start], death_series))
				# for index, death in enumerate(death_series):
				# 	if index >= len(data) and death <= latest_D:
				# 		death_series[index] = None
				uncertainty.append(death_series)
		else:
			population = list(data["Population"])[-1]
			initial = 0
			if tail:
				initial = tail
			initial_residuals = np.zeros(initial)
			residuals = (data[death_metric].values)[initial:] - fit[initial:len(data),7]
			residuals = np.concatenate((initial_residuals,residuals))
			errors = get_param_errors(res, residuals)
			errors[len(p0_params):] = 0
			for i in range(samples):
				sample = np.random.normal(loc=res.x, scale=errors)
				death_series = model_beyond(fit, sample, data, guess_bounds, extrapolate, error_start=error_start)
				# death_series = np.concatenate((fit[:,7][0:len(data)-1], death_series))
				# death_series = np.concatenate((data[death_metric].values[0:len(data)], death_series[-1*error_start:]))
				death_series = np.concatenate((data["daily_deaths"].values[0:len(data)+error_start], death_series))
				for index, death in enumerate(death_series):
					if death < 0:
						death_series[index] = None
				uncertainty.append(death_series)
	else: 
		for i in range(samples):
			sample = np.random.normal(loc=res.x, scale=errors)
			death_series = model(sample, data, extrapolate)
			death_series = death_series[:,7]
			uncertainty.append(death_series)
		
	uncertainty = np.array(uncertainty)
	return uncertainty

def mse_qd(A, B):
	Ap = np.nan_to_num(A)
	Bp = np.nan_to_num(B)
	Ap[A == -np.inf] = 0
	Bp[B == -np.inf] = 0
	Ap[A == np.inf] = 0
	Bp[B == np.inf] = 0
	return mean_squared_error(Ap, Bp)

def plot_model(res, data, extrapolate=14, boundary=None, plot_infectious=False, death_metric="deaths"):   
	s = model(res.x, data, extrapolate=extrapolate)
	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]

	t = np.arange(0, len(data))
	tp = np.arange(0, len(data)+extrapolate)

	p = bokeh.plotting.figure(plot_width=1000,
							  plot_height=600,
							 title = ' PECAIQR Model',
							 x_axis_label = 't (days)',
							 y_axis_label = '# people')

	if plot_infectious:
		p.line(tp, I, color = 'red', line_width = 1, legend = 'Currently Infected')
	p.line(tp, D, color = 'black', line_width = 1, legend = 'Deceased')
	p.line(tp, Q, color = 'green', line_width = 1, legend = 'Quarantined')
	p.line(tp, R, color = 'gray', line_width = 1, legend = 'Removed')
	p.line(tp, P, color = 'blue', line_width = 1, legend = 'Protected')
	p.line(tp, E, color = 'yellow', line_width = 1, legend = 'Exposed')
	p.line(tp, C, color = 'orange', line_width = 1, legend = 'Carrier')
	p.line(tp, A, color = 'brown', line_width = 1, legend = 'Asymptotic')

	# death
	p.circle(t, data[death_metric], color ='black', legend='Real Death')

	# quarantined
	p.circle(t, data['active_cases'], color ='purple', legend='Active Cases')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)

def plot_with_errors_sample(res, p0_params, data, extrapolate=14, boundary=None, plot_infectious=False, error_start=-1, quick=False, tail=False, death_metric="deaths"):
	s = model(res.x, data, len(data)+extrapolate)
	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]

	uncertainty = get_fit_errors(res, p0_params, data, extrapolate=extrapolate, error_start=error_start, quick=quick, tail=tail, death_metric=death_metric)
	
	s1 = np.nanpercentile(uncertainty, 10, axis=0)
	s2 = np.nanpercentile(uncertainty, 90, axis=0)

	t = np.arange(0, len(data))
	tp = np.arange(0, len(data)+extrapolate)
	p = bokeh.plotting.figure(plot_width=1000,
							  plot_height=600,
							 title = ' PECAIQR Model Errors',
							 x_axis_label = 't (days)',
							 y_axis_label = '# people')

	p.varea(x=tp, y1=s1, y2=s2, color='lightblue', fill_alpha=0.4)

	daily_D = []
	for index, death in enumerate(D):
		if index == 0:
			daily_D.append(0)
		else:
			daily_D.append(death-D[index-1])

	p.line(tp, daily_D, color = 'black', line_width = 1, legend = 'Deceased')
	p.circle(t, data["daily_deaths"], color ='black', legend = 'Daily Deaths')

	# quarantined
	# p.circle(t, data['cases'], color ='purple')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)
	return uncertainty


def test_convergence(data_length, pop, predictions):
	converge = True
	deaths = [death[1] for death in predictions]
	min_death = min(deaths)
	max_death = max(deaths)
	if min_death < 0 or max_death > 0.1*pop:
		converge = False
	diff = deaths[-1] - deaths[data_length-1] 
	if diff < 0 or deaths[-1] > 0.3*pop:
		converge = False
	return converge	

def fill_nonconvergent(nonconvergent, data, end, error_start=0, fix_nonconvergent=False):
	counties_dates = []
	counties_death_errors = []
	counties_fips = nonconvergent
	for index, county in enumerate(nonconvergent):
		# if len(str(county)) == 4:
		# 	county = int('0'+str(county))
		county_data = loader.query(data, "fips", county)
		deaths = county_data["deaths"].values
		dates = pd.to_datetime(county_data["date"].values)
		extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')

		death_cdf = []
		if fix_nonconvergent:
			start = 0
			if error_start is not None and error_start < 0:
				deaths = deaths[:error_start]
				start = abs(error_start)

			for percentile in [10, 20, 30, 40, 50, 60, 70]: #make this a separate function
				# latest_D = deaths[-1]
				# bound = (1 + percentile/200)*latest_D
				predictions = [0 for i in range(int(14)+start)]
				predictions = predictions + [0 for i in range(int(extrapolate-14))]
				forecast = list(np.concatenate((deaths, predictions)))
				death_cdf.append(forecast)

			for percentile in [80, 90]:
				if percentile == 80:
					bound = np.mean(deaths)
				else:
					bound = np.mean([d for d in deaths if d>0])
				predictions = [bound for i in range(int(14)+start)]
				predictions = predictions + [0 for i in range(int(extrapolate-14))]
				forecast = list(np.concatenate((deaths, predictions)))
				death_cdf.append(forecast)
		else:
			for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]: #make this a separate function
				predictions = [0 for i in range(int(extrapolate))]
				forecast = list(np.concatenate((deaths, predictions)))
				death_cdf.append(forecast)

		death_cdf = np.transpose(death_cdf)
		counties_dates.append(dates)
		counties_death_errors.append(death_cdf)

	return (counties_dates, counties_death_errors, counties_fips)
		

def fit(data, bias=None, bias_value=0.4, weight=False, plot=False, extrapolate=14, guesses=None, error_start=-1, quick=False, tail=False, fitQ=False, getbounds=False, death_metric="deaths"):
	param_ranges = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
	initial_ranges = [(0,1), (0,1), (0,0.01), (0,0.01), (0,0.01), (0,0.01), (0,0.01)]
	ranges = param_ranges+initial_ranges
	if guesses is None:
		params = [9e-02, 1e-01, 7e-02, 3.e-01, 4.e-01, 1e-01, 1e-01, 3e-01, 4e-01, 7e-02, 2e-04, 8e-02, 7e-03, 2e-02, 2e-04, 2e-06, 4e-03]
		initial_conditions = [7e-01, 2e-01, 4e-08, 7e-03, 1e-08, 3e-20, 7e-06]
		guesses = params+initial_conditions

	else:
		initial_ranges = [(0.1*guesses[17],10*guesses[17]), (0.1*guesses[18],10*guesses[18]), (0.1*guesses[19],10*guesses[19]), (0.1*guesses[20],10*guesses[20]), (0.1*guesses[21],10*guesses[21]), \
		(0.1*guesses[22],10*guesses[22]), (0.1*guesses[23],10*guesses[23])]
		ranges = param_ranges+initial_ranges

	if bias is not None and bias < 0:
		bias = None

	for boundary in [len(data)]:
		res = least_squares(leastsq_qd, guesses, args=(data[:boundary],bias, bias_value, weight, fitQ, death_metric), bounds=np.transpose(np.array(ranges)))
		predictions = get_deaths(res, data, extrapolate=extrapolate)
		convergent_status = test_convergence(len(data), data['Population'].values[0], predictions) 
		if convergent_status == False:
			return (None,None,None)

		death_pdf = []
		if plot:
			plot_model(res, data, extrapolate=extrapolate, boundary=boundary, plot_infectious=True, death_metric=death_metric)
			death_pdf = plot_with_errors_sample(res, guesses[:17], data, extrapolate=extrapolate, boundary=boundary, plot_infectious=False, error_start=error_start, quick=quick, tail=tail, death_metric=death_metric)
		else:
			if getbounds:
				death_pdf = get_fit_errors(res, guesses[:17], data, extrapolate=extrapolate, error_start=error_start, quick=quick, tail=tail, death_metric=death_metric)
			else:
				prediction_fit = [point[1] for point in predictions]
				death_error = quickie(prediction_fit, data, None, error_start=None)
				death_error = np.concatenate((data["daily_deaths"].values[0:1], death_error))
				death_pdf.append(death_error)
	return (predictions, death_pdf, res)


###########################################################
def test(end, bias=False, policy_regime=False, tail_regime=False, weight=True, plot=False, guesses=None, error_start=-1, quick=False, tail=False, fitQ=False, getbounds=True, adaptive=False, death_metric="deaths"):

	counties_dates = []
	counties_death_errors = []
	counties_fips = []
	nonconvergent = []
	parameters = {}

	# us = process_data("/data/us/covid/nyt_us_counties.csv", "/data/us/demographics/county_populations.csv")
	us = loader.load_data("/models/epidemiological/production/us_training_data.csv")
	us_daily = loader.load_data("/data/us/covid/nyt_us_counties_daily.csv")
	policies = loader.load_data("/data/us/other/policies.csv")
	fips_key = loader.load_data("/data/us/processing_data/fips_key.csv", encoding="latin-1")
	# fips_list = fips_key["FIPS"]
	fips_list = [36061] #34017, 17031, 25013, 34023, 36059, 33011      56013,1017, 44007, 42101, 6037 27053
	total = len(fips_list)

	for index, county in enumerate(fips_list):
		print(f"{index+1} / {total}")
		county_data = loader.query(us, "fips", county)
		county_data['daily_deaths'] = loader.query(us_daily, "fips", county)["deaths"]
		county_data['avg_deaths'] = county_data.iloc[:,6].rolling(window=3).mean()
		county_data = county_data[2:]
		
		firstnonzero = next((i for i,value in enumerate(county_data[death_metric].values) if value != 0), None)
		final_death = (county_data["deaths"].values)[-1]
		initial_death = (county_data["deaths"].values)[firstnonzero]
		if firstnonzero is not None:
			if firstnonzero > len(county_data)-7 or final_death-initial_death == 0:
				# add to nonconvergent counties
				nonconvergent.append(county)
				continue
			death_observations = (county_data['daily_deaths'].values)[firstnonzero:]
			if list(death_observations).count(0) > len(death_observations)/2:
				nonconvergent.append(county)
				continue # for fit_single_county use return [county]
			begin = firstnonzero-death_time
			if begin >= 0:
				county_data = county_data[begin:]
				firstnonzero = death_time
				county_data.reset_index(drop=True, inplace=True)
		else:
			continue # dont add to convonvergent counties, just leave blank and submission script will fill it in with all zeros

		if adaptive and death_metric=="deaths":
			actual_deaths = (county_data['deaths'].values)[firstnonzero:]
			moving_deaths = (county_data['avg_deaths'].values)[firstnonzero:]
			residuals = []
			for index in range(1, len(actual_deaths)):
				moving_change = moving_deaths[index] - moving_deaths[index-1]
				if moving_change > 0:
					residue = actual_deaths[index] - moving_deaths[index]
					residue = residue/moving_change
					residuals.append(residue)
			if np.std(residuals) > 0.25:
				print("gottem")
				death_metric = "avg_deaths"

		dates = pd.to_datetime(county_data["date"].values)
		county_policy_regime = policy_regime
		policy_regime_change = -2*death_time
		if bias or policy_regime:
			policy_date = loader.query(policies, "FIPS", county)["stay at home"]
			if len(policy_date) == 0 or math.isnan(policy_date.values[0]):
				county_policy_regime = False
			else:
				policy_date = int(policy_date.values[0])
				policy_regime_change = int((datetime.datetime.fromordinal(policy_date)-dates[0])/np.timedelta64(1, 'D'))
				if policy_regime_change < (death_time-5) or policy_regime_change  > len(county_data) - (death_time+5) or policy_regime_change <= firstnonzero:
					county_policy_regime = False
					policy_regime_change = -2*death_time

		if tail_regime and type(tail_regime)==int:
			tail = tail_regime
			extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')
			county_tail_regime = len(county_data) + tail_regime
			county_tail_regime = max(firstnonzero, county_tail_regime)
			predictions, death_pdf, res = fit(county_data, bias=policy_regime_change+death_time, weight=weight, plot=False, extrapolate=extrapolate, guesses=guesses, fitQ=fitQ, getbounds=getbounds, death_metric=death_metric)
			if res is not None:
				predictions2, death_pdf2, res2 = fit(county_data, bias=county_tail_regime, bias_value=0.01, weight=weight, plot=plot, extrapolate=extrapolate, guesses=res.x, error_start=error_start, quick=quick, tail=tail, fitQ=fitQ, getbounds=getbounds, death_metric=death_metric)
				if res2 is not None:
					predictions, death_pdf, res = predictions2, death_pdf2, res2

		else:
			extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')
			predictions, death_pdf, res = fit(county_data, bias=policy_regime_change+death_time, weight=weight, plot=plot, extrapolate=extrapolate, guesses=guesses, error_start=error_start, quick=quick, tail=tail, fitQ=fitQ, getbounds=getbounds, death_metric=death_metric)

		if res is None:
			# add to nonconvergent counties
			nonconvergent.append(county)
			continue
		print(list(res.x))

		death_cdf = get_death_cdf(death_pdf, extrapolate, switch=quick)
		if death_cdf is None:
			if policy_regime:
				death_pdf = get_fit_errors2(res, guesses[:17], county_data1, county_data2, extrapolate=extrapolate, error_start=error_start, quick=True, tail=tail, death_metric=death_metric)
			else:
				death_pdf = get_fit_errors(res, guesses[:17], county_data, extrapolate=extrapolate, error_start=error_start, quick=True, tail=tail, death_metric=death_metric)
			death_cdf = get_death_cdf(death_pdf, extrapolate, switch=True)
		

		death_cdf = np.transpose(death_cdf)
		counties_dates.append(dates)
		counties_death_errors.append(death_cdf)
		counties_fips.append(county)
		if county in parameters.keys():
			parameters[county].append(res.x)
		else:
			parameters[county] = [res.x]

	if len(nonconvergent) > 0:
		print(f"nonconvergent: {nonconvergent}")
		counties_dates_non, counties_death_errors_non, counties_fips_non = fill_nonconvergent(nonconvergent, us_daily, end, error_start=error_start) 
		counties_dates = counties_dates + counties_dates_non
		for death_cdf in counties_death_errors_non:
			counties_death_errors.append(death_cdf)
		# county_death_errors = counties_death_errors + counties_death_errors2
		counties_fips = counties_fips + counties_fips_non

	output_dict = {"counties_dates": np.array(counties_dates), "counties_death_errors": np.array(counties_death_errors), "counties_fips": np.array(counties_fips), \
	"nonconvergent": nonconvergent, "parameters": parameters}

	return output_dict





class Empty:
    pass 

def generate_single_confidence(input_dict):
	us = input_dict["us"]
	us_daily = input_dict["us_daily"]
	policies = input_dict["policies"]
	county = input_dict["county"]
	end = input_dict["end"]
	bias = input_dict["bias"]
	policy_regime = input_dict["policy_regime"]
	tail_regime = input_dict["tail_regime"]
	weight = input_dict["weight"]
	params = input_dict["params"]
	quick = input_dict["quick"]
	error_start = input_dict["error_start"]
	tail = input_dict["tail"]
	adaptive = input_dict["adaptive"]
	death_metric = input_dict["death_metric"]
	nonconvergent = None 

	if params is None:
		# add to nonconvergent counties
		return [county]

	res = Empty()
	res.x = params

	county_data = loader.query(us, "fips", county)
	county_data['daily_deaths'] = loader.query(us_daily, "fips", county)["deaths"]
	county_data['avg_deaths'] = county_data.iloc[:,6].rolling(window=3).mean()
	county_data = county_data[2:-8]

	firstnonzero = next((index for index,value in enumerate(county_data[death_metric].values) if value != 0), None)
	if firstnonzero is not None:
		if firstnonzero > len(county_data)-7 or (county_data["deaths"].values)[-1]-(county_data["deaths"].values)[firstnonzero] == 0:
			return [county] # add to nonconvergent counties
		death_observations = (county_data['daily_deaths'].values)[firstnonzero:]
		if list(death_observations).count(0) > len(death_observations)/2:
			return [county]
		begin = firstnonzero-death_time
		if begin >= 0:
			county_data = county_data[begin:]
			firstnonzero = death_time
			county_data.reset_index(drop=True, inplace=True)
	else:
		return None # dont add to nonconvergent counties, just leave blank and submission script will fill it in with all zeros

	if adaptive and death_metric == "deaths":
		actual_deaths = (county_data['deaths'].values)[firstnonzero:]
		moving_deaths = (county_data['avg_deaths'].values)[firstnonzero:]
		residuals = []
		for index in range(1, len(actual_deaths)):
			moving_change = moving_deaths[index] - moving_deaths[index-1]
			if moving_change > 0:
				residue = actual_deaths[index] - moving_deaths[index]
				residue = residue/moving_change
				residuals.append(residue)
		if np.std(residuals) >= 0.25:
			death_metric = "avg_deaths"

	dates = pd.to_datetime(county_data["date"].values)
	extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')

	policy_regime_change = -2*death_time
	if bias or policy_regime:
		policy_date = loader.query(policies, "FIPS", county)["stay at home"]
		if len(policy_date) == 0:
			bias = False
			policy_regime = False
		else:
			policy_date = int(policy_date.values[0])
			policy_regime_change = int((datetime.datetime.fromordinal(policy_date)-dates[0])/np.timedelta64(1, 'D'))
			if policy_regime_change < (death_time-5) or policy_regime_change  > len(county_data) - (death_time+5) or policy_regime_change <= firstnonzero:
				bias = False
				policy_regime = False
				policy_regime_change = -2*death_time

	if tail_regime and type(tail_regime)==int:
		tail = tail_regime
	plot_model(res, county_data, extrapolate=extrapolate, plot_infectious=True, death_metric=death_metric)
	death_pdf = plot_with_errors_sample(res, params, county_data, extrapolate=extrapolate, error_start=error_start, quick=True, tail=tail, death_metric=death_metric)
		# death_pdf = get_fit_errors(res, params, county_data, extrapolate=extrapolate, error_start=error_start, quick=True, tail=tail, death_metric=death_metric)
	death_cdf = get_death_cdf(death_pdf, extrapolate, switch=True)
	death_cdf = np.transpose(death_cdf)

	return (dates, death_cdf, county)





def multi_generate_confidence(combined_parameters, end, quick=True, error_start=-14, tail=False, fix_nonconvergent=False):
	#Get date range of April1 to June30 inclusive. Figure out how much to extrapolate
	counties_dates = []
	counties_death_errors = []
	counties_fips = []
	nonconvergent = []

	# us = process_data("/data/us/covid/nyt_us_counties.csv", "/data/us/demographics/county_populations.csv")
	us = loader.load_data("/models/epidemiological/production/us_training_data.csv")
	us_daily = loader.load_data("/data/us/covid/nyt_us_counties_daily.csv")
	policies = loader.load_data("/data/us/other/policies.csv")
	policies = policies.dropna(subset=['stay at home'])
	fips_key = loader.load_data("/data/us/processing_data/fips_key.csv", encoding="latin-1")
	fips_list = fips_key["FIPS"]

	data = []
	for county in fips_list:
		input_dict = {}
		input_dict["us"] = us
		input_dict["us_daily"] = us_daily
		input_dict["policies"] = policies
		input_dict["county"] = county
		input_dict["end"] = end
		input_dict["quick"] = quick
		input_dict["error_start"] = error_start 
		input_dict['tail'] = tail
		# input_dict["params"] = combined_parameters["params"]
		# input_dict["policy_regime"] = combined_parameters["policy_regime"]
		# input_dict["tail_regime"] = combined_parameters["tail_regime"]
		# input_dict["adaptive"] = combined_parameters["adaptive"]
		# input_dict["death_metric"] = combined_parameters["death_metric"]
		county_key = str(county)
		if county_key in list(combined_parameters.keys()): #combined_parameters has an entry for every county from the optimize script submission files. If the file they came from not have parameters for county, feed in None
			if combined_parameters[county_key] is None:
				continue
			for key in list(combined_parameters[county_key].keys()):
				input_dict[key] = (combined_parameters[county_key])[key]
		else:
			continue
		data.append(input_dict)

	if len(data) == 1:
		result = generate_single_confidence(data[0])
		if len(result) == 1:
			nonconvergent.append(result[0]) 
		else:
			(dates, death_cdf, county) = result
			counties_dates.append(dates)
			counties_death_errors.append(death_cdf)
			counties_fips.append(county)
	else:
		pool = Pool(os.cpu_count()) ## According to TA this will saturate more cores in the hpc?
		results = pool.map(generate_single_confidence, data)
	
		for result in results:
			if result is not None:
				if len(result) == 1:
					nonconvergent.append(result[0]) 
				else:
					(dates, death_cdf, county) = result
					counties_dates.append(dates)
					counties_death_errors.append(death_cdf)
					counties_fips.append(county)

	if len(nonconvergent) > 0:
		counties_dates_non, counties_death_errors_non, counties_fips_non = fill_nonconvergent(nonconvergent, us_daily, end, fix_nonconvergent=fix_nonconvergent) 
		counties_dates = counties_dates + counties_dates_non
		for death_cdf in counties_death_errors_non:
			counties_death_errors.append(death_cdf)
		counties_fips = counties_fips + counties_fips_non

	output_dict = {"counties_dates": np.array(counties_dates), "counties_death_errors": np.array(counties_death_errors), "counties_fips": np.array(counties_fips), \
	"nonconvergent": nonconvergent}
	return output_dict





###########################################################


if __name__ == '__main__':
	end = datetime.datetime(2020, 6, 30)
	# guesses = [1.41578513e-01, 1.61248129e-01, 2.48362028e-01, 3.42978127e-01, 5.79023652e-01, 4.64392758e-02, \
	# 9.86745420e-06, 4.83700388e-02, 4.85290835e-01, 3.72688900e-02, 4.92398129e-04, 5.20319673e-02, \
	# 4.16822944e-02, 2.93718207e-02, 2.37765976e-01, 6.38313283e-04, 1.00539865e-04, 7.86113867e-01, \
	# 3.26287443e-01, 8.18317732e-06, 5.43511913e-10, 1.30387168e-04, 3.58953133e-03, 1.57388153e-05]

	guesses = None 
	# guesses = [0.347788756754361, 0.34491677887291705, 0.003154722699912536, 0.4985060201450094, 0.550563984674294, 0.03823471453254752, 0.0015897569370657341, 0.37585163777022107, 0.4154389095590687, 0.08168387387250878, 9.304462310605794e-06, 0.06250864864173467, 0.021263444109332643, 0.05731367025219088, 0.16611190329700892, 0.0013937980057879332, 0.00042326122491193784, 0.7664703213628267, 0.6574011838097715, 2.183979980914606e-07, 1.1781675019848431e-09, 6.883361598806804e-05, 0.009839663454234501, 2.03354867231974e-05]
	# guesses = [0.09450366904673224, 0.0004710760723271189, 0.1400163735156153, 0.22484294992189927, 0.45928280320621817, 0.10881967853088996, 0.04383382741493148, 0.3198398363291732, 0.31772919729568444, 0.18029197284930204, 4.5633833774879513e-10, 0.0001555994932997694, 0.03131462536531793, 0.06482950780667499, 0.09014858524860507, 4.0645019531833214e-09, 0.03245638906145152, 0.739559707217159, 0.24539613824473772, 0.0039457747991040164, 0.0066152618110283355, 0.00023626676203165555, 7.944968195828775e-38, 7.057330849865465e-06]
	test(end, bias=False, policy_regime=False, tail_regime=False, weight=True, plot=True, guesses=guesses, error_start=None, quick=True, tail=False, fitQ=False, adaptive=True, death_metric="deaths")

	# test(end, bias=True, policy_regime=False, tail_regime=-14, weight=True, plot=True, guesses=guesses, error_start=None, quick=True, tail=False, fitQ=True, adaptive=True, death_metric="deaths")


	# params1 = [0.347788756754361, 0.34491677887291705, 0.003154722699912536, 0.4985060201450094, 0.550563984674294, 0.03823471453254752, 0.0015897569370657341, 0.37585163777022107, 0.4154389095590687, 0.08168387387250878, 9.304462310605794e-06, 0.06250864864173467, 0.021263444109332643, 0.05731367025219088, 0.16611190329700892, 0.0013937980057879332, 0.00042326122491193784, 0.7664703213628267, 0.6574011838097715, 2.183979980914606e-07, 1.1781675019848431e-09, 6.883361598806804e-05, 0.009839663454234501, 2.03354867231974e-05]
	# # # params1 = list(0.8*np.array(params1))
	# a_1 = params1[0] #decrease to delay
	# a_2 = params1[1] #decrease to delay
	# a_3 = params1[2] #increase to raise peak
	# b_1 = params1[3] #decrease to delay
	# b_2 = 0.95*params1[4] #decrease to delay
	# b_3 = 2*params1[5] #Increase to delay
	# b_4 = params1[6] #Increase to delay
	# g_a = 0.8*params1[7] #Increase to lower peak
	# g_i = 1.4*params1[8] #Decrease to widen
	# th = 0.1*params1[9] #Decrease to make tail heavier
	# del_a = params1[10]
	# del_i = params1[11] #Increase to flatten
	# r_a = params1[12]
	# r_i = 0.8*params1[13] #Increase to flatten
	# r_q = params1[14]
	# d_i = 1.2*params1[15] #Increase to raise peak
	# d_q = params1[16]
	# P0 = params1[17]
	# E0 = params1[18]
	# C0 = params1[19]
	# A0 = params1[20]
	# I0 = params1[21]
	# Q0 = params1[22]
	# R0 = params1[23]
	# params = [a_1, a_2, a_3, b_1, b_2, b_3, b_4, g_a, g_i, th, del_a, del_i, r_a, r_i, r_q, d_i, d_q, P0, E0, C0, A0, I0, Q0, R0]
	# combined_parameters = {
	# "36061":{"params": params,\
	# "bias":True, "weight":True, "policy_regime":False, "tail_regime":-14, "death_metric":"deaths", "adaptive":True}}

	# output = multi_generate_confidence(combined_parameters, end, quick=True, error_start=None)
	# # print(list(output["counties_death_errors"]))


