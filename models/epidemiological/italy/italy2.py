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



def process_data(data_covid, data_population, save=True):
	covid = loader.load_data(data_covid)
	loader.convert_dates(covid, "Date")
	population = loader.load_data(data_population)
	covid['Population'] = covid.apply(lambda row: loader.query(population, "Region", row.Region)['Population'], axis=1)
	if save:
		covid.to_csv(f"{homedir}" + "/models/epidemiological/italy/italy_training_data.csv")
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


###########################################################

def pecaiqr(dat, t, params, N, max_t):
	# define a time td of social distancing
	# for t > td, divide dI/dt and dA/dt and dQ/dt and dC/dt by 2 
	# might not be able to do this, as there is still one parameter fit, bad to average
	# need to paste together two separate parameter fit policy_regimes, fitted on time<td and time>td. Initial conditions of second fit are the last datapoints before td
	# initial parameters of the second fit are the fitted parameters of the first fit. Perhaps even fix some parameters to the values from the original fit? Look at leastsquares documentation
	# this way i wont need to guess how much social distancing reduces the differentials, and i can also output new parameters to see how they change
	if t >= max_t:
		return [0]*8
	a_1 = params[0]
	a_2 = params[1]
	a_3 = params[2]
	b_1 = params[3]
	b_2 = params[4]
	b_3 = params[5]
	b_4 = params[6]
	g_a = params[7]
	g_i = params[8]
	th = params[9]
	del_a = params[10]
	del_i = params[11]
	r_a = params[12]
	r_i = params[13]
	r_q = params[14]
	d_i = params[15]
	d_q = params[16]

	P = dat[0]
	E = dat[1]
	C = dat[2]
	A = dat[3]
	I = dat[4]
	Q = dat[5]
	R = dat[6]

	dPdt = (- ((a_1+a_2)*C*P)/N) + (-a_3*P + b_4*E)*(N/(P+E))
	dEdt = (- (b_1 * A + b_2 * I) * E / N) + b_3*C + (a_3*P - b_4*E)*(N/(P+E))
	dCdt = -(g_a + g_i)*C + ((b_1 * A + b_2 * I) * E / N) - b_3*C
	dAdt = (a_1 * C * P) / N + g_a*C - (r_a + del_a + th)*A
	dIdt = (a_2 * C * P) / N + g_i*C - ((r_i+d_i)+del_i)*I+th*A
	dQdt = del_a*A + del_i*I - (r_q+d_q)*Q
	dRdt = r_a*A + (r_i+d_i)*I + (r_q+d_q)*Q
	dDdt = d_i*I + d_q*Q

	dzdt = [dPdt, dEdt, dCdt, dAdt, dIdt, dQdt, dRdt, dDdt]
	return dzdt

def model(params, data, extrapolate=-1, offset=0, death_metric="Deaths"):
	N = data['Population'].values[0] # total population
	initial_conditions = N * np.array(params[-7:]) # the parameters are a fraction of the population so multiply by the population
	P0 = initial_conditions[0]
	E0 = initial_conditions[1]
	C0 = initial_conditions[2]
	A0 = initial_conditions[3]
	I0 = initial_conditions[4]
	Q0 = initial_conditions[5]
	# Q0 = data['active_cases'].values[0] #fit to active cases instead
	R0 = initial_conditions[6]
	D0 = abs(data[death_metric].values[0])
	yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	
	n = len(data)
	if extrapolate > 0:
		n += extrapolate
	args = (params, N, n)
	try:
		s = scipy.integrate.odeint(pecaiqr, yz_0, np.arange(offset, n), args=args) #printmsg = True
		# s = scipy.integrate.solve_ivp(fun=lambda t, y: pecaiqr(y, t, params, N, n), t_span=[offset, n], y0=yz_0, t_eval=np.arange(offset, n), method="LSODA")
	except RuntimeError:
		print('RuntimeError', params)
		return np.zeros((n, len(yz_0)))

	return s

def model_ivp(params, data, extrapolate=-1, offset=0, death_metric="Deaths"):
	N = data['Population'].values[0] # total population
	initial_conditions = N * np.array(params[-7:]) # the parameters are a fraction of the population so multiply by the population
	P0 = initial_conditions[0]
	E0 = initial_conditions[1]
	C0 = initial_conditions[2]
	A0 = initial_conditions[3]
	I0 = initial_conditions[4]
	Q0 = initial_conditions[5]
	# Q0 = data['active_cases'].values[0] #fit to active cases instead
	R0 = initial_conditions[6]
	D0 = abs(data[death_metric].values[0])
	yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	
	n = len(data) + extrapolate

	solved = scipy.integrate.solve_ivp(fun=lambda t, y: pecaiqr(y, t, params, N, n), t_span=[offset, n], y0=yz_0, t_eval=np.arange(offset, n), method="LSODA")
	s = solved.y
	status = solved.success
	print(status)
	# if status is False or diff < 0:
	# 	s = None 
	return s	

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
		actual_current = (data["Deaths"].values)[tail:end]
		actual_previous = (data["Deaths"].values)[-1+tail:-1]
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
	offset = len(data)+error_start
	bound_mean, bound_deviation = guess_bounds
	# bound = []
	change_bound = []
	predictions = fit[:,7][(offset-1):]
	scaler = np.random.normal(loc=bound_mean, scale=bound_deviation)
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
def get_fit_errors(res, p0_params, data, extrapolate=14, error_start=-1, quick=False, tail=False, death_metric="Deaths"):
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

def plot_model(res, data, extrapolate=14, boundary=None, plot_infectious=False, death_metric="Deaths"):   
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
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)

def plot_with_errors_sample(res, p0_params, data, extrapolate=14, boundary=None, plot_infectious=False, error_start=-1, quick=False, tail=False, death_metric="Deaths"):
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

	p.varea(x=tp, y1=s1, y2=s2, color='black', fill_alpha=0.2)

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

		

def leastsq_qd(params, data, bias=None, bias_value=0.4, weight=False, fitQ=False, death_metric="Deaths"):
	Ddata = (data[death_metric].values)
	s = model(params, data)

	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]

	last_weight = 1.5
	if weight and weight is not True:
		if weight > 0.5:
			last_weight = weight

	w = 1
	regime_boundary = 0
	if bias is not None:
		regime_boundary = bias
		w0 = np.zeros(regime_boundary)+bias_value
		w = np.concatenate((w0, np.zeros(len(data)-regime_boundary)+1))

	if weight:
		w1 = np.geomspace(0.5,last_weight,len(data)-regime_boundary)
		if bias is not None:
			w = np.concatenate((w[:regime_boundary],w1))
		else:
			w = w1
		

	d_error = D-Ddata
	for i, dval in enumerate(Ddata):
		if dval < 0:
			d_error[i] = 0

	d_error = d_error*w
	return d_error


def fit(data, bias=None, bias_value=0.4, weight=False, plot=False, extrapolate=14, guesses=None, error_start=-1, quick=False, tail=False, fitQ=False, getbounds=False, death_metric="Deaths"):
	param_ranges = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
	initial_ranges = [(0,1), (0,1), (0,0.01), (0,0.01), (0,0.01), (0,0.01), (0,0.01)]
	ranges = param_ranges+initial_ranges
	if guesses is None:
		params = [9e-02, 1e-01, 7e-02, 3.e-01, 4.e-01, 1e-01, 1e-01, 3e-01, 4e-01, 7e-02, 2e-04, 8e-02, 7e-03, 2e-02, 2e-04, 2e-06, 4e-03]
		initial_conditions = [7e-01, 2e-01, 4e-08, 7e-03, 1e-08, 3e-20, 7e-06]
		guesses = params+initial_conditions

	else:
		initial_ranges = [(0.8*guesses[17],1.2*guesses[17]), (0.8*guesses[18],1.2*guesses[18]), (0.8*guesses[19],1.2*guesses[19]), (0.8*guesses[20],1.2*guesses[20]), (0.8*guesses[21],1.2*guesses[21]), \
		(0, 0.01), (0.8*guesses[23],1.2*guesses[23])]
		ranges = param_ranges+initial_ranges

	if bias is not None and bias < 0:
		bias = None

	for boundary in [len(data)]:
		res = least_squares(leastsq_qd, guesses, args=(data[:boundary],bias, bias_value, weight, fitQ, death_metric), bounds=np.transpose(np.array(ranges)))
		predictions = get_deaths(res, data, extrapolate=extrapolate)

		death_pdf = None
		if plot:
			plot_model(res, data, extrapolate=extrapolate, boundary=boundary, plot_infectious=True, death_metric=death_metric)
			death_pdf = plot_with_errors_sample(res, guesses[:17], data, extrapolate=extrapolate, boundary=boundary, plot_infectious=False, error_start=error_start, quick=quick, tail=tail, death_metric=death_metric)
		else:
			if getbounds:
				death_pdf = get_fit_errors(res, guesses[:17], data, extrapolate=extrapolate, error_start=error_start, quick=quick, tail=tail, death_metric=death_metric)

	return (predictions, death_pdf, res)


###########################################################
def test(weight=True, plot=True, cutoff=None):
	# italy = process_data("/data/international/italy/covid/dpc-covid19-ita-regioni.csv", "/models/data/international/italy/demographics/region-populations.csv")
	italy = loader.load_data("/models/epidemiological/italy/italy_training_data.csv")
	lombardia = loader.query(italy, "Region", "Lombardia")
	lombardia['avg_deaths'] = lombardia.iloc[:,-8].rolling(window=3).mean()
	if cutoff is None:
		cutoff = len(lombardia)
	lombardia = lombardia[:cutoff]
	current_deaths = lombardia['Deaths'].values[2:]
	yesterday_deaths = lombardia['Deaths'].values[1:-1]
	lombardia = lombardia[2:]
	lombardia['daily_deaths'] = current_deaths-yesterday_deaths
	# guesses = [6.69209312e-02, 1.10239913e-01, 4.33677422e-02, 3.01411969e-01,
	# 3.55547441e-01, 1.35711130e-01, 1.87415444e-01, 3.40118459e-01,
	# 6.54169531e-01, 5.80742686e-02, 2.66926724e-05, 1.27460914e-01,
	# 3.14216375e-02, 1.33884397e-06, 3.95164660e-02, 5.51770694e-11,
	# 1.27560414e-02, 6.55819545e-01, 1.66249610e-01, 8.78316719e-10,
	# 9.72332562e-03, 1.18016076e-18, 9.50245298e-18, 7.02140155e-06]
	guesses = [3.29142138e-01, 6.10729377e-02, 1.54971604e-01, 4.57604830e-01,
	4.00514413e-01, 2.29936105e-02, 3.12355123e-04, 4.19042120e-02,
	5.20538956e-01, 5.89851095e-05, 7.51938317e-07, 1.48040699e-01,
	2.79663721e-02, 1.22173817e-01, 1.33376981e-01, 9.21393091e-08,
	1.18451662e-03, 8.58246731e-01, 1.02247495e-01, 4.45881513e-10,
	1.92751396e-02, 2.36032152e-18, 3.35111924e-05, 7.02140155e-06]
	# guesses = [1.41578513e-01, 1.61248129e-01, 2.48362028e-01, 3.42978127e-01, 5.79023652e-01, 4.64392758e-02, \
	# 9.86745420e-06, 4.83700388e-02, 4.85290835e-01, 3.72688900e-02, 4.92398129e-04, 5.20319673e-02, \
	# 4.16822944e-02, 2.93718207e-02, 2.37765976e-01, 6.38313283e-04, 1.00539865e-04, 7.86113867e-01, \
	# 3.26287443e-01, 8.18317732e-06, 5.43511913e-10, 1.30387168e-04, 3.58953133e-03, 1.57388153e-05]
	fit(lombardia, guesses=guesses, weight=weight, plot=plot, quick=True, error_start=None, extrapolate=58)


if __name__ == '__main__':
	test(cutoff=-30)

