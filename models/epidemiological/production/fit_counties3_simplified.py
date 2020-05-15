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

death_time = 14

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
		max_total = death_cdf[-1][-1*(extrapolate-1):end]
		max_total_previous = death_cdf[-1][-1*(extrapolate):end-1]
		min_total = death_cdf[0][-1*(extrapolate-1):end]
		min_total_previous = death_cdf[0][-1*(extrapolate):end-1]
		max_daily_change = [i - j for i, j in zip(max_total, max_total_previous)]
		min_daily_change = [i - j for i, j in zip(min_total, min_total_previous)]
		expected_total = death_cdf[4][-1*(extrapolate-1):end]
		expected__total_previous = death_cdf[4][-1*(extrapolate):end-1]
		expected__daily_change = [i - j for i, j in zip(expected_total, expected__total_previous)]

		expected = np.mean(expected__daily_change)
		diff = np.mean(np.array(max_daily_change)-np.array(min_daily_change))
		# ratio = np.mean(np.array(max_daily_change)/np.array(min_daily_change))
		ratio = diff/expected
		if ratio > 0.5:
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
def get_param_errors(res, pop):
	pfit = res.x
	pcov = res.jac
	pcov = np.dot(pcov.T, pcov)
	pcov = np.linalg.pinv(pcov) #uses svd
	pcov = np.diag(pcov)
	rcov = np.cov(res.fun)/pop ##put res.fun/pop inside
	# scaler = res.cost*len(res.fun)/pop 
	# rcov = rcov * scaler
	perr = pcov * rcov
	perr = np.sqrt(perr)
	return perr

###########################################################

def pecaiqr(dat, t, params, N, max_t):
	# define a time td of social distancing
	# for t > td, divide dI/dt and dA/dt and dQ/dt and dC/dt by 2 
	# might not be able to do this, as there is still one parameter fit, bad to average
	# need to paste together two separate parameter fit regimes, fitted on time<td and time>td. Initial conditions of second fit are the last datapoints before td
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

	dPdt = (- ((a_1+a_2)*C*P)/N)
	dEdt = (- (b_1 * A + b_2 * I) * E / N) + b_3*C
	dCdt = -(g_a + g_i)*C + ((b_1 * A + b_2 * I) * E / N) - b_3*C
	dAdt = (a_1 * C * P) / N + g_a*C - (r_a + del_a + th)*A
	dIdt = (a_2 * C * P) / N + g_i*C - ((r_i+d_i)+del_i)*I+th*A
	dQdt = del_a*A + del_i*I - (r_q+d_q)*Q
	dRdt = r_a*A + (r_i+d_i)*I + (r_q+d_q)*Q
	dDdt = d_i*I + d_q*Q

	dzdt = [dPdt, dEdt, dCdt, dAdt, dIdt, dQdt, dRdt, dDdt]
	return dzdt

def model(params, data, extrapolate=-1, offset=0, death_metric="deaths"):
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

def model_ivp(params, data, extrapolate=-1, offset=0, death_metric="deaths"):
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

def model_beyond(fit, params, data, guess_bounds, extrapolate=-1, start=-1):
	offset = len(data)+start
	N = data['Population'].values[0] # total population
	P0 = fit[:,0][offset]
	E0 = fit[:,1][offset]
	C0 = fit[:,2][offset]
	A0 = fit[:,3][offset]
	I0 = fit[:,4][offset]
	Q0 = fit[:,5][offset]
	# Q0 = data['active_cases'].values[0] #fit to active cases instead
	R0 = fit[:,6][offset]

	# D0 = data[death_metric].values[start]
	D0 = fit[:,7][offset]
	yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	
	n = len(data)+extrapolate
	args = (params, N, n)
	try:
		s = scipy.integrate.odeint(pecaiqr, yz_0, np.arange(offset, n), args=args)
		# s = scipy.integrate.solve_ivp(fun=lambda t, y: pecaiqr(y, t, params, N, n), t_span=[offset, n], y0=yz_0)
	except RuntimeError:
		print('RuntimeError', params)
		death_series = quickie(fit, data, guess_bounds, start=start)
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


def estimate_bounds(res, data, fit):
	mean = 1
	deviation = 0.2
	end = len(data)
	firstnonzero = next((i for i,value in enumerate(data["avg_deaths"].values) if value != 0), None)
	actual_current = (data["avg_deaths"].values)[firstnonzero+1:end]
	actual_previous = (data["avg_deaths"].values)[firstnonzero:-1]
	actual_slope = [i - j for i, j in zip(actual_current, actual_previous)]
	fit_current = fit[:,7][firstnonzero+1:end]
	fit_previous = fit[:,7][firstnonzero:-1]
	fit_slope = [i - j for i, j in zip(fit_current, fit_previous)]
	slope_ratio = np.array(actual_slope)/np.array(fit_slope)
	slope_ratio = reject_outliers(slope_ratio)
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

def quickie(fit, data, guess_bounds, start=-1):
	offset = len(data)+start
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
def get_fit_errors(res, p0_params, data, extrapolate=14, start=-1, quick=False, death_metric="deaths"):
	population = list(data["Population"])[-1]
	errors = get_param_errors(res, population)
	errors[len(p0_params):] = 0

	if start is None:
		start = -1*len(data)+1

	fit = model(res.x, data, extrapolate)
	guess_bounds = estimate_bounds(res,data,fit)
	if guess_bounds == (None, None):
		return np.zeros((1,int(len(data)+extrapolate)))

	uncertainty = []
	samples = 100

	if extrapolate > 0 :
		if quick:
			for i in range(samples):
				death_series = quickie(fit, data, guess_bounds, start=start)
				latest_D = (data[death_metric].values)[-1]
				# death_series = np.concatenate((data[death_metric].values[0:len(data)], death_series[-1*start:]))
				death_series = np.concatenate((data["daily_deaths"].values[0:len(data)+start], death_series))
				# for index, death in enumerate(death_series):
				# 	if index >= len(data) and death <= latest_D:
				# 		death_series[index] = None
				uncertainty.append(death_series)
		else:
			for i in range(samples):
				sample = np.random.normal(loc=res.x, scale=errors)
				death_series = model_beyond(fit, sample, data, guess_bounds, extrapolate, start=start)
				latest_D = (data[death_metric].values)[-1]
				# death_series = np.concatenate((fit[:,7][0:len(data)-1], death_series))
				# death_series = np.concatenate((data[death_metric].values[0:len(data)], death_series[-1*start:]))
				death_series = np.concatenate((data["daily_deaths"].values[0:len(data)+start], death_series))
				for index, death in enumerate(death_series):
					if index >= len(data) and death <= latest_D:
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
	p.circle(t, data['active_cases'], color ='purple', legend='Tested Infected')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)

def plot_with_errors_sample(res, p0_params, data, extrapolate=14, boundary=None, plot_infectious=False, start=-1, quick=False, death_metric="deaths"):
	s = model(res.x, data, len(data)+extrapolate)
	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]

	uncertainty = get_fit_errors(res, p0_params, data, extrapolate=extrapolate, start=start, quick=quick, death_metric=death_metric)
	
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
	diff = deaths[-1] - deaths[data_length-1] 
	if diff < 0 or deaths[-1] > 0.3*pop:
		converge = False
	return converge	

def fill_nonconvergent(nonconvergent, data, end, fix_nonconvergent=False):
	counties_dates = []
	counties_death_errors = []
	counties_fips = nonconvergent
	for index, county in enumerate(nonconvergent):
		county_data = loader.query(data, "fips", county)
		deaths = county_data["daily_deaths"].values
		dates = pd.to_datetime(county_data["date"].values)
		extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')

		death_cdf = []
		if fix_nonconvergent:
			latest_D = deaths[-1]
			for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]: #make this a separate function
				# bound = (1 + percentile/200)*latest_D
				bound = np.mean(deaths)
				predictions = [bound for i in range(int(14))]
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
		

def leastsq_qd(params, data, bias=None, weight=False, fitQ=False, death_metric="deaths"):
	Ddata = (data[death_metric].values)
	Qdata = (data['active_cases'].values)
	s = model(params, data)

	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]

	w = 1
	regime_boundary = 0
	if bias is not None:
		regime_boundary = bias+death_time
		w0 = np.zeros(regime_boundary)+0.4
		w = np.concatenate((w0, np.zeros(len(data)-regime_boundary)+1))

	if weight:
		w1 = np.geomspace(0.5,1.5,len(data)-regime_boundary)
		if bias is not None:
			w = np.concatenate((w[:regime_boundary],w1))
		else:
			w = w1
		

	d_error = D-Ddata
	for i, dval in enumerate(Ddata):
		if dval < 0:
			d_error[i] = 0

	d_error = d_error*w

	if fitQ:
		q_error = Q-Qdata
		q_error = 0.1*q_error
		for i, qval in enumerate(Qdata):
			if qval <= 0:
				q_error[i] = 0
		if weight:
			q_error = q_error*w
		return np.concatenate((d_error, q_error))

	return d_error


def fit(data, bias=None, weight=False, plot=False, extrapolate=14, guesses=None, start=-1, quick=False, fitQ=False, getbounds=False, death_metric="deaths"):
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


	for boundary in [len(data)]:
		res = least_squares(leastsq_qd, guesses, args=(data[:boundary],bias, weight, fitQ, death_metric), bounds=np.transpose(np.array(ranges)))
		predictions = get_deaths(res, data, extrapolate=extrapolate)
		convergent_status = test_convergence(len(data), data['Population'].values[0], predictions) 
		if convergent_status == False:
			return (None,None,None)

		death_pdf = None
		if plot:
			plot_model(res, data, extrapolate=extrapolate, boundary=boundary, plot_infectious=True, death_metric=death_metric)
			death_pdf = plot_with_errors_sample(res, guesses[:17], data, extrapolate=extrapolate, boundary=boundary, plot_infectious=False, start=start, quick=quick, death_metric=death_metric)
		else:
			if getbounds:
				death_pdf = get_fit_errors(res, guesses[:17], data, extrapolate=extrapolate, start=start, quick=quick, death_metric=death_metric)

	return (predictions, death_pdf, res)

########################################################## 

def model2(params, data, extrapolate=-1, offset=0):
	N = data['Population'].values[0] # total population
	initial_conditions = N * np.array(params[-8:]) # the parameters are a fraction of the population so multiply by the population
	P0 = initial_conditions[0]
	E0 = initial_conditions[1]
	C0 = initial_conditions[2]
	A0 = initial_conditions[3]
	I0 = initial_conditions[4]
	Q0 = initial_conditions[5]
	# Q0 = data['active_cases'].values[0] #fit to active cases instead
	R0 = initial_conditions[6]
	D0 = initial_conditions[7]
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
	

# returns uncertainty of the fit for all variables
def get_fit_errors2(res, p0_params, original, data, extrapolate=14, start=-1, quick=False, death_metric="deaths"):
	population = list(data["Population"])[-1]
	errors = get_param_errors(res, population)
	errors[len(p0_params):] = 0

	if start is None:
		start = -1*len(data)+1

	fit = model2(res.x, data, extrapolate)
	guess_bounds = estimate_bounds(res,data,fit)
	if guess_bounds == (None, None):
		return np.zeros((1,int(len(data)+extrapolate)))

	uncertainty = []
	samples = 100

	if extrapolate > 0 :
		if quick:
			for i in range(samples):
				death_series = quickie(fit, data, guess_bounds, start=start)
				# latest_D = (data[death_metric].values)[-1]
				death_series = np.concatenate((data["daily_deaths"].values[0:len(data)+start], death_series))
				# for index, death in enumerate(death_series):
				# 	if index >= len(data) and death <= latest_D:
				# 		death_series[index] = None
				death_series = np.concatenate((original["daily_deaths"].values[0:len(original)-death_time], death_series))
				uncertainty.append(death_series)
		else:
			for i in range(samples):
				sample = np.random.normal(loc=res.x, scale=errors)
				death_series = model_beyond(fit, sample, data, guess_bounds, extrapolate, start=start)
				latest_D = (data[death_metric].values)[-1]
				# death_series = np.concatenate((fit[:,7][0:len(data)-1], death_series))
				death_series = np.concatenate((data["daily_deaths"].values[0:len(data)+start], death_series))
				for index, death in enumerate(death_series):
					if index >= len(data) and death <= latest_D:
						death_series[index] = None
				death_series = np.concatenate((original["daily_deaths"].values[0:len(original)-death_time], death_series))
				uncertainty.append(death_series)
	else: 
		for i in range(samples):
			sample = np.random.normal(loc=res.x, scale=errors)
			death_series = model2(sample, data, len(data)+extrapolate)
			death_series = s[:,7]
			uncertainty.append(death_series)
		
	uncertainty = np.array(uncertainty)
	return uncertainty

def plot_model2(res, data, extrapolate=14, boundary=None, plot_infectious=False, death_metric="deaths"):   
	s = model2(res.x, data, extrapolate=extrapolate)
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
	p.circle(t, data['active_cases'], color ='purple', legend='Tested Infected')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)

def plot_with_errors_sample2(res, res_original, p0_params, original, data, extrapolate=14, boundary=None, plot_infectious=False, start=-1, quick=False, death_metric="deaths"):
	s = model2(res.x, data, len(data)+extrapolate)
	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]

	p = bokeh.plotting.figure(plot_width=1000,
							  plot_height=600,
							 title = ' PECAIQR Model Errors',
							 x_axis_label = 't (days)',
							 y_axis_label = '# people')

	t = np.arange(0, len(original)-death_time+len(data))
	tp = np.arange(0, len(original)-death_time+len(data)+extrapolate)

	# # This plots the cumulative death metric
	# D = np.concatenate((original[death_metric].values[0:len(original)-death_time], D))
	# real_D = np.concatenate((original[death_metric].values[0:len(original)-death_time], data[death_metric]))

	# This plots the daily deaths
	D_o = model(res_original.x, original, extrapolate=-1)[:,7]
	daily_D_o = []
	for index, death in enumerate(D_o):
		if index == 0:
			daily_D_o.append(0)
		else:
			daily_D_o.append(death-D_o[index-1])

	daily_D = []
	for index, death in enumerate(D):
		if index == 0:
			daily_D.append(0)
		else:
			daily_D.append(death-D[index-1])
	D = np.concatenate((daily_D_o, daily_D[death_time:]))
	real_D = np.concatenate((original["daily_deaths"].values[0:len(original)-death_time], data["daily_deaths"]))

	p.line(tp, D, color = 'black', line_width = 1, legend = 'Deceased')
	p.circle(t, real_D, color ='black', legend = 'Daily Deaths')

	# uncertainty_0 = get_fit_errors(res_original, p0_params, original, start=None, quick=quick)
	# uncertainty = get_fit_errors2(res, p0_params, original, data, extrapolate=extrapolate, start=start, quick=quick)
	# uncertainty = np.concatenate((uncertainty_o[:,:],uncertainty[:,len(uncertainty_o[0]):]),axis=1)
	# s1 = np.nanpercentile(uncertainty, 10, axis=0)
	# s2 = np.nanpercentile(uncertainty, 90, axis=0)
	# p.varea(x=tp, y1=s1, y2=s2, color='black', fill_alpha=0.2)
	uncertainty_o = get_fit_errors(res_original, p0_params, original, extrapolate=len(data)-death_time, start=None, quick=quick)
	uncertainty = get_fit_errors2(res, p0_params, original, data, extrapolate=extrapolate, start=start, quick=quick)
	s1_o = np.nanpercentile(uncertainty_o, 10, axis=0)
	s2_o = np.nanpercentile(uncertainty_o, 90, axis=0)
	s1 = np.nanpercentile(uncertainty, 10, axis=0)
	s2 = np.nanpercentile(uncertainty, 90, axis=0)
	p.varea(x=tp, y1=s1_o, y2=s2_o, color='black', fill_alpha=0.1)
	p.varea(x=tp, y1=s1, y2=s2, color='black', fill_alpha=0.2)

	# quarantined
	# p.circle(t, data['cases'], color ='purple')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=len(original)-death_time+boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)
	return uncertainty


def get_variables2(res, data, index):
	extrapolate = -1
	if index > len(data)-1:
		extrapolate = index + (index-len(data)+1)
	s = model2(res.x, data, extrapolate=extrapolate)
	P = s[:,0][index]
	E = s[:,1][index]
	C = s[:,2][index]
	A = s[:,3][index]
	I = s[:,4][index]
	Q = s[:,5][index]
	R = s[:,6][index]
	D = s[:,7][index]

	return (P,E,C,A,I,Q,R,D)

def get_deaths2(res, data, extrapolate=14):   
	s = model2(res.x, data, len(data)+extrapolate)
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

def leastsq_qd2(params, data, weight=False, fitQ=False, death_metric="deaths"):
	Ddata = (data[death_metric].values)
	Qdata = (data['active_cases'].values)
	s = model2(params, data)

	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]


	d_error = D-Ddata
	for i, dval in enumerate(Ddata):
		if dval < 0:
			d_error[i] = 0

	if weight:
		w = np.geomspace(0.5,1.5,len(data))
		d_error = d_error*w

	if fitQ:
		q_error = Q-Qdata

		for i, qval in enumerate(Qdata):
			if qval <= 0:
				q_error[i] = 0
		if weight:
			w = np.geomspace(0.5,1.5,len(data))
			q_error = q_error*w
		return np.concatenate((d_error, q_error))

	return d_error


# TODO: The initial ranges from the guesses hav error ValueError: Each lower bound must be strictly less than each upper bound. for philadelphia county 42101
def fit2(original, res_original, data, weight=False, plot=False, extrapolate=14, guesses=None, start=-1, quick=False, fitQ=False, getbounds=False, death_metric="deaths"):
	param_ranges = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
	initial_ranges = [(0.9*guesses[17],1.1*guesses[17]), (0.9*guesses[18],1.1*guesses[18]), (0.9*guesses[19],1.1*guesses[19]), (0.9*guesses[20],1.1*guesses[20]), (0.9*guesses[21],1.1*guesses[21]), \
	(0.9*guesses[22],1.1*guesses[22]), (0.9*guesses[23],1.1*guesses[23]), (0.1*guesses[24],10*guesses[24])]
	ranges = param_ranges+initial_ranges

	for boundary in [len(data)]:
		res = least_squares(leastsq_qd2, guesses, args=(data[:boundary],weight,fitQ,death_metric), bounds=np.transpose(np.array(ranges)))
		predictions = get_deaths2(res, data, extrapolate=extrapolate)
		convergent_status = test_convergence(len(data), data['Population'].values[0], predictions) 
		if convergent_status == False:
			return (None,None,None)
		death_pdf = None

		if plot:
			plot_model2(res, data, extrapolate=extrapolate, boundary=boundary, plot_infectious=True, death_metric=death_metric)
			death_pdf = plot_with_errors_sample2(res, res_original, guesses[:17], original, data, extrapolate=extrapolate, boundary=boundary, plot_infectious=False, start=start, quick=quick, death_metric=death_metric)
		else:
			if getbounds:
				death_pdf = get_fit_errors2(res, guesses[:17], original, data, extrapolate=extrapolate, start=start, quick=quick, death_metric=death_metric)

	return (predictions, death_pdf, res)

###########################################################
def test(end, bias=False, regime=False, weight=True, plot=False, guesses=None, start=-1, quick=False, fitQ=False, getbounds=False, adaptive=False, death_metric="deaths"):

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
	# fips_list = fips_key["FIPS"][0:10]
	fips_list = [36061] #56013,1017, 44007, 42101, 6037 27053 42101
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
			if np.std(residuals) > 0.35:
				print("gottem")
				death_metric = "avg_deaths"

		dates = pd.to_datetime(county_data["date"].values)

		regime_change = None
		if bias or regime:
			policy_date = int(loader.query(policies, "FIPS", county)["stay at home"].values[0])
			regime_change = int((datetime.datetime.fromordinal(policy_date)-dates[0])/np.timedelta64(1, 'D'))

		if regime:
			if policy_date is None or regime_change > len(county_data) - (death_time+5):
				regime=False
				extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')
				predictions, death_pdf, res = fit(county_data, bias=regime_change, weight=weight, plot=plot, extrapolate=extrapolate, guesses=guesses, start=start, quick=quick, fitQ=fitQ, getbounds=True, death_metric=death_metric)

			else:
				county_data1 = county_data[:regime_change+death_time] ## experimental. Assumes first regime will carry over until death_time into future. Used to be just county_data[:regime_change]
				predictions, death_errors1, res0 = fit(county_data1, bias=None, weight=weight, plot=False, extrapolate=0, guesses=guesses, fitQ=fitQ, getbounds=False, death_metric=death_metric)
				first_parameters = (res0.x)[:17]
				first_conditions = get_variables(res0, county_data1, regime_change)
				first_conditions = np.append(first_conditions, (county_data1[death_metric].values)[regime_change]) 
				N = county_data['Population'].values[0]
				first_conditions = first_conditions/N
				parameter_guess = list(first_parameters)+list(first_conditions)
				parameters[county] = [parameter_guess]

				county_data2 = county_data[regime_change:]
				dates2 = dates[regime_change:]
				county_data2.reset_index(drop=True, inplace=True)
				
				for i in range(death_time):
					county_data2.at[i, death_metric] *= -1
				extrapolate = (end-dates2[-1])/np.timedelta64(1, 'D')
				predictions, death_pdf, res  = fit2(county_data1, res0, county_data2, weight=weight, plot=plot, extrapolate=extrapolate, guesses=parameter_guess, start=start, quick=quick, fitQ=fitQ, getbounds=True, death_metric=death_metric)

		else:
			extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')
			predictions, death_pdf, res = fit(county_data, bias=regime_change, weight=weight, plot=plot, extrapolate=extrapolate, guesses=guesses, start=start, quick=quick, fitQ=fitQ, getbounds=True, death_metric=death_metric)


		if res is None:
			# add to nonconvergent counties
			nonconvergent.append(county)
			continue
		print(res.x)

		death_cdf = get_death_cdf(death_pdf, extrapolate, switch=quick)
		if death_cdf is None:
			if regime:
				death_pdf = get_fit_errors2(res, guesses[:17], county_data1, county_data2, extrapolate=extrapolate, start=start, quick=True, death_metric=death_metric)
			else:
				death_pdf = get_fit_errors(res, guesses[:17], county_data, extrapolate=extrapolate, start=start, quick=True, death_metric=death_metric)
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
		counties_dates_non, counties_death_errors_non, counties_fips_non = fill_nonconvergent(nonconvergent, us, end) 
		counties_dates = counties_dates + counties_dates_non
		for death_cdf in counties_death_errors_non:
			counties_death_errors.append(death_cdf)
		# county_death_errors = counties_death_errors + counties_death_errors2
		counties_fips = counties_fips + counties_fips_non

	output_dict = {"counties_dates": np.array(counties_dates), "counties_death_errors": np.array(counties_death_errors), "counties_fips": np.array(counties_fips), \
	"nonconvergent": nonconvergent, "parameters": parameters}

	return output_dict


def fit_single_county(input_dict):
	us = input_dict["us"]
	us_daily = input_dict["us_daily"]
	policies = input_dict["policies"]
	county = input_dict["county"]
	end = input_dict["end"]
	bias = input_dict["bias"]
	regime = input_dict["regime"]
	weight = input_dict["weight"]
	guesses = input_dict["guesses"]
	start= input_dict["start"]
	quick = input_dict["quick"]
	fitQ = input_dict["fitQ"]
	getbounds = input_dict["getbounds"]
	adaptive = input_dict["adaptive"]
	death_metric = input_dict["death_metric"]
	nonconvergent = None 
	parameters = []

	county_data = loader.query(us, "fips", county)
	county_data['daily_deaths'] = loader.query(us_daily, "fips", county)["deaths"]
	county_data['avg_deaths'] = county_data.iloc[:,6].rolling(window=3).mean()
	county_data = county_data[2:]

	if len(county_data) == 0:
		return None # dont add to nonconvergent counties, just leave blank and submission script will fill it in with all zeros

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
		if np.std(residuals) >= 0.2:
			death_metric = "avg_deaths"

	dates = pd.to_datetime(county_data["date"].values)

	regime_change = None
	if bias or regime:
		policy_date = loader.query(policies, "FIPS", county)["stay at home"]
		if len(policy_date) == 0:
			bias = False
			regime = False
		else:
			policy_date = int(policy_date.values[0])
			regime_change = int((datetime.datetime.fromordinal(policy_date)-dates[0])/np.timedelta64(1, 'D'))
			if regime_change < (death_time-5) or regime_change  > len(county_data) - (death_time+5):
				bias = False
				regime = False
				regime_change = None

	if regime:
		county_data1 = county_data[:regime_change+death_time] ## experimental. Assumes first regime will carry over until death_time into future. Used to be just county_data[:regime_change]
		predictions, death_errors1, res0 = fit(county_data1, bias=None, weight=weight, plot=False, extrapolate=0, guesses=guesses, fitQ=fitQ, getbounds=False, death_metric=death_metric)
		first_parameters = (res0.x)[:17]
		first_conditions = get_variables(res0, county_data1, regime_change)
		first_conditions = np.append(first_conditions, (county_data1[death_metric].values)[regime_change]) 
		N = county_data['Population'].values[0]
		first_conditions = first_conditions/N
		parameter_guess = list(first_parameters)+list(first_conditions)
		parameters.append(parameter_guess)

		county_data2 = county_data[regime_change:]
		dates2 = dates[regime_change:]
		county_data2.reset_index(drop=True, inplace=True)
		
		for i in range(death_time):
			county_data2.at[i, death_metric] *= -1
		extrapolate = (end-dates2[-1])/np.timedelta64(1, 'D')
		predictions, death_pdf, res  = fit2(county_data1, res0, county_data2, weight=weight, plot=False, extrapolate=extrapolate, guesses=parameter_guess, start=start, quick=quick, fitQ=fitQ, getbounds=True, death_metric=death_metric)

	else:
		extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')
		predictions, death_pdf, res = fit(county_data, bias=regime_change, weight=weight, plot=False, extrapolate=extrapolate, guesses=guesses, start=start, quick=quick, fitQ=fitQ, getbounds=True, death_metric=death_metric)


	if res is None:
		# add to nonconvergent counties
		return [county]

	parameters.append(list(res.x))
	death_cdf = get_death_cdf(death_pdf, extrapolate, switch=quick)
	if death_cdf is None:
		if regime:
			death_pdf = get_fit_errors2(res, guesses[:17], county_data1, county_data2, extrapolate=extrapolate, start=start, quick=True, death_metric=death_metric)
		else:
			death_pdf = get_fit_errors(res, guesses[:17], county_data, extrapolate=extrapolate, start=start, quick=True, death_metric=death_metric)
		death_cdf = get_death_cdf(death_pdf, extrapolate, switch=True)
	death_cdf = np.transpose(death_cdf)

	print(county)

	return (dates, death_cdf, county, parameters)
	


def multi_submission(end, bias=False, regime=True, weight=True, guesses=None, start=-1, quick=False, fitQ=False, getbounds=False, adaptive=False, death_metric="deaths", fix_nonconvergent=True):
	#Get date range of April1 to June30 inclusive. Figure out how much to extrapolate
	counties_dates = []
	counties_death_errors = []
	counties_fips = []
	nonconvergent = []
	parameters_list = {}

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
		input_dict["bias"] = bias
		input_dict["regime"] = regime
		input_dict["weight"] = weight
		input_dict["guesses"] = guesses
		input_dict["start"] = start
		input_dict["quick"] = quick
		input_dict["fitQ"] = fitQ
		input_dict["getbounds"] = getbounds
		input_dict["adaptive"] = adaptive
		input_dict["death_metric"] = death_metric
		data.append(input_dict)

	pool = Pool(os.cpu_count()) ## According to TA this will saturate more cores in the hpc?
	results = pool.map(fit_single_county, data)
	
	for result in results:
		if result is not None:
			if len(result) == 1:
				nonconvergent.append(result[0]) 
			else:
				(dates, death_cdf, county, parameters) = result
				counties_dates.append(dates)
				counties_death_errors.append(death_cdf)
				counties_fips.append(county)
				parameters_list[county] = parameters

	if len(nonconvergent) > 0:
		print(f"nonconvergent: {nonconvergent}")
		counties_dates_non, counties_death_errors_non, counties_fips_non = fill_nonconvergent(nonconvergent, us, end, fix_nonconvergent=fix_nonconvergent) 
		counties_dates = counties_dates + counties_dates_non
		for death_cdf in counties_death_errors_non:
			counties_death_errors.append(death_cdf)
		counties_fips = counties_fips + counties_fips_non

	output_dict = {"counties_dates": np.array(counties_dates), "counties_death_errors": np.array(counties_death_errors), "counties_fips": np.array(counties_fips), \
	"nonconvergent": nonconvergent, "parameters": parameters_list}
	return output_dict

# figure out how to stitch two gaussians for regime

if __name__ == '__main__':
	end = datetime.datetime(2020, 6, 30)
	guesses = [1.41578513e-01, 1.61248129e-01, 2.48362028e-01, 3.42978127e-01, 5.79023652e-01, 4.64392758e-02, \
	9.86745420e-06, 4.83700388e-02, 4.85290835e-01, 3.72688900e-02, 4.92398129e-04, 5.20319673e-02, \
	4.16822944e-02, 2.93718207e-02, 2.37765976e-01, 6.38313283e-04, 1.00539865e-04, 7.86113867e-01, \
	3.26287443e-01, 8.18317732e-06, 5.43511913e-10, 1.30387168e-04, 3.58953133e-03, 1.57388153e-05]
	test(end, bias=True, regime=False, weight=True, plot=True, guesses=guesses, start=None, quick=True, fitQ=False, adaptive=True, death_metric="deaths")
	# output_dict = fit_counties2_1.submission(end, regime=False, weight=True, guesses=guesses, start=-7, quick=True, fitQ=False)


# Fit DI/dt to daily_cases*(total_pop/tested_pop) and weight the score much lower than 


# i changed range of Q initial condition to (0, 0.3) dont forget to change back!

# Tasks
# =========
# MAKE SUBMISSION SCRIPT
# Make object oriented (dont have to keep calculating model(res.x, params, data) for plotting and error bounds)
# Fit Q to active cases
# Fix equations so variables add up to N
# Dynamically find regime_change from policies spreadsheet
# Submission file
# fit for moving windows of time and see how parameters change. Use random forest regressor from non covid data to find mapping function to target parameter values



