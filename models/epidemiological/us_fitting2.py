import numpy as np
import matplotlib.pyplot as plt
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
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


def process_data(data_covid, data_population, save=True):
	covid = loader.load_data(data_covid)
	loader.convert_dates(covid, "date")
	population = loader.load_data(data_population)
	# pcovid[covid["county"]=='New York City']["fips"].fillna(36061, inplace=True)
	# covid['fips'] = covid.apply(lambda row: 36061, axis=1)
	covid.loc[covid["county"]=='New York City', "fips"]=36061
	covid['Population'] = covid.apply(lambda row: loader.query(population, "FIPS", row.fips)['total_pop'], axis=1)
	covid.dropna(subset=['fips'], inplace=True)
	covid['fips']=covid['fips'].astype(int)
	# covid.apply(pd.to_numeric, errors='ignore')
	if save:
		covid.to_csv("us_training_data.csv")
	return covid

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
	# N = P + E + C + A + I + Q + R

	dPdt = (- ((a_1+a_2)*C*P)/N) + (-a_3*P + b_4*E)*(N/(P+E))
	dEdt = (- (b_1 * A + b_2 * I) * E / N) + b_3*C + (a_3*P - b_4*E)*(N/(P+E))
	dCdt = -(g_a + g_i)*C + ((b_1 * A + b_2 * I) * E / N) - b_3*C
	dAdt = (a_1 * C * P) / N + g_a*C - (r_a + del_a + th)*A
	dIdt = (a_2 * C * P) / N + g_i*C - ((r_i+d_i)+del_i)*I+th*A
	dQdt = del_a*A + del_i*I - (r_q+d_q)*Q
	dRdt = r_a*A + (r_i+d_i)*I + (r_q+d_q)*Q
	dDdt = d_i*I + d_q*Q

	dzdt = [dPdt, dEdt, dCdt, dAdt, dIdt, dQdt, dRdt, dDdt]
	# td = 20
	# scaler = 4 #find what value minimizes error? would have to iterate several least squares. make unique for each d/dt too
	# if t>td:
	# 	dzdt = [dPdt, dEdt, dCdt/scaler, dAdt/scaler, dIdt/scaler, dQdt, dRdt, dDdt]
	return dzdt

def model(params, data, extrapolate=-1):
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
	D0 = abs(data['deaths'].values[0])
	# offset = data['date_processed'].min()
	yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	
	n = len(data)
	if extrapolate > 0:
		n += extrapolate
	args = (params, N, n)

	# R0 = initial_conditions[6]
	# D0 = data['deaths'].values[0]
	# offset = data['date_processed'].min()
	# yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	
	# n = len(data)
	# if extrapolate > 0:
	# 	n += extrapolate
	
	# # Package parameters into a tuple
	# args = (params, N, n, offset)
	
	# Integrate ODEs
	try:
		s = scipy.integrate.odeint(pecaiqr, yz_0, np.arange(0, n), args=args)
	except RuntimeError:
#         print('RuntimeError', params)
		return np.zeros((n, len(yz_0)))

	return s

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
# return params, 1 standard deviation errors
def get_errors(res, p0, data, extrapolate):
	p0 = np.array(p0)
	ysize = len(res.fun)
	cost = 2 * res.cost  # res.cost is half sum of squares!
	popt = res.x
	# Do Moore-Penrose inverse discarding zero singular values.
	_, s, VT = svd(res.jac, full_matrices=False)
	threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
	s = s[s > threshold]
	VT = VT[:s.size]
	pcov = np.dot(VT.T / s**2, VT)

	warn_cov = False
	absolute_sigma = False
	if pcov is None:
		# indeterminate covariance
		pcov = zeros((len(popt), len(popt)), dtype=float)
		pcov.fill(inf)
		warn_cov = True
	elif not absolute_sigma:
		if ysize > p0.size:
			s_sq = cost / (ysize - p0.size)
			pcov = pcov * s_sq
		else:
			pcov.fill(inf)
			warn_cov = True

	if warn_cov:
		print('cannot estimate variance')
		return None
	
	perr = np.sqrt(np.diag(pcov))

	uncertainty = []
	samples = 100
	for i in range(samples):
		sample = np.random.normal(loc=res.x, scale=perr)
		s = model(sample, data, len(data)+extrapolate)
		latest_D = (data["deaths"].values)[-1]
		if s[:,7][len(data)-1] >= latest_D:
			uncertainty.append(s)

	return uncertainty

# returns standard deviation of fitted parameters
def get_param_errors(res, pop):
	pfit = res.x
	pcov = res.jac
	pcov = np.dot(pcov.T, pcov)
	pcov = np.linalg.pinv(pcov) #uses svd
	pcov = np.diag(pcov)
	rcov = np.cov(res.fun)/pop ##put res.fun/pop inside
	perr = pcov * rcov
	perr = np.sqrt(perr)
	return perr

# returns uncertainty of the fit for all variables
def get_fit_errors(res, p0_params, data, extrapolate=14, trim=False):
	population = list(data["Population"])[-1]
	errors = get_param_errors(res, population)
	errors[len(p0_params):] = 0
	uncertainty = []
	samples = 100
	if trim:
		for i in range(samples):
			sample = np.random.normal(loc=res.x, scale=errors)
			s = model(sample, data, len(data)+extrapolate)
			latest_D = (data["deaths"].values)[-1]
			if s[:,7][len(data)-1] >= latest_D:
				uncertainty.append(s)
	else:
		for i in range(samples):
			sample = np.random.normal(loc=res.x, scale=errors)
			s = model(sample, data, len(data)+extrapolate)
			uncertainty.append(s)
		
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

def plot_model(res, data, extrapolate=14, boundary=None, plot_infectious=False):   
	s = model(res.x, data, len(data)+extrapolate)
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
	p.circle(t, data['deaths'], color ='black', legend='Real Death')

	# quarantined
	p.circle(t, data['cases'], color ='purple', legend='Tested Infected')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)

def plot_with_errors_sample(res, p0_params, data, extrapolate=14, boundary=None, plot_infectious=False, trim=False):
	s = model(res.x, data, len(data)+extrapolate)
	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]

	uncertainty = get_fit_errors(res, p0_params, data, extrapolate=extrapolate, trim=trim)
	s1 = np.percentile(uncertainty, 25, axis=0)
	if trim:
		s1 = np.percentile(uncertainty, 0, axis=0)
	s2 = np.percentile(uncertainty, 75, axis=0)

	t = np.arange(0, len(data))
	tp = np.arange(0, len(data)+extrapolate)
	p = bokeh.plotting.figure(plot_width=1000,
							  plot_height=600,
							 title = ' PECAIQR Model Errors',
							 x_axis_label = 't (days)',
							 y_axis_label = '# people')
	if plot_infectious:
		p.varea(x=tp, y1=s1[:, 4], y2=s2[:, 2], color='red', fill_alpha=0.2)
		p.line(tp, I, color = 'red', line_width = 1, legend = 'Currently Infected')
	p.varea(x=tp, y1=s1[:, 7], y2=s2[:, 7], color='black', fill_alpha=0.2)
	p.line(tp, D, color = 'black', line_width = 1, legend = 'Deceased')
	# p.line(tp, Q, color = 'yellow', line_width = 1, legend = 'Quarantined')
	# p.line(tp, R, color = 'green', line_width = 1, legend = 'Recovered')

	# death
	p.circle(t, data['deaths'], color ='black')

	# quarantined
	# p.circle(t, data['cases'], color ='purple')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)
	return uncertainty


def leastsq_qd(params, data, weight=False):
	Ddata = (data['deaths'].values)
	Idata = (data['cases'].values)
	s = model(params, data)

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

	# q_error = Q+R+D-Idata 
	# # use q_error=Q-active_cases. Data for active_cases is falsely 0 for a long time because of bad reporting. Give weight of 0 to q_error for data with active_cases=0 
	if weight:
		# mu, sigma = 1, 0.2
		# w = np.random.normal(mu, sigma, len(error))
		# w = np.sort(w)
		w = np.geomspace(0.5,1.5,len(data))
		d_error = d_error*w
		# q_error = q_error*w

	# error = np.concatenate((d_error, q_error))
	error = d_error

	# Fitting to N = P+E+C+A+I+Q+R shows that equation is not balanced
	# pop = data['Population'].values[0]
	# N = np.empty(len(D))
	# N.fill(pop)
	# n_error = ((N-(P+E+C+A+I+Q+R))/N)**2
	# error = np.concatenate((d_error, n_error))
	
	return error


def fit(data, weight=False, plot=False, extrapolate=14, guesses=None, trim=False):
	param_ranges = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
	initial_ranges = [(0,1), (0,1), (0,0.01), (0,0.01), (0,0.01), (0,0.01), (0,0.01)]
	ranges = param_ranges+initial_ranges
	if guesses is None:
		params = [9e-02, 1e-01, 7e-02, 3.e-01, 4.e-01, 1e-01, 1e-01, 3e-01, 4e-01, 7e-02, 2e-04, 8e-02, 7e-03, 2e-02, 2e-04, 2e-06, 4e-03]
		initial_conditions = [7e-01, 2e-01, 4e-08, 7e-03, 1e-08, 3e-20, 7e-06]
		guesses = params+initial_conditions

	else:
		initial_ranges = [(0.9*guesses[17],1.1*guesses[17]), (0.9*guesses[18],1.1*guesses[18]), (0.9*guesses[19],1.1*guesses[19]), (0.9*guesses[20],1.1*guesses[20]), (0.9*guesses[21],1.1*guesses[21]), \
		(0.9*guesses[22],1.1*guesses[22]), (0.9*guesses[23],1.1*guesses[23])]
		ranges = param_ranges+initial_ranges


	for boundary in [len(data)]:
		res = least_squares(leastsq_qd, guesses, args=(data[:boundary],weight), bounds=np.transpose(np.array(ranges)))
		if plot:
			plot_model(res, data, extrapolate=extrapolate, boundary=boundary, plot_infectious=True)
			errors = plot_with_errors_sample(res, guesses[:17], data, extrapolate=extrapolate, boundary=boundary, plot_infectious=False, trim=trim)
		else:
			errors = get_fit_errors(res, guesses[:17], data, extrapolate=extrapolate, trim=trim)
		predictions = get_deaths(res, data, extrapolate=extrapolate)
		death_errors = errors[:,:,-1]

	return (predictions, death_errors, res)



##########################################################
def model2(params, data, extrapolate=-1):
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
	offset = data['date_processed'].min()
	yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	
	n = len(data)
	if extrapolate > 0:
		n += extrapolate
	args = (params, N, n)

	# R0 = initial_conditions[6]
	# D0 = data['deaths'].values[0]
	# offset = data['date_processed'].min()
	# yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	
	# n = len(data)
	# if extrapolate > 0:
	# 	n += extrapolate
	
	# # Package parameters into a tuple
	# args = (params, N, n, offset)
	
	# Integrate ODEs
	try:
		s = scipy.integrate.odeint(pecaiqr, yz_0, np.arange(0, n), args=args)
	except RuntimeError:
#         print('RuntimeError', params)
		return np.zeros((n, len(yz_0)))

	return s

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

# returns uncertainty of the fit for all variables
def get_fit_errors2(res, p0_params, data, extrapolate=14, trim=False):
	population = list(data["Population"])[-1]
	errors = get_param_errors(res, population)
	errors[len(p0_params):] = 0
	uncertainty = []
	samples = 100
	if trim:
		for i in range(samples):
			sample = np.random.normal(loc=res.x, scale=errors)
			s = model2(sample, data, len(data)+extrapolate)
			latest_D = (data["deaths"].values)[-1]
			if s[:,7][len(data)-1] >= latest_D:
				uncertainty.append(s)
	else:
		for i in range(samples):
			sample = np.random.normal(loc=res.x, scale=errors)
			s = model2(sample, data, len(data)+extrapolate)
			uncertainty.append(s)
		
	uncertainty = np.array(uncertainty)
	return uncertainty


def plot_model2(res, data, extrapolate=14, boundary=None, plot_infectious=False):   
	s = model2(res.x, data, len(data)+extrapolate)
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
	p.circle(t, data['deaths'], color ='black', legend='Real Death')

	# quarantined
	p.circle(t, data['cases'], color ='purple', legend='Tested Infected')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)

def plot_with_errors_sample2(res, p0_params, data, extrapolate=14, boundary=None, plot_infectious=False, trim=False):
	s = model2(res.x, data, len(data)+extrapolate)
	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]

	uncertainty = get_fit_errors2(res, p0_params, data, extrapolate=extrapolate, trim=trim)
	s1 = np.percentile(uncertainty, 25, axis=0)
	if trim:
		s1 = np.percentile(uncertainty, 0, axis=0)
	s2 = np.percentile(uncertainty, 75, axis=0)

	t = np.arange(0, len(data))
	tp = np.arange(0, len(data)+extrapolate)
	p = bokeh.plotting.figure(plot_width=1000,
							  plot_height=600,
							 title = ' PECAIQR Model Errors',
							 x_axis_label = 't (days)',
							 y_axis_label = '# people')
	if plot_infectious:
		p.varea(x=tp, y1=s1[:, 4], y2=s2[:, 2], color='red', fill_alpha=0.2)
		p.line(tp, I, color = 'red', line_width = 1, legend = 'Currently Infected')
	p.varea(x=tp, y1=s1[:, 7], y2=s2[:, 7], color='black', fill_alpha=0.2)
	p.line(tp, D, color = 'black', line_width = 1, legend = 'Deceased')
	# p.line(tp, Q, color = 'yellow', line_width = 1, legend = 'Quarantined')
	# p.line(tp, R, color = 'green', line_width = 1, legend = 'Recovered')

	# death
	p.circle(t, data['deaths'], color ='black')

	# quarantined
	# p.circle(t, data['cases'], color ='purple')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)
	return uncertainty


def leastsq_qd2(params, data, weight=False):
	Ddata = (data['deaths'].values)
	Idata = (data['cases'].values)
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

	# q_error = Q+R+D-Idata 
	# # use q_error=Q-active_cases. Data for active_cases is falsely 0 for a long time because of bad reporting. Give weight of 0 to q_error for data with active_cases=0 
	if weight:
		# mu, sigma = 1, 0.2
		# w = np.random.normal(mu, sigma, len(error))
		# w = np.sort(w)
		w = np.geomspace(0.5,1.5,len(data))
		d_error = d_error*w
		# q_error = q_error*w

	error = d_error
	return error


def fit2(data, guesses, weight=False, plot=False, extrapolate=14, trim=False):
	param_ranges = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
	initial_ranges = [(0.9*guesses[17],1.1*guesses[17]), (0.9*guesses[18],1.1*guesses[18]), (0.9*guesses[19],1.1*guesses[19]), (0.9*guesses[20],1.1*guesses[20]), (0.9*guesses[21],1.1*guesses[21]), \
	(0.9*guesses[22],1.1*guesses[22]), (0.9*guesses[23],1.1*guesses[23]), (0.1*guesses[24],10*guesses[24])]
	ranges = param_ranges+initial_ranges

	for boundary in [len(data)]:
		res = least_squares(leastsq_qd2, guesses, args=(data[:boundary],weight), bounds=np.transpose(np.array(ranges)))
		if plot:
			plot_model2(res, data, extrapolate=extrapolate, boundary=boundary, plot_infectious=True)
			errors = plot_with_errors_sample2(res, guesses[:17], data, extrapolate=extrapolate, boundary=boundary, plot_infectious=False, trim=trim)
		else:
			errors = get_fit_errors2(res, guesses[:17], data, extrapolate=extrapolate, trim=trim)
		predictions = get_deaths2(res, data, extrapolate=extrapolate)
		death_errors = errors[:,:,-1]

	return (predictions, death_errors, res)


###########################################################

def main0(weight=True, plot=False, trim=False):
	#Get date range of April1 to June30 inclusive. Figure out how much to extrapolate
	us = process_data("/data/us/covid/nyt_us_counties.csv", "/data/us/demographics/county_populations.csv")
	fips_key = loader.load_data("/data/us/processing_data/fips_key.csv", encoding="latin-1")
	fips = fips_key["FIPS"]

	for county in [36059, 36061]:
		county_data = loader.query(us, "fips", county)
		firstnonzero = next((index for index,value in enumerate(county_data["deaths"].values) if value != 0), None)
		death_time = 16
		if firstnonzero:
			begin = firstnonzero-death_time
			if begin >= 0:
				county_data = county_data[begin:]
				#county_data.reset_index(drop=True, inplace=True)
		# county_data = county_data[:-5]
		dates = pd.to_datetime(county_data["date"].values)
		start = dates[0].day
		extrapolate = (datetime.datetime(2020, 6, 30)-dates[-1])/np.timedelta64(1, 'D')
		# expand dates using extrapolate
		predictions, death_errors, res = fit(county_data, weight=weight, plot=plot, extrapolate=extrapolate, trim=trim)
		parameters = res.x
		cost = res.cost
		print(parameters)
		print(cost)


		# submission=[]

	# for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]: #make this a separate function
	# 	forecast = list(np.percentile(death_errors, percentile, axis=0))
	# 	submission.append(forecast)

	# submission = np.transpose(submission)
	# print(submission)
	# #output 4 tuple of (date, fips, time, )


def main1(weight=True, plot=False, trim=False, shift=False):
	#Get date range of April1 to June30 inclusive. Figure out how much to extrapolate
	us = process_data("/data/us/covid/nyt_us_counties.csv", "/data/us/demographics/county_populations.csv")
	fips_key = loader.load_data("/data/us/processing_data/fips_key.csv", encoding="latin-1")
	fips = fips_key["FIPS"]

	for county in [36059, 36061]:
		county_data = loader.query(us, "fips", county)
		firstnonzero = next((index for index,value in enumerate(county_data["deaths"].values) if value != 0), None)
		death_time = 16 #find a way to calculate this for each county, possibly using ml or trying to see what shift makes I and D overlap the most
		if firstnonzero:
			begin = firstnonzero-death_time
			if begin >= 0:
				county_data = county_data[begin:]
				#county_data.reset_index(drop=True, inplace=True)

		policy_date = 737506 # get from policies.csv using county fips query
		dates = pd.to_datetime(county_data["date"].values)
		start = dates[0]
		regime_change = int((datetime.datetime.fromordinal(policy_date)-start)/np.timedelta64(1, 'D'))

		county_data1 = county_data[:regime_change] # experiment with regime_change+death_time
		predictions, death_errors1, res1 = fit(county_data1, weight=weight, plot=plot, extrapolate=0, trim=trim)
		first_parameters = (res1.x)[:17]
		first_conditions = get_variables(res1, county_data1, regime_change)
		N = county_data['Population'].values[0]
		first_conditions = first_conditions/N
		parameter_guess = list(first_parameters)+list(first_conditions)

		county_data2 = county_data[regime_change:]
		dates2 = dates[regime_change:]
		county_data2.reset_index(drop=True, inplace=True)
		if shift:
			for i in range(death_time):
				county_data2.at[i, "deaths"] *= -1
		extrapolate = (datetime.datetime(2020, 6, 30)-dates2[-1])/np.timedelta64(1, 'D')
		predictions, death_errors, res2  = fit(county_data2, weight=weight, plot=plot, extrapolate=extrapolate, guesses=parameter_guess, trim=trim)


# Train on the 14 days after regime change and set initial conditions at regime chage to that of previous fit  
def main2(weight=True, plot=False, trim=False, shift=True):
	#Get date range of April1 to June30 inclusive. Figure out how much to extrapolate
	us = process_data("/data/us/covid/nyt_us_counties.csv", "/data/us/demographics/county_populations.csv")
	fips_key = loader.load_data("/data/us/processing_data/fips_key.csv", encoding="latin-1")
	fips = fips_key["FIPS"]

	for county in [36061]:
		county_data = loader.query(us, "fips", county)
		firstnonzero = next((index for index,value in enumerate(county_data["deaths"].values) if value != 0), None)
		death_time = 16 #find a way to calculate this for each county, possibly using ml or trying to see what shift makes I and D overlap the most
		if firstnonzero:
			begin = firstnonzero-death_time
			if begin >= 0:
				county_data = county_data[begin:]
				#county_data.reset_index(drop=True, inplace=True)

		policy_date = 737506 # get from policies.csv using county fips query
		dates = pd.to_datetime(county_data["date"].values)
		start = dates[0]
		regime_change = int((datetime.datetime.fromordinal(policy_date)-start)/np.timedelta64(1, 'D'))

		county_data1 = county_data[:regime_change+death_time] ## experimental. Assumes first regime will carry over until death_time into future. Used to be just county_data[:regime_change]
		predictions, death_errors1, res1 = fit(county_data1, weight=weight, plot=plot, extrapolate=0, trim=trim)
		first_parameters = (res1.x)[:17]
		first_conditions = get_variables(res1, county_data1, regime_change)
		first_conditions = np.append(first_conditions, (county_data1["deaths"].values)[regime_change]) 
		N = county_data['Population'].values[0]
		first_conditions = first_conditions/N
		parameter_guess = list(first_parameters)+list(first_conditions)
		print(parameter_guess)

		county_data2 = county_data[regime_change:]
		dates2 = dates[regime_change:]
		county_data2.reset_index(drop=True, inplace=True)
		if shift:
			for i in range(death_time):
				county_data2.at[i, "deaths"] *= -1
			extrapolate = (datetime.datetime(2020, 6, 30)-dates2[-1])/np.timedelta64(1, 'D')
		predictions, death_errors, res2  = fit2(county_data2, weight=weight, plot=plot, extrapolate=extrapolate, guesses=parameter_guess, trim=trim)

if __name__ == '__main__':
	main0(plot=True, trim=True, weight=True)
	# main1(plot=True, trim=True, weight=False) # I dont think weighting is beneficial
	# main2(plot=True, trim=True, weight=True) 

	main1_2(plot=True, trim=True, weight=False, shift=True)
	main1_2(plot=True, trim=True, weight=True, shift=True)

# Need to make sure the standard deviation parameters are in the bound
# use shivas CDF function to reduce runtime
# Tasks
# Fit Q to active cases
# Fix equations so variables add up to N
# Dynamically find regime_change from policies spreadsheet
# Submission file
# fit for moving windows of time and see how parameters change. Use random forest regressor from non covid data to find mapping function to target parameter values
# make object oriented


