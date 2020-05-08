import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings

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


def process_data(data_covid, data_population, save=True):
	covid = loader.load_data(data_covid)
	loader.convert_dates(covid, "date")
	population = loader.load_data(data_population)
	covid.loc[covid["county"]=='New York City', "fips"]=36061
	covid['Population'] = covid.apply(lambda row: loader.query(population, "FIPS", row.fips)['total_pop'], axis=1)
	covid.dropna(subset=['fips'], inplace=True)
	covid['fips']=covid['fips'].astype(int)
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

def get_death_cdf(death_pdf, switch=True):
	death_cdf = []
	for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]: #make this a separate function
		forecast = list(np.nanpercentile(death_pdf, percentile, axis=0))
		death_cdf.append(forecast)

	if switch == False:
		if np.mean(np.array(death_cdf[-1])/np.array(death_cdf[0])) > 1.5:
			print("recalculate error bounds")
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

def model(params, data, extrapolate=-1, offset=0):
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

def model_ivp(params, data, extrapolate=-1, offset=0):
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

	# D0 = data["deaths"].values[start]
	D0 = fit[:,7][offset]
	yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	
	n = len(data)+extrapolate
	args = (params, N, n)
	try:
		s = scipy.integrate.odeint(pecaiqr, yz_0, np.arange(offset, n), args=args)
		# s = scipy.integrate.solve_ivp(fun=lambda t, y: pecaiqr(y, t, params, N, n), t_span=[offset, n], y0=yz_0)
	except RuntimeError:
		print('RuntimeError', params)
		bound_mean, bound_deviation = guess_bounds
		scaler = np.random.normal(loc=bound_mean, scale=bound_deviation)
		bound = (fit[:,7][offset:])*(1+scaler)
		return bound
		# return np.zeros((n, len(yz_0)))
	return s[:,7]


def estimate_bounds(res, data):
	residuals = res.fun
	normalized_residuals = []
	for index, death in enumerate(data["deaths"].values):
		if death > 0:
			normalized_residuals.append((residuals[index])/death)
	normalized_residuals = reject_outliers(normalized_residuals)
	normalized_residuals = abs(normalized_residuals)
	mean = None
	deviation = None
	if len(normalized_residuals) > 0:
		# mean = sum(normalized_residuals)/len(normalized_residuals)
		mean = 0
		deviation = np.std(normalized_residuals)
	return (mean,deviation)

def quickie(fit, data, guess_bounds, start=-1):
	offset = len(data)+start
	bound_mean, bound_deviation = guess_bounds
	scaler = np.random.normal(loc=bound_mean, scale=bound_deviation)
	bound = (fit[:,7][offset:])*(1+scaler)
	return bound

# returns uncertainty of the fit for all variables
def get_fit_errors(res, p0_params, data, extrapolate=14, start=-1, quick=False):
	population = list(data["Population"])[-1]
	errors = get_param_errors(res, population)
	errors[len(p0_params):] = 0

	guess_bounds = estimate_bounds(res,data)
	if guess_bounds == (None, None):
		return np.zeros((1,int(len(data)+extrapolate)))

	uncertainty = []
	samples = 100

	if extrapolate > 0 :
		fit = model(res.x, data, extrapolate)
		if quick:
			for i in range(samples):
				death_series = quickie(fit, data, guess_bounds, start=start)
				latest_D = (data["deaths"].values)[-1]
				death_series = np.concatenate((data["deaths"].values[0:len(data)], death_series[-1*start:]))
				for index, death in enumerate(death_series):
					if index >= len(data) and death <= latest_D:
						death_series[index] = None
				uncertainty.append(death_series)
		else:
			for i in range(samples):
				sample = np.random.normal(loc=res.x, scale=errors)
				death_series = model_beyond(fit, sample, data, guess_bounds, extrapolate, start=start)
				latest_D = (data["deaths"].values)[-1]
				# death_series = np.concatenate((fit[:,7][0:len(data)-1], death_series))
				death_series = np.concatenate((data["deaths"].values[0:len(data)], death_series[-1*start:]))
				for index, death in enumerate(death_series):
					if index >= len(data) and death <= latest_D:
						death_series[index] = None
				uncertainty.append(death_series)
	else: 
		for i in range(samples):
			sample = np.random.normal(loc=res.x, scale=errors)
			death_series = model(sample, data, len(data)+extrapolate)
			death_series = s[:,7]
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

def plot_model(res, data, extrapolate=14, boundary=None, plot_infectious=False):   
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
	p.circle(t, data['deaths'], color ='black', legend='Real Death')

	# quarantined
	p.circle(t, data['cases'], color ='purple', legend='Tested Infected')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)

def plot_with_errors_sample(res, p0_params, data, extrapolate=14, boundary=None, plot_infectious=False, start=-1, quick=False):
	s = model(res.x, data, len(data)+extrapolate)
	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]

	uncertainty = get_fit_errors(res, p0_params, data, extrapolate=extrapolate, start=start, quick=quick)
	
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

	p.line(tp, D, color = 'black', line_width = 1, legend = 'Deceased')
	p.circle(t, data['deaths'], color ='black')

	# quarantined
	# p.circle(t, data['cases'], color ='purple')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)
	return uncertainty


def test_convergence(data_length, predictions):
	converge = True
	deaths = [death[1] for death in predictions]
	diff = deaths[-1] - deaths[data_length-1] 
	if diff < 0:
		converge = False
	return converge	

def fill_nonconvergent(nonconvergent, data, end):
	counties_dates = []
	counties_death_errors = []
	counties_fips = nonconvergent
	for index, county in enumerate(nonconvergent):
		county_data = loader.query(data, "fips", county)
		deaths = county_data["deaths"].values
		dates = pd.to_datetime(county_data["date"].values)
		extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')

		death_cdf = []
		latest_D = deaths[-1]
		for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]: #make this a separate function
			bound = (1 + percentile/200)*latest_D
			predictions = [bound for i in range(int(extrapolate))]
			forecast = list(np.concatenate((deaths, predictions)))
			death_cdf.append(forecast)
		death_cdf = np.transpose(death_cdf)
		counties_dates.append(dates)
		counties_death_errors.append(death_cdf)

	return (counties_dates, counties_death_errors, counties_fips)
		

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

	if weight:
		w = np.geomspace(0.5,1.5,len(data))
		d_error = d_error*w

	error = d_error	
	return error


def fit(data, weight=False, plot=False, extrapolate=14, guesses=None, start=-1, quick=False, getbounds=False):
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
		predictions = get_deaths(res, data, extrapolate=extrapolate)
		convergent_status = test_convergence(len(data), predictions) 
		if convergent_status == False:
			return (None,None,None)

		death_pdf = None
		if plot:
			plot_model(res, data, extrapolate=extrapolate, boundary=boundary, plot_infectious=True)
			death_pdf = plot_with_errors_sample(res, guesses[:17], data, extrapolate=extrapolate, boundary=boundary, plot_infectious=False, start=start, quick=quick)
		else:
			if getbounds:
				death_pdf = get_fit_errors(res, guesses[:17], data, extrapolate=extrapolate, start=start, quick=quick)

	return (predictions, death_pdf, res)

###########################################################

def test(weight=True, plot=False, guesses=None, start=-1, quick=False, moving=False, getbounds=False):
	#Get date range of April1 to June30 inclusive. Figure out how much to extrapolate
	counties_dates = []
	counties_death_errors = []
	counties_fips = []
	nonconvergent = []

	# us = process_data("/data/us/covid/nyt_us_counties.csv", "/data/us/demographics/county_populations.csv")
	us = loader.load_data("/models/epidemiological/production/us_training_data.csv")
	fips_key = loader.load_data("/data/us/processing_data/fips_key.csv", encoding="latin-1")
	# fips_list = fips_key["FIPS"]
	fips_list = [36061] #[56013, 36061, 36103] #1017
	total = len(fips_list)

	for index, county in enumerate(fips_list):
		print(f"{index+1} / {total}")
		county_data = loader.query(us, "fips", county)
		if moving:
			# county_data['deaths'] = county_data.iloc[:,6].rolling(window=3).mean()
			county_data['avg_deaths'] = county_data.iloc[:,6].rolling(window=3).mean() #pass optional parameters to fitting, calculate real deaths from moving average at the end
			county_data = county_data[2:]
			# #since this is the sum of the last two and the current, extrapolate 1 extra and at the end shift everything 1 to  the left so its effectively previous+current+next

		firstnonzero = next((index for index,value in enumerate(county_data["deaths"].values) if value != 0), None)
		death_time = 16
		if firstnonzero:
			if firstnonzero > len(county_data)-7 or (county_data["deaths"].values)[-1]-(county_data["deaths"].values)[firstnonzero] == 0:
				# add to nonconvergent counties
				nonconvergent.append(county)
				continue
			begin = firstnonzero-death_time
			if begin >= 0:
				county_data = county_data[begin:]
				county_data.reset_index(drop=True, inplace=True)
		else:
			# dont add to convonvergent counties, just leave blank and submission script will fill it in with all zeros
			continue

		dates = pd.to_datetime(county_data["date"].values)
		extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')
		predictions, death_pdf, res = fit(county_data, weight=weight, plot=plot, extrapolate=extrapolate, guesses=guesses, start=start, quick=quick, getbounds=True)
		if res is None:
			# add to nonconvergent counties
			nonconvergent.append(county)
			continue

		death_cdf = get_death_cdf(death_pdf, switch=quick)
		if death_cdf is None:
			death_pdf = get_fit_errors(res, guesses[:17], data, extrapolate=extrapolate, start=start, quick=True)
			death_cdf = get_death_cdf(death_pdf, switch=True)


		death_cdf = np.transpose(death_cdf)
		counties_dates.append(dates)
		counties_death_errors.append(death_cdf)
		counties_fips.append(county)

	if len(nonconvergent) > 0:
		print(nonconvergent)
		counties_dates2, counties_death_errors2, counties_fips2 = fill_nonconvergent(nonconvergent, us, end) 
		counties_dates = counties_dates + counties_dates2
		for death_cdf in counties_death_errors2:
			counties_death_errors.append(death_cdf)
		# county_death_errors = counties_death_errors + counties_death_errors2
		counties_fips = counties_fips + counties_fips2


	return (np.array(counties_dates), np.array(counties_death_errors), np.array(counties_fips))


###########################################################

def submission(end, weight=True, guesses=None, start=-1, quick=False, moving=False):
	#Get date range of April1 to June30 inclusive. Figure out how much to extrapolate
	counties_dates = []
	counties_death_errors = []
	counties_fips = []
	nonconvergent = []
	parameters={}

	# us = process_data("/data/us/covid/nyt_us_counties.csv", "/data/us/demographics/county_populations.csv")
	us = loader.load_data("/models/epidemiological/production/us_training_data.csv")
	fips_key = loader.load_data("/data/us/processing_data/fips_key.csv", encoding="latin-1")
	fips_list = fips_key["FIPS"]
	# fips_list = [56013, 36061, 36103]
	total = len(fips_list)

	for index, county in enumerate(fips_list):
		print(f"{index+1} / {total}")
		county_data = loader.query(us, "fips", county)
		if moving:
			# county_data['deaths'] = county_data.iloc[:,6].rolling(window=3).mean()
			county_data['avg_deaths'] = county_data.iloc[:,6].rolling(window=3).mean() #pass optional parameters to fitting, calculate real deaths from moving average at the end
			county_data = county_data[2:]
			# #since this is the sum of the last two and the current, extrapolate 1 extra and at the end shift everything 1 to  the left so its effectively previous+current+next

		firstnonzero = next((index for index,value in enumerate(county_data["deaths"].values) if value != 0), None)
		death_time = 16
		if firstnonzero:
			if firstnonzero > len(county_data)-7 or (county_data["deaths"].values)[-1]-(county_data["deaths"].values)[firstnonzero] == 0:
				# add to nonconvergent counties
				nonconvergent.append(county)
				continue
			begin = firstnonzero-death_time
			if begin >= 0:
				county_data = county_data[begin:]
				county_data.reset_index(drop=True, inplace=True)
		else:
			# dont add to convonvergent counties, just leave blank and submission script will fill it in with all zeros
			continue

		dates = pd.to_datetime(county_data["date"].values)
		extrapolate = (end-dates[-1])/np.timedelta64(1, 'D')
		predictions, death_pdf, res = fit(county_data, weight=weight, plot=False, extrapolate=extrapolate, guesses=guesses, start=start, quick=quick, getbounds=True)
		if res is None:
			# add to nonconvergent counties
			nonconvergent.append(county)
			continue

		death_cdf = get_death_cdf(death_pdf, switch=quick)
		if death_cdf is None:
			death_pdf = get_fit_errors(res, guesses[:17], data, extrapolate=extrapolate, start=start, quick=True)
			death_cdf = get_death_cdf(death_pdf, switch=True)


		death_cdf = np.transpose(death_cdf)
		counties_dates.append(dates)
		counties_death_errors.append(death_cdf)
		counties_fips.append(county)

	if len(nonconvergent) > 0:
		print(nonconvergent)
		counties_dates2, counties_death_errors2, counties_fips2 = fill_nonconvergent(nonconvergent, us, end) 
		counties_dates = counties_dates + counties_dates2
		for death_cdf in counties_death_errors2:
			counties_death_errors.append(death_cdf)
		# county_death_errors = counties_death_errors + counties_death_errors2
		counties_fips = counties_fips + counties_fips2

	if county in parameters.keys():
			parameters[county].append(list(res.x))
	else:
		parameters[county] = [list(res.x)]

	return (np.array(counties_dates), np.array(counties_death_errors), np.array(counties_fips), nonconvergent, parameters)


if __name__ == '__main__':
	end = datetime.datetime(2020, 6, 30)
	guesses = [1.41578513e-01, 1.61248129e-01, 2.48362028e-01, 3.42978127e-01, 5.79023652e-01, 4.64392758e-02, \
	9.86745420e-06, 4.83700388e-02, 4.85290835e-01, 3.72688900e-02, 4.92398129e-04, 5.20319673e-02, \
	4.16822944e-02, 2.93718207e-02, 2.37765976e-01, 6.38313283e-04, 1.00539865e-04, 7.86113867e-01, \
	3.26287443e-01, 8.18317732e-06, 5.43511913e-10, 1.30387168e-04, 3.58953133e-03, 1.57388153e-05]
	# counties_dates, counties_death_errors, counties_fips = submission(end, weight=True, guesses=guesses, start=-7, quick=True, moving=False)
	test(weight=True, plot=True, guesses=guesses, start=-7, quick=False, moving=False)
	# test(weight=True, plot=False, getbounds=False, guesses=guesses, start=-7, quick=True, moving=False)


# Tasks
# =========
# MAKE SUBMISSION SCRIPT
# Make object oriented (dont have to keep calculating model(res.x, params, data) for plotting and error bounds)
# Fit Q to active cases
# Fix equations so variables add up to N
# Dynamically find regime_change from policies spreadsheet
# Submission file
# fit for moving windows of time and see how parameters change. Use random forest regressor from non covid data to find mapping function to target parameter values



