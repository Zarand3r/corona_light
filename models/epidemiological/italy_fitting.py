import numpy as np
from scipy.integrate import odeint
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
import itertools

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
	loader.convert_dates(covid, "Date")
	population = loader.load_data(data_population)
	covid['Population'] = covid.apply(lambda row: loader.query(population, "Region", row.Region)['Population'], axis=1)
	if save:
		covid.to_csv("italy_training_data.csv")
	return covid

# return params, 1 standard deviation errors
def get_errors(res, p0):
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
	return perr

# returns standard deviation of fitted parameters
def get_param_errors(res, p0):
	pfit = res.x
	pcov = res.jac
	pcov = np.dot(pcov.T, pcov)
	pcov = np.linalg.pinv(pcov) #uses svd
	pcov = np.diag(pcov)
	rcov = np.cov(res.fun)/3750000
	perr = pcov * rcov
	perr = np.sqrt(perr)
	return perr

def mse_qd(A, B):
	Ap = np.nan_to_num(A)
	Bp = np.nan_to_num(B)
	Ap[A == -np.inf] = 0
	Bp[B == -np.inf] = 0
	Ap[A == np.inf] = 0
	Bp[B == np.inf] = 0
	return mean_squared_error(Ap, Bp)

def pecaiqr(dat, t, params, N, max_t, offset):
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

	dPdt = (- ((a_1*a_2)*C*P)/N) + (-a_3*P + b_4*E)*(N/(P+E))
	dEdt = (- ((b_1 * A + b_2 * I) * E) / N) + b_3*C + (a_3*P - b_4*E)*(N/(P+E))
	dCdt = -(g_a + g_i)*C + (((b_1 * A + b_2 * I) * E) / N) - b_3*C
	dAdt = (a_1 * C * P) / N + g_a*C - (r_a + del_a + th)*A
	dIdt = (a_2 * C * P) / N + g_i*C - ((r_i+d_i)+del_i)*I+th*A
	dQdt = del_a*A + del_i*I - (r_q+d_q)*Q
	dRdt = r_a*A + (r_i+d_i)*I + (r_q+d_q)*Q
	dDdt = d_i*I + d_q*Q

	dzdt = [dPdt, dEdt, dCdt, dAdt, dIdt, dQdt, dRdt, dDdt]
	return dzdt

# def plot_pecaiqr():
# 	n = 2000
# 	# params = [0.3, 0.8, 0.1, 0.3, 0.6, 0.1, 0.01, 0.1, 0.8, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.2, 0.5]
# 	params = [0.1, 0.1, 0.1, 0.3, 0.5, 0.1, 0.1, 0.3, 0.5, 0.1, 0.1, 0.1, 0.001, 0.001, 0.001, 0.001, 0.001]
# 	params[:] = [x/1  for x in params]
# 	z0 = [75000, 25000, 100, 100, 100, 10, 10, 0]
# 	t = np.linspace(0,500,n)
# 	z = odeint(pecaiqr,z0,t, args = (params, ))

# 	P = z[:,0]
# 	E = z[:,1]
# 	C = z[:,2]
# 	A = z[:,3]
# 	I = z[:,4]
# 	Q = z[:,5]
# 	R = z[:,6]
# 	D = z[:,7]


# 	# plot results
# 	# plt.plot(t,P,'g--',label='P(t)')
# 	# plt.plot(t,E,'b--',label='E(t)')
# 	plt.plot(t,C,'b--',label='C(t)')
# 	plt.plot(t,A,'y--',label='A(t)')
# 	plt.plot(t,I,'r:',label='I(t)')
# 	plt.plot(t,Q,'r:',label='Q(t)')
# 	plt.plot(t,R,'k--',label='R(t)')
# 	plt.plot(t,D,'k--',label='D(t)')
# 	plt.ylabel('# people')
# 	plt.xlabel('time')
# 	plt.legend(loc='best')
# 	plt.show()

def model_qd(params, data, extrapolate=-1):
	N = data['Population'].values[0] # total population
	initial_conditions = N * np.array(params[-7:]) # the parameters are a fraction of the population so multiply by the population
	P0 = initial_conditions[0]
	E0 = initial_conditions[1]
	C0 = initial_conditions[2]
	A0 = initial_conditions[3]
	I0 = initial_conditions[4]
	Q0 = initial_conditions[5]
	R0 = initial_conditions[6]
	D0 = data['Deaths'].values[0]
	offset = data['date_processed'].min()
	yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	
	n = len(data)
	if extrapolate > 0:
		n += extrapolate
	
	# Package parameters into a tuple
	args = (params, N, n, offset)
	
	# Integrate ODEs
	try:
		s = scipy.integrate.odeint(pecaiqr, yz_0, np.arange(0, n), args=args)
	except RuntimeError:
#         print('RuntimeError', params)
		return np.zeros((n, len(yz_0)))

	return s


def get_deaths(res, data, extrapolate=14):   
	s = model_qd(res.x, data, len(data)+extrapolate)
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
def get_fit_errors(res, p0_params, p0_initial_conditions, data, extrapolate=14):
	errors = get_param_errors(res, list(p0_params) + list(p0_initial_conditions))
	errors[len(p0_params):] = 0
	uncertainty = []
	samples = 100
	for i in range(samples):
		sample = np.random.normal(loc=res.x, scale=errors)
		s = model_qd(sample, data, len(data)+extrapolate)
		uncertainty.append(s)
		
	uncertainty = np.array(uncertainty)
	return uncertainty

def plot_qd(res, p0_params, p0_initial_conditions, data, extrapolate=14, boundary=None, plot_infectious=False):   
	s = model_qd(res.x, data, len(data)+extrapolate)
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

	p = bokeh.plotting.figure(plot_width=600,
							  plot_height=400,
							 title = ' PECAIQR Model',
							 x_axis_label = 't (days)',
							 y_axis_label = '# people')

	if plot_infectious:
		p.line(tp, I, color = 'red', line_width = 1, legend = 'All infected')
	p.line(tp, D, color = 'black', line_width = 1, legend = 'Deceased')
	p.line(tp, Q, color = 'yellow', line_width = 1, legend = 'Quarantined')
	p.line(tp, R, color = 'green', line_width = 1, legend = 'Recovered')

	# death
	p.circle(t, data['Deaths'], color ='black', legend='Real Death')

	# quarantined
	p.circle(t, data['TotalCurrentlyPositive'], color ='purple', legend='Tested Infected')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)

def plot_with_errors_sample(res, p0_params, p0_initial_conditions, data, extrapolate=14, boundary=None, plot_infectious=False):
	s = model_qd(res.x, data, len(data)+extrapolate)
	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]

	uncertainty = get_fit_errors(res, p0_params, p0_initial_conditions, data, extrapolate=14)
	s1 = np.percentile(uncertainty, 25, axis=0)
	s2 = np.percentile(uncertainty, 75, axis=0)

	t = np.arange(0, len(data))
	tp = np.arange(0, len(data)+extrapolate)
	p = bokeh.plotting.figure(plot_width=600,
							  plot_height=400,
							 title = ' PECAIQR Model Errors',
							 x_axis_label = 't (days)',
							 y_axis_label = '# people')
	if plot_infectious:
		p.varea(x=tp, y1=s1[:, 4], y2=s2[:, 2], color='red', fill_alpha=0.2)
	p.varea(x=tp, y1=s1[:, 7], y2=s2[:, 7], color='black', fill_alpha=0.2)
	
	if plot_infectious:
		p.line(tp, I, color = 'red', line_width = 1, legend = 'Currently Infected')
	p.line(tp, D, color = 'black', line_width = 1, legend = 'Deceased')
	# p.line(tp, Q, color = 'yellow', line_width = 1, legend = 'Quarantined')
	# p.line(tp, R, color = 'green', line_width = 1, legend = 'Recovered')

	# death
	p.circle(t, data['Deaths'], color ='black')

	# quarantined
	p.circle(t, data['TotalCurrentlyPositive'], color ='purple')
	
	if boundary is not None:
		vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
		p.renderers.extend([vline])

	p.legend.location = 'top_left'
	bokeh.io.show(p)


def leastsq_qd(params, data, weight=False):
	Ddata = (data['Deaths'].values)
	Idata = (data['TotalCurrentlyPositive'].values)
	s = model_qd(params, data)

	P = s[:,0]
	E = s[:,1]
	C = s[:,2]
	A = s[:,3]
	I = s[:,4]
	Q = s[:,5]
	R = s[:,6]
	D = s[:,7]

	error = D-Ddata
	if weight:
		# mu, sigma = 1, 0.2
		# w = np.random.normal(mu, sigma, len(error))
		# w = np.sort(w)
		w = np.geomspace(0.5,1.5,len(data))
		error = error*w

	return error


def fit(data, weight=False, plot=False, extrapolate=14):
	params = [9e-02, 1e-01, 7e-02, 3.e-01, 4.e-01, 1e-01, 1e-01, 3e-01, 4e-01, 7e-02, 2e-04, 8e-02, 7e-03, 2e-02, 2e-04, 2e-06, 4e-03]
	params[:] = [x/1  for x in params]
	param_ranges = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
	# initial_conditions = [0.74, 0.24, 0.005, 4e-6, 0.0009, 0.0005, 0.0002]
	initial_conditions = [7e-01, 2e-01, 4e-08, 7e-03, 1e-08, 3e-20, 7e-06]
	initial_ranges = [(0,1), (0,1), (0,0.01), (0,0.01), (0,0.01), (0,0.01), (0,0.01)]
	# guesses = params + initial_conditions
	guesses = params+initial_conditions
	ranges = param_ranges+initial_ranges

	for boundary in [len(data)]:
		res = least_squares(leastsq_qd, guesses, args=(data[:boundary],weight), bounds=np.transpose(np.array(ranges)))
		if plot:
			plot_qd(res, params, initial_conditions, data, extrapolate=extrapolate, boundary=boundary, plot_infectious=True)
			plot_with_errors_sample(res, params, initial_conditions, data, extrapolate=extrapolate, boundary=boundary, plot_infectious=False)
		predictions = get_deaths(res, data, extrapolate=extrapolate)
		errors = get_fit_errors(res, params, initial_conditions, data, extrapolate=extrapolate)
		death_errors = errors[:,:,-1]
		parameters = res.x 
		cost = res.cost

	# for boundary in [(0,30), (10,40), (20, 51)]:
	# 	df = data[boundary[0]:boundary[1]]
	# 	res = least_squares(leastsq_qd, guesses, args=(df,weight), bounds=np.transpose(np.array(ranges)))
	# 	if plot:
	# 		plot_qd(res, params, initial_conditions, df, extrapolate=14, boundary=boundary[1]-boundary[0], plot_infectious=True)
	# 		# plot_with_errors_sample(res, params, initial_conditions, df, extrapolate=14, boundary=boundary[1]-boundary[0], plot_infectious=False)
	# 	predictions = get_deaths(res, df, extrapolate=14)
	# 	predictions = [(x+boundary[0],y) for (x,y) in predictions]
	# 	parameters = res.x 
	# 	cost = res.cost
	# 	# store in list right now it only returns the last prediction and parameters
	# 	print(parameters)
	# 	print(predictions)
	# 	print(cost)
	return (predictions, death_errors, parameters, cost)

def main(weight=True, plot=True):
	#Get date range of April1 to June30 inclusive. Figure out how much to extrapolate
	italy = process_data("/models/data/international/italy/covid/dpc-covid19-ita-regioni.csv", "/models/data/international/italy/demographics/region-populations.csv")
	lombardia = loader.query(italy, "Region", "Lombardia")
	fit(lombardia, weight=weight, plot=plot, extrapolate=14)

if __name__ == '__main__':
	main()
	# plot_pecaiqr()



