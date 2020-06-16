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


###########################################################

# def pecaiqr(dat, t, params, N, max_t):
# 	# define a time td of social distancing
# 	# for t > td, divide dI/dt and dA/dt and dQ/dt and dC/dt by 2 
# 	# might not be able to do this, as there is still one parameter fit, bad to average
# 	# need to paste together two separate parameter fit policy_regimes, fitted on time<td and time>td. Initial conditions of second fit are the last datapoints before td
# 	# initial parameters of the second fit are the fitted parameters of the first fit. Perhaps even fix some parameters to the values from the original fit? Look at leastsquares documentation
# 	# this way i wont need to guess how much social distancing reduces the differentials, and i can also output new parameters to see how they change
# 	if t >= max_t:
# 		return [0]*8
# 	a_1 = params[0]
# 	a_2 = params[1]
# 	a_3 = params[2]
# 	b_1 = params[3]
# 	b_2 = params[4]
# 	b_3 = params[5]
# 	b_4 = params[6]
# 	g_a = params[7]
# 	g_i = params[8]
# 	th = params[9]
# 	del_a = params[10]
# 	del_i = params[11]
# 	r_a = params[12]
# 	r_i = params[13]
# 	r_q = params[14]
# 	d_i = params[15]
# 	d_q = params[16]

# 	P = dat[0]
# 	E = dat[1]
# 	C = dat[2]
# 	A = dat[3]
# 	I = dat[4]
# 	Q = dat[5]
# 	R = dat[6]

# 	dPdt = (- ((a_1+a_2)*C*P)/N) + (-a_3*P + b_4*E)*(N/(P+E))
# 	dEdt = (- (b_1 * A + b_2 * I) * E / N) + b_3*C + (a_3*P - b_4*E)*(N/(P+E))
# 	dCdt = -(g_a + g_i)*C + ((b_1 * A + b_2 * I) * E / N) - b_3*C
# 	dAdt = (a_1 * C * P) / N + g_a*C - (r_a + del_a + th)*A
# 	dIdt = (a_2 * C * P) / N + g_i*C - ((r_i+d_i)+del_i)*I+th*A
# 	dQdt = del_a*A + del_i*I - (r_q+d_q)*Q
# 	dRdt = r_a*A + (r_i+d_i)*I + (r_q+d_q)*Q
# 	dDdt = d_i*I + d_q*Q

# 	dzdt = [dPdt, dEdt, dCdt, dAdt, dIdt, dQdt, dRdt, dDdt]
# 	return dzdt

# def model(params, data, extrapolate=-1, offset=0, death_metric="deaths"):
# 	N = data['Population'].values[0] # total population
# 	initial_conditions = N * np.array(params[-7:]) # the parameters are a fraction of the population so multiply by the population
# 	P0 = initial_conditions[0]
# 	E0 = initial_conditions[1]
# 	C0 = initial_conditions[2]
# 	A0 = initial_conditions[3]
# 	I0 = initial_conditions[4]
# 	Q0 = initial_conditions[5]
# 	# Q0 = data['active_cases'].values[0] #fit to active cases instead
# 	R0 = initial_conditions[6]
# 	D0 = abs(data[death_metric].values[0])
# 	yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	
# 	n = len(data)+death_time
# 	if extrapolate > 0:
# 		n += extrapolate
# 	args = (params, N, n)
# 	try:
# 		s = scipy.integrate.odeint(pecaiqr, yz_0, np.arange(offset, n), args=args) #printmsg = True
# 		# deaths = data['deaths'].values[0:14]
# 		# s[:,7] = np.concatenate((deaths, s[:,7][0:-14]))
# 		s[:,7] = np.concatenate((s[:,7][-14:], s[:,7][0:-14]))
# 		s = s[death_time:]
# 		# s = scipy.integrate.solve_ivp(fun=lambda t, y: pecaiqr(y, t, params, N, n), t_span=[offset, n], y0=yz_0, t_eval=np.arange(offset, n), method="LSODA")
# 	except RuntimeError:
# 		print('RuntimeError', params)
# 		return np.zeros((n, len(yz_0)))

# 	return s

def pecaiqr(dat, t, params, N, max_t, Deaths):
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
	# print(dat)
	# D = dat[7][t]

	dPdt = (- ((a_1+a_2)*C*P)/N) + (-a_3*P + b_4*E)*(N/(P+E))
	dEdt = (- (b_1 * A + b_2 * I) * E / N) + b_3*C + (a_3*P - b_4*E)*(N/(P+E))
	dCdt = -(g_a + g_i)*C + ((b_1 * A + b_2 * I) * E / N) - b_3*C
	dAdt = (a_1 * C * P) / N + g_a*C - (r_a + del_a + th)*A
	D_I = 0
	D_Q = 0
	print(t)
	if t >= death_time:
		D_I = Deaths[t-death_time] - d_q*Q
		Q_I = Deaths[t-death_time] - d_i*I
	dIdt = (a_2 * C * P) / N + g_i*C - (r_i+del_i)*I+th*A - D_I
	dQdt = del_a*A + del_i*I - (r_q+d_q)*Q
	dRdt = r_a*A + (r_i+d_i)*I + r_q*Q + D_Q
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
	Deaths = abs(data[death_metric].values)
	D0 = Deaths[0]
	yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D0])
	# D = abs(data[death_metric].values)
	# yz_0 = np.array([P0, E0, C0, A0, I0, Q0, R0, D])
	
	n = len(data)+death_time
	if extrapolate > 0:
		n += extrapolate
	args = (params, N, n, Deaths)
	try:
		s = scipy.integrate.odeint(pecaiqr, yz_0, np.arange(offset, n), args=args) #printmsg = True
		# deaths = data['deaths'].values[0:14]
		# s[:,7] = np.concatenate((deaths, s[:,7][0:-14]))
		s[:,7] = np.concatenate((s[:,7][-14:], s[:,7][0:-14]))
		s = s[death_time:]
		# s = scipy.integrate.solve_ivp(fun=lambda t, y: pecaiqr(y, t, params, N, n), t_span=[offset, n], y0=yz_0, t_eval=np.arange(offset, n), method="LSODA")
	except RuntimeError:
		print('RuntimeError', params)
		return np.zeros((n, len(yz_0)))

	return s


def leastsq_qd(params, data, bias=None, bias_value=0.4, weight=False, fitQ=False, death_metric="deaths"):
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

	if fitQ:
		q_error = Q-Qdata
		q_error = 0.1*q_error
		for i, qval in enumerate(Qdata):
			if qval <= 0:
				q_error[i] = 0
		q_error = q_error*w
		return np.concatenate((d_error, q_error))
	return d_error


