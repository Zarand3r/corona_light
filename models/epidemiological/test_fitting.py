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


def process_data(data_covid, data_population):
    covid = loader.load_data(data_covid)
    loader.convert_dates(covid, "Date")
    population = loader.load_data(data_population)
    covid['Population'] = covid.apply(lambda row: loader.query(population, "Region", row.Region)['Population'], axis=1)
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


def mse_qd(A, B):
    Ap = np.nan_to_num(A)
    Bp = np.nan_to_num(B)
    Ap[A == -np.inf] = 0
    Bp[B == -np.inf] = 0
    Ap[A == np.inf] = 0
    Bp[B == np.inf] = 0
    return mean_squared_error(Ap, Bp)

def model_qd(params, data, tmax=-1):
    # initial conditions
    N = data['Population'].values[0] # total population
    initial_conditions = N * np.array(params[-5:]) # the parameters are a fraction of the population so multiply by the population
    
    # initial conditions
    e0 = initial_conditions[0]
    i0 = initial_conditions[1]
    q0 = initial_conditions[2]
    r0 = initial_conditions[3]
    sa0 = initial_conditions[4]
    
    d0 = data['Deaths'].values[0]
    s0 = N - np.sum(initial_conditions) - d0

    yz_0 = np.array([s0, e0, i0, q0, r0, d0, sa0])
    
    # Package parameters into a tuple
    args = (params, N)
    
    n = len(data)
    if tmax > 0:
        n = tmax
    
    # Integrate ODEs
    s = scipy.integrate.odeint(seirqd, yz_0, np.arange(0, n), args=args)

    return s

def plot_qd(res, p0_params, p0_initial_conditions, data, extrapolate=1, boundary=None, plot_infectious=False):    
    s = model_qd(res.x, data, len(data)*extrapolate)
    S = s[:,0]
    E = s[:,1]
    I = s[:,2]
    Q = s[:,3]
    R = s[:,4]
    D = s[:,5]
    SA = s[:,6]

    t = np.arange(0, len(data))
    tp = np.arange(0, len(data)*extrapolate)

    p = bokeh.plotting.figure(plot_width=600,
                              plot_height=400,
                             title = ' SEIR-QD Model',
                             x_axis_label = 't (days)',
                             y_axis_label = '# people')

    if plot_infectious:
        p.line(tp, I, color = 'red', line_width = 1, legend = 'All infected')
    p.line(tp, D, color = 'black', line_width = 1, legend = 'Deceased')

    # death
    p.circle(t, data['Deaths'], color ='black')

    # quarantined
    p.circle(t, data['TotalCurrentlyPositive'], color ='purple', legend='Tested infected')
    
    if boundary is not None:
        vline = bokeh.models.Span(location=boundary, dimension='height', line_color='black', line_width=3)
        p.renderers.extend([vline])

    p.legend.location = 'top_left'
    bokeh.io.show(p)

def fit_leastsq_qd(params, data):
    Ddata = (data['Deaths'].values)
    Idata = (data['TotalCurrentlyPositive'].values)
    s = model_qd(params, data)

    S = s[:,0]
    E = s[:,1]
    I = s[:,2]
    Q = s[:,3]
    R = s[:,4]
    D = s[:,5]
    SA = s[:,6]
    
    error = np.concatenate((D-Ddata, I - Idata))
    return error


def seirqd(dat, t, params, N):
    beta = params[0] / N
    delta = params[1]
    gamma = params[2]
    alpha = params[3]
    lambda_ = params[4]
    kappa = params[5]
    
    s = dat[0]
    e = dat[1]
    i = dat[2]
    q = dat[3]
    r = dat[4]
    d = dat[5]
    sa = dat[6]
    
    dsdt = - beta * s * i - alpha * s
    dedt = beta * s * i - gamma * e
    didt = gamma * e - lambda_ * i
    dqdt = lambda_ * i - delta * q - kappa * q
    drdt = delta * q
    dddt = kappa * q
    dsadt = alpha * s
    
    # susceptible, exposed, infected, quarantined, recovered, died, unsusceptible
    return [dsdt, dedt, didt, dqdt, drdt, dddt, dsadt]

def main():
    italy = process_data("/models/data/international/italy/covid/dpc-covid19-ita-regioni.csv", "/models/data/international/italy/demographics/region-populations.csv")
    lombardia = loader.query(italy, "Region", "Lombardia")

    params = [2.0, 0.3, 0.2, 0.05, 0.2, 0.03]
    initial_conditions = [0.5e-3, 0.5e-3, 0.3e-3, 0.1e-4, 0.5]
    param_ranges = [(0.5, 3.0), (0.0, 0.5), (0.0, 0.5), (0.01, 0.5), (0.0, 0.5), (0.005, 0.1)]
    initial_ranges = [(1.0e-7, 0.01), (1.0e-7, 0.01), (1.0e-7, 0.01), (1.0e-7, 0.01), (1.0e-7, 0.9)]
    guesses = params + initial_conditions
    ranges = param_ranges + initial_ranges

    start = 10
    step = 4
    ind = 0
    results = []
    one_more = False

    for boundary in range(10, 40, 10):
        res = least_squares(fit_leastsq_qd, guesses, args=(lombardia[:boundary],), bounds=np.transpose(np.array(ranges)))
        plot_qd(res, params, initial_conditions, lombardia, extrapolate=2, boundary=boundary, plot_infectious=True)


if __name__ == '__main__':
   main()

   