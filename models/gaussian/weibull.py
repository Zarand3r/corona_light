# import numpy as np
# from scipy.stats import weibull_min
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1)

# c = 1.79
# mean, var, skew, kurt = weibull_min.stats(c, moments='mvsk')

# x = np.linspace(weibull_min.ppf(0.01, c),
#                 weibull_min.ppf(0.99, c), 100)
# ax.plot(x, weibull_min.pdf(x, c),
#        'r-', lw=5, alpha=0.6, label='weibull_min pdf')

# rv = weibull_min(c)
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

# vals = weibull_min.ppf([0.001, 0.5, 0.999], c)
# np.allclose([0.001, 0.5, 0.999], weibull_min.cdf(vals, c))

# r = weibull_min.rvs(c, size=1000)
# print(r)
# # ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
# ax.scatter()
# ax.legend(loc='best', frameon=False)
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

def weib(x, *p):
    XSsat, Lo, W, s = p
    return XSsat*(1-np.exp(-((x-Lo)/W)**s))

x_data = [10.1, 11.7, 14.3, 20.2, 32.1, 37.1, 45.5, 64.2]
y_data = [2.96e-6, 2.58e-5, 1.72e-4, 1.18e-3, 2.27e-2, 3.26e-2, 3.98e-2, 4.67e-2]
p0 = [5e-2, 0, 35, 3]
coeff, pcov = curve_fit(weib, x_data, y_data, p0=p0)


#https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python