## https://stackoverflow.com/questions/35544233/fit-a-curve-to-a-histogram-in-python

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# First normal distribution parameters
mu0 = 3
std0 = 1
# Second normal distribution parameters
mu1 = 7
std1 = 0.4
## generate samples from distributions
data0 = list(np.random.normal(mu0, std0, 1000))
data1 = list(np.random.normal(mu1, std1, 1000))
data = data0 + data1

## calculate data bins
dens, bin_edges = np.histogram(data, bins=50, density=True)
dens_norm = np.append(0, dens / dens.sum()) # normalise density

peaks = find_peaks(
    dens_norm,
    height    = 0.01,
    threshold = 0.005,
    distance  = (min(bin_edges) + max(bin_edges)) / 4
)
height = peaks[1]['peak_heights'] # list of the heights of the peaks
peak_pos = bin_edges[peaks[0]] # list of the peaks positions

def gaussian(x, a, mu, std):
    return a * np.exp( - (x - mu)**2 / (2*std ** 2))

def bimodal(x, a0, mu0, std0, a1, mu1, std1):
    return gaussian(x, a0, mu0, std0) + gaussian(x, a1, mu1, std1)

if len(peak_pos) > 1:
    fit_params, _ = curve_fit(
        bimodal,
        bin_edges, dens_norm,
        bounds = (
            (
                1e-3, 0.1, 0.01,
                1e-3, 0.1, 0.01
            ),
            (
                1, 15, 2,
                1, 15, 2
            )
        ),
        p0 = (
            1, peak_pos[0], 0.5,
            1, peak_pos[1], 0.5
        ),
        maxfev = 5*10**3
    )
    print(fit_params[:3])
    print(fit_params[3:])
    if fit_params[2] < fit_params[4]:
        main_distribution = gaussian(bin_edges, *fit_params[3:])
    else: main_distribution = gaussian(bin_edges, *fit_params[:3])
    list_main_indices = [
        bin_index
        for bin_index in range(len(bin_edges))
        if main_distribution[bin_index] > 0.002
    ]
    main_edges = [
        bin_edges[bin_index]
        for bin_index in range(len(bin_edges))
        if bin_index in list_main_indices
    ]
    main_data = [
        dens_norm[bin_index]
        for bin_index in range(len(bin_edges))
        if bin_index in list_main_indices
    ]

## fit KDE (this doesn't help...)
# import statsmodels.api as sm
# kde = sm.nonparametric.KDEUnivariate(data)
# kde.fit(bw=0.5)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(bin_edges, dens_norm, drawstyle="steps", color="black", lw=3)
ax.plot(bin_edges, bimodal(bin_edges, *fit_params), drawstyle="steps", color="red", lw=3, alpha=0.75)
# ax.plot(bin_edges, main_distribution, drawstyle="steps", color="blue", lw=3, alpha=0.75)
# ax.plot(kde.support, kde.density, lw=3)
for pos in peak_pos:
    ax.axvline(x=pos, ls="--", lw=2, color="green")
if len(peak_pos) > 1:
    ax.plot(main_edges, main_data, drawstyle="steps", color="blue", lw=3, alpha=0.75)
plt.show()
