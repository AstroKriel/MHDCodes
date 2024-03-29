## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np
import warnings

from scipy.optimize import curve_fit
from scipy import interpolate

## load user defined modules
from TheUsefulModule import WWLists
from TheFittingModule import UserModels

## ###############################################################
## FUNCTIONS THAT INTERPOLATE AND FIT
## ###############################################################
def interpData(x, y, x_interp, interp_kind="cubic"):
  interpolator = interpolate.interp1d(x, y, kind=interp_kind)
  return interpolator(x_interp)

def interpLogLogData(x, y, x_interp, interp_kind="cubic", ax=None):
  interpolator = interpolate.interp1d(np.log10(x), np.log10(y), kind=interp_kind)
  y_interp = np.power(10.0, interpolator(np.log10(x_interp)))
  
  return y_interp

def fitExpFunc(
    data_x, data_y, index_start_fit, index_end_fit,
    ax                = None,
    num_interp_points = 10**2,
    color             = "black",
    linestyle         = "-"
  ):
  ## define fit domain
  data_fit_domain = np.linspace(
    data_x[index_start_fit],
    data_x[index_end_fit],
    int(num_interp_points)
  )[1:-1]
  ## interpolate the non-uniform data
  interp_spline = interpolate.interp1d(
    data_x[index_start_fit : index_end_fit],
    data_y[index_start_fit : index_end_fit],
    kind       = "cubic",
    fill_value = "extrapolate"
  )
  ## save time range being fitted to
  time_start = data_x[index_start_fit]
  time_end   = data_x[index_end_fit]
  ## uniformly sample interpolated data
  data_y_sampled = abs(interp_spline(data_fit_domain))
  ## fit exponential function to sampled data (in log-linear domain)
  fit_params_log, fit_params_cov = curve_fit(
    f     = UserModels.ListOfModels.exp_loge,
    xdata = data_fit_domain,
    ydata = np.log(data_y_sampled)
  )
  gamma_std = max(np.sqrt(np.diag(fit_params_cov))[1], 0.01)
  ## undo log transformation
  fit_params_linear = [
    np.exp(fit_params_log[0]),
    fit_params_log[1]
  ]
  gamma_val = fit_params_linear[1]
  if ax is not None:
    ## initialise the plot domain
    data_x_fit = np.linspace(-10, 500, 10**3)
    ## evaluate exponential
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      data_y_fit = UserModels.ListOfModels.exp_linear(data_x_fit, *fit_params_linear)
    ## find where exponential enters / exists fit range
    index_E_start = WWLists.getIndexClosestValue(data_x_fit, time_start)
    index_E_end   = WWLists.getIndexClosestValue(data_x_fit, time_end)
    str_label = r"$\gamma_{\rm exp} =$ " + "{:.2f}".format(gamma_val) + r" $\pm$ " + "{:.2f}".format(gamma_std)
    ax.plot(
      data_x_fit[index_E_start : index_E_end],
      data_y_fit[index_E_start : index_E_end],
      label=str_label, color=color, ls=linestyle, lw=2, zorder=5
    )
  return gamma_val, gamma_std

def fitLinearFunc(
    ax, data_x, data_y, index_start_fit, index_end_fit,
    num_interp_points = 10**2,
    color             = "black",
    linestyle         = "-"
  ):
  ## define fit domain
  data_fit_domain = np.linspace(
    data_x[index_start_fit],
    data_x[index_end_fit],
    int(num_interp_points)
  )[1:-1]
  ## interpolate the non-uniform data
  interp_spline = interpolate.interp1d(
    data_x[index_start_fit : index_end_fit],
    data_y[index_start_fit : index_end_fit],
    kind       = "cubic",
    fill_value = "extrapolate"
  )
  ## save time range being fitted to
  time_start = data_x[index_start_fit]
  time_end   = data_x[index_end_fit]
  ## uniformly sample interpolated data
  data_y_sampled = abs(interp_spline(data_fit_domain))
  ## fit exponential function to sampled data (in log-linear domain)
  fit_params, fit_params_cov = curve_fit(
    f     = UserModels.ListOfModels.linear_offset,
    xdata = data_fit_domain,
    ydata = data_y_sampled
  )
  ## initialise the plot domain
  data_x_fit = np.linspace(-10, 500, 10**3)
  ## evaluate exponential
  data_y_fit = UserModels.ListOfModels.linear_offset(data_x_fit, *fit_params)
  ## plot fit
  gamma_val = fit_params[1]
  gamma_std = max(np.sqrt(np.diag(fit_params_cov))[1], 0.01)
  str_label = r"$\gamma_{\rm lin} =$ " + "{:.2f}".format(gamma_val) + r" $\pm$ " + "{:.2f}".format(gamma_std)
  ## find where exponential enters / exists fit range
  index_E_start = WWLists.getIndexClosestValue(data_x_fit, time_start)
  index_E_end   = WWLists.getIndexClosestValue(data_x_fit, time_end)
  ax.plot(
    data_x_fit[index_E_start : index_E_end],
    data_y_fit[index_E_start : index_E_end],
    label=str_label, color=color, ls=linestyle, lw=2, zorder=5
  )
  return fit_params[1]

def fitConstFunc(
    data_x, data_y, index_start_fit, index_end_fit,
    num_interp_points = 10**2,
    ax                = None,
    str_label         = "",
    linestyle         = "-"
  ):
  ## define fit domain
  data_fit_domain = np.linspace(
    data_x[index_start_fit],
    data_x[index_end_fit],
    int(num_interp_points)
  )[1:-1]
  ## interpolate the non-uniform data
  interp_spline = interpolate.interp1d(
    data_x[index_start_fit : index_end_fit],
    data_y[index_start_fit : index_end_fit],
    kind       = "cubic",
    fill_value = "extrapolate"
  )
  ## uniformly sample interpolated data
  data_y_sampled = interp_spline(data_fit_domain)
  ## measure average saturation level
  data_y_mean = np.mean(data_y_sampled)
  data_y_std  = max(np.std(data_y_sampled), 0.01)
  ## plot fit
  if ax is not None:
    data_x_sub = data_x[index_start_fit : index_end_fit]
    str_label += "{:.2f}".format(data_y_mean) + r" $\pm$ " + "{:.2f}".format(data_y_std)
    ax.plot(
      data_x_sub,
      [ data_y_mean ] * len(data_x_sub),
      label=str_label, color="black", ls=linestyle, lw=2, zorder=5
    )
  ## return fitted quantities
  return data_y_mean, data_y_std


## END OF LIBRARY