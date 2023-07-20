## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np
import lmfit

# import warnings
# warnings.filterwarnings("error")

from scipy.signal import find_peaks

## not required anymore
import abc, functools
from scipy.special import k0, k1

## load user defined modules
from . import FitFuncs
from TheUsefulModule import WWLists
from ThePlottingModule import PlotFuncs


## ###############################################################
## CLASS OF USEFUL SPECTRA MODELS
## ###############################################################
class KineticSpectraModels():
  ## ######################
  ## KINETIC SPECTRA MODELS
  ## ######################
  def simple_linear(k, A, alpha_cas, alpha_dis, k_nu):
    k = np.array(k)
    return A * k**(alpha_cas) * np.exp(-(k / k_nu)**alpha_dis)

  def simple_loge(k, A, alpha_cas, alpha_dis, k_nu):
    k = np.array(k)
    return np.log(A) + alpha_cas * np.log(k) - (k / k_nu)**alpha_dis

  def bottleneck_linear(k, A, alpha_cas, alpha_bot, alpha_dis, k_nu_bot, k_nu_dis):
    k = np.array(k)
    return A * k**alpha_cas * (1 + (k / k_nu_bot)**alpha_bot) * np.exp(-(k / k_nu_dis)**alpha_dis)

  def bottleneck_loge(k, A, alpha_cas, alpha_bot, alpha_dis, k_nu_bot, k_nu_dis):
    return np.log(A) + alpha_cas * np.log(k) + np.log(1 + (k / k_nu_bot)**alpha_bot) - (k / k_nu_dis)**alpha_dis

class SpectraModels():
  def kinetic_linear(k, A, alpha, k_nu):
    ''' exponential + powerlaw in linear-domain:
        y = A * k^alpha * exp(- k / k_nu)
    '''
    return A * np.array(k)**(alpha) * np.exp(-(np.array(k) / k_nu))

  def kinetic_loge(k, A, alpha, k_nu):
    return np.log(A) + alpha * np.log(k) - (np.array(k) / k_nu)

  def magnetic_linear(k, A, alpha_1, alpha_2, k_eta):
    ''' modified Kulsrud and Anderson 1992 in linear-domain:
        y = A * k^alpha_1 * K0( (k / k_eta)^alpha_2 )
    '''
    return A * np.array(k)**(alpha_1) * k0( (np.array(k) / k_eta)**(alpha_2) )

  def magnetic_loge(k, A, alpha_1, alpha_2, k_eta):
    arg = (np.array(k) / k_eta)**(alpha_2)
    try:
      log_bessel = np.where(
        arg > 50,
        ## approximate ln(K0(...)) with the first two terms from the series expansion of K0(...)
        -arg + np.log(np.sqrt(np.pi/2) * ( np.sqrt(1/arg) - 1/8 * (1/arg)**(3/2) )),
        ## evaluate ln(K0(...))
        np.log(k0(arg))
      )
    except Warning:
      print("Bounds of input arguments:", np.min(arg), np.max(arg))
      raise Exception("Error: failed to fit modified bessel function.")
    return np.log(A) + alpha_1 * np.log(k) + log_bessel

  def k_p_implicit(k, alpha_1, alpha_2, k_eta):
    ''' peak scale of the magnetic energy spectra model (modified Kulsrud and Anderson 1992).
      when y'= 0:
        k_p :=  k = ( alpha_1 / alpha_2 * K0(...) / K1(...) )^(1/alpha_2) * k_eta
    '''
    arg = (np.array(k) / k_eta)**(alpha_2)
    return np.array(k) - ( alpha_1 / alpha_2 * k0(arg) / k1(arg) )**(1/alpha_2) * k_eta

  def magnetic_linear_simple(k, A, alpha_1, alpha_2, k_eta):
    ''' simple model: exponential + powerlaw in linear-domain:
        y = A * k^alpha * exp(- k / k_eta)
    '''
    return A * np.array(k)**(alpha_1) * np.exp( -(np.array(k) / k_eta)**(alpha_2) )

  def magnetic_loge_simple(k, A, alpha_1, alpha_2, k_eta):
    return np.log(A) + alpha_1 * np.log(k) - (np.array(k) / k_eta)**(alpha_2)
    # return np.log(SpectraModels.magnetic_linear_simple(k, A, alpha_1, alpha_2, k_eta))

  def k_p_simple(alpha_1, alpha_2, k_eta):
    return (alpha_1 / alpha_2)**(1/alpha_2) * k_eta

  def tail_linear(k, alpha_1, alpha_2):
    ''' powerlaw in linear-domain
      y = 10^alpha_1 * k^alpha_2
    '''
    return 10**(alpha_1) * np.array(k)**(alpha_2)

  def tail_log10(k_log10, alpha_1, alpha_2):
    ''' powerlaw in log10-domain
      log10(y) = alpha_1 + alpha_2 * log10(k)
    '''
    return alpha_1 + alpha_2 * np.array(k_log10)


## ###############################################################
## FUNCTIONS
## ###############################################################
def fitKinSpectrum(
    list_k, list_power,
    list_power_std   = None,
    ax_fit           = None,
    ax_residuals     = None,
    color            = "black",
    label_spect      = "",
    bool_fix_params = False
  ):
  ## define helper function
  def plotFitResiduals(ax, list_x, list_y, fit_params, func, color="black", label_spect=""):
    list_y_fit = func(list_x, *fit_params)
    ax.plot(
      list_x,
      np.log10(list_y_fit) / np.log10(list_y),
      label=label_spect, ls="-", marker="o", ms=8, color=color, markeredgecolor="black"
    )
  ## define model to fit
  fitFuncLoge   = KineticSpectraModels.simple_loge
  fitFuncLinear = KineticSpectraModels.simple_linear
  fit_model     = lmfit.Model(fitFuncLoge) # fit in log-log domain
  fit_model.set_param_hint("A",         min =  10**(-2.0), value =  10**(-1.0), max =  10**(2.0))
  fit_model.set_param_hint("alpha_cas", min = -8.0,        value = -2.0,        max = -0.1, vary=bool_fix_params)
  fit_model.set_param_hint("alpha_dis", min =  0.1,        value =  1.0,        max =  2.0, vary=not(bool_fix_params))
  fit_model.set_param_hint("k_nu",      min =  0.1,        value =  5.0,        max =  100)
  ## find k-index to stop fitting kinetic energy spectrum
  fit_index_start  = 2
  if min(list_power) - min(list_power) < 10**(-5):
    min_power_offset  = 10**(np.log10(min(list_power)) + 1)
    min_power_target  = max([ 10**(-8), min_power_offset ])
    fit_index_end     = WWLists.getIndexClosestValue(list_power, min_power_target)
  else: fit_index_end = len(list_power) - 1
  ## fit kinetic energy model (in log-linear domain) to subset of data
  fit_results  = fit_model.fit(
    k      = list_k[           fit_index_start : fit_index_end],
    data   = np.log(list_power[fit_index_start : fit_index_end]),
    params = fit_model.make_params(),
  )
  ## extract fitted parameters
  fit_params_values = []
  fit_params_values.append(fit_results.params["A"].value)
  fit_params_values.append(fit_results.params["alpha_cas"].value)
  fit_params_values.append(fit_results.params["alpha_dis"].value)
  fit_params_values.append(fit_results.params["k_nu"].value)
  ## extract uncertainty in fitted parameters
  fit_params_errors = []
  fit_params_errors.append(fit_results.params["A"].stderr)
  fit_params_errors.append(fit_results.params["alpha_cas"].stderr)
  fit_params_errors.append(fit_results.params["alpha_dis"].stderr)
  fit_params_errors.append(fit_results.params["k_nu"].stderr)
  ## compute reduced chi-squared
  list_power_fit = fitFuncLinear(list_k, *fit_params_values)
  num_dof  = 3
  if list_power_std is not None:
    fit_rcs = sum(
      (np.log(list_power) - np.log(list_power_fit))**2 / list_power_std
    ) / num_dof
  else: fit_rcs = None
  ## plot fitted spectrum
  if ax_fit is not None:
    array_k_fit     = np.logspace(-3, 3, 1000)
    array_power_fit = fitFuncLinear(array_k_fit, *fit_params_values)
    PlotFuncs.plotData_noAutoAxisScale(
      ax = ax_fit,
      x  = array_k_fit,
      y  = array_power_fit,
      color=color, ls="-", lw=6, alpha=0.65, zorder=10
    )
  ## plot residuals of fit
  if ax_residuals is not None:
    plotFitResiduals(
      ax          = ax_residuals,
      list_x      = list_k[    fit_index_start : fit_index_end],
      list_y      = list_power[fit_index_start : fit_index_end],
      fit_params  = fit_params_values,
      func        = fitFuncLinear,
      color       = color,
      label_spect = label_spect
    )
  ## return fitted parameters
  return fit_params_values, fit_params_errors, fit_rcs

def getSpectrumPeakScale(list_k, list_power):
  array_k_interp = np.logspace(
    start = np.log10(min(list_k)),
    stop  = np.log10(max(list_k)),
    num   = 3*len(list_power)
  )[1:-1]
  array_power_interp = FitFuncs.interpLogLogData(
    x           = list_k,
    y           = list_power,
    x_interp    = array_k_interp,
    interp_kind = "cubic"
  )
  k_max_interp = array_k_interp[np.argmax(array_power_interp)]
  k_max_raw    = list_k[np.argmax(list_power)]
  return k_max_interp, k_max_raw

def getEquipartitionScale(
    list_times, list_k, list_power_mag_group_t, list_power_kin_group_t,
    tol        = 1e-1,
    ax_spectra = None,
    ax_scales  = None,
    color      = "black",
    label      = r"$k_{\rm eq}$"
  ):
  ## store data for each time realisation where k_eq is defined
  k_eq_group_t       = []
  k_eq_power_group_t = []
  list_time_k_eq     = []
  ## loop through each time realisation
  for time_index in range(len(list_times)):
    ## calculate energy spectrum ratio
    list_power_mag = np.array(list_power_mag_group_t[time_index])
    list_power_kin = np.array(list_power_kin_group_t[time_index])
    list_power_ratio = list_power_mag / list_power_kin
    if ax_spectra is not None: ax_spectra.plot(list_k, list_power_ratio, color=color, ls="-", lw=1.5, alpha=0.1)
    ## calculate where to cutoff the spectrum ratio
    ## (ignore only measure the first peaks)
    list_index_peaks, _ = find_peaks(list_power_ratio)
    if len(list_index_peaks) > 0:
      index_cutoff = min(list_index_peaks)
    else: index_cutoff = len(list_power_ratio) - 1
    ## find points where the spectrum ratio is close to unity
    list_index_k_eq = [
      k_index
      for k_index, E_ratio in enumerate(list_power_ratio[:index_cutoff])
      if abs(E_ratio - 1) <= tol
    ]
    ## measure the first point where the spectrum ratio gets close to unity
    if len(list_index_k_eq) > 0:
      index_k_eq = list_index_k_eq[0]
      k_eq       = list_k[index_k_eq]
      k_eq_power = list_power_ratio[index_k_eq]
      k_eq_group_t.append(k_eq)
      k_eq_power_group_t.append(k_eq_power)
      list_time_k_eq.append(list_times[time_index])
  ## plot measured equipartition scale
  if (len(k_eq_group_t) > 0) and (ax_scales is not None):
    ax_scales.plot(
      list_time_k_eq,
      k_eq_group_t,
      label=label, color=color, ls="-"
    )
  ## return equipartition scale measured for each time realisation
  return k_eq_group_t, k_eq_power_group_t, list_time_k_eq


## END OF LIBRARY