## START OF MODULE


## ###############################################################
## MODULES
## ###############################################################
import abc
import functools
import numpy as np

from datetime import datetime
from scipy.signal import argrelextrema
from scipy.special import k0, k1
from scipy.optimize import curve_fit, fsolve

## user-defined modules
from TheUsefulModule import WWLists


## ###############################################################
## CLASSES FOR STORING DATA
## ###############################################################
class SpectraConvergedScales():
  def __init__(
      self,
      Pm,
      list_k_nu_converged, list_k_eta_converged, list_k_p_converged
    ):
    ## simulation setup information
    self.Pm = Pm
    ## converged scales
    self.list_k_nu_converged  = list_k_nu_converged
    self.list_k_eta_converged = list_k_eta_converged
    self.list_k_p_converged   = list_k_p_converged


class SpectraFit():
  def __init__(
      self,
      ## ################
      ## SIMULATION SETUP
      ## ################
        sim_suite,
        sim_label,
        sim_res,
        Re,
        Rm,
        Pm,
      ## ######################
      ## KINETIC ENERGY SPECTRA
      ## ######################
        ## data
        kin_sim_times,
        kin_list_k_group_t,
        kin_list_power_group_t,
        bool_kin_fixed_model,
        ## fitted spectra
        kin_list_fit_k_group_t,
        kin_list_fit_power_group_t,
        ## measured scale
        k_nu_group_t,
        ## best fit information
        kin_list_fit_params_group_t,
        kin_list_fit_std_group_t,
        kin_fit_k_index_group_t,
        ## history of fitting data
        kin_list_fit_k_range_group_t,
        kin_list_fit_2norm_group_t,
        ## fit time range
        kin_fit_start_t,
        kin_fit_end_t,
      ## #######################
      ## MAGNETIC ENERGY SPECTRA
      ## #######################
        ## data
        mag_sim_times,
        mag_list_k_group_t,
        mag_list_power_group_t,
        bool_mag_fixed_model,
        ## fitted spectra
        mag_list_fit_k_group_t,
        mag_list_fit_power_group_t,
        ## measured scale
        k_eta_group_t,
        k_p_group_t,
        k_max_group_t,
        ## best fit information
        mag_list_fit_params_group_t,
        mag_list_fit_std_group_t,
        mag_fit_k_index_group_t,
        ## history of fitting data
        mag_list_fit_k_range_group_t,
        mag_list_fit_2norm_group_t,
        ## fit time range
        mag_fit_start_t,
        mag_fit_end_t,
      ## #######################
      ## FILE ACCESS INFORMATION
      ## #######################
        date_created = None, # not required to pass
        **kwargs # unused arguments
    ):
    ## stamp when the object was first instantiated
    if date_created is None:
      self.date_created = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    else: self.date_created = date_created
    ## indicate when/if the stored data was updated
    self.date_last_updated = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ## simulation information
    self.sim_suite                    = sim_suite
    self.sim_label                    = sim_label
    self.sim_res                      = sim_res
    self.Re                           = Re
    self.Rm                           = Rm
    self.Pm                           = Pm
    self.bool_kin_fixed_model         = bool_kin_fixed_model
    self.bool_mag_fixed_model         = bool_mag_fixed_model
    ## spectra data
    self.kin_sim_times                = kin_sim_times
    self.mag_sim_times                = mag_sim_times
    self.kin_list_k_group_t           = kin_list_k_group_t
    self.mag_list_k_group_t           = mag_list_k_group_t
    self.kin_list_power_group_t       = kin_list_power_group_t
    self.mag_list_power_group_t       = mag_list_power_group_t
    ## fit time range
    self.kin_fit_start_t              = kin_fit_start_t
    self.mag_fit_start_t              = mag_fit_start_t
    self.kin_fit_end_t                = kin_fit_end_t
    self.mag_fit_end_t                = mag_fit_end_t
    ## fitted spectra
    self.kin_list_fit_k_group_t       = kin_list_fit_k_group_t
    self.mag_list_fit_k_group_t       = mag_list_fit_k_group_t
    self.kin_list_fit_power_group_t   = kin_list_fit_power_group_t
    self.mag_list_fit_power_group_t   = mag_list_fit_power_group_t
    ## measured scales
    self.k_nu_group_t                 = k_nu_group_t
    self.k_eta_group_t                = k_eta_group_t
    self.k_p_group_t                  = k_p_group_t
    self.k_max_group_t                = k_max_group_t
    ## fitted spectra parameters
    self.kin_list_fit_params_group_t  = kin_list_fit_params_group_t
    self.mag_list_fit_params_group_t  = mag_list_fit_params_group_t
    ## uncertainties in fit parameter
    self.kin_list_fit_std_group_t     = kin_list_fit_std_group_t
    self.mag_list_fit_std_group_t     = mag_list_fit_std_group_t
    ## max k-mode fitted (break point) for best fits
    self.kin_fit_k_index_group_t      = kin_fit_k_index_group_t
    self.mag_fit_k_index_group_t      = mag_fit_k_index_group_t
    ## fit information
    self.kin_list_fit_k_range_group_t = kin_list_fit_k_range_group_t
    self.mag_list_fit_k_range_group_t = mag_list_fit_k_range_group_t
    self.kin_list_fit_2norm_group_t   = kin_list_fit_2norm_group_t
    self.mag_list_fit_2norm_group_t   = mag_list_fit_2norm_group_t


## ###############################################################
## CLASS OF USEFUL SPECTRA MODELS
## ###############################################################
class SpectraModels():
  ## ######################
  ## KINETIC SPECTRA MODELS
  ## ######################
  def kinetic_linear(x, a0, a1, a2):
    """ Exponential + power-law in linear-domain
        y   = A  * k^p  * exp(- (1 / k_scale) * k)
          = a0 * x^a1 * exp(- a2 * x)
    """
    return (a0) * np.array(x)**(a1) * np.exp(-a2 * np.array(x))
  def kinetic_loge(x, a0_loge, a1, a2):
    """ Exponential + power-law in log(e)-domain
      ln(y)   = ln(A)   + p  * ln(k) - (1 / k_scale) * k
          = a0_loge + a1 * ln(x) - a2 * x
    """
    return (a0_loge) + (a1) * np.log(x) - a2 * np.array(x)
  def kinetic_loge_fixed(x, a0_loge, a2):
    """ Exponential + Kolmogorov power law in log(e)-domain
      ln(y)   = ln(A)   - 5/3 * ln(k) - (1 / k_scale) * k
          = a0_loge - 5/3 * ln(x) - a2 * x
    """
    return a0_loge + (-5/3) * np.log(x) - a2 * np.array(x)
  ## #######################
  ## MAGNETIC SPECTRA MODELS
  ## #######################
  def magnetic_linear(x, a0, a1, a2):
    ''' Kulsrud and Anderson 1992
      y   = A  * k^p  * K0( (1 / k_scale) * k )
        = a0 * x^a1 * k0(a2 * x)
    '''
    return (a0) * np.array(x)**(a1) * k0(a2 * np.array(x))
  def magnetic_loge(x, a0_loge, a1, a2):
    ''' Kulsrud and Anderson 1992
      ln(y)   = ln(A)   + p  * ln(k) + ln(K0( (1 / k_scale) * k ))
          = a0_loge + a1 * ln(x) + ln(k0(a2 * x))
    '''
    return a0_loge + (a1) * np.log(x) + np.log(k0(a2 * np.array(x)))
  def magnetic_loge_fixed(x, a0_loge, a2):
    ''' Kulsrud and Anderson 1992 + Kazantsev power law
      ln(y)   = ln(A)   + 3/2 * ln(k) + ln(K0( (1 / k_scale) * k ))
          = a0_loge + 3/2 * ln(x) + ln(k0(a2 * x))
    '''
    return a0_loge + (3/2) * np.log(x) + np.log(k0(a2 * np.array(x)))
  def magnetic_simple_loge(x, a0_loge, a1, a2):
    """ Exponential + power-law in linear-domain
        y   = A  * k^p  * exp(- (1 / k_scale) * k)
          = a0 * x^a1 * exp(- a2 * x)
      Note:
        y' = 0 when k_p := k = p * k_scale = a1 / a2
    """
    return (a0_loge) + (a1) * np.log(x) - a2 * np.array(x)
  def magnetic_simple_loge_fixed(x, a0_loge, a2):
    """ Exponential + power-law in linear-domain
        y   = A  * k^(3/2) * exp(- (1 / k_scale) * k)
          = a0 * x^(3/2) * exp(- a2 * x)
      Note:
        y' = 0 when k_p := k = (3/2) * k_scale = (3/2) / a2
    """
    return (a0_loge) + (3/2) * np.log(x) - a2 * np.array(x)
  def k_p_implicit(x, a1, a2):
    ''' Implicit peak scale of the magnetic spectra model (Kulsrud and Anderson 1992)
      From: y'= 0
      ->  k_p = p  * k_scale  * K0( (1 / k_scale) * k_p ) / K1( (1 / k_scale) * k_p )
        x   = a1 * 1/a2     * K0(a2 * x)                / K1(a2 * x)
      ->  0   = x - a1 * 1/a2 * K0(a2 * x) / K1(a2 * x)
    '''
    return np.array(x) - (a1 / a2) * k0(a2 * np.array(x)) / k1(a2 * np.array(x))
  ## ############################
  ## NUMERICAL DISSIPATION REGIME
  ## ############################
  def tail_linear(x, a0, a1):
    ''' powerlaw (power spectra decay) in linear-domain
      y   = 10^a0 * k^a1
    '''
    return 10**(a0) * np.array(x)**(a1)
  def tail_log10(x_log10, a0, a1):
    ''' powerlaw (power spectra decay) in log10-domain
      log10(y) = a0 + a1 * log10(k)
    '''
    return (a0) + (a1) * np.array(x_log10)


## ###############################################################
## ROUTINES FOR FITTING SPECTRA
## ###############################################################
class FitSpectra(metaclass=abc.ABCMeta): # Abstract base class (ABC)
  def __init__(
      self,
      list_sim_times, list_k_group_t, list_power_group_t,
      func_plot,
      k_start              = 1,
      k_fit_from           = 5,
      k_step_size          = 1,
      bool_fit_sub_y_range = False,
      num_decades_to_fit   = 6,
      bool_hide_updates    = False
    ):
    ## ####################
    ## SAVE INPUT VARIABLES
    ## ####################
    ## spectra data
    self.list_sim_times       = list_sim_times
    self.list_k_group_t       = list_k_group_t
    self.list_power_group_t   = list_power_group_t
    ## fitting parameters
    self.bool_fit_sub_y_range = bool_fit_sub_y_range
    self.num_decades_to_fit   = num_decades_to_fit
    self.func_plot            = func_plot
    self.k_start              = k_start
    self.k_fit_from           = k_fit_from
    self.k_step_size          = k_step_size
    ## ############################
    ## INITIALISE OUTPUT PARAMETERS
    ## ############################
    ## fitted spectra
    self.list_fit_k_group_t           = []
    self.list_fit_power_group_t       = []
    self.k_scale_group_t              = [] # fitted dissipation scale
    ## fit information (for each fitted time realisation)
    self.list_best_fit_params_group_t = [] # fitted parameters
    self.list_best_fit_std_group_t    = [] # uncertainties in paramater fits
    self.best_fit_k_index_group_t     = [] # max k-mode fitted (break point) for best fits
    self.list_fit_k_range_group_t     = [] # range of k-modes fitted to
    self.list_fit_2norm_group_t       = [] # objective function evaluated for all possible k break points
    ## ############################
    ## TRY FITTING RANGE OF K-MODES
    ## ############################
    ## for each time realisation
    for _, time_index in WWLists.loopListWithUpdates(self.list_sim_times, bool_hide_updates):
      ## fit spectra and store fit information + best parameters
      self.fitTimeRealisation(time_index)
  def fitTimeRealisation(
      self,
      time_index
    ):
    list_fit_params_group_k = []
    list_params_std_group_k = []
    fit_2norm_group_k       = []
    ## extract the spectra data at a particular time point
    data_k     = self.list_k_group_t[time_index]
    data_power = self.list_power_group_t[time_index]
    ## define a final k-mode to fit to
    if self.bool_fit_sub_y_range:
      ## find the k-mode where the power spectra is the closest to cutoff y-range
      k_end = int(WWLists.getIndexClosestValue(
        np.log10(np.array(data_power)),
        -(self.num_decades_to_fit)
      ))
    ## fit up to the final k-mode
    else: k_end = int(data_k[-1])
    ## ###################################
    ## FIT TO AN INCREASING SUBSET OF DATA
    ## ###################################
    ## save the range of k explored when fitting at t/T
    list_fit_k_range = list(range(self.k_fit_from, k_end, self.k_step_size))
    for k_break in list_fit_k_range:
      ## ###################
      ## SUBSET SPECTRA DATA
      ## ###################
      ## fit spectra model to the first part of the spectra
      data_x_curve_linear = np.array(data_k[   self.k_start : k_break ])
      data_y_curve_loge   = np.log(data_power[ self.k_start : k_break ])
      ## fit dissipation model to the remaining part of the spectra
      data_x_tail_log10   = np.log10(data_k[     k_break : k_end ])
      data_y_tail_log10   = np.log10(data_power[ k_break : k_end ])
      ## weight k-modes
      list_k_weight = [ x**(0.5) for x in data_x_curve_linear ]
      ## fit spectra model
      list_fit_params_curve_loge, fit_params_cov = self.fitSubsettedSpectra(
        data_k       = data_x_curve_linear,
        data_power   = data_y_curve_loge,
        list_weights = list_k_weight
      )
      ## fit spectra in the dissipation regime in log10-linear
      list_fit_params_tail, _ = curve_fit(
        f      = SpectraModels.tail_log10,
        xdata  = data_x_tail_log10,
        ydata  = data_y_tail_log10,
        bounds = ( (-np.inf, -10), (0, 0) ),
        maxfev = 10**3
      )
      ## undo log(e) transformation of fitted parameters
      a0, a1, a2 = self.extractFitParams(list_fit_params_curve_loge)
      ## save fitted (linear) spectra model parameter values
      list_fit_params_curve = [
        np.exp(a0), # undo log(e) transform
        a1, a2
      ]
      ## evaluate fitted spectra (to all points)
      fitted_power = np.array(
        ## spectra curve
        list(self.func_plot(
          data_k[self.k_start : k_break],
          *list_fit_params_curve
        )) + 
        ## spectra tail
        list(SpectraModels.tail_linear(
          data_k[k_break : k_end],
          *list_fit_params_tail
        ))
      )
      ## measure residuals
      fit_2norm = np.sum((
        np.log10(data_power[self.k_start : k_end]) - np.log10(fitted_power)
      )**2) # 2-norm in log10-space
      list_params_std = np.sqrt(np.diag(fit_params_cov)) # std(parameters) from covariance matrix
      ## append fit information
      list_fit_params_group_k.append(list_fit_params_curve)
      list_params_std_group_k.append(list_params_std)
      fit_2norm_group_k.append(fit_2norm)
    ## save fit information
    self.list_fit_2norm_group_t.append(fit_2norm_group_k)
    ## #################
    ## FIND THE BEST FIT
    ## #################
    ## define error cut-off
    max_fit_2norm = 0.5 * ( max(fit_2norm_group_k) + min(fit_2norm_group_k) )
    ## find good fits (minima in the fit error plot)
    if fit_2norm_group_k[0] < np.percentile(fit_2norm_group_k, 84):
      list_minima_index = [ 0 ] # include the first fit in the list of good fits
    else: list_minima_index = []
    ## find local minima (good fits) in the list of fit errors
    list_minima_index.extend(list(
      argrelextrema(
        np.array(fit_2norm_group_k),
        np.less
      )[0]
    ))
    ## if there are a few good fits
    if len(list_minima_index) > 1:
      ## find fits with little error
      list_good_minima = [
        minima_index
        for minima_index in list_minima_index
        if (
          fit_2norm_group_k[minima_index] < max_fit_2norm
          and
          1 / list_fit_params_group_k[minima_index][2] < len(data_k)
        )
      ]
      ## if there are any fits with little error
      if len(list_good_minima) > 0:
        ## use the fit that fitted to the most data
        index_best_fit = list_good_minima[-1]
      ## if there are no fits with little error, then choose the last good fit
      else: index_best_fit = list_minima_index[-1]
    ## if there is only one good fit, then use it
    elif len(list_minima_index) == 1:
      index_best_fit = list_minima_index[0]
    ## otherwise, if there are no good fits, then use the fit with the smallest error
    else: index_best_fit = WWLists.getIndexListMin(fit_2norm_group_k)
    ## save best fit information
    list_best_fit_params = list_fit_params_group_k[index_best_fit]
    list_best_fit_std    = list_params_std_group_k[index_best_fit]
    best_fit_k_index     = list_fit_k_range[index_best_fit]
    self.list_fit_k_range_group_t.append(list_fit_k_range)
    self.best_fit_k_index_group_t.append(best_fit_k_index)
    self.list_best_fit_params_group_t.append(list_best_fit_params)
    self.list_best_fit_std_group_t.append(list_best_fit_std)
    ## save fitted scales
    a0_b, a1_b, a2_b = list_best_fit_params # get best fit parameters
    powerlaw_slope   = a1_b
    k_scale          = 1 / a2_b
    self.measurePeakScales(
      a1         = a1_b,
      a2         = a2_b,
      k_p_guess  = powerlaw_slope * k_scale,
      data_power = data_power
    )
    self.k_scale_group_t.append(k_scale)
    ## ###################
    ## SAVE FITTED SPECTRA
    ## ###################
    list_fit_k     = list(np.linspace(1, self.list_k_group_t[time_index][-1], 10**3))
    list_fit_power = list(self.func_plot(list_fit_k, *list_best_fit_params))
    self.list_fit_k_group_t.append(list_fit_k)
    self.list_fit_power_group_t.append(list_fit_power)
  ## ############################
  ## EMPTY METHOD IMPLEMENTATIONS
  ## ############################
  def measurePeakScales(
      self,
      a1, a2, k_p_guess, data_power
    ):
    return
  ## ################
  ## ABSTRACT METHODS
  ## ################
  ## the following methods all need to be implemented by any child-classes
  @abc.abstractmethod
  def fitSubsettedSpectra(
      self,
      data_k, data_power, list_weights
    ):
    pass
  @abc.abstractmethod
  def extractFitParams(self):
    pass


class FitKinSpectra(FitSpectra):
  def __init__(
      self,
      list_sim_times, list_k_group_t, list_power_group_t,
      bool_fit_fixed_model = False,
      k_start              = 1,
      k_fit_from           = 5,
      k_step_size          = 1,
      bool_fit_sub_y_range = False,
      num_decades_to_fit   = 6,
      bool_hide_updates    = False
    ):
    self.bool_fit_fixed_model = bool_fit_fixed_model
    fit_bounds = (
      ( np.log(10**(-10)), -5.0, 1/30 ),
      ( np.log(10**(2)),    5.0, 1/0.01 )
    )
    if self.bool_fit_fixed_model:
      ## fit spectra with a fixed power-law exponent
      self.func_fit = SpectraModels.kinetic_loge_fixed
      ## parameter bounds (excluding power-law exponent)
      self.fit_bounds = (
        ( fit_bounds[0][0], fit_bounds[0][2] ),
        ( fit_bounds[1][0], fit_bounds[1][2] )
      )
    else:
      ## fit all parameters in the spectra model
      self.func_fit   = SpectraModels.kinetic_loge
      self.fit_bounds = fit_bounds
    ## call parent class and pass fitting instructions
    FitSpectra.__init__(
      self,
      ## pass input spectra information
      list_sim_times       = list_sim_times,
      list_k_group_t       = list_k_group_t,
      list_power_group_t   = list_power_group_t,
      func_plot            = SpectraModels.kinetic_linear, # complete kinetic spectra model
      k_start              = k_start,
      k_fit_from           = k_fit_from,
      k_step_size          = k_step_size,
      bool_fit_sub_y_range = bool_fit_sub_y_range,
      num_decades_to_fit   = num_decades_to_fit,
      bool_hide_updates    = bool_hide_updates
    )
  def fitSubsettedSpectra(
      self,
      data_k, data_power, list_weights
    ):
    ## fit kinetic energy spectra with simple model
    list_fit_params_curve_loge, fit_params_cov = curve_fit(
      f      = self.func_fit,
      xdata  = data_k,
      ydata  = data_power,
      bounds = self.fit_bounds,
      sigma  = list_weights,
      maxfev = 10**3
    )
    ## return fit parameters
    return list_fit_params_curve_loge, fit_params_cov
  def extractFitParams(self, list_fit_params_curve_loge):
    if self.bool_fit_fixed_model:
      a1 = -5/3 # kolmogorov exponent
      a0, a2 = list_fit_params_curve_loge
    else: a0, a1, a2 = list_fit_params_curve_loge
    return a0, a1, a2
  def getFitDict(self):
    ## save fit output
    return {
      ## times fitted to
      "kin_sim_times":self.list_sim_times,
      ## spectra data fitted to
      "kin_list_k_group_t":self.list_k_group_t,
      "kin_list_power_group_t":self.list_power_group_t,
      ## fitted spectra
      "kin_list_fit_k_group_t":self.list_fit_k_group_t,
      "kin_list_fit_power_group_t":self.list_fit_power_group_t,
      ## fitted scales
      "k_nu_group_t":self.k_scale_group_t,
      ## fit information
      "bool_kin_fixed_model":self.bool_fit_fixed_model,
      "kin_list_fit_params_group_t":self.list_best_fit_params_group_t,
      "kin_list_fit_std_group_t":self.list_best_fit_std_group_t,
      "kin_fit_k_index_group_t":self.best_fit_k_index_group_t,
      "kin_list_fit_k_range_group_t":self.list_fit_k_range_group_t,
      "kin_list_fit_2norm_group_t":self.list_fit_2norm_group_t
    }


class FitMagSpectra(FitSpectra):
  def __init__(
      self,
      list_sim_times, list_k_group_t, list_power_group_t,
      bool_fit_fixed_model = False,
      k_start              = 1,
      k_fit_from           = 5,
      k_step_size          = 1,
      bool_hide_updates    = False
    ):
    self.bool_fit_fixed_model = bool_fit_fixed_model
    self.k_p_group_t   = [] # fitted peak scale
    self.k_max_group_t = [] # measured peak scale
    fit_bounds = (
      ( np.log(10**(-10)), -3.0, 1/100 ),
      ( np.log(10**(2)),    3.0, 1/0.01 )
    )
    ## fitting parameters
    if self.bool_fit_fixed_model:
      ## fit spectra with a fixed power-law exponent
      self.func_fit_simple = SpectraModels.magnetic_simple_loge_fixed
      self.func_fit        = SpectraModels.magnetic_loge_fixed
      ## parameter bounds (excluding power-law exponent)
      self.fit_bounds = (
        ( fit_bounds[0][0], fit_bounds[0][2] ),
        ( fit_bounds[1][0], fit_bounds[1][2] )
      )
    else:
      ## fit all parameters in the spectra model
      self.func_fit_simple = SpectraModels.magnetic_simple_loge
      self.func_fit        = SpectraModels.magnetic_loge
      self.fit_bounds      = fit_bounds
    ## call parent class and pass fitting instructions
    FitSpectra.__init__(
      self,
      list_sim_times       = list_sim_times,
      list_k_group_t       = list_k_group_t,
      list_power_group_t   = list_power_group_t,
      func_plot            = SpectraModels.magnetic_linear,
      k_start              = k_start,
      k_fit_from           = k_fit_from,
      k_step_size          = k_step_size,
      bool_hide_updates    = bool_hide_updates
    )
  def fitSubsettedSpectra(
      self,
      data_k, data_power, list_weights
    ):
    ## beat the Kulsrud and Anderson 1992 model into fitting the magnetic spectra
    ## first fit with a simple model (spectra motivated)
    list_fit_params_guess, _ = curve_fit(
      f      = self.func_fit_simple,
      xdata  = data_k,
      ydata  = data_power,
      bounds = self.fit_bounds,
      sigma  = list_weights,
      maxfev = 10**3
    )
    ## then fit the Kulsrud and Anderson 1992 model + pass the parameter guess from the simple model
    list_fit_params_curve_loge, fit_params_cov = curve_fit(
      f      = self.func_fit,
      xdata  = data_k,
      ydata  = data_power,
      bounds = self.fit_bounds,
      p0     = list_fit_params_guess,
      sigma  = list_weights,
      maxfev = 10**4
    )
    ## return fit parameters
    return list_fit_params_curve_loge, fit_params_cov
  def extractFitParams(self, list_fit_params_curve_loge):
    if self.bool_fit_fixed_model:
      a1 = 3/2 # kazantsev exponent
      a0, a2 = list_fit_params_curve_loge
    else: a0, a1, a2 = list_fit_params_curve_loge
    return a0, a1, a2
  def measurePeakScales(
      self,
      a1, a2, k_p_guess, data_power
    ):
    try:
      ## fit peak scale from the Kulsrud and Anderson 1992 model
      k_p = fsolve(
        functools.partial(
          SpectraModels.k_p_implicit,
          a1 = a1,
          a2 = a2
        ),
        x0 = k_p_guess # give a guess
      )[0]
    except (RuntimeError, ValueError):
      ## use the guess
      k_p = k_p_guess
    ## measured true peak scale (for reference)
    k_max = np.argmax(data_power) + 1
    ## save scales
    self.k_p_group_t.append(k_p)
    self.k_max_group_t.append(k_max)
  def getFitDict(self):
    ## save fit output
    return {
      ## times fitted to
      "mag_sim_times":self.list_sim_times,
      ## spectra data fitted to
      "mag_list_k_group_t":self.list_k_group_t,
      "mag_list_power_group_t":self.list_power_group_t,
      ## fitted spectra
      "mag_list_fit_k_group_t":self.list_fit_k_group_t,
      "mag_list_fit_power_group_t":self.list_fit_power_group_t,
      ## fitted scales
      "k_eta_group_t":self.k_scale_group_t,
      "k_p_group_t":self.k_p_group_t,
      "k_max_group_t":self.k_max_group_t,
      ## fit information
      "bool_mag_fixed_model":self.bool_fit_fixed_model,
      "mag_list_fit_params_group_t":self.list_best_fit_params_group_t,
      "mag_list_fit_std_group_t":self.list_best_fit_std_group_t,
      "mag_fit_k_index_group_t":self.best_fit_k_index_group_t,
      "mag_list_fit_k_range_group_t":self.list_fit_k_range_group_t,
      "mag_list_fit_2norm_group_t":self.list_fit_2norm_group_t
    }


## END OF MODULE