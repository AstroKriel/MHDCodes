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
      k_nu_converged,
      k_eta_converged,
      k_p_converged,
      k_nu_std,
      k_eta_std,
      k_p_std
    ):
    self.k_nu_converged  = k_nu_converged
    self.k_eta_converged = k_eta_converged
    self.k_p_converged   = k_p_converged
    self.k_nu_std        = k_nu_std
    self.k_eta_std       = k_eta_std
    self.k_p_std         = k_p_std


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
        kin_bool_fit_fixed_model,
        ## data
        kin_list_sim_times,
        kin_list_k_group_t,
        kin_list_power_group_t,
        ## fitted spectra
        kin_list_fit_k_group_t,
        kin_list_fit_power_group_t,
        ## measured scale
        k_nu_group_t,
        ## best fit information
        kin_list_fit_params_group_t,
        kin_list_fit_std_group_t,
        kin_max_k_mode_fitted_group_t,
        ## history of fitting data
        kin_fit_k_start,
        kin_list_fit_k_range_group_t,
        kin_list_fit_2norm_group_t,
        ## fit time range
        kin_fit_time_start,
        kin_fit_time_end,
      ## #######################
      ## MAGNETIC ENERGY SPECTRA
      ## #######################
        mag_bool_fit_fixed_model,
        ## data
        mag_list_sim_times,
        mag_list_k_group_t,
        mag_list_power_group_t,
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
        mag_max_k_mode_fitted_group_t,
        ## history of fitting data
        mag_fit_k_start,
        mag_list_fit_k_range_group_t,
        mag_list_fit_2norm_group_t,
        ## fit time range
        mag_fit_time_start,
        mag_fit_time_end,
      ## #######################
      ## FILE ACCESS INFORMATION
      ## #######################
        date_created = None, # not required to define
        **kwargs             # unused arguments
    ):
    ## stamp when the object was first instantiated
    if date_created is None:
      self.date_created = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    else: self.date_created = date_created
    ## indicate when/if the stored data was updated
    self.date_last_updated = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ## simulation information
    self.sim_suite                     = sim_suite
    self.sim_label                     = sim_label
    self.sim_res                       = sim_res
    self.Re                            = Re
    self.Rm                            = Rm
    self.Pm                            = Pm
    self.kin_bool_fit_fixed_model      = kin_bool_fit_fixed_model
    self.mag_bool_fit_fixed_model      = mag_bool_fit_fixed_model
    ## raw spectra data
    self.kin_list_sim_times            = kin_list_sim_times
    self.mag_list_sim_times            = mag_list_sim_times
    self.kin_list_k_group_t            = kin_list_k_group_t
    self.mag_list_k_group_t            = mag_list_k_group_t
    self.kin_list_power_group_t        = kin_list_power_group_t
    self.mag_list_power_group_t        = mag_list_power_group_t
    ## time range corresponding with the domain of interest (kinematic regime)
    self.kin_fit_time_start            = kin_fit_time_start
    self.mag_fit_time_start            = mag_fit_time_start
    self.kin_fit_time_end              = kin_fit_time_end
    self.mag_fit_time_end              = mag_fit_time_end
    ## fitted spectra
    self.kin_list_fit_k_group_t        = kin_list_fit_k_group_t
    self.mag_list_fit_k_group_t        = mag_list_fit_k_group_t
    self.kin_list_fit_power_group_t    = kin_list_fit_power_group_t
    self.mag_list_fit_power_group_t    = mag_list_fit_power_group_t
    ## measured scales
    self.k_nu_group_t                  = k_nu_group_t
    self.k_eta_group_t                 = k_eta_group_t
    self.k_p_group_t                   = k_p_group_t
    self.k_max_group_t                 = k_max_group_t
    ## fitted spectra parameters + uncertainties
    self.kin_list_fit_params_group_t   = kin_list_fit_params_group_t
    self.mag_list_fit_params_group_t   = mag_list_fit_params_group_t
    self.kin_list_fit_std_group_t      = kin_list_fit_std_group_t
    self.mag_list_fit_std_group_t      = mag_list_fit_std_group_t
    ## max k-mode fitted (break point) for the best fit from each time realisation
    self.kin_fit_k_start               = kin_fit_k_start
    self.mag_fit_k_start               = mag_fit_k_start
    self.kin_max_k_mode_fitted_group_t = kin_max_k_mode_fitted_group_t
    self.mag_max_k_mode_fitted_group_t = mag_max_k_mode_fitted_group_t
    ## k-range the fitter tried fitting to
    self.kin_list_fit_k_range_group_t  = kin_list_fit_k_range_group_t
    self.mag_list_fit_k_range_group_t  = mag_list_fit_k_range_group_t
    ## fit information
    self.kin_list_fit_2norm_group_t    = kin_list_fit_2norm_group_t
    self.mag_list_fit_2norm_group_t    = mag_list_fit_2norm_group_t


## ###############################################################
## CLASS OF USEFUL SPECTRA MODELS
## ###############################################################
class SpectraModels():
  ## ######################
  ## KINETIC SPECTRA MODELS
  ## ######################
  def kinetic_linear(x, a0, a1, a2):
    """ Exponential + powerlaw in linear-domain
        y = A  * k^p  * exp(- (1 / k_scale) * k)
          = a0 * x^a1 * exp(- a2 * x)
        where x = k, a0 = A, a1 = p, a2 = 1/k_scale
    """
    return (a0) * np.array(x)**(a1) * np.exp(-a2 * np.array(x))

  def kinetic_loge(x, a0_loge, a1, a2):
    """ Exponential + powerlaw in log(e)-domain
      ln(y) = ln(A)   + p  * ln(k) - (1 / k_scale) * k
            = a0_loge + a1 * ln(x) - a2 * x
      where x = k, a0_loge = ln(A), a1 = p, a2 = 1/k_scale
    """
    return (a0_loge) + (a1) * np.log(x) - a2 * np.array(x)

  ## #######################
  ## MAGNETIC SPECTRA MODELS
  ## #######################
  def magnetic_linear(x, a0, a1, a2):
    ''' Kulsrud and Anderson 1992
      y = A  * k^p  * K0( (1 / k_scale) * k )
        = a0 * x^a1 * k0(a2 * x)
      where x = k, a0 = A, a1 = p, a2 = 1/k_scale
    '''
    return (a0) * np.array(x)**(a1) * k0(a2 * np.array(x))

  def magnetic_loge(x, a0_loge, a1, a2):
    ''' Kulsrud and Anderson 1992
      ln(y) = ln(A)   + p  * ln(k) + ln(K0( (1 / k_scale) * k ))
            = a0_loge + a1 * ln(x) + ln(k0(a2 * x))
      where x = k, a0_loge = ln(A), a1 = p, a2 = 1/k_scale
    '''
    return a0_loge + (a1) * np.log(x) + np.log(k0(a2 * np.array(x)))

  def magnetic_simple_loge(x, a0_loge, a1, a2):
    """ Exponential + powerlaw in linear-domain
        y = A  * k^p  * exp(- (1 / k_scale) * k)
          = a0 * x^a1 * exp(- a2 * x)
        where x = k, a0 = A, a1 = p, a2 = 1/k_scale
      Note:
        y' = 0 when k_p := k = p * k_scale = a1 / a2
    """
    return (a0_loge) + (a1) * np.log(x) - a2 * np.array(x)

  def k_p_implicit(x, a1, a2):
    ''' Implicit peak scale of the magnetic energy spectra model (Kulsrud and Anderson 1992)
      From: y'= 0
      ->  k_p = p  * k_scale  * K0( (1 / k_scale) * k_p ) / K1( (1 / k_scale) * k_p )
          x   = a1 * 1/a2     * K0(a2 * x) / K1(a2 * x)
      ->  0   = x - a1 * 1/a2 * K0(a2 * x) / K1(a2 * x)
          where x = k, a1 = p, a2 = 1/k_scale
    '''
    return np.array(x) - (a1 / a2) * k0(a2 * np.array(x)) / k1(a2 * np.array(x))

  ## ############################
  ## NUMERICAL DISSIPATION REGIME
  ## ############################
  def tail_linear(x, a0, a1):
    ''' powerlaw in linear-domain
      y   = 10^a0 * k^a1
    '''
    return 10**(a0) * np.array(x)**(a1)

  def tail_log10(x_log10, a0, a1):
    ''' powerlaw in log10-domain
      log10(y) = a0 + a1 * log10(k)
    '''
    return (a0) + (a1) * np.array(x_log10)


## ###############################################################
## ROUTINE FOR FITTING SPECTRA
## ###############################################################
class FitSpectra(metaclass=abc.ABCMeta): # abstract base class
  ## default fit parameters
  bool_fit_sub_y_range = False
  num_decades_to_fit   = 6

  def __init__(self, bool_hide_updates=False):
    ## initialise fitted spectra
    self.list_fit_k_group_t        = []
    self.list_fit_power_group_t    = []
    self.k_scale_group_t           = [] # fitted dissipation scale
    ## initialise fit information (for each fitted time realisation)
    self.list_fit_params_group_t   = [] # fitted parameters
    self.list_fit_std_group_t      = [] # uncertainty in paramater fits
    self.max_k_mode_fitted_group_t = [] # max k-mode fitted (break point) for best fits
    self.list_fit_k_range_group_t  = [] # range of k-modes fitted to
    self.list_fit_2norm_group_t    = [] # 2-norm evaluated for all possible k break points
    ## for each time realisation try fitting a range of k-modes
    for _, time_index in WWLists.loopListWithUpdates(self.list_sim_times, bool_hide_updates):
      ## extract the spectra data at a particular time point
      data_k     = self.list_k_group_t[time_index]
      data_power = self.list_power_group_t[time_index]
      ## define a k-mode to stop fitting the spectra energy model
      if self.bool_fit_sub_y_range:
        ## find the k-mode where the energy spectra is closest to the cut-off energy
        k_end = int(WWLists.getIndexClosestValue(
          np.log10(np.array(data_power)),
          -(self.num_decades_to_fit)
        ))
      ## fit up to the final k-mode
      else: k_end = int(data_k[-1])
      ## fit spectra and store fit information + best parameters
      self.fitTimeRealisation(data_k, data_power, k_end)

  def fitTimeRealisation(
      self,
      data_k, data_power, k_end
    ):
    fit_2norm_group_k       = []
    list_fit_params_group_k = []
    list_params_std_group_k = []
    list_fit_k_range        = list(range(self.k_break_from, k_end, self.k_step_size))
    ## fit to an increasing subset of the data
    for k_break in list_fit_k_range:
      list_fit_params, list_fit_params_std, fit_2norm = self.__fitSpectra(data_k, data_power, k_break, k_end)
      ## store fit information
      list_fit_params_group_k.append(list_fit_params)     # fitted parameters
      list_params_std_group_k.append(list_fit_params_std) # uncertainty in fit parameters
      fit_2norm_group_k.append(fit_2norm)                 # 2-norm of fit
    ## store 2-norm of all attempted fits
    self.list_fit_2norm_group_t.append(fit_2norm_group_k)
    ## store list of max k-modes (break points) explored
    self.list_fit_k_range_group_t.append(list_fit_k_range)
    ## find the best fit
    best_fit_index = self.__findBestFit(fit_2norm_group_k, list_fit_params_group_k)
    ## extract best fit information
    bf_max_k_mode_fitted = list_fit_k_range[best_fit_index]        # maximum k-mode fitted
    bf_list_fit_params   = list_fit_params_group_k[best_fit_index] # fit params
    bf_list_fit_std      = list_params_std_group_k[best_fit_index] # uncertainty in fit params
    ## store best fit information
    self.max_k_mode_fitted_group_t.append(bf_max_k_mode_fitted)
    self.list_fit_params_group_t.append(bf_list_fit_params)
    self.list_fit_std_group_t.append(bf_list_fit_std)
    ## save scales measured from best fit
    self.__saveBestFit(data_k, data_power, list_fit_params=bf_list_fit_params)

  def __fitSpectra(
      self,
      data_k, data_power, k_break, k_end
    ):
    ## fit spectra model to the first part of the spectra
    data_x_curve_linear = np.array(data_k[   self.k_start : k_break ])
    data_y_curve_loge   = np.log(data_power[ self.k_start : k_break ])
    ## fit dissipation model to the remaining part of the spectra
    data_x_tail_log10   = np.log10(data_k[     k_break : k_end ])
    data_y_tail_log10   = np.log10(data_power[ k_break : k_end ])
    ## weight k-modes
    list_k_weight = [ x**(0.5) for x in data_x_curve_linear ]
    ## fit spectra model
    list_fit_params_curve_loge, fit_params_cov = self.auxFitSpectra(
      data_k       = data_x_curve_linear,
      data_power   = data_y_curve_loge,
      list_weights = list_k_weight
    )
    ## undo log(e) transformation of fitted parameters
    a0, a1, a2 = self.auxGetFitParams(list_fit_params_curve_loge)
    ## save fitted (linear) spectra model parameter values
    list_fit_params_curve = [
      np.exp(a0), # undo log(e) transform
      a1, a2
    ]
    ## measure parameter uncertainty (std) from fit covariance matrix
    list_fit_params_std = np.sqrt(np.diag(fit_params_cov))
    ## fit numerical dissipation regime in log10-linear
    list_fit_params_tail, _ = curve_fit(
      f      = SpectraModels.tail_log10,
      xdata  = data_x_tail_log10,
      ydata  = data_y_tail_log10,
      bounds = ( (-np.inf, -10), (0, 0) ),
      maxfev = 10**3
    )
    ## generate full fitted spectra (model + numerical dissipation)
    fitted_power = np.array(
      ## model part of spectrum
      list(self.func_plot(
        data_k[self.k_start : k_break],
        *list_fit_params_curve
      )) + 
      ## dissipation part of the spectrum
      list(SpectraModels.tail_linear(
        data_k[k_break : k_end],
        *list_fit_params_tail
      ))
    )
    ## measure residuals (2-norm) in log10-space
    fit_2norm = np.sum((
      np.log10(data_power[self.k_start : k_end]) - np.log10(fitted_power)
    )**2)
    ## return fit information
    return list_fit_params_curve, list_fit_params_std, fit_2norm

  def __findBestFit(
      self,
      fit_2norm_group_k,
      list_fit_params_group_k
    ):
    list_good_fit_indices = []
    ## define cut-off 2-norm error
    max_fit_2norm = 0.5 * ( max(fit_2norm_group_k) + min(fit_2norm_group_k) )
    ## define cut-off dissipation scale
    max_k_scale = 1 / len(self.list_k_group_t[0])
    ## include the first fit as a good fits if it meets a less strict criteria
    if fit_2norm_group_k[0] < np.percentile(fit_2norm_group_k, 84):
      list_good_fit_indices.append(0)
    ## find local minima (good fits) in the fit error (2-norm) vs. trialed max k-mode domain
    list_good_fit_indices.extend(list(argrelextrema(np.array(fit_2norm_group_k), np.less)[0]))
    num_good_fits = len(list_good_fit_indices)
    ## if there are a few good fits
    if num_good_fits > 1:
      ## subset good fits with additional criteria
      list_good_fit_indices_reduced = [
        minima_index
        for minima_index in list_good_fit_indices
        if  (fit_2norm_group_k[minima_index] < max_fit_2norm) and
            (list_fit_params_group_k[minima_index][2] > max_k_scale)
      ]
      ## if there are any good fits with a little amount of 2-norm error
      ## then use the fit that used the largest portion of data
      if len(list_good_fit_indices_reduced) > 0: best_fit_index = list_good_fit_indices_reduced[-1]
      ## if there are no fits with little error, then choose the last good fit
      else: best_fit_index = list_good_fit_indices[-1]
    ## if there is only one good fit, then use it
    elif num_good_fits == 1: best_fit_index = list_good_fit_indices[0]
    ## if there are no good fits, then use the fit with the smallest error
    else: best_fit_index = WWLists.getIndexListMin(fit_2norm_group_k)
    ## return best fit index
    return best_fit_index

  def __saveBestFit(
      self,
      data_k, data_power,
      list_fit_params
    ):
    ## extract best fitted parameters
    _, a1_b, a2_b  = list_fit_params
    powerlaw_slope = a1_b     # power-law exponent
    k_scale        = 1 / a2_b # dissipation scale
    ## measure peak of spectra if a method has been implemented
    self.auxGetPeakScales(
      a1         = a1_b,
      a2         = a2_b,
      k_p_guess  = powerlaw_slope * k_scale,
      data_power = data_power
    )
    ## store measured dissipation scale
    self.k_scale_group_t.append(k_scale)
    ## store fitted spectra
    list_fit_k     = list(np.linspace(data_k[0], data_k[-1], 10**3))
    list_fit_power = list(self.func_plot(list_fit_k, *list_fit_params))
    self.list_fit_k_group_t.append(list_fit_k)
    self.list_fit_power_group_t.append(list_fit_power)

  ## ############################
  ## EMPTY METHOD IMPLEMENTATIONS
  ## ############################
  def auxGetPeakScales(
      self,
      a1, a2, k_p_guess, data_power
    ):
    return

  ## ################
  ## ABSTRACT METHODS
  ## ################
  ## the following methods all need to be implemented by any child-classes
  @abc.abstractmethod
  def auxFitSpectra(
      self,
      data_k, data_power, list_weights
    ):
    pass

  @abc.abstractmethod
  def auxGetFitParams(self):
    pass


## ###############################################################
## FIT KINETIC ENERGY SPECTRA
## ###############################################################
class FitKinSpectra(FitSpectra):
  def __init__(
      self,
      list_sim_times,
      list_k_group_t,
      list_power_group_t,
      bool_fit_fixed_model = False,
      k_start              = 1,
      k_break_from         = 5,
      k_step_size          = 1,
      k_end                = None,
      bool_fit_sub_y_range = False,
      num_decades_to_fit   = 6,
      bool_hide_updates    = False
    ):
    ## store input data
    self.list_sim_times       = list_sim_times
    self.list_k_group_t       = list_k_group_t
    self.list_power_group_t   = list_power_group_t
    ## store fit parameters
    self.fit_bounds           = (
      ( np.log(10**(-10)), -5.0, 1/30   ),
      ( np.log(10**(2)),    5.0, 1/0.01 )
    )
    self.bool_fit_fixed_model = bool_fit_fixed_model
    self.func_fit             = SpectraModels.kinetic_loge
    self.func_plot            = SpectraModels.kinetic_linear
    self.k_start              = k_start
    self.k_break_from         = k_break_from
    self.k_step_size          = k_step_size
    self.k_end                = k_end
    self.bool_fit_sub_y_range = bool_fit_sub_y_range
    self.num_decades_to_fit   = num_decades_to_fit
    ## call parent class and pass fitting instructions
    FitSpectra.__init__(self, bool_hide_updates)

  def auxFitSpectra(
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

  def auxGetFitParams(self, list_fit_params_curve_loge):
    if self.bool_fit_fixed_model:
      # a1 = -5/3 # kolmogorov exponent
      a1 = -2 # burgulence exponent
      a0, a2 = list_fit_params_curve_loge
    else: a0, a1, a2 = list_fit_params_curve_loge
    return a0, a1, a2

  def getFitDict(self):
    ## save fit output
    return {
      ## times fitted to
      "kin_list_sim_times":            self.list_sim_times,
      ## spectra data fitted to
      "kin_list_k_group_t":            self.list_k_group_t,
      "kin_list_power_group_t":        self.list_power_group_t,
      ## fitted spectra
      "kin_list_fit_k_group_t":        self.list_fit_k_group_t,
      "kin_list_fit_power_group_t":    self.list_fit_power_group_t,
      ## fitted scales
      "k_nu_group_t":                  self.k_scale_group_t,
      ## fit information
      "kin_bool_fit_fixed_model":      self.bool_fit_fixed_model,
      "kin_list_fit_params_group_t":   self.list_fit_params_group_t,
      "kin_list_fit_std_group_t":      self.list_fit_std_group_t,
      "kin_fit_k_start":               self.k_start,
      "kin_max_k_mode_fitted_group_t": self.max_k_mode_fitted_group_t,
      "kin_list_fit_k_range_group_t":  self.list_fit_k_range_group_t,
      "kin_list_fit_2norm_group_t":    self.list_fit_2norm_group_t
    }


## ###############################################################
## FIT MAGNETIC ENERGY SPECTRA
## ###############################################################
class FitMagSpectra(FitSpectra):
  def __init__(
      self,
      list_sim_times,
      list_k_group_t,
      list_power_group_t,
      bool_fit_fixed_model = False,
      k_start              = 1,
      k_break_from         = 5,
      k_step_size          = 1,
      k_end                = None,
      bool_hide_updates    = False
    ):
    ## store input data
    self.list_sim_times       = list_sim_times
    self.list_k_group_t       = list_k_group_t
    self.list_power_group_t   = list_power_group_t
    ## store fit parameters
    self.fit_bounds           = (
      ( np.log(10**(-10)), -3.0, 1/100  ),
      ( np.log(10**(2)),    3.0, 1/0.01 )
    )
    self.bool_fit_fixed_model = bool_fit_fixed_model
    self.func_fit_simple      = SpectraModels.magnetic_simple_loge
    self.func_fit             = SpectraModels.magnetic_loge
    self.func_plot            = SpectraModels.magnetic_linear
    self.k_start              = k_start
    self.k_break_from         = k_break_from
    self.k_step_size          = k_step_size
    self.k_end                = k_end
    ## initialise (other) measured scales
    self.k_p_group_t          = [] # fitted peak scale
    self.k_max_group_t        = [] # measured (raw) peak scale
    ## call parent class and pass fitting instructions
    FitSpectra.__init__(self, bool_hide_updates)

  def auxFitSpectra(
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

  def auxGetFitParams(self, list_fit_params_curve_loge):
    if self.bool_fit_fixed_model:
      a1 = 3/2 # kazantsev exponent
      a0, a2 = list_fit_params_curve_loge
    else: a0, a1, a2 = list_fit_params_curve_loge
    return a0, a1, a2

  def auxGetPeakScales(
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
      "mag_list_sim_times":            self.list_sim_times,
      ## spectra data fitted to
      "mag_list_k_group_t":            self.list_k_group_t,
      "mag_list_power_group_t":        self.list_power_group_t,
      ## fitted spectra
      "mag_list_fit_k_group_t":        self.list_fit_k_group_t,
      "mag_list_fit_power_group_t":    self.list_fit_power_group_t,
      ## fitted scales
      "k_eta_group_t":                 self.k_scale_group_t,
      "k_p_group_t":                   self.k_p_group_t,
      "k_max_group_t":                 self.k_max_group_t,
      ## fit information
      "mag_bool_fit_fixed_model":      self.bool_fit_fixed_model,
      "mag_list_fit_params_group_t":   self.list_fit_params_group_t,
      "mag_list_fit_std_group_t":      self.list_fit_std_group_t,
      "mag_fit_k_start":               self.k_start,
      "mag_max_k_mode_fitted_group_t": self.max_k_mode_fitted_group_t,
      "mag_list_fit_k_range_group_t":  self.list_fit_k_range_group_t,
      "mag_list_fit_2norm_group_t":    self.list_fit_2norm_group_t
    }


## END OF MODULE