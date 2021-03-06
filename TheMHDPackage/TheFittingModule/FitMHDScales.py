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
        kin_k_index_fit_from,
        kin_list_fit_k_index_range_group_t,
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
        mag_k_index_fit_from,
        mag_list_fit_k_index_range_group_t,
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
    self.sim_suite                           = sim_suite
    self.sim_label                           = sim_label
    self.sim_res                             = sim_res
    self.Re                                  = Re
    self.Rm                                  = Rm
    self.Pm                                  = Pm
    self.kin_bool_fit_fixed_model            = kin_bool_fit_fixed_model
    self.mag_bool_fit_fixed_model            = mag_bool_fit_fixed_model
    ## raw spectra data
    self.kin_list_sim_times                  = kin_list_sim_times
    self.mag_list_sim_times                  = mag_list_sim_times
    self.kin_list_k_group_t                  = kin_list_k_group_t
    self.mag_list_k_group_t                  = mag_list_k_group_t
    self.kin_list_power_group_t              = kin_list_power_group_t
    self.mag_list_power_group_t              = mag_list_power_group_t
    ## time range corresponding with the domain of interest (kinematic regime)
    self.kin_fit_time_start                  = kin_fit_time_start
    self.mag_fit_time_start                  = mag_fit_time_start
    self.kin_fit_time_end                    = kin_fit_time_end
    self.mag_fit_time_end                    = mag_fit_time_end
    ## fitted spectra
    self.kin_list_fit_k_group_t              = kin_list_fit_k_group_t
    self.mag_list_fit_k_group_t              = mag_list_fit_k_group_t
    self.kin_list_fit_power_group_t          = kin_list_fit_power_group_t
    self.mag_list_fit_power_group_t          = mag_list_fit_power_group_t
    ## measured scales
    self.k_nu_group_t                        = k_nu_group_t
    self.k_eta_group_t                       = k_eta_group_t
    self.k_p_group_t                         = k_p_group_t
    self.k_max_group_t                       = k_max_group_t
    ## fitted spectra parameters + uncertainties
    self.kin_list_fit_params_group_t         = kin_list_fit_params_group_t
    self.mag_list_fit_params_group_t         = mag_list_fit_params_group_t
    self.kin_list_fit_std_group_t            = kin_list_fit_std_group_t
    self.mag_list_fit_std_group_t            = mag_list_fit_std_group_t
    ## max k-mode fitted (break point) for the best fit from each time realisation
    self.kin_k_index_fit_from                = kin_k_index_fit_from
    self.mag_k_index_fit_from                = mag_k_index_fit_from
    self.kin_max_k_mode_fitted_group_t       = kin_max_k_mode_fitted_group_t
    self.mag_max_k_mode_fitted_group_t       = mag_max_k_mode_fitted_group_t
    ## k-range the fitter tried fitting to
    self.kin_list_fit_k_index_range_group_t  = kin_list_fit_k_index_range_group_t
    self.mag_list_fit_k_index_range_group_t  = mag_list_fit_k_index_range_group_t
    ## fit information
    self.kin_list_fit_2norm_group_t          = kin_list_fit_2norm_group_t
    self.mag_list_fit_2norm_group_t          = mag_list_fit_2norm_group_t


## ###############################################################
## CLASS OF USEFUL SPECTRA MODELS
## ###############################################################
class SpectraModels():
  ## ######################
  ## KINETIC SPECTRA MODELS
  ## ######################
  def kinetic_linear(k, A, alpha, ell_nu):
    ''' exponential + powerlaw in linear-domain:
        y = A  * k^alpha * exp(- k / k_nu)
      where ell_nu = 1/k_nu.
    '''
    return A * np.array(k)**(alpha) * np.exp(-(ell_nu * np.array(k)))

  def kinetic_loge(k, A_loge, alpha, ell_nu):
    return A_loge + alpha * np.log(k) - (ell_nu * np.array(k))
    # return np.log(SpectraModels.kinetic_linear(k, np.exp(A_loge), alpha, ell_nu))

  ## #######################
  ## MAGNETIC SPECTRA MODELS
  ## #######################
  def magnetic_linear(k, A, alpha_1, alpha_2, ell_eta):
    ''' modified Kulsrud and Anderson 1992 in linear-domain:
        y = A * k^alpha_1 * K0( (k / k_eta)^alpha_2 )
      where ell_eta = 1/k_eta.
    '''
    return A * np.array(k)**(alpha_1) * k0( (ell_eta * np.array(k))**(alpha_2) )

  def magnetic_loge(k, A_loge, alpha_1, alpha_2, ell_eta):
    arg = (ell_eta * np.array(k))**(alpha_2)
    log_mod_bessel = np.where(
      arg > 10,
      ## approximate ln(K0(...)) with the first two terms from the series expansion of K0(...)
      -arg + np.log( np.sqrt(np.pi/2) * (
        np.sqrt(1/arg) - 1/8 * (1/arg)**(3/2)
      )),
      ## evaluate ln(K0(...))
      np.log(k0(arg))
    )
    return A_loge + alpha_1 * np.log(k) + log_mod_bessel
    # return np.log(SpectraModels.magnetic_linear(k, np.exp(A_loge), alpha, ell_eta))

  def k_p_implicit(k, alpha_1, alpha_2, ell_eta):
    ''' peak scale of the magnetic energy spectra model (modified Kulsrud and Anderson 1992).
      when y'= 0:
        k_p :=  k = alpha * [K0(k / k_eta) / K1(k / k_eta)] * k_eta
      where ell_eta = 1/k_eta.
    '''
    return np.array(k) - (
      alpha_1 / alpha_2 * (
        k0( (ell_eta * np.array(k))**(alpha_2) ) / k1( (ell_eta * np.array(k))**(alpha_2) )
    ))**(1/alpha_2) * 1/ell_eta
  
  def magnetic_simple_linear(k, A, alpha_1, alpha_2, ell_eta):
    ''' simple model: exponential + powerlaw in linear-domain:
        y = A * k^alpha * exp(- k / k_eta)
      where ell_eta = 1/k_eta.
    '''
    return A * np.array(k)**(alpha_1) * np.exp( -(ell_eta * np.array(k))**(alpha_2) )

  def magnetic_simple_loge(k, A_loge, alpha_1, alpha_2, ell_eta):
    return A_loge + alpha_1 * np.log(k) - (ell_eta * np.array(k))**(alpha_2)
    # return np.log(SpectraModels.magnetic_simple_linear(k, np.exp(A_loge), alpha, ell_eta))
  
  def k_p_simple(alpha_1, alpha_2, ell_eta):
    return (alpha_1 / alpha_2)**(1/alpha_2) * 1/ell_eta

  ## ############################
  ## NUMERICAL DISSIPATION REGIME
  ## ############################
  def tail_linear(x, alpha_1, alpha_2):
    ''' powerlaw in linear-domain
      y = 10^alpha_1 * k^alpha_2
    '''
    return 10**(alpha_1) * np.array(x)**(alpha_2)

  def tail_log10(x_log10, alpha_1, alpha_2):
    ''' powerlaw in log10-domain
      log10(y) = alpha_1 + alpha_2 * log10(k)
    '''
    return alpha_1 + alpha_2 * np.array(x_log10)


## ###############################################################
## ROUTINE FOR FITTING SPECTRA
## ###############################################################
class FitSpectra(metaclass=abc.ABCMeta): # abstract base class
  ## default fit parameters
  bool_fit_sub_y_range = False
  num_decades_to_fit   = 6

  def __init__(self, bool_hide_updates=False):
    ## initialise fitted spectra
    self.list_fit_k_group_t             = []
    self.list_fit_power_group_t         = []
    self.k_scale_group_t                = [] #  fitted dissipation scale
    ## initialise fit information (for each fitted time realisation)
    self.list_fit_params_group_t        = [] # fitted parameters
    self.list_fit_std_group_t           = [] # uncertainty in paramater fits
    self.max_k_mode_fitted_group_t      = [] # max k-mode fitted (break point) for best fits
    self.list_fit_k_index_range_group_t = [] # range of k-modes fitted to
    self.list_fit_2norm_group_t         = [] # 2-norm evaluated for all possible k break points
    ## for each time realisation try fitting a range of k-modes
    for _, time_index in WWLists.loopListWithUpdates(self.list_sim_times, bool_hide_updates):
      ## extract the spectra data at a particular time point
      data_k     = self.list_k_group_t[time_index]
      data_power = self.list_power_group_t[time_index]
      ## fit spectra and store fit information + best parameters
      self.fitTimeRealisation(data_k, data_power)

  def fitTimeRealisation(
      self,
      data_k, data_power
    ):
    fit_2norm_group_k       = []
    list_fit_params_group_k = []
    list_params_std_group_k = []
    ## find the first k-mode such that a certain number of decades will be fitted
    k_index_break_Estart = int(WWLists.getIndexClosestValue(
      np.log10(np.array(data_power) / np.sum(data_power)),
      -3
    ))
    ## define a k-mode to stop fitting
    if self.bool_fit_sub_y_range:
      ## find the k-mode where the energy spectra is closest to the cut-off energy
      k_index_break_end = int(WWLists.getIndexClosestValue(
        np.log10(np.array(data_power)),
        -(self.num_decades_to_fit)
      ))
    ## nearly fit up to the final k-mode
    else: k_index_break_end = len(data_k) - 5
    ## create the range of k-modes to break at (i.e., switch from fitting curve to tail)
    list_fit_k_index_range  = list(range(
      max(
        self.k_index_break_from,
        k_index_break_Estart
      ),                       # start index
      k_index_break_end,       # end index
      self.k_index_break_step  # index increment
    ))
    # ## check that there are enough k-modes to fit to beyond the peak-scale
    # k_max_index_plus_decade = int(10**(np.log10(np.argmax(data_power)+1) + 0.5))
    ## fit to an increasing subset of the data
    for k_index_break_fit in list_fit_k_index_range:
      # if (k_index_break_fit < k_max_index_plus_decade) and (len(self.list_k_group_t[0]) > (100 // 2)):
      #   list_fit_params_group_k.append([ ])
      #   list_params_std_group_k.append([ ])
      #   fit_2norm_group_k.append(np.nan)
      #   continue
      ## fit model to a subset of the data (acts as a guess if a simpler model is fitted to the data in this step)
      list_fit_params, list_fit_params_std, fit_2norm = self.__fitSpectra(
        data_k            = data_k,
        data_power        = data_power,
        k_index_break_fit = k_index_break_fit
      )
      ## store fit information
      list_fit_params_group_k.append(list_fit_params)     # fitted parameters
      list_params_std_group_k.append(list_fit_params_std) # uncertainty in fit parameters
      fit_2norm_group_k.append(fit_2norm)                 # 2-norm of fit
    ## store 2-norm of all attempted fits
    self.list_fit_2norm_group_t.append(fit_2norm_group_k)
    ## store list of k-indices explored
    self.list_fit_k_index_range_group_t.append(list_fit_k_index_range)
    ## find the best guess-fit
    bf_index = self.__findBestFit(fit_2norm_group_k, list_fit_params_group_k)
    bf_k_index_break_fit = list_fit_k_index_range[bf_index]
    ## double check the fitted parameters are good
    bf_list_fit_params, bf_list_fit_std, _ = self.__fitSpectra(
      data_k            = data_k,
      data_power        = data_power,
      k_index_break_fit = bf_k_index_break_fit,
      list_guess_params = list_fit_params_group_k[bf_index] # best guess parameters
    )
    ## store the best fit information
    self.max_k_mode_fitted_group_t.append(data_k[bf_k_index_break_fit])
    self.list_fit_params_group_t.append(bf_list_fit_params)
    self.list_fit_std_group_t.append(bf_list_fit_std)
    ## store scales measured from the best fit
    self.auxSaveScales(bf_list_fit_params, data_power=data_power)
    ## store the best fitted spectra
    list_fit_k     = list(np.linspace(data_k[0], data_k[-1], 10**3))
    list_fit_power = list(self.func_plot(list_fit_k, *bf_list_fit_params))
    self.list_fit_k_group_t.append(list_fit_k)
    self.list_fit_power_group_t.append(list_fit_power)

  def __fitSpectra(
      self,
      data_k, data_power,
      k_index_break_fit,
      list_guess_params = None
    ):
    ## fit spectra model to the first part of the spectra
    data_x_curve_linear = np.array(data_k[   self.k_index_fit_from : k_index_break_fit ])
    data_y_curve_loge   = np.log(data_power[ self.k_index_fit_from : k_index_break_fit ])
    ## fit dissipation model to the remaining part of the spectra
    data_x_tail_log10   = np.log10(data_k[     k_index_break_fit : ])
    data_y_tail_log10   = np.log10(data_power[ k_index_break_fit : ])
    ## weight k-modes
    list_k_weight = [ x**(0.5) for x in data_x_curve_linear ]
    ## fit spectra model
    if list_guess_params is None:
      list_fit_params_curve_loge, fit_params_cov = self.auxFitSpectraGuess(
        data_k          = data_x_curve_linear,
        data_power_loge = data_y_curve_loge,
        list_weights    = list_k_weight
      )
    else:
      list_fit_params_curve_loge, fit_params_cov = self.auxFitSpectraFinal(
        data_k            = data_x_curve_linear,
        data_power_loge   = data_y_curve_loge,
        list_weights      = list_k_weight,
        list_guess_params = list_guess_params # pass guess
      )
    ## undo log(e) transformation of fitted model parameters
    list_fit_params_curve = [
      np.exp(list_fit_params_curve_loge[0]), # undo log(e) transform
      *list_fit_params_curve_loge[1:]
    ]
    ## measure parameter uncertainty (std) from fit covariance matrix
    list_fit_params_std = np.sqrt(np.diag(fit_params_cov))
    ## fit numerical dissipation regime in log10-linear
    list_fit_params_tail, _ = curve_fit(
      SpectraModels.tail_log10,
      xdata  = data_x_tail_log10,
      ydata  = data_y_tail_log10,
      bounds = ( (-np.inf, -10), (0, 0) ),
      maxfev = 10**3
    )
    ## generate full fitted spectra (model + numerical dissipation)
    fitted_power = np.array(
      ## model part of the spectrum
      list(self.func_plot(
        data_k[self.k_index_fit_from : k_index_break_fit],
        *list_fit_params_curve
      )) + 
      ## numerical dissipation part of the spectrum
      list(SpectraModels.tail_linear(
        data_k[k_index_break_fit : ],
        *list_fit_params_tail
      ))
    )
    ## measure residuals (2-norm) in log10-space
    fit_2norm = np.sum((
      np.log10(data_power[self.k_index_fit_from : ]) - np.log10(fitted_power)
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
    max_fit_2norm = 0.5 * ( WWLists.getListMax(fit_2norm_group_k) + WWLists.getListMin(fit_2norm_group_k) )
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
  
  ## ################
  ## ABSTRACT METHODS
  ## ################
  ## the following methods need to be implemented by every instance of this class
  @abc.abstractmethod
  def auxFitSpectraGuess(
      self,
      data_k, data_power_loge, list_weights
    ):
    pass

  ## ############################
  ## EMPTY METHOD EMPLIMENTATIONS
  ## ############################
  ## implement the following methods if necessary

  ## by default the final spectra fit routine is the same as the guess-implimentation
  def auxFitSpectraFinal(
      self,
      data_k, data_power_loge, list_weights, list_guess_params
    ):
    return self.auxFitSpectraGuess(data_k, data_power_loge, list_weights)

  def auxSaveScales(self, **kwargs):
    return


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
      k_index_fit_from     = 1,
      k_index_break_from   = 5,
      k_index_break_step   = 1,
      bool_fit_sub_y_range = False,
      num_decades_to_fit   = 6,
      bool_hide_updates    = False
    ):
    ## store input data
    self.list_sim_times       = list_sim_times
    self.list_k_group_t       = list_k_group_t
    self.list_power_group_t   = list_power_group_t
    ## store fit parameters
    self.bool_fit_fixed_model = bool_fit_fixed_model # TODO: implement this functionality
    self.func_plot            = SpectraModels.kinetic_linear
    self.k_index_fit_from     = k_index_fit_from
    self.k_index_break_from   = k_index_break_from
    self.k_index_break_step   = k_index_break_step
    self.bool_fit_sub_y_range = bool_fit_sub_y_range
    self.num_decades_to_fit   = num_decades_to_fit
    ## call parent class and pass fitting instructions
    FitSpectra.__init__(self, bool_hide_updates)

  def auxFitSpectraGuess(
      self,
      data_k, data_power_loge, list_weights
    ):
    list_fit_params_curve_loge, fit_params_cov = curve_fit(
      SpectraModels.kinetic_loge,
      xdata  = data_k,
      ydata  = data_power_loge,
      bounds = (
        # log(A), alpha, 1/k_nu
        ( -15,    -10.0,  1/100   ),
        (   2,    -0.1,   1/0.001 )
      ),
      sigma  = list_weights,
      maxfev = 10**4
    )
    ## return fit parameters
    return list_fit_params_curve_loge, fit_params_cov

  def auxSaveScales(self, list_fit_params, **kwargs):
    ## extract best fit parameters
    _, _, ell_nu = list_fit_params
    k_nu = 1 / ell_nu
    ## store measured dissipation scale
    self.k_scale_group_t.append(k_nu)

  def getFitDict(self):
    ## save fit output
    return {
      ## times fitted to
      "kin_list_sim_times":                 self.list_sim_times,
      ## spectra data fitted to
      "kin_list_k_group_t":                 self.list_k_group_t,
      "kin_list_power_group_t":             self.list_power_group_t,
      ## fitted spectra
      "kin_list_fit_k_group_t":             self.list_fit_k_group_t,
      "kin_list_fit_power_group_t":         self.list_fit_power_group_t,
      ## fitted scales
      "k_nu_group_t":                       self.k_scale_group_t,
      ## fit information
      "kin_bool_fit_fixed_model":           self.bool_fit_fixed_model,
      "kin_list_fit_params_group_t":        self.list_fit_params_group_t,
      "kin_list_fit_std_group_t":           self.list_fit_std_group_t,
      "kin_k_index_fit_from":               self.k_index_fit_from,
      "kin_max_k_mode_fitted_group_t":      self.max_k_mode_fitted_group_t,
      "kin_list_fit_k_index_range_group_t": self.list_fit_k_index_range_group_t,
      "kin_list_fit_2norm_group_t":         self.list_fit_2norm_group_t
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
      k_index_fit_from     = 1,
      k_index_break_from   = 5,
      k_index_break_step   = 1,
      bool_hide_updates    = False
    ):
    ## store input data
    self.list_sim_times       = list_sim_times
    self.list_k_group_t       = list_k_group_t
    self.list_power_group_t   = list_power_group_t
    ## store fit parameters
    self.bool_fit_fixed_model = bool_fit_fixed_model # TODO: implement this functionality
    self.func_plot            = SpectraModels.magnetic_linear
    self.log_bounds           = (
      # log(A), alpha_1,  alpha_2, 1/k_eta
      ( -15,    0.01,     0.01,    0.001 ),
      (   2,    10.0,     2.0,     100   )
    )
    self.k_index_fit_from     = k_index_fit_from
    self.k_index_break_from   = k_index_break_from
    self.k_index_break_step   = k_index_break_step
    ## initialise (other) measured scales
    self.k_p_group_t          = [] # fitted peak scale
    self.k_max_group_t        = [] # measured (raw) peak scale
    ## call parent class and pass fitting instructions
    FitSpectra.__init__(self, bool_hide_updates)

  def auxFitSpectraGuess(
      self,
      data_k, data_power_loge, list_weights
    ):
    ## beat the Kulsrud and Anderson 1992 model into fitting the magnetic spectra
    ## (Step 1) fit with a simple model (spectra motivated)
    list_fit_params_curve_loge, fit_params_cov = curve_fit(
      SpectraModels.magnetic_simple_loge,
      xdata  = data_k,
      ydata  = data_power_loge,
      bounds = self.log_bounds,
      sigma  = list_weights,
      maxfev = 10**4
    )
    ## return fit parameters
    return list_fit_params_curve_loge, fit_params_cov

  def auxFitSpectraFinal(
      self,
      data_k, data_power_loge, list_weights, list_guess_params
    ):
    ## log-transform guess paramaters
    list_guess_params_loge = [ np.log(list_guess_params[0]), *list_guess_params[1:] ]
    ## beat the Kulsrud and Anderson 1992 model into fitting the magnetic spectra
    ## (Step 2) fit the Kulsrud and Anderson 1992 model + pass the parameter guess from the simple model
    list_fit_params_curve_loge, fit_params_cov = curve_fit(
      SpectraModels.magnetic_loge,
      xdata  = data_k,
      ydata  = data_power_loge,
      bounds = self.log_bounds,
      p0     = list_guess_params_loge,
      sigma  = list_weights,
      maxfev = 10**5
    )
    ## return fit parameters
    return list_fit_params_curve_loge, fit_params_cov

  def auxSaveScales(self, list_fit_params, **kwargs):
    ## extract best fit parameters
    _, alpha_1, alpha_2, ell_eta = list_fit_params
    ## evaluate scales
    k_eta     = 1 / ell_eta
    k_p_guess = SpectraModels.k_p_simple(alpha_1, alpha_2, ell_eta)
    ## fit peak scale from the modified Kulsrud and Anderson 1992 model
    try:
      k_p = fsolve(
        functools.partial(
          SpectraModels.k_p_implicit,
          alpha_1 = alpha_1,
          alpha_2 = alpha_2,
          ell_eta = ell_eta
        ),
        x0 = k_p_guess # give a guess
      )[0]
    ## if the solver fails, then use the initial guess
    except (RuntimeError, ValueError):
      k_p = k_p_guess
    ## measured true peak scale (for reference)
    k_max = np.argmax(kwargs["data_power"]) + 1
    ## store measured scales
    self.k_scale_group_t.append(k_eta)
    self.k_p_group_t.append(k_p)
    self.k_max_group_t.append(k_max)

  def getFitDict(self):
    ## save fit output
    return {
      ## times fitted to
      "mag_list_sim_times":                 self.list_sim_times,
      ## spectra data fitted to
      "mag_list_k_group_t":                 self.list_k_group_t,
      "mag_list_power_group_t":             self.list_power_group_t,
      ## fitted spectra
      "mag_list_fit_k_group_t":             self.list_fit_k_group_t,
      "mag_list_fit_power_group_t":         self.list_fit_power_group_t,
      ## fitted scales
      "k_eta_group_t":                      self.k_scale_group_t,
      "k_p_group_t":                        self.k_p_group_t,
      "k_max_group_t":                      self.k_max_group_t,
      ## fit information
      "mag_bool_fit_fixed_model":           self.bool_fit_fixed_model,
      "mag_list_fit_params_group_t":        self.list_fit_params_group_t,
      "mag_list_fit_std_group_t":           self.list_fit_std_group_t,
      "mag_k_index_fit_from":               self.k_index_fit_from,
      "mag_max_k_mode_fitted_group_t":      self.max_k_mode_fitted_group_t,
      "mag_list_fit_k_index_range_group_t": self.list_fit_k_index_range_group_t,
      "mag_list_fit_2norm_group_t":         self.list_fit_2norm_group_t
    }


## END OF MODULE