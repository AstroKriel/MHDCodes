## START OF MODULE


## ###############################################################
## DEPENDENCIES: REQUIRED MODULES
## ###############################################################
import functools
import numpy as np

from scipy.signal import argrelextrema
from scipy.optimize import curve_fit, fsolve

from datetime import datetime

from TheFittingModule.UserModels import SpectraModels
from TheUsefulModule import WWLists


## ###############################################################
## CLASSES STORING DATA
## ###############################################################
class SpectraScales():
  def __init__(
      self,
      Pm,
      list_k_nu_converged, list_k_eta_converged, list_k_max_converged
    ):
    ## simulation setup information
    self.Pm = Pm
    ## converged scales
    self.list_k_nu_converged  = list_k_nu_converged
    self.list_k_eta_converged = list_k_eta_converged
    self.list_k_max_converged = list_k_max_converged

class SpectraFit():
  def __init__(
      self,
      ## ################
      ## SIMULATION SETUP
      ## ###########
        ## identifiers
        sim_suite, sim_label, Re, Rm, sim_res,
      ## ##########################
      ## VELOCITY SPECTRA VARIABLES
      ## ###########
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
      ## ##########################
      ## MAGNETIC SPECTRA VARIABLES
      ## ###########
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
    ):
    ## stamp when the spectra file was made
    self.date_analysed = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ## simulation information
    self.kin_sim_times = kin_sim_times
    self.mag_sim_times = mag_sim_times
    self.sim_suite = sim_suite
    self.sim_label = sim_label
    self.sim_res = sim_res
    self.Re = Re
    self.Rm = Rm
    self.Pm = Rm / Re
    self.bool_kin_fixed_model = bool_kin_fixed_model
    self.bool_mag_fixed_model = bool_mag_fixed_model
    ## simulation data
    self.kin_list_k_group_t = kin_list_k_group_t
    self.mag_list_k_group_t = mag_list_k_group_t
    self.kin_list_power_group_t = kin_list_power_group_t
    self.mag_list_power_group_t = mag_list_power_group_t
    ## fitting time range
    self.kin_fit_start_t = kin_fit_start_t
    self.mag_fit_start_t = mag_fit_start_t
    self.kin_fit_end_t   = kin_fit_end_t
    self.mag_fit_end_t   = mag_fit_end_t
    ## fitted spectras
    self.kin_list_fit_k_group_t = kin_list_fit_k_group_t
    self.mag_list_fit_k_group_t = mag_list_fit_k_group_t
    self.kin_list_fit_power_group_t = kin_list_fit_power_group_t
    self.mag_list_fit_power_group_t = mag_list_fit_power_group_t
    ## measured scales
    self.k_nu_group_t  = k_nu_group_t
    self.k_eta_group_t = k_eta_group_t
    self.k_max_group_t = k_max_group_t
    ## fitted spectra parameters
    self.kin_list_fit_params_group_t = kin_list_fit_params_group_t
    self.mag_list_fit_params_group_t = mag_list_fit_params_group_t
    ## uncertainties in parameter fits
    self.kin_list_fit_std_group_t = kin_list_fit_std_group_t
    self.mag_list_fit_std_group_t = mag_list_fit_std_group_t
    ## break point (k index) of best fits
    self.kin_fit_k_index_group_t = kin_fit_k_index_group_t
    self.mag_fit_k_index_group_t = mag_fit_k_index_group_t
    ## fit quality information
    self.kin_list_fit_k_range_group_t = kin_list_fit_k_range_group_t
    self.mag_list_fit_k_range_group_t = mag_list_fit_k_range_group_t
    self.kin_list_fit_2norm_group_t = kin_list_fit_2norm_group_t
    self.mag_list_fit_2norm_group_t = mag_list_fit_2norm_group_t


## ###############################################################
## FITTING SPECTRA FUNCTIONS
## ###############################################################
class FitSpectra():
  def __init__(
      self,
      list_k_group_t, list_power_group_t, list_sim_times,
      func_fit_simple, func_fit, func_plot, fit_bounds,
      bool_fit_fixed_model, bool_fit_magnetic_spectra,
      log_Ek_range = 6,
      bool_fit_sub_Ek_range = False,
      bool_hide_updates    = False
    ):
    ## --------------------
    ## SAVE INPUT VARIABLES
    ## --------------------
    ## spectra data (only fit to time realisations where turbulence has developed)
    list_subset_indices = [
      time_index for time_index, time_val in enumerate(list_sim_times)
      if time_val > 2
    ]
    self.list_k_group_t     = WWLists.subsetListByIndices(list_k_group_t,     list_subset_indices)
    self.list_power_group_t = WWLists.subsetListByIndices(list_power_group_t, list_subset_indices)
    self.list_sim_times     = WWLists.subsetListByIndices(list_sim_times,   list_subset_indices)
    ## fitting parameters
    self.bool_fit_magnetic_spectra = bool_fit_magnetic_spectra
    self.bool_fit_fixed_model      = bool_fit_fixed_model
    self.bool_fit_sub_Ek_range     = bool_fit_sub_Ek_range
    self.log_Ek_range              = log_Ek_range
    self.func_fit_simple           = func_fit_simple
    self.func_fit                  = func_fit
    self.func_plot                 = func_plot
    self.fit_bounds                = fit_bounds
    ## ----------------
    ## INITIALISE LISTS
    ## ----------------
    ## fitted spectra
    self.list_fit_k_group_t = []
    self.list_fit_power_group_t = []
    ## fitted scales
    self.k_scale_group_t = []
    self.k_max_group_t   = []
    ## fit information (for each time realisation)
    self.list_best_fit_params_group_t = [] # best fit parameters
    self.list_best_fit_std_group_t    = [] # list of fit param std
    self.best_fit_k_index_group_t     = [] # break point (k index) of best fit
    self.list_fit_k_range_group_t     = [] # number of points fitted to
    self.list_fit_2norm_group_t       = [] # objective func evaluated for all possible k break points
    ## -----------
    ## FIT SPECTRA
    ## -----------
    ## for each time slice
    for _, time_index in WWLists.loopListWithUpdates(self.list_sim_times, bool_hide_updates):
      ## fit spectra and store information
      self.fitToSpectra(time_index)
  def fitToSpectra(
      self,
      time_index,
      start_index = 2, # TODO: k=1 for magnetic and k=3 for kinetic energy spectra fits
      step_index  = 1, # change index to mode
      end_index   = None
    ):
    ## -----------------------------
    ## INITIALISE FITTING PARAMETERS
    ## -----------------------------
    list_fit_params_group_k = []
    list_params_std_group_k = []
    fit_2norm_group_k       = []
    ## load data to fit to
    data_k     = self.list_k_group_t[time_index]
    data_power = self.list_power_group_t[time_index]
    ## check that an end k-mode has been defined
    if end_index is None:
      if self.bool_fit_sub_Ek_range:
        ## find the k mode where E(k) is the closest to 10^(-6)
        end_index = WWLists.getIndexClosestValue(
          np.log10(np.array(data_power)),
          -self.log_Ek_range
        )
      else: end_index = len(data_k[:-1]) - 2
    ## ---------------------
    ## FIT TO SUBSET OF DATA
    ## ---------------------
    ## save the range of k explored when fitting at t/T
    list_fit_k_range = range(start_index, end_index, step_index)
    for break_index in list_fit_k_range:
      ## subset spectra curve
      x_data_curve_linear = np.array(data_k[:break_index])
      y_data_curve_loge   = np.log(data_power[:break_index])
      ## subset spectra tail
      x_data_tail_log10 = np.log10(data_k[break_index:])
      y_data_tail_log10 = np.log10(data_power[break_index:])
      ## calculate weighting of points
      list_data_weight = [ x**(0.5) for x in x_data_curve_linear ]
      ## beat the Kulsrud and Anderson 1992 model into fitting the magnetic spectra
      if self.bool_fit_magnetic_spectra:
        ## first, fit magnetic spectra with a simple model (spectra motivated)
        list_fit_params_guess, _ = curve_fit(
          self.func_fit_simple,
          x_data_curve_linear, y_data_curve_loge,
          bounds = self.fit_bounds,
          sigma  = list_data_weight,
          maxfev = 10**3
        )
        ## fit with the Kulsrud and Anderson 1992 model + pass guess from the simple model
        list_fit_params_curve_loge, mat_fit_params_cov = curve_fit(
          self.func_fit,
          x_data_curve_linear, y_data_curve_loge,
          bounds = self.fit_bounds,
          p0     = list_fit_params_guess,
          sigma  = list_data_weight,
          maxfev = 10**4
        )
      ## fit kinetic energy spectra with fewer fit instructions
      else:
        ## fit kinetic energy spectra with simple model
        list_fit_params_curve_loge, mat_fit_params_cov = curve_fit(
          self.func_fit,
          x_data_curve_linear, y_data_curve_loge,
          bounds = self.fit_bounds,
          sigma  = list_data_weight,
          maxfev = 10**3
        )
      ## fit spectra tail component in log10-linear
      list_fit_params_tail, _ = curve_fit(
        SpectraModels.tail_log10,
        x_data_tail_log10, y_data_tail_log10,
        bounds = ( (-np.inf, -10), (0, 0) ),
        maxfev = 10**3
      )
      ## undo log(e) transformation of fitted parameters
      if self.bool_fit_fixed_model:
        a0, a2 = list_fit_params_curve_loge # extract fitted spectra model parameters
        ## get fixed power-law exponent value
        if self.bool_fit_magnetic_spectra:
          a1 = 3/2 # in magnetic spectra model
        else: a1 = -5/3 # in kinetic spectra model
      else: a0, a1, a2 = list_fit_params_curve_loge # extract complete spectra model parameters
      ## save fitted (linear) spectra model parameter values
      list_fit_params_curve = [
        np.exp(a0), # undo log(e) transform
        a1, a2
      ]
      ## calculate fitted spectra to all points
      fitted_power = np.array(
        ## spectra curve
        list(self.func_plot(
          data_k[:break_index],
          *list_fit_params_curve
        )) + 
        ## spectra tail
        list(SpectraModels.tail_linear(
          data_k[break_index:],
          *list_fit_params_tail
        ))
      )
      ## measure residuals
      fit_2norm = np.sum(( np.log10(data_power[:]) - np.log10(fitted_power) )**2) # 2-norm in log10-space
      list_params_std = np.sqrt(np.diag(mat_fit_params_cov)) # std in parameters from covariance matrix
      ## append fit information
      list_fit_params_group_k.append(list_fit_params_curve)
      list_params_std_group_k.append(list_params_std)
      fit_2norm_group_k.append(fit_2norm)
    ## save fit information
    self.list_fit_2norm_group_t.append(fit_2norm_group_k)
    ## -----------------
    ## FIND THE BEST FIT
    ## -----------------
    ## define error cut-off
    max_fit_2norm = 0.5 * ( max(fit_2norm_group_k) + min(fit_2norm_group_k) )
    ## find good fits (minima in the fit error plot)
    if fit_2norm_group_k[0] < np.percentile(fit_2norm_group_k, 84):
      list_minima_index = [ 0 ] # include the first fit in list of good fits
    else: list_minima_index = []
    ## find local minima (good fits) in the list of fit errors
    list_minima_index.extend(list(
      argrelextrema(
        np.array(fit_2norm_group_k),
        np.less
      )[0]
    ))
    ## if there are many good fits
    if len(list_minima_index) > 1:
      ## find good fits with little error
      list_good_minima = [
        minima_index
        for minima_index in list_minima_index
        if (
          fit_2norm_group_k[minima_index] < max_fit_2norm # small error
          and
          1 / list_fit_params_group_k[minima_index][2] < 20 # dissipation scale is reasonable
        )
      ]
      ## if there are many reasonable fits
      if len(list_good_minima) > 0:
        ## use the fit that fitted to the most data
        index_best_fit = list_good_minima[-1]
      ## if there are no reasonable fits, then choose the final reasonable fit
      else: index_best_fit = list_minima_index[-1]
    ## if there is only one reasonable fit, then use it
    elif len(list_minima_index) == 1:
      index_best_fit = list_minima_index[0]
    ## otherwise, if there are no good fits, then use the fit with the smallest error
    else: index_best_fit = WWLists.getIndexListMin(fit_2norm_group_k)
    ## save best fit information
    list_best_fit_params = list_fit_params_group_k[index_best_fit]
    list_best_fit_std = list_params_std_group_k[index_best_fit]
    best_fit_k_index = list_fit_k_range[index_best_fit]
    self.list_fit_k_range_group_t.append(list_fit_k_range)
    self.best_fit_k_index_group_t.append(best_fit_k_index)
    self.list_best_fit_params_group_t.append(list_best_fit_params)
    self.list_best_fit_std_group_t.append(list_best_fit_std)
    ## save fitted scales
    a0_b, a1_b, a2_b = list_best_fit_params # get best fit parameters
    powerlaw_exp = a1_b
    k_scale  = 1 / a2_b
    if self.bool_fit_magnetic_spectra:
      ## measure k_max for Kulsrud and Anderson 1992 model
      try:
        k_max = fsolve(
          functools.partial(
            SpectraModels.k_max_implicit,
            a1 = a1_b,
            a2 = a2_b
          ),
          x0 = powerlaw_exp * k_scale # give a guess
        )[0]
      except (RuntimeError, ValueError): k_max = powerlaw_exp * k_scale
      ## save k_max
      self.k_max_group_t.append(k_max)
    self.k_scale_group_t.append(k_scale)
    ## -------------------
    ## SAVE FITTED SPECTRA
    ## -------------------
    list_fit_k = list(np.linspace(1, self.list_k_group_t[time_index][-1], 10**3))
    list_fit_power = list(self.func_plot(list_fit_k, *list_best_fit_params))
    self.list_fit_k_group_t.append(list_fit_k)
    self.list_fit_power_group_t.append(list_fit_power)

class FitVelSpectra(FitSpectra):
  def __init__(
      self,
      list_k_group_t, list_power_group_t, list_sim_times,
      bool_fit_fixed_model = False,
      bool_fit_sub_Ek_range = False,
      log_Ek_range = 6,
      bool_hide_updates = False
    ):
    ## plotting function: complete kinetic spectra model
    func_plot = SpectraModels.kinetic_linear
    ## fitting parameters
    self.bool_fit_fixed_model = bool_fit_fixed_model
    if bool_fit_fixed_model:
      ## bounds for fitting kinetic model with fixed power-law exponent
      fit_bounds = (
        ( np.log(10**(-10)), 1/30 ),
        ( np.log(10**(2)),   1/0.01 )
      )
      ## fit with fixed kinetic spectra model
      func_fit = SpectraModels.kinetic_loge_fixed
    else:
      ## bounds for fitting complete kinetic spectra model
      fit_bounds = (
        ( np.log(10**(-10)), -5.0, 1/30 ),
        ( np.log(10**(2)),    5.0, 1/0.01 )
      )
      ## fit with complete kinetic spectra model
      func_fit = SpectraModels.kinetic_loge
    ## call parent class and pass fitting instructions
    FitSpectra.__init__(
      self,
      ## pass input spectra information
      list_k_group_t, list_power_group_t, list_sim_times,
      ## pass fitting parameters
      func_fit_simple  = None,
      func_fit         = func_fit,
      func_plot        = func_plot,
      fit_bounds       = fit_bounds,
      bool_fit_magnetic_spectra = False,
      bool_fit_fixed_model  = bool_fit_fixed_model,
      bool_fit_sub_Ek_range = bool_fit_sub_Ek_range,
      log_Ek_range          = log_Ek_range,
      ## hide terminal output
      bool_hide_updates = bool_hide_updates
    )
  def getFitArgs(self):
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
      list_k_group_t, list_power_group_t, list_sim_times,
      bool_fit_fixed_model = False,
      bool_hide_updates = False
    ):
    ## plotting function: complete magnetic spectra model
    func_plot = SpectraModels.magnetic_linear
    ## fitting parameters
    self.bool_fit_fixed_model = bool_fit_fixed_model
    if bool_fit_fixed_model:
      ## fit with fixed magnetic spectra models
      func_fit_simple = SpectraModels.magnetic_simple_loge_fixed
      func_fit = SpectraModels.magnetic_loge_fixed
      ## bounds for fitting magnetic model with fixed power-law exponent
      fit_bounds = (
        ( np.log(10**(-10)), 1/100 ),
        ( np.log(10**(2)),   1/0.01 )
      )
    else:
      ## fit with complete magnetic spectra model
      func_fit_simple = SpectraModels.magnetic_simple_loge
      func_fit = SpectraModels.magnetic_loge
      ## bounds for fitting complete magnetic spectra model
      fit_bounds = (
        ( np.log(10**(-10)), -3.0, 1/100 ),
        ( np.log(10**(2)),    3.0, 1/0.01 )
      )
    ## call parent class and pass fitting instructions
    FitSpectra.__init__(
      self,
      ## pass input spectra information
      list_k_group_t, list_power_group_t, list_sim_times,
      ## pass fitting parameters
      func_fit_simple  = func_fit_simple,
      func_fit         = func_fit,
      func_plot        = func_plot,
      fit_bounds       = fit_bounds,
      bool_fit_fixed_model = bool_fit_fixed_model,
      bool_fit_magnetic_spectra = True,
      ## hide terminal output
      bool_hide_updates = bool_hide_updates
    )
  def getFitArgs(self):
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