## START OF MODULE


## ###############################################################
## MODULES
## ###############################################################
import numpy as np

from scipy import stats
from scipy.optimize import curve_fit

from TheUsefulModule import WWLists


## ###############################################################
## FITTING TO LISTS OF DISTRIBUTIONS
## ###############################################################
def resampleFrom1DKDE(
    input_distributions,
    num_resamp = 10**3
  ):
  ## initialise list of resampled points
  list_resampled = []
  ## if resampling from a list of distributions
  if isinstance(input_distributions[0], list):
    ## then for each distribution in the list of distributions
    for sub_list_points in input_distributions:
      ## resample from the distribution (KDE)
      resampled_points = WWLists.flattenList(
        stats.gaussian_kde(sub_list_points).resample(size=num_resamp).tolist()
      )
      ## append resampled points
      list_resampled.append(resampled_points) # returns a list of lists
  ## otherwise resample from a single distribution
  else:
    ## resample from the distribution (KDE)
    resampled_points = WWLists.flattenList(
      stats.gaussian_kde(input_distributions).resample(size=num_resamp).tolist()
    )
    ## append resampled points
    list_resampled.append(
      resampled_points
    )
  ## save resampled points
  return list_resampled

def fitToDistributions(
    input_x, input_y, func,
    p0     = None,
    bounds = None,
    errors = None,
    maxfev = 10**3
  ):
  ## check that the inputs are distributions
  bool_x_list = isinstance(input_x[0], (list, np.ndarray)) and len(list(input_x[0])) > 1
  bool_y_list = isinstance(input_y[0], (list, np.ndarray)) and len(list(input_y[0])) > 1
  ## check how many fits need to be performed
  if bool_x_list: num_fits = len(input_x[0])
  if bool_y_list: num_fits = len(input_y[0])
  ## check that the input distribution has at least one dimension
  if not bool_x_list and not bool_y_list:
    raise Exception("You need to input a list of distributions in at least one dimension.")
  ## if a 2D distribition has been provided, check that the number of point in each dimenesion is the same
  if bool_x_list and bool_y_list:
    if not len(list(input_x)) == len(list(input_y)):
      raise Exception("You provided a list of 2D distributions, but there weren't the same number of elements in each dimension. The number of components were: '{}' and '{}'.".format(len(list(input_x)), len(list(input_y))))
  ## #######################
  ## INTERPRET INPUT: X-DATA
  ## #########
  ## regig list of distributions into list of sample sets
  if bool_x_list:
    fit_x = [
      [ sub_list[sample_index] for sub_list in input_x ]
      for sample_index in range(num_fits)
    ]
  else: fit_x = input_x
  ## #######################
  ## INTERPRET INPUT: Y-DATA
  ## #########
  ## regig list of distributions into list of sample sets
  if bool_y_list:
    fit_y = [
      [ sub_list[sample_index] for sub_list in input_y ]
      for sample_index in range(num_fits)
    ]
  else: fit_y = input_y
  ## #######################
  ## INTERPRET INPUT: ERRORS
  ## #########
  ## regig list of distributions into list of sample sets
  if errors is not None:
    fit_errors = [
      [ sub_list[sample_index] for sub_list in errors ]
      for sample_index in range(num_fits)
    ]
  else: fit_errors = None
  ## #######################
  ## FITTING TO DISTRIUTIONS
  ## ##########
  ## initialise list of fitted parameters
  list_fit_params_group = []
  list_fit_errors_group = []
  ## repeatedly fit function to resampled points from KDE
  fit_params = None # error checking
  for fit_index in range(num_fits):
    ## try to fit to sampled points
    try: fit_params, fit_cov = curve_fit(
      func,
      np.array(fit_x[fit_index]) if bool_x_list else np.array(fit_x),
      np.array(fit_y[fit_index]) if bool_y_list else np.array(fit_y),
      p0     = p0,
      bounds = bounds if bounds is not None else (-np.inf, np.inf),
      sigma  = fit_errors[fit_index] if fit_errors is not None else None,
      maxfev = maxfev
    )
    except (RuntimeError, ValueError): continue
    ## append the fitted parameters
    list_fit_params_group.append(fit_params)
    list_fit_errors_group.append(np.sqrt(np.diag(fit_cov)))
  ## error checking
  if fit_params is None:
    raise Exception("Fitter wasn't able to fit successfully.")
  ## regig fit paramaters and associated errors
  ## distribution for each parameter: [ [a0_0, ..., a0_m], ... [an_0, ..., an_m] ] for n parameters and m fits
  list_fit_params_organised_group = [ [ 
      list_fit_params[param_index]
      for list_fit_params in list_fit_params_group
    ] for param_index in range(len(fit_params))
  ]
  list_fit_errors_organised_group = [ [ 
      fit_2norm_group_k[param_index]
      for fit_2norm_group_k in list_fit_errors_group
    ] for param_index in range(len(fit_params))
  ]
  ## return organised fit information
  return list_fit_params_organised_group, list_fit_errors_organised_group


## END OF MODULE