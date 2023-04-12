## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np


## ###############################################################
## CLASS OF USEFUL FUNCTIONS
## ###############################################################
class ListOfModels():
  ## ---------
  ## CONSTANTS
  ## ---------
  def constant(x, a0):
    return a0

  ## ------
  ## LINEAR
  ## ------
  def linear(x, a0):
    """ linear function in linear-domain:
      y = a0 * x
    """
    return a0 * np.array(x)

  def linear_offset(x, a0, a1):
    """ linear function with offset in linear-domain:
      y = a0 * x + a1
    """
    return a0 * np.array(x) + a1

  ## ---------
  ## POWER LAW
  ## ---------
  def powerlaw_linear(x, a0, a1):
    """ power-law in linear-domain:
      y = A  * k^p
        = a0 * x^a1
    """
    return a0 * np.array(x)**a1

  def powerlaw_log10(x_log10, a0, a1):
    """ power-law in log10-domain:
      log10(y) = log10(A) + log10(k^p)
          =  log10(A) + p  * log10(k)
          =  a0       + a1 * log10(x)
    """
    return a0 + a1 * np.array(x_log10)

  ## -----------
  ## EXPONENTIAL
  ## -----------
  def exp_linear(x, a0, a1):
    """ exponential in linear-domain:
      y = a0 * exp(a1 * x)
    """
    return a0 * np.exp(a1 * np.array(x))

  def exp_loge(x, a0, a1):
    """ exponential in log(e)-domain:
      log(y)  = ln(A0) + A1 * x
              = a0     + a1 * x
    """
    return a0 + a1 * np.array(x)

  ## -------------
  ## DISTRIBUTIONS
  ## -------------
  def gaussian(x, a, mu, std):
    return a * np.exp( - (np.array(x) - mu)**2 / (2*std ** 2))

  def bimodal(x, a0, mu0, std0, a1, mu1, std1):
    return ListOfModels.gaussian(x, a0, mu0, std0) + ListOfModels.gaussian(x, a1, mu1, std1)

  ## ---------------
  ## LOGISTIC GROWTH
  ## ---------------
  def logistic_growth_increasing(x, a0, a1, a2):
    """ logistic model (increasing)
    """
    return a0 * (1 - np.exp( -(np.array(x) / a1)**a2 ))

  def logistic_growth_decreasing(x, a0, a1, a2):
    """ logistic model (decreasing)
    """
    return a0 / (1 - np.exp( -(np.array(x) / a1)**a2 ))


## END OF LIBRARY