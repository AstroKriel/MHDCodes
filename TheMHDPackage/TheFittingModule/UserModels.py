## START OF MODULE


## ###############################################################
## DEPENDENCIES: REQUIRED MODULES
## ###############################################################
import numpy as np

from scipy.special import k0, k1


## ###############################################################
## CLASS OF USEFUL FUNCTIONS
## ###############################################################
class SpectraModels():
  ## ---------------
  ## KINETIC SPECTRA
  ## ---------------
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
  ## ----------------
  ## MAGNETIC SPECTRA
  ## ----------------
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
  ## ---------------------
  ## NUMERICAL DISSIPATION
  ## ---------------------
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
      y   = A  * k^p
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
      y = a0 * exp(-a1 * x)
    """
    return a0 * np.exp(-a1 * np.array(x))
  def exp_loge(x, a0, a1):
    """ exponential in log(e)-domain:
      log(y)  = ln(A0) - A1 * x
          = a0     - a1 * x
    """
    return a0 - a1 * np.array(x)
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
    return a0 * (1 - np.exp( -(np.array(x) / a1)**a2 ))
  def logistic_growth_decreasing(x, a0, a1, a2):
    return a0 / (1 - np.exp( -(np.array(x) / a1)**a2 ))


## END OF MODULE