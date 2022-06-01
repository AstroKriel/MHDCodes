## START OF MODULE


## ###############################################################
## MODULES
## ###############################################################
from scipy.stats import norm


## ###############################################################
## WORKING WITH DISTRIBUTIONS
## ###############################################################
def sampleGaussFromQuantiles(
    p1, p2,
    x1, x2,
    num_samples = 10**3
  ):
  ''' From: Cook (2010)
    'Determining distribution parameters from quantiles'
  '''
  ## calculate the inverse of the CFD
  cdf_inv_p1 = norm.ppf(p1)
  cdf_inv_p2 = norm.ppf(p2)
  ## calculate mean of the normal distribution
  norm_mean = (
      (x1 * cdf_inv_p2) - (x2 * cdf_inv_p1)
    ) / (cdf_inv_p2 - cdf_inv_p1)
  ## calculate standard deviation of the normal distribution
  norm_std = (x2 - x1) / (cdf_inv_p2 - cdf_inv_p1)
  ## return sampled points
  return norm(loc=norm_mean, scale=norm_std).rvs(size=num_samples)


## END OF MODULE