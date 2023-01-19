## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
from scipy import stats
from scipy.stats import norm

## load user defined modules
from TheUsefulModule import WWLists


## ###############################################################
## FUNCTIONS
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
    list_resampled.append(resampled_points)
  ## save resampled points
  return list_resampled

def sampleGaussFromQuantiles(
    p1, p2,
    x1, x2,
    num_samples = 10**3
  ):
  ## calculate the inverse of the CFD
  cdf_inv_p1 = stats.norm.ppf(p1)
  cdf_inv_p2 = stats.norm.ppf(p2)
  ## calculate mean of the normal distribution
  norm_mean = (
      (x1 * cdf_inv_p2) - (x2 * cdf_inv_p1)
    ) / (cdf_inv_p2 - cdf_inv_p1)
  ## calculate standard deviation of the normal distribution
  norm_std = (x2 - x1) / (cdf_inv_p2 - cdf_inv_p1)
  ## return sampled points
  return stats.norm(
    loc   = norm_mean,
    scale = norm_std
  ).rvs(size=num_samples)


## END OF LIBRARY