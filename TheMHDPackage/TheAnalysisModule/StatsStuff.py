## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np


## ###############################################################
## FUNCTIONS
## ###############################################################
def sampleGaussFromQuantiles(p1, p2, x1, x2,num_samples=10**3):
  ## calculate the inverse of the cumulative distribution function (CDF)
  cdf_inv_p1 = np.sqrt(2) * np.erfinv(2 * p1 - 1)
  cdf_inv_p2 = np.sqrt(2) * np.erfinv(2 * p2 - 1)
  ## calculate the mean and standard deviation of the normal distribution
  norm_mean = ((x1 * cdf_inv_p2) - (x2 * cdf_inv_p1)) / (cdf_inv_p2 - cdf_inv_p1)
  norm_std = (x2 - x1) / (cdf_inv_p2 - cdf_inv_p1)
  ## generate sampled points from the normal distribution
  samples = norm_mean + norm_std * np.random.randn(num_samples)
  return samples

def computePDF(data, num_bins, weights=None):
  min_value = np.min(data)
  max_value = np.max(data)
  ## generate bin edges
  bin_edges = np.linspace(min_value, max_value, num_bins+1)
  ## initialise an array to store the counts for each quantized bin
  bin_counts = np.zeros(num_bins, dtype=float)
  ## use binary search to determine the bin index for each element in the data
  bin_indices = np.searchsorted(bin_edges, data) - 1
  ## increment the corresponding bin count for each element
  if weights is None:
    np.add.at(bin_counts, bin_indices, 1)
  else: np.add.at(bin_counts, bin_indices, weights)
  ## compute the probability density function
  pdf = np.append(0, bin_counts / np.sum(bin_counts))
  ## return the bin edges and the computed pdf
  return bin_edges, pdf


## END OF LIBRARY