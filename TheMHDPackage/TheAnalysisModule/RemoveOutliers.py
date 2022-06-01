## START OF MODULE


## ###############################################################
## MODULES
## ###############################################################
import numpy as np

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from TheUsefulModule import WWLists
from TheFittingModule.UserModels import ListOfModels


## ###############################################################
## CLEANING DATA
## ###############################################################
def getGaussianScales(
    list_scales,
    ax         = None,
    num_bins   = 10,
    num_points = 1000,
    bool_debug = False
  ):
  ## calculate density of data
  dens, bin_edges = np.histogram(list_scales, bins=num_bins, density=True)
  ## normalise density
  dens_norm = np.array(np.append(0, dens / dens.sum()))
  ## find peak indices
  peaks = find_peaks(
    dens_norm,
    height   = 0.05,
    distance = 4
  )
  ## find frequently measured scales
  peak_pos = bin_edges[peaks[0]] # list of the peaks positions
  height = peaks[1]["peak_heights"] # list of the peaks values
  ## plot measured distribution peaks
  if bool_debug:
    for pos in peak_pos:
      ax.axhline(y=pos, ls="--", lw=2, color="green")
  ## define bounds for distribution fits
  bounds_lower = [ 0.001, 0.1, 0.01 ]
  bounds_upper = [ 1, 15, 2 ]
  ## fit a bi-modal distribution
  if len(peak_pos) > 1:
    ## get index of peak at smallest scale
    index_peak = WWLists.getIndexListMin(peak_pos)
    ## fit bi-modal distribution
    fit_params, _ = curve_fit(
      ListOfModels.gaussian,
      bin_edges, dens_norm,
      bounds = [
        bounds_lower,
        bounds_upper
      ],
      p0 = [ height[index_peak], peak_pos[index_peak], 0.5 ],
      maxfev = 5*10**3
    )
  ## fit a gaussian distribution
  else:
    fit_params, _ = curve_fit(
      ListOfModels.gaussian,
      bin_edges, dens_norm,
      bounds = [
        bounds_lower,
        bounds_upper
      ],
      p0 = [ height[0], peak_pos[0], 0.5 ],
      maxfev = 5*10**3
    )
  ## extract good distribution + resample
  resampled_scales = np.random.normal(*fit_params[1:3], num_points)
  ## check fit was good
  if bool_debug:
    ## get fitted distribution
    fitted_ditribution = ListOfModels.gaussian(bin_edges, *fit_params)
    ## plot fitted distribution
    if bool_debug:
      ax.hist(
        fitted_ditribution,
        histtype = "step",
        bins  = bin_edges,
        color = "blue",
        fill  = True,
        alpha = 0.2
      )
    ## find indices corresponding with the fitted distribution
    list_main_indices = [
      bin_index
      for bin_index in range(len(bin_edges))
      if fitted_ditribution[bin_index] > 0
    ]
    list_main_indices.append(list_main_indices[-1]+1)
    ## subset bin edges
    main_edges = [
      bin_edges[bin_index]
      for bin_index in range(len(bin_edges))
      if bin_index in list_main_indices
    ]
    ## subset bin values
    main_data = [
      dens_norm[bin_index]
      for bin_index in range(len(bin_edges))
      if bin_index in list_main_indices
    ]
    ## plot subsetted data
    ax.hist(
      main_data,
      histtype = "step",
      bins  = main_edges,
      color = "red",
      fill  = True,
      alpha = 0.2
    )
  ## return resampled data: that's gaussian(!)
  return resampled_scales

def cleanMeasuredScales(
    list_scales,
    list_times    = [],
    bounds_scales = [0.01, 20]
  ):
  ## if list of time points was also provided, then subset them, also
  bool_subset_time = False
  if len(list_times) > 0: bool_subset_time = True
  ## get indices of obvious outliers (measured scales that are unrealistic)
  list_index_realistic = WWLists.getIndexListInBounds(
    list_scales,
    bounds = bounds_scales # [ # 
    #     min(getGaussianScales(list_scales)),
    #     max(getGaussianScales(list_scales))
    # ]
  )
  ## remove outliers
  if bool_subset_time:
    list_times_subset = WWLists.subsetListByIndices(list_times,  list_index_realistic)
  list_scales_subset = WWLists.subsetListByIndices(list_scales, list_index_realistic)
  ## calculate PDF of subsetted data
  dens, bin_edges = np.histogram(list_scales_subset, bins=10, density=True)
  dens_norm = np.append(0, dens / dens.sum()) # normalise density
  ## remove low density points if a lot of data is conatined in only a few bins
  while (max(dens_norm) > 0.345):
    ## get list of scales that are measured in high density
    list_frequent_scales = WWLists.subsetListByIndices(
      bin_edges,
      WWLists.flattenList([ 
        [ bin_index-1, bin_index ] # edges of the bin (lower and upper)
        for bin_index, bin_val in enumerate(dens_norm)
        if 0 < bin_val
      ]) # get PDF bin-indices associated with scales that are measured in high density
    )
    ## get data-indices associated with scales that are measured in high density
    list_index_frequent = WWLists.getIndexListInBounds(
      list_scales_subset,
      [ min(list_frequent_scales), max(list_frequent_scales) ]
    )
    ## remove scales that appear in low density
    if bool_subset_time:
      list_times_subset = WWLists.subsetListByIndices(list_times_subset, list_index_frequent)
    list_scales_subset = WWLists.subsetListByIndices(list_scales_subset, list_index_frequent)
    ## check that distribution is well represented with 10 bins
    dens, bin_edges = np.histogram(list_scales_subset, bins=10, density=True)
    dens_norm = np.append(0, dens / dens.sum()) # normalise density
  ## return subsetted dataset
  if bool_subset_time:
    ## if set of time points were also provided
    return list_times_subset, list_scales_subset
  ## if only scales were provided
  return list_scales_subset


## END OF MODULE