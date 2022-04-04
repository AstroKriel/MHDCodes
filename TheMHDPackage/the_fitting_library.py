#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import functools

from tqdm.auto import tqdm
from datetime import datetime

from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from scipy.optimize import curve_fit, fsolve
from scipy.special import k0, k1

## user defined libraries
from the_useful_library import *
from the_loading_library import *

## for debugging
# from ipdb import set_trace as stop


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
                y' = 0 when k_max := k = p * k_scale = a1 / a2
        """
        return (a0_loge) + (a1) * np.log(x) - a2 * np.array(x)
    def magnetic_simple_loge_fixed(x, a0_loge, a2):
        """ Exponential + power-law in linear-domain
                y   = A  * k^(3/2) * exp(- (1 / k_scale) * k)
                    = a0 * x^(3/2) * exp(- a2 * x)
            Note:
                y' = 0 when k_max := k = (3/2) * k_scale = (3/2) / a2
        """
        return (a0_loge) + (3/2) * np.log(x) - a2 * np.array(x)
    def k_max_implicit(x, a1, a2):
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
        """ exponential with offset in linear-domain:
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
                vel_sim_times,
                vel_list_k_group_t,
                vel_list_power_group_t,
                bool_vel_fixed_model,
                ## fitted spectra
                vel_list_fit_k_group_t,
                vel_list_fit_power_group_t,
                ## measured scale
                k_nu_group_t,
                ## best fit information
                vel_list_fit_params_group_t,
                vel_list_fit_std_group_t,
                vel_fit_k_index_group_t,
                ## history of fitting data
                vel_list_fit_k_range_group_t,
                vel_list_fit_2norm_group_t,
                ## fit time range
                vel_fit_start_t,
                vel_fit_end_t,
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
        self.vel_sim_times = vel_sim_times
        self.mag_sim_times = mag_sim_times
        self.sim_suite = sim_suite
        self.sim_label = sim_label
        self.sim_res = sim_res
        self.Re = Re
        self.Rm = Rm
        self.Pm = Rm / Re
        self.bool_vel_fixed_model = bool_vel_fixed_model
        self.bool_mag_fixed_model = bool_mag_fixed_model
        ## simulation data
        self.vel_list_k_group_t = vel_list_k_group_t
        self.mag_list_k_group_t = mag_list_k_group_t
        self.vel_list_power_group_t = vel_list_power_group_t
        self.mag_list_power_group_t = mag_list_power_group_t
        ## fitting time range
        self.vel_fit_start_t = vel_fit_start_t
        self.mag_fit_start_t = mag_fit_start_t
        self.vel_fit_end_t   = vel_fit_end_t
        self.mag_fit_end_t   = mag_fit_end_t
        ## fitted spectras
        self.vel_list_fit_k_group_t = vel_list_fit_k_group_t
        self.mag_list_fit_k_group_t = mag_list_fit_k_group_t
        self.vel_list_fit_power_group_t = vel_list_fit_power_group_t
        self.mag_list_fit_power_group_t = mag_list_fit_power_group_t
        ## measured scales
        self.k_nu_group_t  = k_nu_group_t
        self.k_eta_group_t = k_eta_group_t
        self.k_max_group_t = k_max_group_t
        ## fitted spectra parameters
        self.vel_list_fit_params_group_t = vel_list_fit_params_group_t
        self.mag_list_fit_params_group_t = mag_list_fit_params_group_t
        ## uncertainties in parameter fits
        self.vel_list_fit_std_group_t = vel_list_fit_std_group_t
        self.mag_list_fit_std_group_t = mag_list_fit_std_group_t
        ## break point (k index) of best fits
        self.vel_fit_k_index_group_t = vel_fit_k_index_group_t
        self.mag_fit_k_index_group_t = mag_fit_k_index_group_t
        ## full fitting information for all time realisations
        self.vel_list_fit_k_range_group_t = vel_list_fit_k_range_group_t
        self.mag_list_fit_k_range_group_t = mag_list_fit_k_range_group_t
        self.vel_list_fit_2norm_group_t = vel_list_fit_2norm_group_t
        self.mag_list_fit_2norm_group_t = mag_list_fit_2norm_group_t


## ###############################################################
## CLEANING DATA
## ###############################################################
def cleanMeasuredScales(
        list_scales,
        list_times    = [],
        bounds_scales = [0.01, 20]
    ):
    ## if list of time points was also provided, then subset them, also
    bool_subset_time = False
    if len(list_times) > 0: bool_subset_time = True
    ## get indices of obvious outliers (measured scales that are unrealistic)
    list_index_realistic = getIndexListInBounds(
        list_scales,
        bounds = bounds_scales # [ # 
        #     min(getGaussianScales(list_scales)),
        #     max(getGaussianScales(list_scales))
        # ]
    )
    ## remove outliers
    if bool_subset_time:
        list_times_subset = subsetListByIndices(list_times,  list_index_realistic)
    list_scales_subset = subsetListByIndices(list_scales, list_index_realistic)
    ## calculate PDF of subsetted data
    dens, bin_edges = np.histogram(list_scales_subset, bins=10, density=True)
    dens_norm = np.append(0, dens / dens.sum()) # normalise density
    ## remove low density points if a lot of data is conatined in only a few bins
    while (max(dens_norm) > 0.345):
        ## get list of scales that are measured in high density
        list_frequent_scales = subsetListByIndices(
            bin_edges,
            flattenList([ 
                [ bin_index-1, bin_index ] # edges of the bin (lower and upper)
                for bin_index, bin_val in enumerate(dens_norm)
                if 0 < bin_val
            ]) # get PDF bin-indices associated with scales that are measured in high density
        )
        ## get data-indices associated with scales that are measured in high density
        list_index_frequent = getIndexListInBounds(
            list_scales_subset,
            [ min(list_frequent_scales), max(list_frequent_scales) ]
        )
        ## remove scales that appear in low density
        if bool_subset_time:
            list_times_subset = subsetListByIndices(list_times_subset, list_index_frequent)
        list_scales_subset = subsetListByIndices(list_scales_subset, list_index_frequent)
        ## check that distribution is well represented with 10 bins
        dens, bin_edges = np.histogram(list_scales_subset, bins=10, density=True)
        dens_norm = np.append(0, dens / dens.sum()) # normalise density
    ## return subsetted dataset
    if bool_subset_time:
        ## if set of time points were also provided
        return list_times_subset, list_scales_subset
    ## if only scales were provided
    return list_scales_subset

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
        index_peak = getIndexListMin(peak_pos)
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
            resampled_points = flattenList(
                stats.gaussian_kde(sub_list_points).resample(size=num_resamp).tolist()
            )
            ## append resampled points
            list_resampled.append(resampled_points) # returns a list of lists
    ## otherwise resample from a single distribution
    else:
        ## resample from the distribution (KDE)
        resampled_points = flattenList(
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
        self.list_k_group_t     = subsetListByIndices(list_k_group_t,     list_subset_indices)
        self.list_power_group_t = subsetListByIndices(list_power_group_t, list_subset_indices)
        self.list_sim_times     = subsetListByIndices(list_sim_times,   list_subset_indices)
        ## fitting parameters
        self.bool_fit_magnetic_spectra = bool_fit_magnetic_spectra
        self.bool_fit_fixed_model = bool_fit_fixed_model
        self.bool_fit_sub_Ek_range = bool_fit_sub_Ek_range
        self.log_Ek_range = log_Ek_range
        self.func_fit_simple = func_fit_simple
        self.func_fit   = func_fit
        self.func_plot  = func_plot
        self.fit_bounds = fit_bounds
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
        self.list_best_fit_params_group_t = [] # (best) fit parameters
        self.list_best_fit_std_group_t    = [] # list of fit param std
        self.best_fit_k_index_group_t     = [] # break point (k index) of best fits
        self.list_fit_k_range_group_t     = [] # list of number of points fitted to
        self.list_fit_2norm_group_t       = [] # list of fit errors (obj func) evaluated for all possible k break points, for all fits
        ## -----------
        ## FIT SPECTRA
        ## -----------
        ## for each time slice
        for _, time_index in loopListWithUpdates(self.list_sim_times, bool_hide_updates):
            ## fit spectra and store information
            self.fitToSpectraSubset(time_index)
    def fitToSpectraSubset(
            self,
            time_index,
            start_index = 2,
            step_index  = 1,
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
                end_index = getIndexClosestValue(
                    np.log10(np.array(data_power)),
                    -self.log_Ek_range
                )
            else: end_index = len(data_k[:-1]) - 2
        ## ---------------------
        ## FIT TO SUBSET OF DATA
        ## ---------------------
        ## save the range of k explored when fitting at t/T
        list_fit_k_range = range(start_index, end_index, step_index)
        for break_index in range(start_index, end_index, step_index):
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
            ## fit velocity spectra with fewer fit instructions
            else:
                ## fit velocity spectra with simple model
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
        else: index_best_fit = getIndexListMin(fit_2norm_group_k)
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
            "vel_sim_times":self.list_sim_times,
            ## spectra data fitted to
            "vel_list_k_group_t":self.list_k_group_t,
            "vel_list_power_group_t":self.list_power_group_t,
            ## fitted spectra
            "vel_list_fit_k_group_t":self.list_fit_k_group_t,
            "vel_list_fit_power_group_t":self.list_fit_power_group_t,
            ## fitted scales
            "k_nu_group_t":self.k_scale_group_t,
            ## fit information
            "bool_vel_fixed_model":self.bool_fit_fixed_model,
            "vel_list_fit_params_group_t":self.list_best_fit_params_group_t,
            "vel_list_fit_std_group_t":self.list_best_fit_std_group_t,
            "vel_fit_k_index_group_t":self.best_fit_k_index_group_t,
            "vel_list_fit_k_range_group_t":self.list_fit_k_range_group_t,
            "vel_list_fit_2norm_group_t":self.list_fit_2norm_group_t
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


## END OF LIBRARY