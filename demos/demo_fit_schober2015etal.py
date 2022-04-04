import os
import re
import functools
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from scipy.signal import argrelextrema

from scipy.special import k0, k1
from scipy.optimize import curve_fit, fsolve


## ###############################################################
## WORKING WITH FILES
## ###############################################################
def makeFilter(
        str_contains       = None,
        str_not_contains   = None,
        str_startswith     = None,
        str_endswith       = None,
        file_index_placing = None,
        file_start_index   = 0,
        file_end_index     = np.inf,
        str_split_by       = "_"
    ):
    """ makeFilter
    PURPOSE: Create a filter condition for files that look a particular way.
    """
    def meetsCondition(element):
        ## if str_contains specified, then look for condition
        if str_contains is not None: bool_contains = element.__contains__(str_contains)
        else: bool_contains = True # don't consider condition
        ## if str_not_contains specified, then look for condition
        if str_not_contains is not None: bool_not_contains = not(element.__contains__(str_not_contains))
        else: bool_not_contains = True # don't consider condition
        ## if str_startswith specified, then look for condition
        if str_startswith is not None: bool_startswith = element.startswith(str_startswith)
        else: bool_startswith = True # don't consider condition
        ## if str_endswith specified, then look for condition
        if str_endswith is not None: bool_endswith = element.endswith(str_endswith)
        else: bool_endswith = True # don't consider condition
        ## make sure that the file has the right structure
        if (
                bool_contains and 
                bool_not_contains and 
                bool_startswith and 
                bool_endswith and 
                ( file_index_placing is not None )
            ):
            ## make sure that simulation file is in time range
            ## check that the file name has at least the required number of "spacers"
            if len(element.split(str_split_by)) > abs(file_index_placing):
                bool_time_after  = (
                    int(element.split(str_split_by)[file_index_placing]) >= file_start_index
                )
                bool_time_before = (
                    int(element.split(str_split_by)[file_index_placing]) <= file_end_index
                )
                ## if the file meets the required conditions
                if (bool_time_after and bool_time_before): return True
        ## otherwise don"t look at the file
        else: return False
    return meetsCondition

def getFilesFromFolder(
        folder_directory, 
        str_contains       = None,
        str_startswith     = None,
        str_endswith       = None,
        str_not_contains   = None,
        file_index_placing = None,
        file_start_index   = 0,
        file_end_index     = np.inf
    ):
    ''' getFilesFromFolder
    PURPOSE: Return the names of files that meet the required conditions in the specified folder.
    '''
    myFilter = makeFilter(
        str_contains,
        str_not_contains,
        str_startswith,
        str_endswith,
        file_index_placing,
        file_start_index,
        file_end_index
    )
    return list(filter(myFilter, sorted(os.listdir(folder_directory))))

def createFilepath(folder_names):
    """ creatFilePath
    PURPOSE: Concatinate a list of folder names into a filepath string.
    """
    return re.sub( '/+', '/', "/".join([folder for folder in folder_names if folder != ""]) )

def loopListWithUpdates(list_elems, bool_hide_updates=False):
    lst_len = len(list_elems)
    return zip(
        list_elems,
        tqdm(
            range(lst_len),
            total   = lst_len - 1,
            disable = (lst_len < 3) or bool_hide_updates
        )
    )

## ###############################################################
## FUNCTIONS FOR LOADING DATA
## ###############################################################
def loadSpectra(filepath_data):
    data_file = open(filepath_data).readlines() # load in data
    data      = np.array([x.strip().split() for x in data_file[6:]]) # store all data: [row, col]
    try:
        data_x = np.array(list(map(float, data[:, 1]))) # variable: wave number (k)
        data_y = np.array(list(map(float, data[:, 15]))) # variable: power spectrum
        bool_failed_to_read = False
    except:
        bool_failed_to_read = True
        data_x = []
        data_y = []
    return data_x, data_y, bool_failed_to_read

def loadListSpectra(
        filepath_data,
        str_spectra_type  = "mag",
        plots_per_eddy    = 10,
        file_start_index  = 2,
        file_end_index    = np.inf,
        bool_hide_updates = False
    ):
    ## initialise list of spectra data
    k_group_times       = []
    power_group_times   = []
    list_sim_times      = []
    list_failed_to_load = []
    ## filter for spectra files
    spectra_filenames = getFilesFromFolder(
        filepath_data, 
        str_contains = "Turb_hdf5_plt_cnt_",
        str_endswith = "spect_" + str_spectra_type + "s.dat",
        file_index_placing = -3,
        file_start_index   = file_start_index,
        file_end_index     = file_end_index
    )
    ## loop over each of the spectra files
    for filename, _ in loopListWithUpdates(spectra_filenames, bool_hide_updates):
        ## load data
        spectra_k, spectra_power, bool_failed_to_read = loadSpectra(
            createFilepath([
                filepath_data,
                filename
            ])
        )
        ## check data was read correctly
        if bool_failed_to_read:
            list_failed_to_load.append(filename)
            continue
        ## append data
        k_group_times.append(spectra_k)
        power_group_times.append(spectra_power)
        list_sim_times.append( float(filename.split("_")[-3]) / plots_per_eddy )
    ## list those files that failed to load
    if len(list_failed_to_load) > 0:
        print("\tFailed to read in the following files:", "\n\t\t> ".join(
            [" "] + list_failed_to_load
        ))
    ## return spectra data
    return k_group_times, power_group_times, list_sim_times

## ###############################################################
## FUNCTIONS FOR MODELING SPECTRA
## ###############################################################
def spectra_kriel_linear(x, a0, a1, a2):
    """ exponential + power-law in linear-domain:
            y   = A  * k^p  * exp(- (1 / k_scale) * k)
                = a0 * x^a1 * exp(- a2 * x)
        note:
            (y' = 0) when (k_max := k = p * k_scale = a1 / a2)
    """
    return a0 * np.array(x)**a1 * np.exp(-a2 * np.array(x))

def spectra_kriel_loge(x, a0, a1, a2):
    """ exponential + power-law in log(e)-domain:
        ln(y)   = ln(A) + p  * ln(k) - (1 / k_scale) * k
                = a0    + a1 * ln(x) - a2 * x
    """
    return a0 + a1 * np.log(x) - a2 * np.array(x)

def spectra_schober_linear(x, a0, a1, a2):
    ''' Kulsrud and Anderson 1992
        y   = A  * k^p  * K0( (1 / k_scale) * k )
            = a0 * x^a1 * k0(a2 * x)
    '''
    return a0 * np.array(x)**a1 * k0(a2 * np.array(x))

def spectra_schober_loge(x, a0_loge, a1, a2):
    ''' Kulsrud and Anderson 1992
        ln(y)   = ln(A)   + p  * ln(k) + ln(K0( (1 / k_scale) * k ))
                = a0_loge + a1 * ln(x) + ln(k0(a2 * x))
    '''
    return a0_loge + a1 * np.log(x) + np.log(k0(a2 * np.array(x)))

def k_max_implicit(x, a1, a2):
    ''' implicit peak scale of the magnetic spectra model (Kulsrud and Anderson 1992)
        from: y'= 0
        ->  k_p = p  * k_eta * K0( (1 / k_scale) * k_p ) / K1( (1 / k_scale) * k_p )
            x   = a1 * 1/a2  * K0(a2 * x)                / K1(a2 * x)
        ->  0   = x - a1 * 1/a2 * K0(a2 * x) / K1(a2 * x)
    '''
    return np.array(x) - (a1 / a2) * k0(a2 * np.array(x)) / k1(a2 * np.array(x))

def spectra_tail_linear(x, a0, a1):
    ''' powerlaw (power spectra decay) in linear-domain
        y   = 10^a0 * k^a1
    '''
    return 10**a0 * np.array(x)**a1

def spectra_tail_log10(x_log10, a0, a1):
    ''' powerlaw (power spectra decay) in log10-domain
        log10(y) = a0 + a1 * log10(k)
    '''
    return a0 + a1 * x_log10

## ###############################################################
## FUNCTIONS FOR FITTING SPECTRA
## ###############################################################
class FitSpectra():
    def __init__(
            self,
            list_k_group_times, list_power_group_times, list_sim_times,
            bool_kriel_model,
            bool_hide_updates = False
        ):
        ## ####################
        ## SAVE INPUT VARIABLES
        ## #######
        ## spectra data
        self.list_k_group_times     = list_k_group_times
        self.list_power_group_times = list_power_group_times
        self.list_sim_times         = list_sim_times
        ## fitting parameters
        self.bool_kriel_model = bool_kriel_model
        if bool_kriel_model:
            self.func_fit  = spectra_kriel_loge # fit in log(e)-linear space
            self.func_plot = spectra_kriel_linear
        else:
            self.func_fit  = spectra_schober_loge # fit in log(e)-linear space
            self.func_plot = spectra_schober_linear
        ## ################
        ## INITIALISE LISTS
        ## #######
        ## fitted spectra
        self.list_fit_k_group_times     = []
        self.list_fit_power_group_times = []
        ## fitted scales
        self.k_scale_group_times = []
        self.k_max_group_times   = []
        ## fit information (for each time realisation)
        self.fit_params_group_times          = [] # (best) fit parameters
        self.fit_num_points_group_times      = [] # number of points fitted to
        self.list_fit_num_points_group_times = [] # list of number of points fitted to
        self.list_fit_errors_group_times     = [] # list of fit errors
        ## ###########
        ## FIT SPECTRA
        ## #######
        ## for each time slice
        for time_val, time_index in loopListWithUpdates(self.list_sim_times, bool_hide_updates):
            ## fit spectra and store information
            self.fitToSpectraSubset(time_index)
    def fitToSpectraSubset(
            self,
            time_index,
            start_num_points = 10,
            step_num_points  = 1,
            end_num_points   = None
        ):
        ## #############################
        ## INITIALISE FITTING PARAMETERS
        ## #############################
        list_fit_params     = []
        list_fit_num_points = []
        list_fit_errors     = []
        ## load data to fit to
        data_k     = self.list_k_group_times[time_index]
        data_power = self.list_power_group_times[time_index]
        ## check that a maximum sample size has been defined
        if end_num_points is None:
            end_num_points = len(data_k[:-1]) - 10
        ## #####################
        ## FIT TO SUBSET OF DATA
        ## #####################
        for break_point in range(start_num_points, end_num_points, step_num_points):
            ## subset spectra curve
            x_data_curve_log = np.array(data_k[:break_point])
            y_data_curve_log = np.log(data_power[:break_point])
            ## subset spectra tail
            x_data_tail_log = np.log10(data_k[break_point:])
            y_data_tail_log = np.log10(data_power[break_point:])
            ## fit spectra curve in log(e)-linear space
            fit_params_curve, _ = curve_fit(
                self.func_fit,
                x_data_curve_log, y_data_curve_log,
                bounds = (
                    ( np.log(10**(-15)), -3, 1/10 ),
                    ( np.log(10**(-5)),   3, 1/0.1 )
                ),
                maxfev = 10**3
            )
            ## fit spectra tail component in log10-linear
            fit_params_tail, _ = curve_fit(
                spectra_tail_log10,
                x_data_tail_log, y_data_tail_log,
                bounds = ( (-np.inf, -10), (0, 0) ),
                maxfev = 10**3
            )
            ## undo log(e) transformation of fitted parameters
            a0, a1, a2 = fit_params_curve # extract model parameters
            fit_params_curve = [
                np.exp(a0), # undo log transformation
                a1, a2
            ]
            ## calculate fitted full spectra
            fitted_power = np.array(
                ## spectra curve
                list(self.func_plot(
                    data_k[:break_point],
                    *fit_params_curve
                )) + 
                ## spectra tail
                list(spectra_tail_linear(
                    data_k[break_point:],
                    *fit_params_tail
                ))
            )
            ## measure residuals
            fit_error = np.sum(( np.log10(data_power) - np.log10(fitted_power) )**2) # 2-norm
            ## append fit information
            list_fit_params.append(fit_params_curve)
            list_fit_num_points.append(break_point)
            list_fit_errors.append(fit_error)
        ## save fit information
        self.list_fit_num_points_group_times.append(list_fit_num_points) 
        self.list_fit_errors_group_times.append(list_fit_errors)
        ## #################
        ## FIND THE BEST FIT
        ## #################
        ## define error cut-off
        maximum_error = 0.5 * ( max(list_fit_errors) + min(list_fit_errors) )
        ## find good fits (minima in the fit error plot)
        if list_fit_errors[0] < np.percentile(list_fit_errors, 84):
            list_minima_index = [ 0 ] # include the first fit in list of good fits
        else: list_minima_index = []
        ## find local minima (good fits) in the list of fit errors
        list_minima_index.extend(list(
            argrelextrema(
                np.array(list_fit_errors),
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
                    list_fit_errors[minima_index] < maximum_error # small error
                    and
                    1 / list_fit_params[minima_index][2] < 20 # dissipation scale is reasonable
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
        else: index_best_fit = getIndexListMin(list_fit_errors)
        ## save best fit information
        best_fit_params     = list_fit_params[index_best_fit]
        best_fit_num_points = list_fit_num_points[index_best_fit]
        self.fit_params_group_times.append(best_fit_params)
        self.fit_num_points_group_times.append(best_fit_num_points)
        ## save fitted scales
        a0_b, a1_b, a2_b = best_fit_params # get best fit parameters
        powerlaw = a1_b
        k_scale  = 1 / a2_b
        if self.bool_kriel_model:
            ## measure k_max for kriel model
            k_max = powerlaw * k_scale
        else:
            ## measure k_max for schober model
            k_max = fsolve(
                functools.partial(
                    k_max_implicit,
                    a1=a1_b, a2=a2_b
                ),
                x0 = powerlaw * k_scale
            )[0]
        self.k_max_group_times.append(k_max)
        self.k_scale_group_times.append(k_scale)
        ## ###################
        ## SAVE FITTED SPECTRA
        ## ###################
        fit_k = np.linspace(1, self.list_k_group_times[time_index][-1], 10**3)
        fit_power = self.func_plot(fit_k, *best_fit_params)
        self.list_fit_k_group_times.append(fit_k)
        self.list_fit_power_group_times.append(fit_power)

        # fig, ax = plt.subplots(figsize=(5, 3))
        # ax.plot(data_k, data_power, "k.")
        # ax.plot(
        #     data_k[:best_fit_num_points],
        #     self.func_plot(
        #         data_k[:best_fit_num_points],
        #         *best_fit_params
        #     ),
        #     "b--"
        # )
        # ax.axvline(x=k_scale, linestyle="--", color="red")
        # ax.axvline(x=k_max, linestyle="--", color="black")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_xlim([1, 100])
        # # ax.set_ylim([10**(-11), 10**(-1)])
        # ax.set_ylim([10**(-16), 10**(-8)])
        # plt.show()
        # print(fit_params_curve)
        # stop()

class FitMagSpectra(FitSpectra):
    def __init__(
            self,
            list_k_group_times, list_power_group_times, list_sim_times,
            bool_kriel_model,
            bool_hide_updates = False
        ):
        ## call parent class and pass fitting functions
        FitSpectra.__init__(
            self,
            list_k_group_times, list_power_group_times, list_sim_times,
            bool_kriel_model = bool_kriel_model,
            bool_hide_updates = bool_hide_updates
        )
    def getFitArgs(self):
        ## save fit output
        return {
            ## times fitted to
            "mag_sim_times":self.list_sim_times,
            ## spectra data fitted to
            "mag_k":self.list_k_group_times,
            "mag_power":self.list_power_group_times,
            ## fitted spectra
            "mag_fit_k":self.list_fit_k_group_times,
            "mag_fit_power":self.list_fit_power_group_times,
            ## fitted scales
            "k_eta":self.k_scale_group_times,
            "k_max":self.k_max_group_times,
            ## fit information
            "mag_fit_params":self.fit_params_group_times,
            "mag_fit_num_points":self.fit_num_points_group_times,
            "mag_list_fit_num_points":self.list_fit_num_points_group_times,
            "mag_list_fit_errors":self.list_fit_errors_group_times
        }


## load magnetic spectra data
filepath_data = "/Users/dukekriel/Documents/Projects/TurbulentDynamo/data/Re500/288/Pm2/"
mag_k_group_times, mag_power_group_times, mag_sim_times = loadListSpectra(
    filepath_data,
    str_spectra_type  = "mag"
)
## fit kriel model
kriel_fit = FitMagSpectra(mag_k_group_times[20:], mag_power_group_times[20:], mag_sim_times[20:], bool_kriel_model=True)
kriel_args = kriel_fit.getFitArgs()
## fit schober model
schober_fit = FitMagSpectra(mag_k_group_times[20:], mag_power_group_times[20:], mag_sim_times[20:], bool_kriel_model=False)
schober_args = schober_fit.getFitArgs()

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
## plot kriel fits
axs[0].plot(kriel_args["k_eta"], "b.")
axs[1].plot(kriel_args["k_max"], "b.")
## plot schober fits
axs[0].plot(schober_args["k_eta"], "r.")
axs[1].plot(schober_args["k_max"], "r.")
plt.show()


## ###############################################################
## SIMPLE PROGRAM
## ###############################################################
# filename = "Turb_hdf5_plt_cnt_0220_spect_mags.dat"
# data_k, data_power = loadSpectra(filepath_data + filename)
# data_k_sub = data_k[:60]
# data_power_sub = data_power[:60]
# weighting = [ x**(0.5) for x in data_k_sub ]

# ## fit exponential model (log(e)-linear space)
# params_kriel_loge, _ = curve_fit(
#     spectra_kriel_loge,
#     data_k_sub, np.log(data_power_sub),
#     sigma = weighting
# )
# ## undo transform of log(e) fit
# a0, a1, a2 = params_kriel_loge
# params_kriel_linear = [ np.exp(a0), a1, a2 ]

# ## fit bessel model (log(e) space)
# params_schober_loge, _ = curve_fit(
#     spectra_schober_loge,
#     data_k_sub, np.log(data_power_sub),
#     sigma = weighting
# )
# ## undo transform of log(e) fit
# a0, a1, a2 = params_schober_loge
# params_schober_linear = [ np.exp(a0), a1, a2 ]

# ## plot data and fits
# fg, ax = plt.subplots(figsize = (7, 4))
# ax.plot(data_k, data_power, "k.", markersize=10)
# ax.plot(
#     data_k, spectra_kriel_linear(data_k, *params_kriel_linear),
#     color="b", linestyle="-", linewidth=2, label="exp model (log-fit)"
# )
# ax.plot(
#     data_k, spectra_schober_linear(data_k, *params_schober_linear),
#     color="r", linestyle="--", linewidth=2, label="bessel model (log-fit)"
# )
# ax.legend()
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim([1, 60])
# ax.set_ylim([10**(-13), 10**(-5)])
# plt.show()
