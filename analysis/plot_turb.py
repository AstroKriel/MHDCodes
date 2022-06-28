#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit


## ###############################################################
## PREPARE TERMINAL/WORKSPACE/CODE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend('agg') # use a non-interactive plotting backend


bool_new_data = True
## ###############################################################
## FUNCTIONS
## ###############################################################
def funcLabelNotation(number, sig_fig=2):
  ret_string = "{0:.{1:d}e}".format(number, sig_fig)
  a, b = ret_string.split("e")
  return a + "\\times 10^{" + str(int(b)) + r"}"

def funcFindDomain(list_data_group, list_start_point, list_end_point):
  ## initialise lists
  list_index_start = []
  list_index_end   = []
  ## for each simulation
  for sim_index in range(len(list_data_group)):
    ## find indices when to start fitting
    index_start = getIndexClosestValue(list_data_group[sim_index], list_start_point[sim_index])
    ## find indices when to stop fitting
    index_end = getIndexClosestValue(list_data_group[sim_index], list_end_point[sim_index])
    ## check that there are a suffient number of points to fit to
    num_points = len(list_data_group[sim_index][index_start : index_end])
    if num_points < 3:
      ## if no sufficient domain to fit to, then don't define a fit range
      index_start = None
      index_end = None
      print("\t> Insufficient number of points ('{:d}') to fit to for sim. ({:d}).".format(
        num_points, sim_index
      ))
    ## save information
    list_index_start.append(index_start)
    list_index_end.append(index_end)
  ## return the list of indices
  return list_index_start, list_index_end

def funcFitExp(
    ax,
    data_x, data_y,
    index_start_fit, index_end_fit,
    color = "black"
  ):
  ## ##################
  ## SAMPLING DATA
  ## ########
  # define fit domain
  data_fit_domain = np.linspace(
    data_x[index_start_fit],
    data_x[index_end_fit],
    10**2
  )
  ## interpolate the non-uniform data
  interp_spline = make_interp_spline(
    data_x[index_start_fit : index_end_fit],
    data_y[index_start_fit : index_end_fit]
  )
  ## uniformly sample interpolated data
  data_sampled_y = interp_spline(data_fit_domain)
  ## ##################
  ## FITTING DATA
  ## #######
  ## fit exponential function to sampled data (in log-linear domain)
  fit_params_log, _ = curve_fit(
    ListOfModels.exp_loge,
    data_fit_domain,
    np.log(data_sampled_y)
  )
  ## undo log transformation
  fit_params_linear = [
    np.exp(fit_params_log[0]),
    fit_params_log[1]
  ]
  ## ##################
  ## PLOTTING DATA
  ## ########
  ## initialise the plot domain
  data_plot_domain = np.linspace(0, 100, 10**3)
  ## evaluate exponential
  data_E_exp = ListOfModels.exp_linear(
    data_plot_domain,
    *fit_params_linear
  )
  ## find where exponential enters / exists fit range
  index_E_start = getIndexClosestValue(data_E_exp, data_y[index_start_fit])
  index_E_end = getIndexClosestValue(data_E_exp, data_y[index_end_fit])
  ## create line data
  line_fitted = [ np.column_stack((
    data_plot_domain[index_E_start : index_E_end],
    data_E_exp[index_E_start : index_E_end]
  )) ]
  ## plot shadow line
  ax.add_collection(
    LineCollection(line_fitted, colors="black", ls="-", linewidth=3, zorder=9),
    autolim = False # ignore line when setting axis bounds
  )
  ## plot fitted exponential
  ax.add_collection(
    LineCollection(line_fitted,colors=color, ls="-", linewidth=1, zorder=10),
    autolim = False # ignore line when setting axis bounds
  )

def funcPlotTime(
    list_time_group, list_data_y_group, list_sim_labels,
    filepath_plot, fig_name,
    y_label      = None,
    bool_log_y   = False,
    bool_fit_exp = False,
    list_index_start_fit  = [None],
    list_index_end_fit    = [None]
  ):
  num_sims = len(list_sim_labels)
  ## extract colours from Cmasher's colormap
  cmasher_colormap = plt.get_cmap("cmr.tropical", num_sims)
  my_colormap = cmasher_colormap(np.linspace(0, 1, num_sims))
  ## create figure
  _, ax = plt.subplots(constrained_layout=True)
  ## #########################
  ## LOOP OVER EACH SIMULATION
  ## #########
  ## initialise boolean
  bool_plotted_y_bounds = False
  ## for each simulation
  for sim_index in range(num_sims):
    ## plot time evolving data
    ax.plot(
      list_time_group[sim_index],
      list_data_y_group[sim_index],
      color     = my_colormap[sim_index],
      linestyle = "-"
    )
    ## fit exponential to data
    if bool_fit_exp:
      ## check that fit indices have been defined
      if list_index_start_fit[sim_index] is None: continue
      if list_index_end_fit[sim_index]   is None: continue
      ## TODO: remove after debugging
      ax.plot(
        list_time_group[sim_index][list_index_start_fit[sim_index]],
        list_data_y_group[sim_index][list_index_start_fit[sim_index]],
        "ko", ms=5
      )
      ax.plot(
        list_time_group[sim_index][list_index_end_fit[sim_index]],
        list_data_y_group[sim_index][list_index_end_fit[sim_index]],
        "ko", ms=5
      )
      ## fit exponential
      funcFitExp(
        ax,
        data_x          = list_time_group[sim_index],
        data_y          = list_data_y_group[sim_index],
        index_start_fit = list_index_start_fit[sim_index],
        index_end_fit   = list_index_end_fit[sim_index],
        color           = my_colormap[sim_index]
      )
      ## plot lines bounding the fit range: y = [E_min, E_max]
      if not bool_plotted_y_bounds:
        ## toggle boolean
        bool_plotted_y_bounds = True
        ## lower limit
        ax.axhline(
          y=list_data_y_group[sim_index][list_index_start_fit[sim_index]],
          ls=(0, (7.5, 5)), color="black", lw=1
        )
        ## upper limit
        ax.axhline(
          y=list_data_y_group[sim_index][list_index_end_fit[sim_index]],
          ls=(0, (7.5, 5)), color="black", lw=1
        )
  ## #########################
  ## LABEL PLOT
  ## #########
  ## add legend
  if None not in list_sim_labels:
    addLegend(
      ax = ax,
      loc  = "upper right",
      bbox = (0.95, 1.0),
      artists = ["o"],
      colors  = my_colormap[:num_sims],
      legend_labels  = [
        list_elem.replace("_", "\_")
        for list_elem in list_sim_labels
      ]
    )
  ## add axis labels
  ax.set_xlabel(r"$t / t_\mathrm{eddy}$", fontsize=22)
  ax.set_ylabel(y_label, fontsize=22)
  ## scale y-axis
  if bool_log_y: ax.set_yscale("log")
  ## #########################
  ## SAVE IMAGE
  ## #########
  fig_filepath = createFilepath([filepath_plot, fig_name])
  plt.savefig(fig_filepath)
  plt.close()
  print("\t> Figure saved:", fig_name)

def funcPlotErrorbars(
    filepath_plot,
    list_data_x, list_data_y_group,
    list_index_start, list_index_end,
    list_sim_labels, fig_name,
    bool_log_x = False,
    bool_log_y = False,
    label_x = None,
    label_y = None
  ):
  ## extract colours from Cmasher's colormap
  num_sims = len(list_sim_labels)
  cmasher_colormap = plt.get_cmap("cmr.tropical", num_sims)
  my_colormap = cmasher_colormap(np.linspace(0, 1, num_sims))
  ## create figure
  _, ax = plt.subplots(constrained_layout=True)
  ## #########################
  ## LOOP OVER EACH SIMULATION
  ## #########
  for sim_index in range(num_sims):
    ## plot distribution
    plotErrorBar(
      ax,
      data_x = list_data_x[sim_index],
      data_y = list_data_y_group[sim_index][
        list_index_start[sim_index] : list_index_end[sim_index]
      ],
      color  = my_colormap[sim_index]
    )
  ## #########################
  ## LABEL THE FIGURE
  ## #########
  ## add legend
  addLegend(
    ax = ax,
    loc  = "upper right",
    bbox = (0.95, 1.0),
    artists = ["o"],
    colors  = my_colormap[:num_sims],
    legend_labels  = [ 
      list_elem.replace("_", "\_") if ("_" in list_elem)
      else list_elem
      for list_elem in list_sim_labels
    ]
  )
  ## scale axis
  if bool_log_x: ax.set_xscale("log")
  if bool_log_y: ax.set_yscale("log")
  ## label plot
  ax.set_xlabel(label_x, fontsize=22)
  ax.set_ylabel(label_y, fontsize=22)
  ## #########################
  ## SAVE FIGURE
  ## #########
  fig_filepath = createFilepath([
    filepath_plot, 
    fig_name
  ])
  plt.savefig(fig_filepath)
  plt.close()
  print("\t> Figure saved:", fig_name)

def funcMeasureStats(
    list_data_y_group,
    list_index_start = None,
    list_index_end   = None
  ):
  ## check if a domain to subset over has been defined
  bool_subset = (list_index_start is None) or (list_index_end is None)
  ## loop over each simulation
  num_sims = len(list_data_y_group)
  for sim_index in range(num_sims):
    ## subset data point
    if bool_subset:
      data_y = list_data_y_group[sim_index][
        list_index_start[sim_index] : list_index_end[sim_index]
      ]
    ## otherwise, just get the data
    else: data_y = list_data_y_group[sim_index]
    ## print statistics
    # print("{:s} \pm {:s}".format(
    #     funcLabelNotation(np.mean(data_y), sig_fig=1),
    #     funcLabelNotation(np.std(data_y),  sig_fig=2),
    # ))
    str_median = ("{0:.2g}").format(np.mean(data_y))
    if "." not in str_median:
      str_median = ("{0:.1f}").format(np.mean(data_y))
    if "." in str_median:
      if ("0" == str_median[0]) and (len(str_median.split(".")[1]) == 1):
        str_median = ("{0:.2f}").format(np.mean(data_y))
      num_decimals = len(str_median.split(".")[1])
    str_std = ("{0:."+str(num_decimals)+"f}").format(np.std(data_y))
    print("${} \pm {}$".format(str_median, str_std))

def funcMeasureGrowth(
    filepath_plot,
    list_Re, list_Rm, list_Pm,
    data_x_group, data_y_group,
    list_sim_labels, fig_name,
    list_index_start_fit = [None],
    list_index_end_fit   = [None]
  ):
  num_sims = len(list_sim_labels)
  ## extract colours from Cmasher's colormap
  cmasher_colormap = plt.get_cmap("cmr.tropical", num_sims)
  my_colormap = cmasher_colormap(np.linspace(0, 1, num_sims))
  ## create figure
  fig, axs = plt.subplots(3, 1, figsize=(7, 14))
  ## #########################
  ## LOOP OVER EACH SIMULATION
  ## #########
  for sim_index in range(num_sims):
    ## check that a fitting domain has been provided
    if list_index_start_fit[sim_index] is None: continue
    if list_index_end_fit[sim_index]   is None: continue
    ## get data
    data_x = data_x_group[sim_index]
    data_y = data_y_group[sim_index]
    index_start_fit = list_index_start_fit[sim_index]
    index_end_fit   = list_index_end_fit[sim_index]
    ## ##################
    ## SAMPLE DATA
    ## ######
    # define fit domain
    data_fit_domain = np.linspace(
      data_x[index_start_fit],
      data_x[index_end_fit],
      10**2
    )
    ## interpolate the non-uniform data
    interp_spline = make_interp_spline(
      data_x[index_start_fit : index_end_fit],
      data_y[index_start_fit : index_end_fit]
    )
    ## uniformly sample interpolated data
    data_sampled_y = interp_spline(data_fit_domain)
    ## ##################
    ## FIT DATA
    ## ###
    ## fit exponential function to sampled data (in log-linear domain)
    fit_params_log, fit_cov = curve_fit(
      ListOfModels.exp_loge,
      data_fit_domain,
      np.log(data_sampled_y)
    )
    ## undo log transformation
    fit_params_linear = [
      np.exp(fit_params_log[0]),
      fit_params_log[1]
    ]
    ## get fit value
    gamma_fit = fit_params_linear[1] * -1 # remove the negative in the model
    gamma_std = np.sqrt(np.diag(fit_cov))[1]
    ## ##################
    ## PLOT DATA
    ## ####
    str_median = ("{0:.2g}").format(gamma_fit)
    if "." not in str_median:
      str_median = ("{0:.1f}").format(gamma_fit)
    if "." in str_median:
      if ("0" == str_median[0]) and (len(str_median.split(".")[1]) == 1):
        str_median = ("{0:.2f}").format(gamma_fit)
      num_decimals = len(str_median.split(".")[1])
    str_std = ("{0:."+str(num_decimals)+"f}").format(gamma_std)
    print("${} \pm {}$".format(str_median, str_std))
    # print(
    #     funcLabelNotation(gamma_fit, sig_fig=1) + " \pm " +
    #     funcLabelNotation(gamma_std, sig_fig=2)
    # )
    ## plot Re distribution
    axs[0].errorbar(
      list_Re[sim_index],
      gamma_fit,
      yerr = gamma_std,
      color  = my_colormap[sim_index],
      fmt="o", markersize=7, markeredgecolor="black", capsize=7.5,
      elinewidth=2, linestyle="None", zorder=5
    )
    ## plot Re distribution
    axs[1].errorbar(
      list_Rm[sim_index],
      gamma_fit,
      yerr = gamma_std,
      color  = my_colormap[sim_index],
      fmt="o", markersize=7, markeredgecolor="black", capsize=7.5,
      elinewidth=2, linestyle="None", zorder=5
    )
    ## plot Re distribution
    axs[2].errorbar(
      list_Pm[sim_index],
      gamma_fit,
      yerr = gamma_std,
      color  = my_colormap[sim_index],
      fmt="o", markersize=7, markeredgecolor="black", capsize=7.5,
      elinewidth=2, linestyle="None", zorder=5
    )
  ## #########################
  ## LABEL PLOT
  ## #########
  ## add legend
  if None not in list_sim_labels:
    addLegend(
      ax  = axs[0],
      loc = "upper right",
      artists = ["o"],
      colors  = my_colormap[:num_sims],
      legend_labels  = [
        list_elem.replace("_", "\_")
        for list_elem in list_sim_labels
      ]
    )
  ## add axis labels
  axs[0].set_xlabel(r"Re", fontsize=22)
  axs[1].set_xlabel(r"Rm", fontsize=22)
  axs[2].set_xlabel(r"Pm", fontsize=22)
  axs[0].set_ylabel(r"$\Gamma_k$", fontsize=22)
  axs[1].set_ylabel(r"$\Gamma_k$", fontsize=22)
  axs[2].set_ylabel(r"$\Gamma_k$", fontsize=22)
  ## #########################
  ## SAVE IMAGE
  ## #########
  fig_filepath = createFilepath([filepath_plot, fig_name])
  plt.savefig(fig_filepath)
  plt.close()
  print("\t> Figure saved:", fig_name)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## ############################
  ## INPUT COMMAND LINE ARGUMENTS
  ## ############################
  parser = MyParser()
  ## ------------------- DEFINE OPTIONAL ARGUMENTS
  args_opt = parser.add_argument_group(description='Optional processing arguments:') # optional argument group
  ## define typical input requirements
  bool_args = {"required":False, "type":str2bool, "nargs":"?", "const":True}
  list_args = {"required":False, "nargs":"+"}
  ## 
  ## directory information
  args_opt.add_argument("-vis_folder", type=str, default="vis_folder", required=False)
  ## simulation information
  args_opt.add_argument("-sim_labels", type=str, default=[None], **list_args)
  args_opt.add_argument("-Re",         type=int, default=[None], **list_args)
  args_opt.add_argument("-Rm",         type=int, default=[None], **list_args)
  args_opt.add_argument("-Pm",         type=int, default=[None], **list_args)
  ## which domains should be fitted?
  args_opt.add_argument("-plot_E_grow", default=False, **bool_args)
  args_opt.add_argument("-plot_E_sat",  default=False, **bool_args)
  ## subsetting time ranges
  args_opt.add_argument("-E_ave_start", default=[1e-7],   **list_args, type=float)
  args_opt.add_argument("-E_ave_end",   default=[1e-2],   **list_args, type=float)
  args_opt.add_argument("-T_ave_start", default=[2],      **list_args, type=float)
  args_opt.add_argument("-T_ave_end",   default=[np.inf], **list_args, type=float)
  ## plot domain
  args_opt.add_argument("-plot_start", type=float, default=1,      required=False)
  args_opt.add_argument("-plot_end",   type=float, default=np.inf, required=False)
  ## ------------------- DEFINE REQUIRED ARGUMENTS
  args_req = parser.add_argument_group(description='Required processing arguments:') # required argument group
  ## define inputs
  args_req.add_argument("-base_path",   type=str,   required=True)
  args_req.add_argument("-sim_folders", type=str,   required=True, nargs="+")
  args_req.add_argument("-fig_name",    type=str,   required=True)
  args_req.add_argument("-t_turb",      type=float, required=True, nargs="+")
  ## ---------------------------- OPEN ARGUMENTS
  args = vars(parser.parse_args())
  ## ---------------------------- SAVE PARAMETERS
  ## simulation information
  list_t_eddy = args["t_turb"]
  list_Re     = args["Re"]
  list_Rm     = args["Rm"]
  list_Pm     = args["Pm"]
  ## which plpys should be made
  bool_plot_E_grow = args["plot_E_grow"]
  bool_plot_E_sat  = args["plot_E_sat"]
  ## fitting / subsetting information
  list_E_grow_start = args["E_ave_start"]
  list_E_grow_end   = args["E_ave_end"]
  list_E_sat_start  = args["T_ave_start"]
  list_E_sat_end    = args["T_ave_end"]
  ## plot domain
  plot_start = args["plot_start"]
  plot_end   = args["plot_end"]
  ## ---------------------------- SIMULATION PARAMETERS
  filepath_base    = args["base_path"]
  list_sim_folders = args["sim_folders"]
  list_sim_labels  = args["sim_labels"]
  folder_plot      = args["vis_folder"]
  fig_name         = args["fig_name"]

  ## ######################
  ## INITIALISING VARIABLES
  ## ######################
  print("Interpreting inputs...")
  num_sims = len(list_sim_folders)
  ## simulation information
  extendInputList(list_sim_labels, "list_sim_labels", num_sims, list_sim_folders)
  extendInputList(list_t_eddy,     "list_t_eddy",     num_sims)
  extendInputList(list_Re,         "list_Re",         num_sims)
  extendInputList(list_Rm,         "list_Rm",         num_sims)
  extendInputList(list_Pm,         "list_Pm",         num_sims)
  ## fit / subset information
  extendInputList(list_E_grow_start, "list_E_grow_start", num_sims)
  extendInputList(list_E_grow_end,   "list_E_grow_end",   num_sims)
  extendInputList(list_E_sat_start,  "list_E_sat_start",  num_sims)
  extendInputList(list_E_sat_end,    "list_E_sat_end",    num_sims)
  ## check that a valid list of Pm values have been defined
  if bool_plot_E_grow:
    if None in list_Pm:
      raise Exception("'Pm' values have not been defined.")
    elif len(list_Pm) < len(list_sim_folders):
      raise Exception("Only '{:d}' Pm values have been defined, but there are '{:d}' simulations.".format(
        len(list_Pm),
        len(list_sim_folders)
      ))
  print(" ")

  ## #####################
  ## PREPARING DIRECTORIES
  ## #####################
  ## folders where spectra data files are stored for each simulation
  list_filepaths_data = []
  for sim_index in range(num_sims):
    list_filepaths_data.append(createFilepath([ filepath_base, list_sim_folders[sim_index] ]))
  ## create folder where the figures will be saved
  filepath_plot = createFilepath([filepath_base, folder_plot])
  createFolder(filepath_plot)

  ## ##########################################
  ## PRINT CONFIGURATION INFORMATION TO CONSOLE
  ## ##########################################
  ## print directory information
  printInfo("Base filepath:", filepath_base)
  printInfo("Figure folder:", filepath_plot)
  printInfo("Figure name:",   fig_name)
  print("Working with the following directories:")
  for sim_index in range(len(list_filepaths_data)):
    print("\t({:d})".format(sim_index), list_filepaths_data[sim_index])
  print(" ")
  ## print simulation information
  printInfo("t_turb:",     list_t_eddy)
  printInfo("Sim labels:", list_sim_labels)
  ## print plot information
  printInfo("Plot domain:", [plot_start, plot_end])
  if bool_plot_E_grow or bool_plot_E_sat:
    printInfo("Re:", list_Re)
    printInfo("Rm:", list_Rm)
    printInfo("Pm:", list_Pm)
  if bool_plot_E_grow:
    printInfo("(E_B / E_K) kin. E_start:", list_E_grow_start)
    printInfo("(E_B / E_K) kin. E_end:  ", list_E_grow_end)
  if bool_plot_E_sat:
    printInfo("(E_B / E_K) sat. T_start:", list_E_sat_start)
    printInfo("(E_B / E_K) sat. T_end:  ", list_E_sat_end)
  print(" ")

  ## ##################################
  ## LOAD TIME EVOLVING DATA QUANTITIES
  ## ##################################
  ## initialise list of data
  list_data_time_group = []
  list_data_Mach_group = []
  list_data_E_B_group  = []
  list_data_E_K_group  = []
  print("Loading 'Turb.dat' data...")
  for sim_index in range(len(list_filepaths_data)):
    ## load 'Mach' data
    data_time, data_Mach = loadTurbData(
      filepath_data = list_filepaths_data[sim_index],
      var_y      = 13 if bool_new_data else 8, # 13 (new), 8 (old)
      t_turb     = list_t_eddy[sim_index],
      time_start = plot_start,
      time_end   = plot_end
    )
    ## load 'E_B' data
    data_time, data_E_B = loadTurbData(
      filepath_data = list_filepaths_data[sim_index],
      var_y      = 11 if bool_new_data else 29, # 11 (new), 29 (old)
      t_turb     = list_t_eddy[sim_index],
      time_start = plot_start,
      time_end   = plot_end
    )
    ## load 'E_K' data
    data_time, data_E_K = loadTurbData(
      filepath_data = list_filepaths_data[sim_index],
      var_y      = 9 if bool_new_data else 6, # 9 (new), 6 (old)
      t_turb     = list_t_eddy[sim_index],
      time_start = plot_start,
      time_end   = plot_end
    )
    ## append data
    list_data_time_group.append(data_time)
    list_data_Mach_group.append(data_Mach)
    list_data_E_B_group.append(data_E_B)
    list_data_E_K_group.append(data_E_K)
  ## find ratio: 'E_B / E_K'
  list_data_E_ratio_group = [
    [ (E_B / E_K) for E_B, E_K in zip(list_data_E_B, list_data_E_K) ]
    for list_data_E_B, list_data_E_K in zip(list_data_E_B_group, list_data_E_K_group)
  ]
  ## get bounds of kinematic regime
  list_index_start_E_grow = []
  list_index_end_E_grow   = []
  if bool_plot_E_grow:
    print("Finding kinematic domain...")
    list_index_start_E_grow, list_index_end_E_grow = funcFindDomain(
      list_data_group  = list_data_E_ratio_group,
      list_start_point = list_E_grow_start,
      list_end_point   = list_E_grow_end
    )
    print(" ")
  ## get bounds of saturated regime
  list_index_start_E_sat = []
  list_index_end_E_sat   = []
  if bool_plot_E_sat:
    print("Finding saturated domain...")
    list_index_start_E_sat, list_index_end_E_sat = funcFindDomain(
      list_data_group  = list_data_time_group,
      list_start_point = list_E_sat_start,
      list_end_point   = list_E_sat_end
    )
    print(" ")

  ## ###################
  ## PLOT TIME EVOLUTION
  ## ###################
  print("Plotting data...")
  ## plot 'E_B / E_K' evolution
  funcPlotTime(
    list_data_time_group, list_data_E_ratio_group, list_sim_labels,
    filepath_plot,
    fig_name = createName([fig_name, "E_ratio.pdf"]),
    y_label      = r"$E_{\rm{mag}} / E_{\rm{kin}}$",
    bool_log_y   = True,
    bool_fit_exp = bool_plot_E_grow,
    list_index_start_fit = list_index_start_E_grow if bool_plot_E_grow else [None],
    list_index_end_fit   = list_index_end_E_grow   if bool_plot_E_grow else [None]
  )
  ## plot 'Mach' evolution
  funcPlotTime(
    list_data_time_group, list_data_Mach_group, list_sim_labels,
    filepath_plot,
    fig_name = createName([ fig_name, "Mach.pdf" ]),
    y_label  = r"$\mathcal{M}$"
  )
  print(" ")

  ## ########################################
  ## MEASURING STATISTICS IN KINEMATIC REGIME
  ## ########################################
  if bool_plot_E_grow:
    ## plot 'E_B'
    print("Measuring magnetic growth rate...")
    funcMeasureGrowth(
      filepath_plot,
      list_Re, list_Rm, list_Pm,
      data_x_group = list_data_time_group,
      data_y_group = list_data_E_B_group,
      list_sim_labels = list_sim_labels,
      fig_name        = createName([fig_name, "Bfield_growth_rate.pdf"]),
      list_index_start_fit = list_index_start_E_grow,
      list_index_end_fit   = list_index_end_E_grow
    )
    print(" ")
    ## plot 'Mach'
    print("Measuring Mach values...")
    funcMeasureStats(
      list_data_y_group = list_data_Mach_group,
      list_index_start  = list_index_start_E_grow,
      list_index_end    = list_index_end_E_grow
    )
    funcPlotErrorbars(
      filepath_plot,
      list_data_x = list_Pm,
      list_data_y_group = list_data_Mach_group,
      list_index_start  = list_index_start_E_grow,
      list_index_end    = list_index_end_E_grow,
      list_sim_labels   = list_sim_labels,
      fig_name   = createName([fig_name, "Mach_kinematic_phase.pdf"]),
      bool_log_x = True,
      bool_log_y = False,
      label_x    = r"Pm",
      label_y    = r"$\mathcal{M}$"
    )
    print(" ")

  ## ########################################
  ## MEASURING STATISTICS IN SATURATED REGIME
  ## ########################################
  if bool_plot_E_sat:
    ## plot 'E_B / E_K'
    print("Measuring saturated ratio...")
    funcMeasureStats(
      list_data_y_group = list_data_E_ratio_group,
      list_index_start  = list_index_start_E_sat,
      list_index_end    = list_index_end_E_sat
    )
    print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM