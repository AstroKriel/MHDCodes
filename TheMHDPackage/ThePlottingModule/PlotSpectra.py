## START OF MODULE


## ###############################################################
## DEPENDENCIES: REQUIRED MODULES
## ###############################################################
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import v

from tqdm import tqdm
from matplotlib.gridspec import GridSpec

from ThePlottingModule import PlotFuncs
from TheUsefulModule import WWLists, WWFnF
# from TheAnalysisModule import RemoveOutliers


## ###############################################################
## SPECTRA STATISTICS OF SPECTRA FITS FOR SINGLE SIMULATION
## ###############################################################
class PlotSpectraFit():
  ''' Plotting fitted spectra.
  '''
  def __init__(
      self,
      spectra_obj
    ):
    ## save spectra object
    self.spectra_obj = spectra_obj
    ## check that the spectra object has been labelled
    if (self.spectra_obj.sim_suite is None) or (self.spectra_obj.sim_label is None):
      raise Exception("Spectra object should have a suite ({:}) and label ({:}) defned.".format(
        self.spectra_obj.sim_suite,
        self.spectra_obj.sim_label
      ))
    ## save the times when both the velocity and magnetic spectra were fitted
    self.sim_times = WWLists.getCommonElements(
      self.spectra_obj.vel_sim_times,
      self.spectra_obj.mag_sim_times
    )
    ## find which part of the data to collect statistics about
    self.bool_plot_ave_scales = False # initialise as false
    bool_vel_fit = (self.spectra_obj.vel_fit_start_t is not None) and (self.spectra_obj.vel_fit_end_t is not None)
    bool_mag_fit = (self.spectra_obj.mag_fit_start_t is not None) and (self.spectra_obj.mag_fit_end_t is not None)
    if bool_vel_fit and bool_mag_fit:
      ## if a fitting domain has been defined
      self.bool_plot_ave_scales = True
      ## find indices of velocity fit time range
      self.vel_index_start = WWLists.getIndexClosestValue(self.sim_times, self.spectra_obj.vel_fit_start_t)
      self.vel_index_end   = WWLists.getIndexClosestValue(self.sim_times, self.spectra_obj.vel_fit_end_t)
      ## find indices of magnetic fit time range
      self.mag_index_start = WWLists.getIndexClosestValue(self.sim_times, self.spectra_obj.mag_fit_start_t)
      self.mag_index_end   = WWLists.getIndexClosestValue(self.sim_times, self.spectra_obj.mag_fit_end_t)
      ## subset data
      k_nu_subset_t  = self.spectra_obj.k_nu_group_t[self.vel_index_start:self.vel_index_end]
      k_eta_subset_t = self.spectra_obj.k_eta_group_t[self.mag_index_start:self.mag_index_end]
      k_max_subset_t = self.spectra_obj.k_max_group_t[self.mag_index_start:self.mag_index_end]
      ## calculate mean of measured scales
      self.k_nu_mean  = np.mean(k_nu_subset_t)
      self.k_eta_mean = np.mean(k_eta_subset_t)
      self.k_max_mean = np.mean(k_max_subset_t)
      ## calculate 1-sigma of measured scales
      self.k_nu_std  = np.std(k_nu_subset_t)
      self.k_eta_std = np.std(k_eta_subset_t)
      self.k_max_std = np.std(k_max_subset_t)
  def plotMeasuredScales(
      self,
      filepath_plot
    ):
    ## ########################
    ## CREATE FIGURE
    ## ########
    ## initialise figure
    fig, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.25)
    ## initialise figure y-range
    range_y = [ 0.05, 150 ]
    ## extract data
    data_x_ax0 = self.spectra_obj.vel_sim_times
    data_x_ax1 = self.spectra_obj.mag_sim_times
    data_x_ax2 = self.spectra_obj.mag_sim_times
    data_y_ax0 = self.spectra_obj.k_nu_group_t
    data_y_ax1 = self.spectra_obj.k_eta_group_t
    data_y_ax2 = self.spectra_obj.k_max_group_t
    ## #######################
    ## PLOTTING DATA
    ## ########
    ## show data within the fit range
    if self.bool_plot_ave_scales:
      ## #######################
      ## CLEANING DATA
      ## ########
      ## remove outliers in k_nu
      data_x_ax0_subset = data_x_ax0[self.vel_index_start : self.vel_index_end]
      data_y_ax0_subset = data_y_ax0[self.vel_index_start : self.vel_index_end]
      ## remove outliers in k_eta
      data_x_ax1_subset = data_x_ax1[self.mag_index_start : self.mag_index_end]
      data_y_ax1_subset = data_y_ax1[self.mag_index_start : self.mag_index_end]
      ## remove outliers in k_max
      data_x_ax2_subset = data_x_ax2[self.mag_index_start : self.mag_index_end]
      data_y_ax2_subset = data_y_ax2[self.mag_index_start : self.mag_index_end]
      ## plot full dataset
      ax0.plot(data_x_ax0, data_y_ax0, "k.", alpha=0.1)
      ax1.plot(data_x_ax1, data_y_ax1, "k.", alpha=0.1)
      ax2.plot(data_x_ax2, data_y_ax2, "k.", alpha=0.1)
      ## plot subsetted data (in fit range)
      ax0.plot(data_x_ax0_subset, data_y_ax0_subset, "r.", alpha=0.2)
      ax1.plot(data_x_ax1_subset, data_y_ax1_subset, "r.", alpha=0.2)
      ax2.plot(data_x_ax2_subset, data_y_ax2_subset, "r.", alpha=0.2)
      ## distribution of k_nu (in fit range)
      ax0_inset = PlotFuncs.insetPDF(
        ax0,
        data_y_ax0_subset,
        label_x = r"$k_\nu$ (subsetted)",
        label_y = r"$P(k_\nu)$"
      )
      ax0_inset.axvline(
        x=np.percentile(data_y_ax0_subset, 16),
        ls="--", color="k"
      )
      ax0_inset.axvline(
        x=np.percentile(data_y_ax0_subset, 50),
        ls="--", color="k"
      )
      ax0_inset.axvline(
        x=np.percentile(data_y_ax0_subset, 84),
        ls="--", color="k"
      )
      ## distribution of k_eta (in fit range)
      ax1_inset = PlotFuncs.insetPDF(
        ax1,
        data_y_ax1_subset,
        label_x = r"$k_\eta$ (subsetted)",
        label_y = r"$P(k_\eta)$"
      )
      ax1_inset.axvline(
        x=np.percentile(data_y_ax1_subset, 16),
        ls="--", color="k"
      )
      ax1_inset.axvline(
        x=np.percentile(data_y_ax1_subset, 50),
        ls="--", color="k"
      )
      ax1_inset.axvline(
        x=np.percentile(data_y_ax1_subset, 84),
        ls="--", color="k"
      )
      ## distribution of k_p (in fit range)
      ax2_inset = PlotFuncs.insetPDF(
        ax2,
        data_y_ax2_subset,
        label_x = r"$k_p$ (subsetted)",
        label_y = r"$P(k_p)$"
      )
      ax2_inset.axvline(
        x=np.percentile(data_y_ax2_subset, 16),
        ls="--", color="k"
      )
      ax2_inset.axvline(
        x=np.percentile(data_y_ax2_subset, 50),
        ls="--", color="k"
      )
      ax2_inset.axvline(
        x=np.percentile(data_y_ax2_subset, 84),
        ls="--", color="k"
      )
    else:
      ## plot data
      ax0.plot(data_x_ax0, data_y_ax0, "k.")
      ax1.plot(data_x_ax1, data_y_ax1, "k.")
      ax2.plot(data_x_ax2, data_y_ax2, "k.")
    ## label axis
    ax0.set_xlabel(r"$t / T$",  fontsize=20)
    ax1.set_xlabel(r"$t / T$",  fontsize=20)
    ax2.set_xlabel(r"$t / T$",  fontsize=20)
    ax0.set_ylabel(r"$k_\nu$",  fontsize=20)
    ax1.set_ylabel(r"$k_\eta$", fontsize=20)
    ax2.set_ylabel(r"$k_p$", fontsize=20)
    ## ########################
    ## SCALE FIGURE AXIS
    ## ########
    ## set figure axis range
    ax0.set_ylim(range_y)
    ax1.set_ylim(range_y)
    ax2.set_ylim(range_y)
    ## scale figure axis
    ax0.set_yscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    PlotFuncs.FixLogAxis(ax0, bool_fix_y_axis=True)
    PlotFuncs.FixLogAxis(ax1, bool_fix_y_axis=True)
    PlotFuncs.FixLogAxis(ax2, bool_fix_y_axis=True)
    ## check if any points are outside of the figure window
    PlotFuncs.showDataOutsideAxis(ax0, data_x_ax0, data_y_ax0, range_y)
    PlotFuncs.showDataOutsideAxis(ax0, data_x_ax1, data_y_ax1, range_y)
    PlotFuncs.showDataOutsideAxis(ax0, data_x_ax2, data_y_ax2, range_y)
    ## ########################
    ## SAVE FIGURE
    ## ########
    ## create figure name
    fig_name = WWFnF.createName([
      self.spectra_obj.sim_suite,
      self.spectra_obj.sim_label,
      "check_MeasuredScales"
    ]) + ".pdf"
    ## create filepath where figure will be saved
    filepath_fig = WWFnF.createFilepath([
      filepath_plot,
      fig_name
    ])
    ## save figure
    plt.savefig(filepath_fig)
    print("\t> Figure saved: " + fig_name)
    # close plot
    plt.close(fig)
  def plotNumFitPoints(
      self,
      filepath_plot
    ):
    ## ########################
    ## INITIALISE DATA
    ## ########
    ## left axis
    data_x_ax0  = self.spectra_obj.vel_sim_times
    data_y_ax0  = self.spectra_obj.vel_fit_k_index_group_t
    range_y_ax0 = [ 0.5, max(self.spectra_obj.vel_list_k_group_t[0]) ]
    ## right axis
    data_x_ax1  = self.spectra_obj.mag_sim_times
    data_y_ax1  = self.spectra_obj.mag_fit_k_index_group_t
    range_y_ax1 = [ 0.5, max(self.spectra_obj.mag_list_k_group_t[0]) ]
    ## ########################
    ## CREATE FIGURE
    ## ########
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.2)
    ## show data within the fit range
    if self.bool_plot_ave_scales:
      ## left axis
      PlotFuncs.insetPlot(
        ax0,
        data_x_ax0[self.vel_index_start:self.vel_index_end],
        data_y_ax0[self.vel_index_start:self.vel_index_end],
        label_x    = r"$t / T$ (sub-domain)",
        label_y    = r"Last fitted k-mode",
        range_y    = range_y_ax0,
        bool_log_y = True
      )
      ## right axis
      PlotFuncs.insetPlot(
        ax1,
        data_x_ax1[self.mag_index_start:self.mag_index_end],
        data_y_ax1[self.mag_index_start:self.mag_index_end],
        label_x    = r"$t / T$ (sub-domain)",
        range_y    = range_y_ax1,
        bool_log_y = True
      )
      ## plot data
      ax0.plot(data_x_ax0, data_y_ax0, "k.", alpha=0.1)
      ax1.plot(data_x_ax1, data_y_ax1, "k.", alpha=0.1)
      ## plot data in fit range
      ax0.plot(
        data_x_ax0[self.vel_index_start:self.vel_index_end],
        data_y_ax0[self.vel_index_start:self.vel_index_end],
        "r.", alpha=0.2
      )
      ax1.plot(
        data_x_ax1[self.mag_index_start:self.mag_index_end],
        data_y_ax1[self.mag_index_start:self.mag_index_end],
        "r.", alpha=0.2
      )
    else:
      ## plot data
      ax0.plot(data_x_ax0, data_y_ax0, "k.")
      ax1.plot(data_x_ax1, data_y_ax1, "k.")
    ## label axis
    ax0.set_xlabel(r"$t / T$", fontsize=20)
    ax1.set_xlabel(r"$t / T$", fontsize=20)
    ax0.set_ylabel(r"Number of points fitted", fontsize=20)
    ## ########################
    ## SCALE FIGURE AXIS
    ## ########
    ## set axis y-range
    ax0.set_ylim(range_y_ax0)
    ax1.set_ylim(range_y_ax1)
    ## scale figure axis
    ax0.set_yscale("log")
    ax1.set_yscale("log")
    PlotFuncs.FixLogAxis(ax0, bool_fix_y_axis=True)
    PlotFuncs.FixLogAxis(ax1, bool_fix_y_axis=True)
    ## check if any points are outside of the figure window
    PlotFuncs.showDataOutsideAxis(ax0, data_x_ax0, data_y_ax0, range_y_ax0)
    ## check if any points are outside of the figure window
    PlotFuncs.showDataOutsideAxis(ax1, data_x_ax1, data_y_ax1, range_y_ax1)
    ## ########################
    ## SAVE FIGURE
    ## ########
    ## create figure name
    fig_name = WWFnF.createName([
      self.spectra_obj.sim_suite,
      self.spectra_obj.sim_label,
      "check_NumFitPoints"
    ]) + ".pdf"
    ## create filepath where figure will be saved
    filepath_fig = WWFnF.createFilepath([
      filepath_plot,
      fig_name
    ])
    ## save figure
    plt.savefig(filepath_fig)
    print("\t> Figure saved: " + fig_name)
    # close plot
    plt.close(fig)
  def plotFit2Norm_NumFitPoints(
      self,
      filepath_plot, target_time
    ):
    ## get fit index associated with the target time
    fit_index = WWLists.getIndexClosestValue(self.sim_times, target_time)
    ## ########################
    ## CREATE FIGURE
    ## ########
    ## initialise figure
    fig_scales = plt.figure(figsize=(12, 5))
    fig_grids = GridSpec(ncols=2, nrows=1, figure=fig_scales)
    ax0 = fig_scales.add_subplot(fig_grids[0])
    ax1 = fig_scales.add_subplot(fig_grids[1])
    ## plot data
    ax0.plot(
      self.spectra_obj.vel_list_fit_k_range_group_t[fit_index],
      self.spectra_obj.vel_list_fit_2norm_group_t[fit_index],
      "k."
    )
    ax1.plot(
      self.spectra_obj.mag_list_fit_k_range_group_t[fit_index],
      self.spectra_obj.mag_list_fit_2norm_group_t[fit_index],
      "k."
    )
    ## ########################
    ## LABEL FIGURE
    ## ########
    ## add chosen number of fitted lines
    ax0.axvline(
      x = self.spectra_obj.vel_fit_k_index_group_t[fit_index],
      ls="--", color="k"
    )
    ax1.axvline(
      x = self.spectra_obj.mag_fit_k_index_group_t[fit_index],
      ls="--", color="k"
    )
    ## add time label
    ax1.text(
      0.95, 0.95, r"$t / T = {}$".format(str(target_time)),
      va="top", ha="right", transform=ax1.transAxes, fontsize=16
    )
    ## add spectra label
    ax0.text(
      0.05, 0.95, r"Velocity energy spectra",
      va="top", ha="left", transform=ax0.transAxes, fontsize=16
    )
    ## add spectra label
    ax1.text(
      0.05, 0.95, r"Magnetic power spectra",
      va="top", ha="left", transform=ax1.transAxes, fontsize=16
    )
    ## label axis
    ax0.set_xlabel(r"Number of points fitted", fontsize=20)
    ax1.set_xlabel(r"Number of points fitted", fontsize=20)
    ax0.set_ylabel(r"$\sum_{i = 1}^{N} |y_i - f(x_i)|^2$", fontsize=20)
    ## ########################
    ## SAVE FIGURE
    ## ########
    ## create figure name
    fig_name = WWFnF.createName([
      self.spectra_obj.sim_suite,
      self.spectra_obj.sim_label,
      "check_FitError_NumFitPoints_{0:04}".format(int(fit_index))
    ]) + ".pdf"
    ## create filepath where figure will be saved
    filepath_fig = WWFnF.createFilepath([
      filepath_plot,
      fig_name
    ])
    ## save figure
    plt.savefig(filepath_fig)
    print("\t> Figure saved: " + fig_name)
    # close plot
    plt.close(fig_scales)
  def plotSpectra_TargetTime(
      self,
      filepath_plot, target_time
    ):
    ## get fit index associated with the target time
    fit_index = WWLists.getIndexClosestValue(self.sim_times, target_time)
    ## create figure name
    fig_name = WWFnF.createName([
      self.spectra_obj.sim_suite,
      self.spectra_obj.sim_label,
      "check_SpectraFit={0:04}".format(int(fit_index))
    ]) + ".pdf"
    ## initialise spectra evolution figure
    fig, ax = plt.subplots(constrained_layout=True)
    ## plot spectra data
    self.plotSpectra(
      fig, ax,
      filepath_plot, fit_index, fig_name,
      y_min = 1e-18,
      y_max = 1e2,
      x_min = 0.1,
      x_max = 200
    )
    ## close figure
    plt.close(fig)
    ## print information to the terminal
    print("\t> Figure saved: " + fig_name)
  def plotSpectraEvolution(
      self,
      filepath_plot,
      plot_index_start  = 0,
      plot_index_step   = 1,
      bool_hide_updates = False
    ):
    ## initialise spectra evolution figure
    fig, ax = plt.subplots()
    ## loop over each time slice
    for time_index in tqdm(
        range(plot_index_start, len(self.sim_times), plot_index_step),
        miniters = (len(self.sim_times) - plot_index_start) / 10,
        disable  = bool_hide_updates or (len(self.sim_times) < 3)
      ):
      self.plotSpectra(fig, ax, filepath_plot, time_index)
    ## close figure
    plt.close(fig)
  def plotSpectra(
      self,
      fig, ax, filepath_plot, time_index,
      fig_name = None,
      y_min = 1e-21,
      y_max = 1e2,
      x_min = 10**(-1),
      x_max = 300
    ):
    ## ###############################
    ## PLOT SPECTRA DATA
    ## #############
    ax.plot(
      self.spectra_obj.vel_list_k_group_t[time_index],
      self.spectra_obj.vel_list_power_group_t[time_index],
      label=r"vel-spectra", color="blue", ls="", marker=".", markersize=8
    )
    ax.plot(
      self.spectra_obj.mag_list_k_group_t[time_index],
      self.spectra_obj.mag_list_power_group_t[time_index],
      label=r"mag-spectra", color="red", ls="", marker=".", markersize=8
    )
    ## ###############################
    ## PLOT FITTED SPECTRAS
    ## #############
    ## plot fitted spectra
    ax.plot(
      self.spectra_obj.vel_list_fit_k_group_t[time_index],
      self.spectra_obj.vel_list_fit_power_group_t[time_index],
      label=r"vel-spectra (fitted)", color="blue", linestyle="--", dashes=(5, 2.5), linewidth=2
    )
    ax.plot(
      self.spectra_obj.mag_list_fit_k_group_t[time_index],
      self.spectra_obj.mag_list_fit_power_group_t[time_index],
      label=r"mag-spectra (fitted)", color="red", linestyle="--", dashes=(5, 2.5), linewidth=2
    )
    ## plot measured scales
    ax.axvline(x=self.spectra_obj.k_nu_group_t[time_index],  ls="--", color="blue",  label=r"$k_\nu$")
    ax.axvline(x=self.spectra_obj.k_eta_group_t[time_index], ls="--", color="red",   label=r"$k_\eta$")
    ax.axvline(x=self.spectra_obj.k_max_group_t[time_index], ls="--", color="black", label=r"$k_p$")
    ## plot measured scales if a time range to average over has been defined
    if self.bool_plot_ave_scales:
      ax.fill_betweenx(
        np.linspace(y_min, y_max, 100),
        (self.k_nu_mean - self.k_nu_std),
        (self.k_nu_mean + self.k_nu_std),
        facecolor="blue", alpha=0.15, zorder=1
      )
      ax.fill_betweenx(
        np.linspace(y_min, y_max, 100),
        (self.k_eta_mean - self.k_eta_std),
        (self.k_eta_mean + self.k_eta_std),
        facecolor="red", alpha=0.15, zorder=1
      )
      ax.fill_betweenx(
        np.linspace(y_min, y_max, 100),
        (self.k_max_mean - self.k_max_std),
        (self.k_max_mean + self.k_max_std),
        facecolor="black", alpha=0.15, zorder=1
      )
    ## ###############################
    ## LABEL FIGURE
    ## #############
    ## label spectra models
    PlotFuncs.plotLabelBox(
      fig, ax,
      ## box placement
      box_alignment = (0.0, 0.0),
      xpos = 0.025,
      ypos = 0.025,
      ## label appearance
      alpha = 0.25,
      ## list of labels to place in box
      list_fig_labels = [
        ## velocity spectra fit
        r"$\mathcal{P}_{\rm vel}(k) = A k^{\alpha_1} \exp\left\{-\frac{k}{k_\nu}\right\}$",
        (
          r"$A = $ "+"{:.2e}".format(self.spectra_obj.vel_list_fit_params_group_t[time_index][0]) +
          r", $\alpha_\mathrm{kin} = $ "+"{:.2f}".format(self.spectra_obj.vel_list_fit_params_group_t[time_index][1]) +
          r", $k_\nu = $ "+"{:.2f}".format(1 / self.spectra_obj.vel_list_fit_params_group_t[time_index][2])
        ),
        ## magnetic spectra fit
        r"$\mathcal{P}_{\rm mag}(k) = A k^{\alpha_1} K_0\left\{-\frac{k}{k_\eta}\right\}$",
        (
          r"$A = $ "+"{:.2e}".format(self.spectra_obj.mag_list_fit_params_group_t[time_index][0]) +
          r", $\alpha_\mathrm{mag} = $ "+"{:.2f}".format(self.spectra_obj.mag_list_fit_params_group_t[time_index][1]) +
          r", $k_\eta = $ "+"{:.2f}".format(1 / self.spectra_obj.mag_list_fit_params_group_t[time_index][2]) + 
          r", $k_p = $ "+"{:.2f}".format(self.spectra_obj.k_max_group_t[time_index])
        )
      ]
    )
    ## add time stamp
    ax.text(0.025, 0.975,
      r"$t / t_{\rm eddy} = $ "+"{:.1f}".format(self.sim_times[time_index]), 
      va="top", ha="left", transform=ax.transAxes, fontsize=16
    )
    ## add legend
    ax.legend(frameon=True, loc="upper right", facecolor="white", framealpha=0.5, fontsize=12)
    ## adjust figure axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## label axes
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\mathcal{P}$")
    ## ###############################
    ## SAVE FIGURE
    ## #############
    ## make sure that a name for the figure has been defined
    if fig_name is None:
      fig_name = WWFnF.createName([
        self.spectra_obj.sim_suite,
        self.spectra_obj.sim_label,
        "spectra_fit={0:04}".format(int(time_index))
      ]) + ".png"
    ## save the figure
    plt.savefig(
      WWFnF.createFilepath([
        filepath_plot,
        fig_name
      ]),
      dpi = 100
    )
    ## clear figure and axis
    fig.artists.clear()
    ax.clear()
  def aniSpectra(
      self,
      filepath_plot,
      filepath_ani_movie
    ):
    PlotFuncs.aniEvolution(
      filepath_plot,
      filepath_ani_movie,
      WWFnF.createName([ self.spectra_obj.sim_suite, self.spectra_obj.sim_label, "spectra_fit=%*.png" ]),
      WWFnF.createName([ self.spectra_obj.sim_suite, self.spectra_obj.sim_label, "ani_spectra_fit.mp4" ])
    )


## ###############################################################
## PLOT SPECTRA EVOLUTION
## ###############################################################
class PlotSpectra():
  '''
  Plotting raw spectra data.
  '''
  def __init__(self,
      ## spectra data
      vel_k, vel_power, mag_k, mag_power,
      ## frame information
      sim_times, sim_name,
      ## where to save plos / animation
      filepath_frames, filepath_ani
    ):
    self.vel_k = vel_k
    self.mag_k = mag_k
    self.vel_power = vel_power
    self.mag_power = mag_power
    self.sim_times = sim_times
    self.sim_name  = sim_name
    self.filepath_frames = filepath_frames
    self.filepath_ani    = filepath_ani
  def plotSpectra(self, bool_hide_updates):
    '''
    Plot the evolution of the spectra.
    '''
    ## plot evolution of spectra
    y_min = 1e-20
    y_max = 10
    x_min = 10**(-1)
    x_max = max(len(self.vel_k[0]), len(self.mag_k[0]))
    ## initialise spectra evolution figure
    fig, ax = plt.subplots(constrained_layout=True)
    ## loop over each time slice
    for time_index in tqdm(
        range(len(self.sim_times)),
        disable = bool_hide_updates or (len(self.sim_times) < 3)
      ):
      ## #################
      ## PLOT SPECTRA DATA
      ## #######
      ax.plot(
        self.vel_k[time_index],
        self.vel_power[time_index],
        label=r"vel-spectra", color="blue", ls="", marker=".", markersize=8
      )
      ax.plot(
        self.mag_k[time_index],
        self.mag_power[time_index],
        label=r"mag-spectra", color="red", ls="", marker=".", markersize=8
      )
      ## ############
      ## LABEL FIGURE
      ## #######
      ## add time stamp
      ax.text(0.975, 0.975, 
        r"$t/t_{\rm eddy} = $ "+"{:.1f}".format(self.sim_times[time_index]), 
        va="top", ha="right", transform=ax.transAxes, fontsize=16
      )
      ## add legend
      ax.legend(frameon=True, loc="upper left", facecolor="white", framealpha=0.5, fontsize=12)
      ## adjust figure axes
      ax.set_xlim(x_min, x_max)
      ax.set_ylim(y_min, y_max)
      ax.set_xscale("log")
      ax.set_yscale("log")
      ## label axes
      ax.set_xlabel(r"$k$")
      ax.set_ylabel(r"$\mathcal{P}$")
      ## ###########
      ## SAVE FIGURE
      ## ######
      tmp_name = WWFnF.createFilepath([
        self.filepath_frames,
        WWFnF.createName([
          self.sim_name,
          "spectra={0:04}".format(int(time_index))
        ])+".png"
      ])
      plt.savefig(tmp_name)
      ## clear axis
      ax.clear()
    ## once plotting had finished - close figure
    plt.close()
  def aniSpectra(self):
    '''
    Animate the spectra frames.
    '''
    PlotFuncs.aniEvolution(
      self.filepath_frames,
      self.filepath_ani,
      WWFnF.createName([ self.sim_name, "spectra=%*.png" ]),
      WWFnF.createName([ self.sim_name, "ani_spectra.mp4" ])
    )


## END OF MODULE