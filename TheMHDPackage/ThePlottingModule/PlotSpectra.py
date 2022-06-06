## START OF MODULE


## ###############################################################
## MODULES
## ###############################################################
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from ThePlottingModule import PlotFuncs
from TheUsefulModule import WWLists, WWFnF


## ###############################################################
## PLOT TIME-AVERAGED SPECTRA
## ###############################################################
class PlotAveSpectra():
  def __init__(
      self,
      ax, spectra_fits_obj, time_range
    ):
    ## save figure axis
    self.ax = ax
    ## extract indices corresponding with the time range
    kin_sim_times = spectra_fits_obj.kin_sim_times
    mag_sim_times = spectra_fits_obj.mag_sim_times
    if len(kin_sim_times) > 0:
      bool_kin_spectra_fitted = True
      kin_index_start = WWLists.getIndexClosestValue(kin_sim_times, time_range[0])
      kin_index_end   = WWLists.getIndexClosestValue(kin_sim_times, time_range[1])
    else: bool_kin_spectra_fitted = False
    if len(mag_sim_times) > 0:
      bool_mag_spectra_fitted = True
      mag_index_start = WWLists.getIndexClosestValue(mag_sim_times, time_range[0])
      mag_index_end   = WWLists.getIndexClosestValue(mag_sim_times, time_range[1])
    else: bool_mag_spectra_fitted = False
    ## load spectra data
    if bool_kin_spectra_fitted:
      list_kin_power     = spectra_fits_obj.kin_list_power_group_t[kin_index_start : kin_index_end]
      list_kin_fit_power = spectra_fits_obj.kin_list_fit_power_group_t[kin_index_start : kin_index_end]
      bool_kin_spectra_fitted = len(list_kin_fit_power) > 0
    if bool_mag_spectra_fitted:
      self.list_k        = spectra_fits_obj.mag_list_k_group_t[0]
      list_mag_power     = spectra_fits_obj.mag_list_power_group_t[mag_index_start : mag_index_end]
      self.list_fit_k    = spectra_fits_obj.mag_list_fit_k_group_t[0]
      list_mag_fit_power = spectra_fits_obj.mag_list_fit_power_group_t[mag_index_start : mag_index_end]
      bool_mag_spectra_fitted = len(list_mag_fit_power) > 0
    ## calculate kinetic energy spectra plots
    if bool_kin_spectra_fitted:
      ## normalise spectra data
      list_norm_kin_power = [
        np.array(kin_power) / np.sum(kin_power)
            for kin_power in list_kin_power
      ]
      ## normalise spectra fit
      list_norm_kin_fit_power = [
        np.array(kin_fit_power) / np.sum(kin_power)
            for kin_fit_power, kin_power in zip(
              list_kin_fit_power, list_kin_power
            )
      ]
      ## plot kinematic spectra
      self.plotSpectra(
        list_power     = list_norm_kin_power,
        list_fit_power = list_norm_kin_fit_power,
        label = "kin-spectra",
        color = "blue"
      )
    ## calculate kinetic energy spectra plots
    if bool_mag_spectra_fitted:
      ## normalise spectra data
      list_norm_mag_power = [
        np.array(mag_power) / np.sum(mag_power)
            for mag_power in list_mag_power
      ]
      ## normalise spectra fit
      list_norm_mag_fit_power = [
        np.array(mag_fit_power) / np.sum(mag_power)
            for mag_fit_power, mag_power in zip(
              list_mag_fit_power, list_mag_power
            )
      ]
      ## plot magnetic spectra
      self.plotSpectra(
        list_power     = list_norm_mag_power,
        list_fit_power = list_norm_mag_fit_power,
        label = "mag-spectra",
        color = "red"
      )
    ## label and tune figure
    if bool_kin_spectra_fitted or bool_mag_spectra_fitted:
      ## add legend
      self.ax.legend(frameon=False, loc="lower left", fontsize=14)
      ## log axis
      self.ax.set_xlabel(r"$k$")
      self.ax.set_ylabel(r"$\mathcal{P}(k)$")
      self.ax.set_xscale("log")
      self.ax.set_yscale("log")
      ## add log axis-ticks
      PlotFuncs.addLogAxisTicks(ax, bool_major_ticks=True, max_num_major_ticks=5)
  def plotSpectra(
      self,
      list_power, list_fit_power,
      label, color
    ):
    ## plot time-averaged spectra
    self.ax.plot(
      self.list_k, np.median(list_power, axis=0),
      label=label, color=color, ls="-", lw=1.5, zorder=3
    )
    ## plot time-averaged spectra
    self.ax.fill_between(
      self.list_k,
      1.2 * np.percentile(list_power, 16, axis=0),
      1.2 * np.percentile(list_power, 84, axis=0),
      facecolor=color, alpha=0.3, zorder=1
    )
    ## plot time-averaged spectra fit
    self.ax.plot(
      self.list_fit_k, np.median(list_fit_power, axis=0),
      color="black", ls="-.", lw=2, zorder=5
    )

## ###############################################################
## SPECTRA STATISTICS OF SPECTRA FITS FOR A SINGLE SIMULATION
## ###############################################################
class PlotSpectraFit():
  ''' Plotting fitted spectra.
  '''
  def __init__(self, spectra_fits_obj):
    ## save spectra object
    self.spectra_fits_obj = spectra_fits_obj
    ## check that the spectra object has been labelled
    if (self.spectra_fits_obj.sim_suite is None) or (self.spectra_fits_obj.sim_label is None):
      raise Exception("Spectra object should have a suite ({:}) and label ({:}) defned.".format(
        self.spectra_fits_obj.sim_suite,
        self.spectra_fits_obj.sim_label
      ))
    ## save the times when both the kinetic energy and magnetic spectra were fitted
    self.sim_times = WWLists.getCommonElements(
      self.spectra_fits_obj.kin_sim_times,
      self.spectra_fits_obj.mag_sim_times
    )
    ## flags indicated in figure / animation name
    self.fig_flags = ""
    if self.spectra_fits_obj.bool_kin_fit_fixed_model:
      self.fig_flags += "_fk"
    if self.spectra_fits_obj.bool_mag_fit_fixed_model:
      self.fig_flags += "_fm"
  def plotSpectra_TargetTime(
      self,
      filepath_plot, target_time
    ):
    ## get fit index associated with the target time
    fit_index = WWLists.getIndexClosestValue(self.sim_times, target_time)
    ## create figure name
    fig_name = WWFnF.createName([
      self.spectra_fits_obj.sim_suite,
      self.spectra_fits_obj.sim_label,
      "check_SpectraFit={0:04}".format(int(fit_index))
    ]) + ".pdf"
    ## initialise spectra evolution figure
    fig, ax = plt.subplots(constrained_layout=True)
    ## plot spectra data
    self.plotAnnotatedSpectra(
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
      plot_index_start       = 0,
      plot_index_step        = 1,
      bool_delete_old_frames = False,
      bool_hide_updates      = False
    ):
    ## remove old frames
    if bool_delete_old_frames:
      print("\t> Deleting old spectra frames...")
      os.system("rm {}/*{}*{}*".format(
        filepath_plot,
        self.spectra_fits_obj.sim_label,
        self.fig_flags
      ))
    ## create frames of spectra evolution
    print("\t> Plotting spectra frames...")
    print("\t(Total of '{:d}' spectra fits. Plotting every '{:d}' fit(s) from fit-index '{:d}')".format(
        len(WWLists.getCommonElements(
          self.spectra_fits_obj.kin_sim_times,
          self.spectra_fits_obj.mag_sim_times
        )),
        plot_index_step,
        plot_index_start
      )
    )
    ## initialise spectra evolution figure
    fig, ax = plt.subplots()
    ## loop over each time slice
    for time_index in tqdm(
        range(plot_index_start, len(self.sim_times), plot_index_step),
        miniters = (len(self.sim_times) - plot_index_start) / 10,
        disable  = bool_hide_updates or (len(self.sim_times) < 3)
      ):
      self.plotAnnotatedSpectra(fig, ax, filepath_plot, time_index)
    ## close figure
    plt.close(fig)
  def plotAnnotatedSpectra(
      self,
      fig, ax, filepath_plot, time_index,
      fig_name = None,
      y_min = 1e-21,
      y_max = 1e2,
      x_min = 10**(-1),
      x_max = 300
    ):
    ## #################
    ## PLOT SPECTRA DATA
    ## #################
    ax.plot(
      self.spectra_fits_obj.kin_list_k_group_t[time_index],
      self.spectra_fits_obj.kin_list_power_group_t[time_index],
      label=r"kin-spectra", color="blue", ls="", marker=".", markersize=8
    )
    ax.plot(
      self.spectra_fits_obj.mag_list_k_group_t[time_index],
      self.spectra_fits_obj.mag_list_power_group_t[time_index],
      label=r"mag-spectra", color="red", ls="", marker=".", markersize=8
    )
    ## ####################
    ## PLOT FITTED SPECTRAS
    ## ####################
    ## plot fitted spectra
    ax.plot(
      self.spectra_fits_obj.kin_list_fit_k_group_t[time_index],
      self.spectra_fits_obj.kin_list_fit_power_group_t[time_index],
      label=r"kin-spectra (fitted)", color="blue", linestyle="--", dashes=(5, 2.5), linewidth=2
    )
    ax.plot(
      self.spectra_fits_obj.mag_list_fit_k_group_t[time_index],
      self.spectra_fits_obj.mag_list_fit_power_group_t[time_index],
      label=r"mag-spectra (fitted)", color="red", linestyle="--", dashes=(5, 2.5), linewidth=2
    )
    ## plot measured scales
    ax.axvline(x=self.spectra_fits_obj.k_nu_group_t[time_index],  ls="--", color="blue",  label=r"$k_\nu$")
    ax.axvline(x=self.spectra_fits_obj.k_eta_group_t[time_index], ls="--", color="red",   label=r"$k_\eta$")
    ax.axvline(x=self.spectra_fits_obj.k_p_group_t[time_index],   ls="--", color="black", label=r"$k_p$")
    ## #################
    ## ADD FIGURE LABELS
    ## #################
    ## kinetic energy spectra labels
    str_kin_spectra = r"$\mathcal{P}_{\rm kin}(k) = A_{\rm kin} k^{\alpha_{\rm kin}} \exp\left\{-\frac{k}{k_\nu}\right\}$"
    str_A_kin       = r"$A_{\rm kin} = $ "+"{:.2e}".format(self.spectra_fits_obj.kin_list_fit_params_group_t[time_index][0])
    str_alpha_kin   = r"$\alpha_\mathrm{kin} = $ "+"{:.2f}".format(self.spectra_fits_obj.kin_list_fit_params_group_t[time_index][1])
    str_k_nu        = r"$k_\nu = $ "+"{:.2f}".format(1 / self.spectra_fits_obj.k_nu_group_t[time_index])
    ## magnetic energy spectra labels
    str_mag_spectra = r"$\mathcal{P}_{\rm mag}(k) = A_{\rm mag} k^{\alpha_{\rm mag}} K_0\left\{-\frac{k}{k_\eta}\right\}$"
    str_A_mag       = r"$A_{\rm mag} = $ "+"{:.2e}".format(self.spectra_fits_obj.mag_list_fit_params_group_t[time_index][0])
    str_alpha_mag   = r"$\alpha_\mathrm{mag} = $ "+"{:.2f}".format(self.spectra_fits_obj.mag_list_fit_params_group_t[time_index][1])
    str_k_eta       = r"$k_\eta = $ "+"{:.2f}".format(1 / self.spectra_fits_obj.k_eta_group_t[time_index])
    str_k_p         = r"$k_p = $ "+"{:.2f}".format(self.spectra_fits_obj.k_p_group_t[time_index])
    PlotFuncs.plotLabelBox(
      fig, ax,
      ## box placement
      box_alignment = (0.0, 0.0),
      xpos = 0.025,
      ypos = 0.025,
      ## label appearance
      alpha    = 0.0,
      fontsize = 14,
      ## list of labels to place in box
      list_fig_labels = [
        str_kin_spectra,
        str_A_kin + r", " + str_alpha_kin + r", " + str_k_nu,
        str_mag_spectra,
        str_A_mag + r", " + str_alpha_mag + r", " + str_k_eta + r", " + str_k_p
      ]
    )
    ## add legend
    ax.legend(frameon=False, loc="upper left", facecolor="white", framealpha=0.0, fontsize=14)
    ## add time stamp
    ax.text(0.975, 0.975,
      r"$t / t_{\rm turb} = $ "+"{:.1f}".format(self.sim_times[time_index]), 
      va="top", ha="right", transform=ax.transAxes, fontsize=16
    )
    ## adjust figure axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## label axes
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\mathcal{P}(k)$")
    ## #############
    ## SAVE SNAPSHOT
    ## #############
    ## make sure that a name for the figure has been defined
    if fig_name is None:
      fig_name = WWFnF.createName([
        self.spectra_fits_obj.sim_suite,
        self.spectra_fits_obj.sim_label,
        "spectra_fit={0:04}".format(int(time_index)),
        self.fig_flags,
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
      filepath_frames,
      filepath_ani_movie,
      bool_hide_updates = False
    ):
    print("\t> Animating spectra frames...")
    PlotFuncs.aniEvolution(
      filepath_frames    = filepath_frames,
      filepath_ani_movie = filepath_ani_movie,
      input_name = WWFnF.createName([
        self.spectra_fits_obj.sim_suite,
        self.spectra_fits_obj.sim_label,
        "spectra_fit=%*",
        self.fig_flags
      ]) + ".png",
      output_name = WWFnF.createName([
        self.spectra_fits_obj.sim_suite,
        self.spectra_fits_obj.sim_label,
        "ani_spectra_fit",
        self.fig_flags
      ]) + ".mp4",
      bool_hide_updates  = bool_hide_updates
    )


## ###############################################################
## PLOT SPECTRA EVOLUTION
## ###############################################################
class PlotSpectra():
  '''
  Plotting raw spectra data.
  '''
  def __init__(self,
      kin_k, kin_power,
      mag_k, mag_power,
      sim_times,
      fig_name,
      filepath_frames,
      filepath_ani_movie
    ):
    self.kin_k              = kin_k
    self.mag_k              = mag_k
    self.kin_power          = kin_power
    self.mag_power          = mag_power
    self.sim_times          = sim_times
    self.fig_name           = fig_name
    self.filepath_frames    = filepath_frames
    self.filepath_ani_movie = filepath_ani_movie
  def plotSpectra(self, bool_hide_updates=False):
    '''
    Plot the evolution of the spectra.
    '''
    ## plot evolution of spectra
    y_min = 1e-20
    y_max = 10
    x_min = 10**(-1)
    x_max = max(len(self.kin_k[0]), len(self.mag_k[0]))
    ## initialise spectra evolution figure
    fig, ax = plt.subplots(constrained_layout=True)
    ## loop over each time slice
    for time_index in tqdm(
        range(len(self.sim_times)),
        disable = bool_hide_updates or (len(self.sim_times) < 3)
      ):
      ## #################
      ## PLOT SPECTRA DATA
      ## #################
      ax.plot(
        self.kin_k[time_index],
        self.kin_power[time_index],
        label=r"kin-spectra", color="blue", ls="", marker=".", markersize=8
      )
      ax.plot(
        self.mag_k[time_index],
        self.mag_power[time_index],
        label=r"mag-spectra", color="red", ls="", marker=".", markersize=8
      )
      ## ############
      ## LABEL FIGURE
      ## ############
      ## add time stamp
      ax.text(0.975, 0.975, 
        r"$t/t_{\rm turb} = $ "+"{:.1f}".format(self.sim_times[time_index]), 
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
      ## #############
      ## SAVE SNAPSHOT
      ## #############
      tmp_name = WWFnF.createFilepath([
        self.filepath_frames,
        WWFnF.createName([
          self.fig_name,
          "spectra={0:04}".format(int(time_index))
        ])+".png"
      ])
      plt.savefig(tmp_name)
      ## clear axis
      ax.clear()
    ## close figure once plotting has finished
    plt.close()
  def aniSpectra(self, bool_hide_updates=False):
    '''
    Animate the spectra frames.
    '''
    PlotFuncs.aniEvolution(
      filepath_frames    = self.filepath_frames,
      filepath_ani_movie = self.filepath_ani_movie,
      input_name         = WWFnF.createName([ self.fig_name, "spectra=%*.png" ]),
      output_name        = WWFnF.createName([ self.fig_name, "ani_spectra.mp4" ]),
      bool_hide_updates  = bool_hide_updates
    )


## END OF MODULE