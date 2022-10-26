## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from ThePlottingModule import PlotFuncs
from TheUsefulModule import WWLists, WWFnF


## ###############################################################
## PLOT TIME-AVERAGED SPECTRA
## ###############################################################
class PlotAveSpectraFit():
  def __init__(
      self,
      ax, fits_obj, time_range
    ):
    ## save figure axis
    self.ax = ax
    ## extract indices corresponding with the time range
    kin_list_sim_times = fits_obj.kin_list_sim_times
    mag_list_sim_times = fits_obj.mag_list_sim_times
    ## plot kinetic energy spectra
    if len(kin_list_sim_times) > 0:
      kin_time_index_start = WWLists.getIndexClosestValue(kin_list_sim_times, time_range[0])
      kin_time_index_end   = WWLists.getIndexClosestValue(kin_list_sim_times, time_range[1])
      ## plot kinematic spectra
      self.__plotSpectra(
        label                  = "kin-spectra",
        color                  = "blue",
        list_k                 = fits_obj.kin_list_k_group_t[0],
        list_power_group_t     = fits_obj.kin_list_power_group_t[kin_time_index_start : kin_time_index_end],
        list_fit_k             = fits_obj.kin_list_fit_k_group_t[0],
        list_fit_power_group_t = fits_obj.kin_list_fit_power_group_t[kin_time_index_start : kin_time_index_end],
        min_fit_k_index        = fits_obj.kin_k_index_fit_from,
        max_fit_k              = np.median(
          fits_obj.kin_max_k_mode_fitted_group_t[kin_time_index_start : kin_time_index_end]
        )
      )
    ## plot magnetic energy spectra
    if len(mag_list_sim_times) > 0:
      mag_time_index_start = WWLists.getIndexClosestValue(mag_list_sim_times, time_range[0])
      mag_time_index_end   = WWLists.getIndexClosestValue(mag_list_sim_times, time_range[1])
      ## plot magnetic spectra
      self.__plotSpectra(
        label                  = "mag-spectra",
        color                  = "red",
        list_k                 = fits_obj.mag_list_k_group_t[0],
        list_power_group_t     = fits_obj.mag_list_power_group_t[mag_time_index_start : mag_time_index_end],
        list_fit_k             = fits_obj.mag_list_fit_k_group_t[0],
        list_fit_power_group_t = fits_obj.mag_list_fit_power_group_t[mag_time_index_start : mag_time_index_end],
        min_fit_k_index        = fits_obj.mag_k_index_fit_from,
        max_fit_k              = np.median(
          fits_obj.mag_max_k_mode_fitted_group_t[mag_time_index_start : mag_time_index_end]
        )
      )
    ## label and tune figure
    ## add legend
    self.ax.legend(frameon=False, loc="lower left", fontsize=14)
    ## log axis
    self.ax.set_xlabel(r"$k$")
    self.ax.set_ylabel(r"$\mathcal{P}(k)$")
    self.ax.set_xscale("log")
    self.ax.set_yscale("log")
    ## add log axis-ticks
    PlotFuncs.addLogAxisTicks(ax, bool_major_ticks=True, num_major_ticks=5)

  def __plotSpectra(
      self,
      list_k, list_power_group_t,
      list_fit_k, list_fit_power_group_t,
      min_fit_k_index, max_fit_k,
      label, color
    ):
    ## normalise spectra data
    list_power_norm_group_t = [
      np.array(list_power) / np.sum(list_power)
      for list_power in list_power_group_t
    ]
    ## normalise spectra fit
    list_fit_power_norm_group_t = [
      np.array(list_fit_power) / np.sum(list_power)
      for list_fit_power, list_power in zip(
        list_fit_power_group_t, list_power_group_t
      )
    ]
    ## plot time-averaged spectra
    self.ax.plot(
      list_k, np.median(list_power_norm_group_t, axis=0),
      label=label, color=color, ls="-", lw=1.5, zorder=3
    )
    ## plot time-averaged spectra
    self.ax.fill_between(
      list_k,
      np.percentile(list_power_norm_group_t, 16, axis=0),
      np.percentile(list_power_norm_group_t, 84, axis=0),
      facecolor=color, alpha=0.3, zorder=1
    )
    ## find fit subset index range
    fit_k_index_start       = WWLists.getIndexClosestValue(list_fit_k, list_k[min_fit_k_index])
    fit_k_index_end         = WWLists.getIndexClosestValue(list_fit_k, max_fit_k)
    list_fit_power_norm_ave = np.median(list_fit_power_norm_group_t, axis=0)[fit_k_index_start : fit_k_index_end]
    ## plot time-averaged spectra fit
    self.ax.plot(
      list_fit_k[fit_k_index_start : fit_k_index_end],
      list_fit_power_norm_ave,
      color="black", ls="-.", lw=2, zorder=5
    )
    ## plot first data point
    self.ax.plot(
      list_fit_k[fit_k_index_start : fit_k_index_end][0],
      list_fit_power_norm_ave[0],
      color="black", marker="o", ms=5, zorder=7
    )
    ## plot last data point
    self.ax.plot(
      list_fit_k[fit_k_index_start : fit_k_index_end][-1],
      list_fit_power_norm_ave[-1],
      color="black", marker="o", ms=5, zorder=7
    )


## ###############################################################
## SPECTRA STATISTICS OF SPECTRA FITS FOR A SINGLE SIMULATION
## ###############################################################
class PlotSpectraFit():
  ''' Plotting fitted spectra.
  '''
  def __init__(self, fits_obj):
    ## save spectra object
    self.fits_obj = fits_obj
    ## check that the spectra object has been labelled
    if (self.fits_obj.sim_suite is None) or (self.fits_obj.sim_label is None):
      Exception("Spectra object should have a suite ({:}) and label ({:}) defned.".format(
        self.fits_obj.sim_suite,
        self.fits_obj.sim_label
      ))
    ## save the times when both the kinetic energy and magnetic spectra were fitted
    self.sim_times = WWLists.getCommonElements(
      self.fits_obj.kin_list_sim_times,
      self.fits_obj.mag_list_sim_times
    )
    self.fig_tags = ""

  def createFigTag(self, bool_plot_kin, bool_plot_mag):
    fig_tags = ""
    ## plot kinetic energy spectra
    if bool_plot_kin:
      fig_tags += "_kin"
      if not(bool_plot_mag):
        fig_tags += "_only"
      ## fitted with fixed model
      if self.fits_obj.kin_bool_fit_fixed_model:
        fig_tags += "_fk"
    ## plot magnetic energy spectra
    if bool_plot_mag:
      fig_tags += "_mag"
      if not(bool_plot_kin):
        fig_tags += "_only"
      ## fitted with fixed model
      if self.fits_obj.mag_bool_fit_fixed_model:
        fig_tags += "_fm"
    ## return label annotation (tags)
    return fig_tags

  def plotSpectra_TargetTime(
      self,
      filepath_vis, target_time
    ):
    ## get fit index associated with the target time
    fit_index = WWLists.getIndexClosestValue(self.sim_times, target_time)
    ## create figure name
    fig_name = WWFnF.createName([
      self.fits_obj.sim_suite,
      self.fits_obj.sim_label,
      "check_SpectraFit={0:04}".format(int(fit_index))
    ]) + ".pdf"
    ## initialise spectra evolution figure
    fig, ax = plt.subplots(constrained_layout=True)
    ## plot spectra data
    self.__plotAnnotatedSpectra(
      fig, ax,
      filepath_vis, fit_index, fig_name,
      y_min = 1e-18,
      y_max = 1e2,
      x_min = 0.1,
      x_max = 200
    )
    ## close figure
    plt.close(fig)
    ## print information to the terminal
    print("\t> Figure saved:", fig_name)

  def plotSpectraEvolution(
      self,
      filepath_frames, filepath_movie,
      plot_index_start       = 0,
      plot_index_step        = 1,
      bool_plot_kin          = True,
      bool_plot_mag          = True,
      bool_adjust_y_axis     = True,
      bool_hide_updates      = False
    ):
    ## reset flags (this is gross, but is necessary to keep files organised)
    self.fig_tags = self.createFigTag(bool_plot_kin, bool_plot_mag)
    ## initialise spectra evolution figure
    fig, ax = plt.subplots()
    ## loop over each time slice
    for time_index in tqdm(
        range(plot_index_start, len(self.sim_times), plot_index_step),
        miniters = (len(self.sim_times) - plot_index_start) / 10,
        disable  = bool_hide_updates or (len(self.sim_times) < 3)
      ):
      ## plot annotated figure
      self.__plotAnnotatedSpectra(
        fig                = fig,
        ax                 = ax,
        filepath_vis      = filepath_frames,
        time_index         = time_index,
        bool_plot_kin      = bool_plot_kin,
        bool_plot_mag      = bool_plot_mag,
        bool_adjust_y_axis = bool_adjust_y_axis
      )
    ## close figure object
    plt.close(fig)
    ## animate frames
    self.__aniSpectra(
      filepath_frames = filepath_frames,
      filepath_movie  = filepath_movie
    )

  def __plotAnnotatedSpectra(
      self,
      fig, ax, filepath_vis, time_index,
      fig_name           = None,
      y_min              = 1e-21,
      y_max              = 1e2,
      x_min              = 10**(-1),
      x_max              = 300,
      bool_plot_kin      = True,
      bool_plot_mag      = True,
      bool_adjust_y_axis = True
    ):
    ## #################
    ## PLOT SPECTRA DATA
    ## #################
    if bool_plot_kin:
      ## plot data
      ax.plot(
        self.fits_obj.kin_list_k_group_t[time_index],
        self.fits_obj.kin_list_power_group_t[time_index],
        color="blue", ls="", marker=".", markersize=8, alpha=0.3, zorder=3
      )
      ## plot fitted data
      ax.plot(
        self.fits_obj.kin_list_k_group_t[time_index][
          int(self.fits_obj.kin_k_index_fit_from) :
          int(self.fits_obj.kin_max_k_mode_fitted_group_t[time_index])
        ],
        self.fits_obj.kin_list_power_group_t[time_index][
          int(self.fits_obj.kin_k_index_fit_from) :
          int(self.fits_obj.kin_max_k_mode_fitted_group_t[time_index])
        ],
        label=r"kin-spectra", color="blue", ls="", marker=".", markersize=8, zorder=5
      )
    if bool_plot_mag:
      ## plot data
      ax.plot(
        self.fits_obj.mag_list_k_group_t[time_index],
        self.fits_obj.mag_list_power_group_t[time_index],
        color="red", ls="", marker=".", markersize=8, alpha=0.3, zorder=3
      )
      ## plot fitted data
      ax.plot(
        self.fits_obj.mag_list_k_group_t[time_index][
          int(self.fits_obj.mag_k_index_fit_from) :
          int(self.fits_obj.mag_max_k_mode_fitted_group_t[time_index])
        ],
        self.fits_obj.mag_list_power_group_t[time_index][
          int(self.fits_obj.mag_k_index_fit_from) :
          int(self.fits_obj.mag_max_k_mode_fitted_group_t[time_index])
        ],
        label=r"mag-spectra", color="red", ls="", marker=".", markersize=8, zorder=5
      )
    ## ####################
    ## PLOT FITTED SPECTRAS
    ## ####################
    ## plot fitted spectra
    if bool_plot_kin:
      ax.plot(
        self.fits_obj.kin_list_fit_k_group_t[time_index],
        self.fits_obj.kin_list_fit_power_group_t[time_index],
        label=r"kin-spectra (fitted)", color="blue", linestyle="--", dashes=(5, 2.5), linewidth=2, scaley=False
      )
    if bool_plot_mag:
      ax.plot(
        self.fits_obj.mag_list_fit_k_group_t[time_index],
        self.fits_obj.mag_list_fit_power_group_t[time_index],
        label=r"mag-spectra (fitted)", color="red", linestyle="--", dashes=(5, 2.5), linewidth=2, scaley=False
      )
    ## plot measured scales
    if bool_plot_kin:
      ax.axvline(x=self.fits_obj.k_nu_group_t[time_index], ls="--", color="blue", label=r"$k_\nu$")
    if bool_plot_mag:
      k_eta         = self.fits_obj.k_eta_group_t[time_index]
      k_eta_alpha_2 = k_eta**( 1 / self.fits_obj.mag_list_fit_params_group_t[time_index][2] )
      k_p           = self.fits_obj.k_p_group_t[time_index]
      k_max         = self.fits_obj.k_max_group_t[time_index]
      ax.axvline(x=k_eta,         ls="--", color="red",    label=r"$k_\eta$")
      ax.axvline(x=k_eta_alpha_2, ls="--", color="black",  label=r"$k_\eta^{\alpha_{{\rm mag}, 2}}$")
      ax.axvline(x=k_p,           ls="--", color="green",  label=r"$k_{\rm p}$")
      ax.axvline(x=k_max,         ls="--", color="purple", label=r"$k_{\rm max}$")
    ## #################
    ## ADD FIGURE LABELS
    ## #################
    list_fig_labels = []
    ## kinetic energy spectra labels
    if bool_plot_kin:
      str_kin_spectra  = r"$\mathcal{P}_{\rm kin}(k) = A_{\rm kin} k^{\alpha_{\rm kin}} \exp\left\{-\frac{k}{k_\nu}\right\}$"
      val_A_kin        = self.fits_obj.kin_list_fit_params_group_t[time_index][0]
      val_alpha_kin    = self.fits_obj.kin_list_fit_params_group_t[time_index][1]
      val_k_nu         = self.fits_obj.k_nu_group_t[time_index]
      str_A_kin        = r"$A_{\rm kin} = $ "+"{:.1e}".format(val_A_kin)
      str_alpha_kin    = r"$\alpha_{\rm kin} = $ "+"{:.1f}".format(val_alpha_kin)
      str_k_nu         = r"$k_\nu = $ "+"{:.1f}".format(val_k_nu)
      list_fig_labels += [
        str_kin_spectra,
        rf"{str_A_kin}, {str_alpha_kin}, {str_k_nu}"
      ]
    ## magnetic energy spectra labels
    if bool_plot_mag:
      str_mag_spectra   = r"$\mathcal{P}_{\rm mag}(k) = A_{\rm mag} k^{\alpha_{{\rm mag}, 1}} \exp\left\{ -\left(\frac{k}{k_\eta}\right)^{\alpha_{{\rm mag}, 2}} \right\}$"
      val_A_mag         = self.fits_obj.mag_list_fit_params_group_t[time_index][0]
      val_alpha_mag_1   = self.fits_obj.mag_list_fit_params_group_t[time_index][1]
      val_alpha_mag_2   = self.fits_obj.mag_list_fit_params_group_t[time_index][2]
      val_k_eta         = self.fits_obj.k_eta_group_t[time_index]
      val_k_eta_alpha_2 = val_k_eta**(1/val_alpha_mag_2)
      val_k_p           = self.fits_obj.k_p_group_t[time_index]
      val_k_max         = self.fits_obj.k_max_group_t[time_index]
      str_A_mag         = r"$A_{\rm mag} = $ "+"{:.2e}".format(val_A_mag)
      str_alpha_mag_1   = r"$\alpha_{{\rm mag}, 1} = $ "+"{:.1f}".format(val_alpha_mag_1)
      str_alpha_mag_2   = r"$\alpha_{{\rm mag}, 2} = $ "+"{:.1f}".format(val_alpha_mag_2)
      str_k_eta         = r"$k_\eta = $ "+"{:.2f}".format(val_k_eta)
      str_k_eta_alpha_2 = r"$k_\eta^{1/\alpha_{{\rm mag}, 2}} = $ "+"{:.2f}".format(val_k_eta_alpha_2)
      str_k_p           = r"$k_{\rm p} = $ "+"{:.1f}".format(val_k_p)
      str_k_max         = r"$k_{\rm max} = $ "+"{:.1f}".format(val_k_max)
      list_fig_labels  += [
        str_mag_spectra,
        rf"{str_alpha_mag_1}, {str_alpha_mag_2}, {str_k_eta}, {str_k_eta_alpha_2}, {str_k_p}"
      ]
    PlotFuncs.plotBoxOfLabels(
      fig, ax,
      box_alignment   = (0.0, 0.0),
      xpos            = 0.025,
      ypos            = 0.025,
      alpha           = 0.0,
      fontsize        = 14,
      list_fig_labels = list_fig_labels
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
    if bool_adjust_y_axis:
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
        self.fits_obj.sim_suite,
        self.fits_obj.sim_label,
        "spectra_fit={0:04}".format(int(time_index)),
        self.fig_tags,
      ]) + ".png"
    ## save the figure
    fig.savefig(
      WWFnF.createFilepath([ filepath_vis, fig_name ]),
      dpi = 150
    )
    ## clear figure and axis
    fig.artists.clear()
    ax.clear()

  def __aniSpectra(
      self,
      filepath_frames,
      filepath_movie
    ):
    PlotFuncs.aniEvolution(
      filepath_frames = filepath_frames,
      filepath_movie  = filepath_movie,
      input_name = WWFnF.createName([
        self.fits_obj.sim_suite,
        self.fits_obj.sim_label,
        "spectra_fit=%*",
        self.fig_tags
      ]) + ".png",
      output_name = WWFnF.createName([
        self.fits_obj.sim_suite,
        self.fits_obj.sim_label,
        "spectra_fit",
        self.fig_tags
      ]) + ".mp4"
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
      filepath_movie
    ):
    self.kin_k              = kin_k
    self.mag_k              = mag_k
    self.kin_power          = kin_power
    self.mag_power          = mag_power
    self.sim_times          = sim_times
    self.fig_name           = fig_name
    self.filepath_frames    = filepath_frames
    self.filepath_movie     = filepath_movie

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
    _, ax = plt.subplots()
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
        r"$t / t_{\rm turb} = $ "+"{:.1f}".format(self.sim_times[time_index]), 
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
      ## save the figure
      plt.savefig(tmp_name, dpi=150)
      ## clear axis
      ax.clear()
    ## close figure once plotting has finished
    plt.close()
    ## animate frames
    self.__aniSpectra()

  def __aniSpectra(self):
    '''
    Animate the spectra frames.
    '''
    PlotFuncs.aniEvolution(
      filepath_frames = self.filepath_frames,
      filepath_movie  = self.filepath_movie,
      input_name      = WWFnF.createName([ self.fig_name, "spectra=%*.png" ]),
      output_name     = WWFnF.createName([ self.fig_name, "spectra.mp4" ])
    )


## END OF LIBRARY