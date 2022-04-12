import numpy as np

## load old user defined modules
from the_matplotlib_styler import *
from OldModules.the_useful_library import *
from OldModules.the_fitting_library import *
from OldModules.the_plotting_library import *


def funcFitScale(
        fit_args_ax,
    ):
    ## fit with linear line
    plotKDEFit(
        **fit_args_ax,
        func_label = "linear",
        func_fit   = ListOfModels.linear,
        func_plot  = ListOfModels.linear,
        num_resamp = 10**3,
        num_fit    = 10**3,
        plot_args  = { "x":0.05, "y":0.95, "va":"top", "ha":"left", "color":"black" }
    )
    ## fit with power-law
    plotKDEFit(
        **fit_args_ax,
        func_label = "PowerLaw",
        func_fit   = ListOfModels.powerlaw_log10,
        func_plot  = ListOfModels.powerlaw_linear,
        bool_log_fit = True,
        list_indices_unlog = [0],
        num_resamp = 10**3,
        num_fit    = 10**3,
        bounds     = ((np.log(1e-2), 0), (np.log(1), 2)),
        p0         = (np.log(0.2), 1),
        num_digits = 2,
        plot_args  = { "x":0.95, "y":0.05, "va":"bottom", "ha":"right", "color":"blue" }
    )

def main():
    ## generate data
    print("generating data...")
    num_points  = 5
    list_data_x = np.logspace(1, num_points, num_points)
    list_data_y = [
        np.logspace(point_index, 0.5 + point_index, num_points)
        for point_index in range(num_points)
    ]
    ## initialise figure
    fig, ax = plt.subplots(figsize=(7, 4))
    ## plot scale distributions
    print("plotting data...")
    for point_index in range(num_points):
        plotErrorBar(
            ax,
            data_x = list_data_x[point_index],
            data_y = list_data_y[point_index],
            marker = "o"
        )
    ## fit
    print("fitting data...")
    funcFitScale(
        {
            "ax":ax,
            "var_str":r"$x$",
            "input_x":list_data_x,
            "input_y":list_data_y
        }
    )
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## save plot
    plt.savefig("fitting_log.png")
    print("saved figure.")

if __name__ == "__main__":
    main()
    sys.exit()

