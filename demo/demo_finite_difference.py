## ###############################################################
## MODULES
## ###############################################################
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from ThePlottingModule import PlotFuncs

def relError(rel_val, val):
  return np.where(
    rel_val == 0,
    (rel_val - val),
    (rel_val - val) / rel_val
  )

def getRelErrorStats(rel_val, val):
  rel_error = copy.deepcopy(relError(rel_val, val))
  rel_error_flat = rel_error.flatten()
  index_inset = 3
  return {
    "rel_error_abs_max" : np.nanmax(np.abs(rel_error_flat)),
    "rel_error_min"     : np.nanmin(rel_error_flat),
    "rel_error_max"     : np.nanmax(rel_error_flat),
    "rel_error_p16"     : np.nanpercentile(rel_error_flat, 16),
    "rel_error_p50"     : np.nanpercentile(rel_error_flat, 50),
    "rel_error_p84"     : np.nanpercentile(rel_error_flat, 84),
    "2norm"             : np.sum((
      rel_val[index_inset:-index_inset,index_inset:-index_inset]
      - val[index_inset:-index_inset,index_inset:-index_inset]
    )**2)
  }

class FirstDerivatives():
  ''' index: [row, col] '''

  def fd1o(field, cell_index, unit_step, Nres, inv_step, cell_width):
    ''' d(field)/d(step): first order (1o), forward difference (fd)
      inv_step = 1 / cell_width where cell_width = box_width / Nres
    '''
    a = inv_step
    a = 1 / cell_width
    return a * (
        field[(cell_index[0] + unit_step[0]) % Nres,
              (cell_index[1] + unit_step[1]) % Nres]
      - field[ cell_index[0], cell_index[1]]
    )

  def bd1o(field, cell_index, unit_step, Nres, inv_step, cell_width):
    ''' d(field)/d(step): first order (1o), backward difference (bd)
      inv_step = 1 / cell_width where cell_width = box_width / Nres
    '''
    a = inv_step
    a = 1 / cell_width
    return a * (
        field[ cell_index[0], cell_index[1]]
      - field[(cell_index[0] - unit_step[0]) % Nres,
              (cell_index[1] - unit_step[1]) % Nres]
    )

  def cd2o(field, cell_index, unit_step, Nres, inv_2step, cell_width):
    ''' d(field)/d(step): second order (2o), centered difference (cd)
      inv_2step = 1 / (2 * cell_width) where cell_width = box_width / Nres
    '''
    a = inv_2step
    a = 1 / (2 * cell_width)
    return a * (
        field[(cell_index[0] + unit_step[0]) % Nres,
              (cell_index[1] + unit_step[1]) % Nres]
      - field[(cell_index[0] - unit_step[0]) % Nres,
              (cell_index[1] - unit_step[1]) % Nres]
    )

  def cd4o(field, cell_index, unit_step, Nres, inv_12step, cell_width):
    ''' d(field)/d(step): second order (2o), centered difference (cd)
      inv_2step = 1 / (2 * cell_width) where cell_width = box_width / Nres
    '''
    a = inv_12step
    a = 1 / (12 * cell_width)
    return a * (
      -   field[(cell_index[0] + 2*unit_step[0]) % Nres,
                (cell_index[1] + 2*unit_step[1]) % Nres]
      + 8*field[(cell_index[0] +   unit_step[0]) % Nres,
                (cell_index[1] +   unit_step[1]) % Nres]
      - 8*field[(cell_index[0] -   unit_step[0]) % Nres,
                (cell_index[1] -   unit_step[1]) % Nres]
      +   field[(cell_index[0] - 2*unit_step[0]) % Nres,
                (cell_index[1] - 2*unit_step[1]) % Nres]
    )


class SecondDerivatives():
  ''' index: [row, col] '''

  def cd2o(field, cell_index, unit_step, Nres, inv_step_sq, cell_width):
    ''' d^2(field)/d(step)^2: second order (2o), centered difference (cd)
      inv_2step = 1 / (2 * cell_width**2) where cell_width = box_width / Nres
    '''
    a = inv_step_sq
    a = 1 / (2 * cell_width**2)
    return a * (
          field[(cell_index[0] + unit_step[0]) % Nres,
                (cell_index[1] + unit_step[1]) % Nres]
      - 2*field[cell_index[0], cell_index[1]]
      +   field[(cell_index[0] - unit_step[0]) % Nres,
                (cell_index[1] - unit_step[1]) % Nres]
    )

  def cd4o(field, cell_index, unit_step, Nres, inv_12step_sq, cell_width):
    ''' d^2(field)/d(step)^2: fouth order (4o), centered difference (cd)
      inv_2step = 1 / (2 * cell_width**2) where cell_width = box_width / Nres
    '''
    a = inv_12step_sq
    # a = 1 / (12 * cell_width**2)
    return a * (
      -    field[ (cell_index[0] + 2*unit_step[0]) % Nres,
                  (cell_index[1] + 2*unit_step[1]) % Nres]
      + 16*field[ (cell_index[0] +   unit_step[0]) % Nres,
                  (cell_index[1] +   unit_step[1]) % Nres]
      - 30*field[cell_index[0], cell_index[1]]
      + 16*field[ (cell_index[0] -   unit_step[0]) % Nres,
                  (cell_index[1] -   unit_step[1]) % Nres]
      -    field[ (cell_index[0] - 2*unit_step[0]) % Nres,
                  (cell_index[1] - 2*unit_step[1]) % Nres]
    )


def plot_df_dx(field_x, field_y, dfield_dx_exact, dfield_dx_approx, num_cells, field_name):
  cbar_bounds = [
    0.8 * np.min(dfield_dx_exact),
    1.2 * np.max(dfield_dx_exact)
  ]
  rel_error     = relError(dfield_dx_exact, dfield_dx_approx)
  dict_error    = getRelErrorStats(dfield_dx_exact, dfield_dx_approx)
  error_abs_max = dict_error["rel_error_abs_max"]
  error_min     = dict_error["rel_error_min"]
  error_max     = dict_error["rel_error_max"]
  error_p16     = dict_error["rel_error_p16"]
  error_p50     = dict_error["rel_error_p50"]
  error_p84     = dict_error["rel_error_p84"]
  two_norm      = dict_error["2norm"]
  print("Plotting first derivatives...")
  print("relative error:")
  print(f"\t> min: {error_min:.3f}")
  print(f"\t> perc-16: {error_p16:.3f}")
  print(f"\t> perc-50: {error_p50:.3f}")
  print(f"\t> perc-84: {error_p84:.3f}")
  print(f"\t> max: {error_max:.3f}")
  print(f"\t> 2-norm: {two_norm:.3f}")
  fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13,10))
  PlotFuncs.plot2DField(
    fig                 = fig,
    ax                  = axs[0,0],
    field_slice_x1      = field_x,
    field_slice_x2      = field_y,
    quiver_step         = num_cells // 10,
    NormType            = colors.Normalize,
    cbar_bounds         = [-0.5, 1.5],
    cmap_name           = "cmr.arctic_r",
    bool_plot_magnitude = True,
    bool_plot_quiver    = True,
    bool_add_colorbar   = True,
    cbar_title          = r"$||$" + field_name + r"$||$"
  )
  PlotFuncs.plot2DField(
    fig                 = fig,
    ax                  = axs[0,1],
    field_slice_x1      = dfield_dx_exact,
    field_slice_x2      = np.zeros_like(dfield_dx_exact),
    NormType            = colors.Normalize,
    cbar_bounds         = cbar_bounds,
    cmap_name           = "cmr.arctic_r",
    bool_plot_magnitude = True,
    bool_add_colorbar   = True,
    cbar_title          = r"exact ${\rm d}f/{\rm d}x$"
  )
  PlotFuncs.plot2DField(
    fig                 = fig,
    ax                  = axs[1,1],
    field_slice_x1      = dfield_dx_approx,
    field_slice_x2      = np.zeros_like(dfield_dx_approx),
    NormType            = colors.Normalize,
    cmap_name           = "cmr.arctic_r",
    cbar_bounds         = cbar_bounds,
    bool_plot_magnitude = True,
    bool_add_colorbar   = True,
    cbar_title          = r"approx. ${\rm d}f/{\rm d}x$"
  )
  PlotFuncs.plot2DField(
    fig                 = fig,
    ax                  = axs[1,0],
    field_slice_x1      = rel_error,
    field_slice_x2      = np.zeros_like(rel_error),
    NormType            = colors.Normalize,
    cmap_name           = "cmr.fusion",
    cbar_title          = "rel. error",
    cbar_bounds         = [ -error_abs_max, error_abs_max ],
    bool_plot_magnitude = True,
    bool_add_colorbar   = True,
  )
  ## save figure
  fig_name = f"demo_df_dx_ncells={num_cells:.0f}.png"
  fig.savefig(fig_name)
  plt.close(fig)
  print("Saved figure:", fig_name)
  print(" ")


def plot_d2f_dx2(field_x, field_y, d2field_dx2_exact, d2field_dx2_approx, num_cells, field_name):
  cbar_bounds = [
    0.8 * np.min(d2field_dx2_exact),
    1.2 * np.max(d2field_dx2_exact)
  ]
  rel_error     = relError(d2field_dx2_exact, d2field_dx2_approx)
  dict_error    = getRelErrorStats(d2field_dx2_exact, d2field_dx2_approx)
  error_abs_max = dict_error["rel_error_abs_max"]
  error_min     = dict_error["rel_error_min"]
  error_max     = dict_error["rel_error_max"]
  error_p16     = dict_error["rel_error_p16"]
  error_p50     = dict_error["rel_error_p50"]
  error_p84     = dict_error["rel_error_p84"]
  two_norm      = dict_error["2norm"]
  print("Plotting second derivatives...")
  print("relative error:")
  print(f"\t> min: {error_min:.3f}")
  print(f"\t> perc-16: {error_p16:.3f}")
  print(f"\t> perc-50: {error_p50:.3f}")
  print(f"\t> perc-84: {error_p84:.3f}")
  print(f"\t> max: {error_max:.3f}")
  print(f"\t> 2-norm: {two_norm:.3f}")
  fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13,10))
  PlotFuncs.plot2DField(
    fig                 = fig,
    ax                  = axs[0,0],
    field_slice_x1      = field_x,
    field_slice_x2      = field_y,
    quiver_step         = num_cells // 10,
    NormType            = colors.Normalize,
    cbar_bounds         = [-0.5, 1.5],
    cmap_name           = "cmr.arctic_r",
    bool_plot_magnitude = True,
    bool_plot_quiver    = True,
    bool_add_colorbar   = True,
    cbar_title          = r"$||$" + field_name + r"$||$"
  )
  PlotFuncs.plot2DField(
    fig                 = fig,
    ax                  = axs[0,1],
    field_slice_x1      = d2field_dx2_exact,
    field_slice_x2      = np.zeros_like(d2field_dx2_exact),
    NormType            = colors.Normalize,
    cbar_bounds         = cbar_bounds,
    cmap_name           = "cmr.arctic_r",
    bool_plot_magnitude = True,
    bool_add_colorbar   = True,
    cbar_title          = r"exact ${\rm d}^2f/{\rm d}x^2$"
  )
  PlotFuncs.plot2DField(
    fig                 = fig,
    ax                  = axs[1,1],
    field_slice_x1      = d2field_dx2_approx,
    field_slice_x2      = np.zeros_like(d2field_dx2_approx),
    NormType            = colors.Normalize,
    cbar_bounds         = cbar_bounds,
    cmap_name           = "cmr.arctic_r",
    bool_plot_magnitude = True,
    bool_add_colorbar   = True,
    cbar_title          = r"approx. ${\rm d}^2f/{\rm d}x^2$"
  )
  PlotFuncs.plot2DField(
    fig                 = fig,
    ax                  = axs[1,0],
    field_slice_x1      = rel_error,
    field_slice_x2      = np.zeros_like(rel_error),
    NormType            = colors.Normalize,
    cbar_bounds         = [ -error_abs_max, error_abs_max ],
    cbar_title          = "rel. error",
    cmap_name           = "cmr.fusion",
    bool_plot_magnitude = True,
    bool_add_colorbar   = True,
  )
  ## save figure
  fig_name = f"demo_d2f_dx2_ncells={num_cells:.0f}.png"
  fig.savefig(fig_name)
  plt.close(fig)
  print("Saved figure:", fig_name)
  print(" ")


def computeDerivatives(num_cells):
  ## define field
  x = np.linspace(-1.0, 1.0, num_cells)
  y = np.linspace(-1.0, 1.0, num_cells)
  X, Y = np.meshgrid(x, -y)
  field_name = r"$x^2$"
  field_x = X**4
  field_y = Y**2
  df_dx_exact = lambda x_, y_: 4*x_**3
  d2f_dx2_exact = lambda x_, y_: 12*x_**2
  ## compute grid information
  box_size_x   = np.max(x) - np.min(x)
  cell_width_x = box_size_x / num_cells
  inv_x        = 1 / cell_width_x
  inv_2x       = 1 / (2  * cell_width_x)
  inv_12x      = 1 / (12 * cell_width_x)
  inv_x_sq     = 1 / (2  * cell_width_x**2)
  inv_12x_sq   = 1 / (12 * cell_width_x**2)
  ## initialise exact derivatives
  dfield_dx_exact   = np.zeros_like(field_x)
  d2field_dx2_exact = np.zeros_like(field_x)
  ## initialise approx first derivatives
  dfield_dx_fd1o = np.zeros_like(field_x)
  dfield_dx_bd1o = np.zeros_like(field_x)
  dfield_dx_cd2o = np.zeros_like(field_x)
  dfield_dx_cd4o = np.zeros_like(field_x)
  ## initialise approx second derivatives
  d2field_dx2_cd2o = np.zeros_like(field_x)
  d2field_dx2_cd4o = np.zeros_like(field_x)
  ## compute derivatives
  message = f"Computing derivatives with num_cells = {num_cells}"
  print(message)
  print("="*len(message))
  print(" ")
  for index_x in range(num_cells):
    for index_y in range(num_cells):
      ## exact derivatives
      dfield_dx_exact[index_x, index_y]   = df_dx_exact(X[index_x, index_y], Y[index_x, index_y])
      d2field_dx2_exact[index_x, index_y] = d2f_dx2_exact(X[index_x, index_y], Y[index_x, index_y])
      ## df/dx first-order, forward difference
      dfield_dx_fd1o[index_x, index_y] = FirstDerivatives.fd1o(
        field      = field_x,
        cell_index = [index_x, index_y],
        unit_step  = [0, 1],
        Nres       = num_cells,
        cell_width = cell_width_x,
        inv_step   = inv_x
      )
      ## df/dx first-order, backward difference
      dfield_dx_bd1o[index_x, index_y] = FirstDerivatives.bd1o(
        field      = field_x,
        cell_index = [index_x, index_y],
        unit_step  = [0, 1],
        Nres       = num_cells,
        cell_width = cell_width_x,
        inv_step   = inv_x
      )
      ## df/dx second-order, centered difference
      dfield_dx_cd2o[index_x, index_y] = FirstDerivatives.cd2o(
        field      = field_x,
        cell_index = [index_x, index_y],
        unit_step  = [0, 1],
        Nres       = num_cells,
        cell_width = cell_width_x,
        inv_2step  = inv_2x
      )
      ## df/dx fourth-order, centered difference
      dfield_dx_cd4o[index_x, index_y] = FirstDerivatives.cd4o(
        field      = field_x,
        cell_index = [index_x, index_y],
        unit_step  = [0, 1],
        Nres       = num_cells,
        cell_width = cell_width_x,
        inv_12step = inv_12x
      )
      ## d^2/dx^2 second-order, centered difference
      d2field_dx2_cd2o[index_x, index_y] = SecondDerivatives.cd2o(
        field       = field_x,
        cell_index  = [index_x, index_y],
        unit_step   = [0, 1],
        Nres        = num_cells,
        cell_width  = cell_width_x,
        inv_step_sq = inv_x_sq
      )
      ## d^2/dx^2 fourth-order, centered difference
      d2field_dx2_cd4o[index_x, index_y] = SecondDerivatives.cd4o(
        field         = field_x,
        cell_index    = [index_x, index_y],
        unit_step     = [0, 1],
        Nres          = num_cells,
        cell_width    = cell_width_x,
        inv_12step_sq = inv_12x_sq
      )
  ## compute relative error statistics
  dict_error_stats = {
    "df_dx_fd1o"   : getRelErrorStats(dfield_dx_exact, dfield_dx_fd1o),
    "df_dx_bd1o"   : getRelErrorStats(dfield_dx_exact, dfield_dx_bd1o),
    "df_dx_cd2o"   : getRelErrorStats(dfield_dx_exact, dfield_dx_cd2o),
    "df_dx_cd4o"   : getRelErrorStats(dfield_dx_exact, dfield_dx_cd4o),
    "d2f_dx2_cd2o" : getRelErrorStats(d2field_dx2_exact, d2field_dx2_cd2o),
    "d2f_dx2_cd4o" : getRelErrorStats(d2field_dx2_exact, d2field_dx2_cd4o)
  }
  ## choose which approx derivatives to have a look at
  dfield_dx_approx   = dfield_dx_cd2o
  d2field_dx2_approx = d2field_dx2_cd2o
  ## plot data
  plot_df_dx(field_x,   field_y, dfield_dx_exact,   dfield_dx_approx,   num_cells, field_name)
  plot_d2f_dx2(field_x, field_y, d2field_dx2_exact, d2field_dx2_approx, num_cells, field_name)
  return dict_error_stats


def main():
  ## define helper plot function
  def plotErrorBar(ax, num_cells, list_stats, color, marker, label):
    if bool_labelled: label=None
    ax.plot(
      num_cells, list_stats["2norm"],
      color=color, marker=marker, ms=7, ls="", label=label
    )
  ## initialise figure
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
  box_width = 2
  list_num_cells = [ 10, 20, 50, 100, 500, 1000 ]
  bool_labelled = False
  for num_cells in list_num_cells:
    ## compute error information
    dict_error_stats = computeDerivatives(num_cells)
    ## plot scalings for first derivatives
    plotErrorBar(axs[0], box_width/num_cells, dict_error_stats["df_dx_fd1o"], "red",   "D", "1-order, forward dif.")
    plotErrorBar(axs[0], box_width/num_cells, dict_error_stats["df_dx_bd1o"], "blue",  "s", "1-order, backward dif.")
    plotErrorBar(axs[0], box_width/num_cells, dict_error_stats["df_dx_cd2o"], "green", "o", "2-order, centered dif.")
    plotErrorBar(axs[0], box_width/num_cells, dict_error_stats["df_dx_cd4o"], "black", "^", "4-order, centered dif.")
    ## plot scalings for second derivatives
    plotErrorBar(axs[1], box_width/num_cells, dict_error_stats["d2f_dx2_cd2o"], "green", "o", "2-order, centered dif.")
    plotErrorBar(axs[1], box_width/num_cells, dict_error_stats["d2f_dx2_cd4o"], "black", "^", "4-order, centered dif.")
    bool_labelled = True
    print(" ")
  ## adjust figure axis
  x = np.logspace(-5, 0, 100)
  list_weight = [ 0.01, 1 ]
  for col_index in range(2):
    PlotFuncs.plotData_noAutoAxisScale(
      ax    = axs[col_index],
      x     = x,
      y     = list_weight[col_index] * x**(-1),
      ls    = "-",
      lw    = 2,
      label = r"$\propto (\Delta x)^{-1}$"
    )
    PlotFuncs.plotData_noAutoAxisScale(
      ax    = axs[col_index],
      x     = x,
      y     = list_weight[col_index] * x**(-2),
      ls    = ":",
      lw    = 2,
      label = r"$\propto (\Delta x)^{-2}$"
    )
    axs[col_index].set_xscale("log")
    axs[col_index].set_xscale("log")
    axs[col_index].set_yscale("log")
    axs[col_index].set_xlabel(r"$\Delta x$")
    axs[col_index].set_ylabel("2-norm")
  ## add legend
  PlotFuncs.addLegend(
    ax   = axs[0],
    loc  = "lower left",
    bbox = (0.01, 0.01),
    fontsize = 16
  )
  PlotFuncs.addLegend(
    ax   = axs[1],
    loc  = "upper right",
    bbox = (0.99, 0.99),
    fontsize = 16
  )
  ## save figure
  fig_name = f"demo_differencing_scaling.png"
  fig.savefig(fig_name)
  plt.close(fig)
  print("Saved figure:", fig_name)


## PROGRAM ENTRY POINT
if __name__ == "__main__":
  main()


## END OF PROGRAM