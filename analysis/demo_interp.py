import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def f_data(x, A, nu, k):
    return A * np.exp(-k*x) * np.cos(2*np.pi * nu * x)

fig, ax = plt.subplots()
list_params = 10, 4, 2
array_x = np.linspace(0, 0.5, 8)
array_y = f_data(array_x, *list_params)
ax.plot(array_x, array_y, ls="", marker="o", c="black", label="raw data")

list_plot_styles = [ "b-", "r-", "g-", "k-" ]
list_interp_kind = [ "nearest", "linear", "quadratic", "cubic" ]
x_interp = np.linspace(min(array_x), max(array_x), 100)
for interp_kind, plot_style in zip(list_interp_kind, list_plot_styles):
    y_interp = interp1d(array_x, array_y, kind=interp_kind)(x_interp)
    ax.plot(x_interp, y_interp, plot_style, label=interp_kind)
plt.legend()
plt.show()
