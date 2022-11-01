import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

def interpLogData(x, y, x_interp):
  interpolator_linear = scipy.interpolate.interp1d(
    np.log10(x), np.log10(y), kind="linear"
  )
  return np.power(10.0, interpolator_linear(np.log10(x_interp)))

def main():
  x = [1, 2, 3] # np.linspace(x_min, x_max, num_points)
  y = np.exp(x)
  x_interp = np.logspace(np.log10(1.0), np.log10(3.0), 30)
  y_interp = interpLogData(x, y, x_interp)
  _, ax = plt.subplots()
  ax.plot(x, y, 'o', zorder=1)
  ax.plot(x_interp, y_interp, '.', zorder=3)
  ax.set_xscale("log")
  ax.set_yscale("log") 
  plt.show()

if __name__ == "__main__":
  main()

## end of demo program