## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np


## ###############################################################
## FUNCTIONS
## ###############################################################
class GetLabel():
  def frac(str_numer, str_denom):
    return r"$\dfrac{}{}$".format(
      "{" + str_numer.replace("$", "") + "}",
      "{" + str_denom.replace("$", "") + "}"
    )

  def percentiles(list_vals, num_sig_digits=1):
    perc_16 = np.percentile(list_vals, 16)
    perc_50 = np.percentile(list_vals, 50)
    perc_84 = np.percentile(list_vals, 84)
    diff_lo = perc_50 - perc_16
    diff_hi = perc_84 - perc_50
    str_minus = ("-{:."+str(num_sig_digits)+"g}").format(diff_lo)
    str_plus  = ("+{:."+str(num_sig_digits)+"g}").format(diff_hi)
    num_decimals_minus = 0
    num_decimals_plus  = 0
    if "." in str_minus: num_decimals_minus = len(str_minus.split(".")[1])
    if "." in str_plus:  num_decimals_plus  = len(str_plus.split(".")[1])
    num_decimals_val = max([
      num_decimals_minus,
      num_decimals_plus
    ])
    str_val = ("{:."+str(num_decimals_val)+"f}").format(perc_50)
    return r"${}_{}^{}$\;".format(
      str_val,
      "{" + str_minus + "}",
      "{" + str_plus + "}"
    )

  def spectrum(spect_field, spect_quantity=""):
    if   "tot" in spect_quantity: label_comp = ", tot"
    elif "lgt" in spect_quantity: label_comp = ", \parallel"
    elif "trv" in spect_quantity: label_comp = ", \perp"
    else:                         label_comp = ""
    return r"$\widehat{\mathcal{P}}_{\rm " + spect_field + label_comp + r"}(k,t)$"

  def timeAve(label):
    return r"$\langle$ " + label + r" $\rangle_{\forall t/t_{\rm turb}}$"

  def log10(label):
    return r"$\log_{10}\left(\right.$" + label + r"$\left.\right)$"


## END OF LIBRARY