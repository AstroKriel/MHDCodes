## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np

## load user routines
from TheUsefulModule import WWLists


## ###############################################################
## FUNCTIONS
## ###############################################################
def getString(val, num_sig_digits):
  if val < 1.0: return ("{:."+str(num_sig_digits)+"g}").format(val)
  else: return str(round(val))

class GetLabel():
  def frac(str_numer, str_denom):
    return r"$\dfrac{}{}$".format(
      "{" + str_numer.replace("$", "") + "}",
      "{" + str_denom.replace("$", "") + "}"
    )
  
  def modes(list_vals, num_sig_digits=1):
    if len([val for val in list_vals if val is not None]) < 5: return
    list_vals = WWLists.replaceNoneWNan(list_vals)
    val_std   = np.nanstd(list_vals)
    val_mean  = np.nanmean(list_vals)
    str_std   = getString(val_std, num_sig_digits)
    num_decimals_val = 0
    if "." in str_std: num_decimals_val = len(str_std.split(".")[1])
    str_mean = ("{:."+str(num_decimals_val)+"f}").format(val_mean)
    return str_mean + r" $\pm$ " + str_std

  def percentiles(list_vals, num_sig_digits=1):
    list_vals = WWLists.replaceNoneWNan(list_vals)
    val_perc_16 = np.nanpercentile(list_vals, 16)
    val_perc_50 = np.nanpercentile(list_vals, 50)
    val_perc_84 = np.nanpercentile(list_vals, 84)
    diff_lo = val_perc_50 - val_perc_16
    diff_hi = val_perc_84 - val_perc_50
    str_minus = "-" + getString(diff_lo, num_sig_digits)
    str_plus  = "+" + getString(diff_hi, num_sig_digits)
    num_decimals_minus = 0
    num_decimals_plus  = 0
    if "." in str_minus: num_decimals_minus = len(str_minus.split(".")[1])
    if "." in str_plus:  num_decimals_plus  = len(str_plus.split(".")[1])
    num_decimals_val = max([
      num_decimals_minus,
      num_decimals_plus
    ])
    str_perc_50 = ("{:."+str(num_decimals_val)+"f}").format(val_perc_50)
    return r"${}_{}^{}$\;".format(
      str_perc_50,
      "{" + str_minus + "}",
      "{" + str_plus + "}"
    )

  def spectrum(spect_field, spect_comp=""):
    if   "tot" in spect_comp: label_comp = ", tot"
    elif "lgt" in spect_comp: label_comp = ", \parallel"
    elif "trv" in spect_comp: label_comp = ", \perp"
    else:                     label_comp = ""
    return r"$\widehat{\mathcal{P}}_{\rm " + spect_field + label_comp + r"}(k,t)$"

  def timeAve(label):
    return r"$\langle$ " + label + r" $\rangle_{\forall t/t_{\rm turb}}$"

  def log10(label):
    return r"$\log_{10}\left(\right.$" + label + r"$\left.\right)$"


## END OF LIBRARY