## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np


## ###############################################################
## FUNCTIONS
## ###############################################################
class GetLabel():
  @classmethod
  def frac(cls, str_numer, str_denom):
    return r"$\dfrac{}{}$".format(
      "{" + str_numer.replace("$", "") + "}",
      "{" + str_denom.replace("$", "") + "}"
    )

  @classmethod
  def percentiles(cls, list_vals, num_sig_digits=1):
    perc_16 = np.percentile(list_vals, 16)
    perc_50 = np.percentile(list_vals, 50)
    perc_84 = np.percentile(list_vals, 84)
    diff_lo = perc_50 - perc_16
    diff_hi = perc_84 - perc_50
    str_lo = ("-{:."+str(num_sig_digits)+"g}").format(diff_lo)
    str_hi = ("+{:."+str(num_sig_digits)+"g}").format(diff_hi)
    num_decimals_lo = 0
    num_decimals_hi = 0
    if "." in str_lo: num_decimals_lo = len(str_lo.split(".")[1])
    if "." in str_hi: num_decimals_hi = len(str_hi.split(".")[1])
    num_decimals = max([num_decimals_lo, num_decimals_hi])
    str_val = ("{:."+str(num_decimals)+"f}").format(perc_50)
    return r"${}_{}^{}$\;".format(
      str_val,
      "{" + str_lo + "}",
      "{" + str_hi + "}"
    )

  @classmethod
  def spectrum(cls, spect_field, spect_quantity=""):
    if   "tot" in spect_quantity: label_comp = ", tot"
    elif "lgt" in spect_quantity: label_comp = ", \parallel"
    elif "trv" in spect_quantity: label_comp = ", \perp"
    else:                         label_comp = ""
    return r"$\widehat{\mathcal{P}}_{\rm " + spect_field + label_comp + r"}(k,t)$"

  @classmethod
  def timeAve(cls, label):
    return r"$\langle$ " + label + r" $\rangle_{\forall t/t_{\rm turb}}$"

  @classmethod
  def log10(cls, label):
    return r"$\log_{10}\left(\right.$" + label + r"$\left.\right)$"


## END OF LIBRARY