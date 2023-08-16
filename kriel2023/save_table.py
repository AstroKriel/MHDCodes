#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import sys

## load user defined modules
from TheUsefulModule import WWObjs


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def createMeasurement_float(val):
  str_val = f"{val:.1e}"
  amplitude = str_val.split("e")[0]
  exponent  = str_val.split("e")[1]
  if abs(int(exponent)) > 0:
    return f"{amplitude} \\times 10^{{{int(exponent)}}}"
  else: return f"{amplitude}"

def createMeasurement_error(elem, num_decimals=None):
  if (num_decimals is None):
    std = "{:.1g}".format(elem["std"])
    if ("." in std):
      num_decimals = len(std.split(".")[1])
    else: num_decimals = 0
  else: std = "{:.{}f}".format(elem["std"], num_decimals)
  val = "{:.{}f}".format(elem["val"], num_decimals)
  return f"{val} \pm {std}"

def createMeasurement_error_nres(elem, idx, num_decimals=None):
  if (num_decimals is None):
    std = "{:.1g}".format(elem["fit_std"][idx])
    if ("." in std):
      num_decimals = len(std.split(".")[1])
    else: num_decimals = 0
  else: std = "{:.{}f}".format(elem["fit_std"][idx], num_decimals)
  val = "{:.{}f}".format(elem["fit_params"][idx], num_decimals)
  return f"{val} \pm {std}"

def printMainData(fp, dict_all_data, sim_name):
  dict_sim_data = dict_all_data[sim_name]
  Re            = int(dict_sim_data["Re"])
  Rm            = int(dict_sim_data["Rm"])
  Pm            = int(dict_sim_data["Pm"])
  sim_id        = "$\\mathcal{M}$" + sim_name.split("Mach")[1].split("_")[0] + f"Re{Re:d}Pm{Pm:d}"
  nu            = createMeasurement_float(dict_sim_data["nu"])
  eta           = createMeasurement_float(dict_sim_data["eta"])
  Mach          = createMeasurement_error(dict_sim_data["Mach"])
  E_growth_rate = createMeasurement_error(dict_sim_data["E_growth_rate"])
  E_ratio_sat   = createMeasurement_error(dict_sim_data["E_ratio_sat"])
  k_nu          = createMeasurement_error(dict_sim_data["k_nu_vel"]["inf"], 1)
  k_eta         = createMeasurement_error(dict_sim_data["k_eta_cur"]["inf"], 0)
  k_p           = createMeasurement_error(dict_sim_data["k_p"]["inf"], 0)
  list_sims = sorted([
    int(sim_res)
    for sim_res in dict_sim_data["k_p"]
    if (dict_sim_data["k_p"][sim_res]["val"] is not None) and ("i" not in sim_res)
  ])
  list_extra_sims = [
    f"{sim_res}^3"
    for sim_res in list_sims
    if sim_res > 300
  ]
  if len(list_extra_sims) > 0:
    extra_sims = "$" + ", ".join(list_extra_sims) + "$"
  else: extra_sims = "--"
  fp.write(sim_id+"\n")
  fp.write(f"\t& ${Re:d}$ & ${Rm:d}$ & ${Pm:d}$"+"\n")
  fp.write(f"\t& ${nu}$ & ${eta}$ & ${Mach}$"+"\n")
  fp.write(f"\t& ${E_growth_rate}$ & ${E_ratio_sat}$"+"\n")
  fp.write(f"\t& ${k_nu}$ & ${k_eta}$ & ${k_p}$"+"\n")
  fp.write(f"\t& {extra_sims} \\\\"+"\n")

def printNresData(fp, dict_all_data, sim_name):
  dict_sim_data = dict_all_data[sim_name]
  Re            = int(dict_sim_data["Re"])
  Pm            = int(dict_sim_data["Pm"])
  sim_id        = "$\\mathcal{M}$" + sim_name.split("Mach")[1].split("_")[0] + f"Re{Re:d}Pm{Pm:d}"
  k_nu_Ncrit    = createMeasurement_error_nres(dict_sim_data["k_nu_vel"]["inf"], 1, 1)
  k_eta_Ncrit   = createMeasurement_error_nres(dict_sim_data["k_eta_cur"]["inf"], 1, 1)
  k_p_Ncrit     = createMeasurement_error_nres(dict_sim_data["k_p"]["inf"], 1, 1)
  k_nu_R        = createMeasurement_error_nres(dict_sim_data["k_nu_vel"]["inf"], 2, 1)
  k_eta_R       = createMeasurement_error_nres(dict_sim_data["k_eta_cur"]["inf"], 2, 1)
  k_p_R         = createMeasurement_error_nres(dict_sim_data["k_p"]["inf"], 2, 1)
  list_sims = sorted([
    int(sim_res)
    for sim_res in dict_sim_data["k_p"]
    if (dict_sim_data["k_p"][sim_res]["val"] is not None) and ("i" not in sim_res)
  ])
  fp.write(sim_id+"\n")
  fp.write(f"\t& ${k_nu_Ncrit}$ & ${k_nu_R}$"+"\n")
  fp.write(f"\t&& ${k_eta_Ncrit}$ & ${k_eta_R}$"+"\n")
  fp.write(f"\t&& ${k_p_Ncrit}$ & ${k_p_R}$ \\\\"+"\n")


## ###############################################################
## MAIN TABLE
## ###############################################################
def saveMainTable():
  filepath_table = f"{FILEPATH_OUTPUT}/dataset_main_table.txt"
  print("Loading datasets...")
  dict_scales_group_sim = WWObjs.readJsonFile2Dict(
    filepath = FILEPATH_OUTPUT,
    filename = "dataset.json"
  )
  ## print Mach=0.3 simulation data
  with open(filepath_table, "w") as fp:
    ## add heading
    fp.write(
      "\\hline" + "\n" +
      "\\hline" + "\n" +
      "\\multicolumn{13}{c}{$\Mach = 0.3$} \\\\" + "\n" +
      "\\hline" + "\n"
    )
    ## add simulation stats
    for sim_name in dict_scales_group_sim:
      if round(dict_scales_group_sim[sim_name]["Mach"]["val"]) < 1:
        printMainData(fp, dict_scales_group_sim, sim_name)
  ## print Mach=1 simulation data
  with open(filepath_table, "a") as fp:
    ## add heading
    fp.write(
      "\\hline" + "\n" +
      "\\multicolumn{13}{c}{$\Mach = 1$} \\\\" + "\n" +
      "\\hline" + "\n"
    )
    ## add simulation stats
    for sim_name in dict_scales_group_sim:
      if round(dict_scales_group_sim[sim_name]["Mach"]["val"]) == 1:
        printMainData(fp, dict_scales_group_sim, sim_name)
  ## print Mach=5 simulation data
  with open(filepath_table, "a") as fp:
    fp.write(
      "\\hline" + "\n" +
      "\\multicolumn{13}{c}{$\Mach = 5$} \\\\" + "\n" +
      "\\hline" + "\n"
    )
    ## add simulation stats
    for sim_name in dict_scales_group_sim:
      if round(dict_scales_group_sim[sim_name]["Mach"]["val"]) == 5:
        printMainData(fp, dict_scales_group_sim, sim_name)
  ## print Mach=10 simulation data
  with open(filepath_table, "a") as fp:
    ## add heading
    fp.write(
      "\\hline" + "\n" +
      "\\multicolumn{13}{c}{$\Mach = 10$} \\\\" + "\n" +
      "\\hline" + "\n"
    )
    ## add simulation stats
    for sim_name in dict_scales_group_sim:
      if round(dict_scales_group_sim[sim_name]["Mach"]["val"]) == 10:
        printMainData(fp, dict_scales_group_sim, sim_name)
    ## end of table
    fp.write(
      "\\hline" + "\n" +
      "\\hline"
    )
  ## display progress
  print("Saved:", filepath_table)


## ###############################################################
## CONVERGENCE TABLE
## ###############################################################
def saveNresTable():
  filepath_table = f"{FILEPATH_OUTPUT}/dataset_Nres_table.txt"
  print("Loading datasets...")
  dict_scales_group_sim = WWObjs.readJsonFile2Dict(
    filepath = FILEPATH_OUTPUT,
    filename = "dataset.json"
  )
  ## print Mach=0.3 simulation data
  with open(filepath_table, "w") as fp:
    ## add heading
    fp.write(
      "\\hline" + "\n" +
      "\\hline" + "\n" +
      "\\multicolumn{9}{c}{$\Mach = 0.3$} \\\\" + "\n" +
      "\\hline" + "\n"
    )
    ## add simulation stats
    for sim_name in dict_scales_group_sim:
      if round(dict_scales_group_sim[sim_name]["Mach"]["val"]) < 1:
        printNresData(fp, dict_scales_group_sim, sim_name)
  ## print Mach=1 simulation data
  with open(filepath_table, "a") as fp:
    ## add heading
    fp.write(
      "\\hline" + "\n" +
      "\\multicolumn{9}{c}{$\Mach = 1$} \\\\" + "\n" +
      "\\hline" + "\n"
    )
    ## add simulation stats
    for sim_name in dict_scales_group_sim:
      if round(dict_scales_group_sim[sim_name]["Mach"]["val"]) == 1:
        printNresData(fp, dict_scales_group_sim, sim_name)
  ## print Mach=5 simulation data
  with open(filepath_table, "a") as fp:
    fp.write(
      "\\hline" + "\n" +
      "\\multicolumn{9}{c}{$\Mach = 5$} \\\\" + "\n" +
      "\\hline" + "\n"
    )
    ## add simulation stats
    for sim_name in dict_scales_group_sim:
      if round(dict_scales_group_sim[sim_name]["Mach"]["val"]) == 5:
        printNresData(fp, dict_scales_group_sim, sim_name)
  ## print Mach=10 simulation data
  with open(filepath_table, "a") as fp:
    ## add heading
    fp.write(
      "\\hline" + "\n" +
      "\\multicolumn{9}{c}{$\Mach = 10$} \\\\" + "\n" +
      "\\hline" + "\n"
    )
    ## add simulation stats
    for sim_name in dict_scales_group_sim:
      if round(dict_scales_group_sim[sim_name]["Mach"]["val"]) == 10:
        printNresData(fp, dict_scales_group_sim, sim_name)
    ## end of table
    fp.write(
      "\\hline" + "\n" +
      "\\hline"
    )
  ## display progress
  print("Saved:", filepath_table)


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
FILEPATH_OUTPUT = "/home/586/nk7952/MHDCodes/kriel2023/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  saveMainTable()
  saveNresTable()
  sys.exit()


## END OF PROGRAM