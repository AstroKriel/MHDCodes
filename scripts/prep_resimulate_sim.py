#!/bin/env python3

## written by Christoph Federrath, 2020
import argparse
from tempfile import mkstemp
from shutil import move, copyfile
import numpy as np
import os
import h5py


# ============= readh5 =============
def readh5(filename, datasetname):
  # first test if filen exists
  if os.path.isfile(filename) == False:
    print("Error: file '"+filename+"' does not exist. Exiting.")
    exit()
  # open HDF5 file
  f = h5py.File(filename, "r")
  # check if dataset name exists in hdf5 file
  if (datasetname in f.keys()) == False:
    print("Error: dataset name '"+datasetname+"' in file '"+filename+"' does not exist. Exiting.")
    exit()
  # grab the dataset as an np array
  dset = f[datasetname]
  data = np.array(dset)
  #close file
  f.close()
  return data
# ============= end: read =============

# ================= read_runtime_parameters ===================
def read_runtime_parameters(flash_file):
  params_dsets = ['integer runtime parameters', \
          'real runtime parameters', \
          'logical runtime parameters', \
          'string runtime parameters']
  runtime_parameters = dict()
  for dset in params_dsets:
    data = readh5(flash_file, dset)
    for i in range(0, len(data)):
      datstr = data[i][0].strip().decode()
      if dset == 'string runtime parameters':
        datval = data[i][1].strip().decode()
      else:
        datval = data[i][1]
      runtime_parameters[datstr] = datval
  return runtime_parameters
# ================ end: read_runtime_parameters ===============

# ======================= read_scalars ========================
def read_scalars(flash_file):
  scalars_dsets = ['integer scalars', \
          'real scalars', \
          'logical scalars', \
          'string scalars']
  scalars = dict()
  for dset in scalars_dsets:
    data = readh5(flash_file, dset)
    for i in range(0, len(data)):
      datstr = data[i][0].strip().decode()
      if dset == 'string scalars':
        datval = data[i][1].strip().decode()
      else:
        datval = data[i][1]
      scalars[datstr] = datval
  return scalars
# ==================== end: read_scalars ======================

def overwrite_value(filename, search_str, set_value):
  debug = True
  fh, tempfile = mkstemp()
  with open(tempfile, 'w') as ftemp:
    with open(filename, 'r') as f:
      found_line = False
      for line in f:
        # replace
        if line.lower().find(search_str.lower())==0:
          found_line = True
          if debug==True: print(filename+": found line   : "+line.rstrip())
          i = line.find("=")
          newline = line[0:i+1]+" "+str(set_value)+"\n"
          line = newline
          if debug==True: print(filename+": replaced with: "+line.rstrip())
        # add lines to temporary output file
        ftemp.write(line)
      if not found_line:
        # add a new line to the end of the file
        line = "\n"+search_str+" = "+str(set_value)+"\n"
        if debug==True: print(filename+": search string not found; creating line at end of file:")
        print(line.lstrip().rstrip())
        ftemp.write(line)
  os.remove(filename)
  move(tempfile, filename)
  os.chmod(filename, 0o644)


# ===== MAIN Start =====
# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Prepare FLASH for re-simulation.')
  parser.add_argument("inputfile", help="FLASH plt or chk file to re-simulate from")
  args = parser.parse_args()
  # make a backup copy of flash.par
  print("Copying 'flash.par' to 'flash.par_resim_backup' as backup.")
  copyfile("flash.par", "flash.par_resim_backup")
  # read runtime parameters and scalars
  rp = read_runtime_parameters(args.inputfile)
  sc = read_scalars(args.inputfile)
  # overwrite flash.par entries
  overwrite_value("flash.par", "sim_input_file", "\""+args.inputfile+"\"")
  overwrite_value("flash.par", "tinitial", sc["time"])
  overwrite_value("flash.par", "dtinit", sc["dt"])
  overwrite_value("flash.par", "xmin", rp["xmin"])
  overwrite_value("flash.par", "xmax", rp["xmax"])
  overwrite_value("flash.par", "ymin", rp["ymin"])
  overwrite_value("flash.par", "ymax", rp["ymax"])
  overwrite_value("flash.par", "zmin", rp["zmin"])
  overwrite_value("flash.par", "zmax", rp["zmax"])
  overwrite_value("flash.par", "plotFileNumber", sc["plotfilenumber"])
  # overwrite_value("flash.par", "basenm", "\""+"RS_"+rp["basenm"].rstrip()+"\"")

# ===== MAIN End =====
