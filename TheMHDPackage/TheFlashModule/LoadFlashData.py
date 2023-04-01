## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import h5py
import numpy as np

## load user defined modules
from TheUsefulModule import WWFnF, WWLists
from TheFlashModule import FileNames


## ###############################################################
## FUNCTIONS FOR LOADING FLASH DATA
## ###############################################################
def reformatFlashField(field, num_blocks, num_procs):
  """ reformatFlashField
  PURPOSE:
    This code reformats a field from being structured as [iprocs*jprocs*kprocs, nzb, nxb, nyb]
    to: [kprocs*nzb, iprocs*nxb, jprocs*nyb]
  INPUTS:
    > field      : FLASH field
    > num_blocks : number of blocks
    > num_procs  : number of processors per block, in each spatial dimension
  """
  ## extract number of blocks in each direction
  nxb = num_blocks[0]
  nyb = num_blocks[1]
  nzb = num_blocks[2]
  ## extract number of processors in each direction
  iprocs = num_procs[0]
  jprocs = num_procs[1]
  kprocs = num_procs[2]
  ## initialise the organised field
  field_sorted = np.zeros([nxb*iprocs, nyb*jprocs, nzb*kprocs])
  ## reorganise unsorted field
  for k in range(kprocs):
    for j in range(jprocs):
      for i in range(iprocs):
        field_sorted[
          k * nzb : (k+1) * nzb,
          i * nxb : (i+1) * nxb,
          j * nyb : (j+1) * nyb
        ] = field[j + (i * iprocs) + (k * jprocs * jprocs)]
  return field_sorted

def loadFlashDataCube(
    filepath_file, num_blocks, num_procs, field_name,
    bool_print_h5keys = False
  ):
  ## open hdf5 file stream
  with h5py.File(filepath_file, "r") as h5file:
    ## create list of field-keys to extract from hdf5 file
    list_keys_stored = list(h5file.keys())
    list_keys_used = [
      key
      for key in list_keys_stored
      if key.startswith(field_name)
    ]
    ## check which keys are stored
    if bool_print_h5keys: 
      print("--------- All the keys stored in the FLASH hdf5 file:\n\t" + "\n\t".join(list_keys_stored))
      print("--------- All the keys that were used: " + str(list_keys_used))
    ## extract fields from hdf5 file
    data_x, data_y, data_z = np.array([
      h5file[key]
      for key in list_keys_used
    ])
    ## close file stream
    h5file.close()
  ## reformat data
  data_sorted_x = reformatFlashField(data_x, num_blocks, num_procs)
  data_sorted_y = reformatFlashField(data_y, num_blocks, num_procs)
  data_sorted_z = reformatFlashField(data_z, num_blocks, num_procs)
  ## return spatial-components of data
  return data_sorted_x, data_sorted_y, data_sorted_z


def loadAllFlashDataCubes(
    filepath, field_name, dict_sim_inputs,
    start_time     = 0,
    end_time       = np.inf,
    bool_debug     = False
  ):
  outputs_per_t_turb = dict_sim_inputs["outputs_per_t_turb"]
  ## get all plt files in the directory
  filenames = WWFnF.getFilesFromFilepath(
    filepath              = filepath,
    filename_contains     = FileNames.FILENAME_FLASH_DATA_CUBE,
    filename_not_contains = "spect",
    loc_file_index        = -1,
    file_start_index      = outputs_per_t_turb * start_time,
    file_end_index        = outputs_per_t_turb * end_time
  )
  ## find min and max colorbar limits
  col_min_val = np.nan
  col_max_val = np.nan
  ## save field slices and simulation times
  field_group_t = []
  list_turb_times = []
  for filename, _ in WWLists.loopListWithUpdates(filenames):
    ## load dataset
    field_magnitude = loadFlashDataCube(
      filepath_file = f"{filepath}/{filename}",
      num_blocks    = dict_sim_inputs["num_blocks"],
      num_procs     = dict_sim_inputs["num_procs"],
      field_name    = field_name
    )
    # ## check the dimensions of 
    # if bool_debug:
    #   print( len(field_magnitude) )
    #   print( len(field_magnitude[0]) )
    ## append slice of field magnitude
    field_group_t.append( field_magnitude[:,:] )
    ## append the simulation time
    list_turb_times.append( float(filename.split("_")[-1]) / outputs_per_t_turb )
    ## find data magnitude bounds
    col_min_val = np.nanmin([
      np.nanmin(field_magnitude[:,:]),
      col_min_val
    ])
    col_max_val = np.nanmax([
      np.nanmax(field_magnitude[:,:]),
      col_max_val
    ])
  print(" ")
  return col_min_val, col_max_val, field_group_t, list_turb_times

def loadVIData(
    filepath, t_turb,
    field_index  = None,
    field_name   = None,
    time_start   = 1,
    time_end     = np.inf,
    bool_debug   = False,
    bool_verbose = True
  ):
  ## define which quantities to read in
  time_index = 0
  if field_index is None:
    ## check that a variable name has been provided
    if field_name is None: raise Exception("ERROR: need to provide either a field-index or field-name")
    ## check which formatting the output file uses
    with open(f"{filepath}/{FileNames.FILENAME_FLASH_VI_DATA}", "r") as fp:
      file_first_line = fp.readline()
      bool_format_new = "#01_time" in file_first_line.split() # new version of file indexes from 1
    ## get index of field in file
    if   "kin"  in field_name.lower(): field_index = 9  if bool_format_new else 6
    elif "mag"  in field_name.lower(): field_index = 11 if bool_format_new else 29
    elif "mach" in field_name.lower(): field_index = 13 if bool_format_new else 8
    else: raise Exception(f"ERROR: reading in {FileNames.FILENAME_FLASH_VI_DATA}")
  ## initialise quantities to track traversal
  data_time  = []
  data_field = []
  prev_time  = np.inf
  with open(f"{filepath}/{FileNames.FILENAME_FLASH_VI_DATA}", "r") as fp:
    num_fields = len(fp.readline().split())
    ## read data in backwards
    for line in reversed(fp.readlines()):
      data_split_columns = line.replace("\n", "").split()
      ## only read where every field has been processed
      if not(len(data_split_columns) == num_fields): continue
      ## ignore comments
      if "#" in data_split_columns[time_index]:  continue
      if "#" in data_split_columns[field_index]: continue
      ## compute time in units of eddy turnover time
      this_time = float(data_split_columns[time_index]) / t_turb
      ## only read data that has progressed in time
      if this_time < prev_time:
        data_val = float(data_split_columns[field_index])
        ## something might have gone wrong: it is very unlikely to encounter a 0-value exactly
        if (data_val == 0.0) and (0 < this_time):
          warning_message = f"{FileNames.FILENAME_FLASH_VI_DATA}: value of field-index = {field_index} is 0.0 at time = {this_time}"
          if bool_debug: raise Exception(f"Error: {warning_message}")
          if bool_verbose: print(f"Warning: {warning_message}")
          continue
        ## store data
        data_time.append(this_time)
        data_field.append(data_val)
        ## step backwards
        prev_time = this_time
  ## re-order data
  data_time  = data_time[::-1]
  data_field = data_field[::-1]
  ## subset data based on provided time bounds
  index_start = WWLists.getIndexClosestValue(data_time, time_start)
  index_end   = WWLists.getIndexClosestValue(data_time, time_end)
  data_time_subset  = data_time[index_start  : index_end]
  data_field_subset = data_field[index_start : index_end]
  return data_time_subset, data_field_subset

def loadSpectrum(filepath_file, spect_field, spect_comp="total"):
  with open(filepath_file, "r") as fp:
    dataset = fp.readlines()
    ## read main dataset
    data = np.array([
      lines.strip().split() # remove leading/trailing whitespace + separate by whitespace-delimiter
      for lines in dataset[6:] # read from after header
    ])
    ## get the indices assiated with fields of interest
    k_index = 1
    if   "lgt" in spect_comp.lower(): field_index = 11 # longitudinal
    elif "trv" in spect_comp.lower(): field_index = 13 # transverse
    elif "tot" in spect_comp.lower(): field_index = 15 # total
    else: raise Exception(f"Error: {spect_comp} is an invalid spectra component (should be lgt, trv, or tot)")
    ## read fields from file
    try:
      data_k     = np.array(list(map(float, data[:, k_index]))) 
      data_power = np.array(list(map(float, data[:, field_index])))
      if   "vel" in spect_field.lower(): data_power = data_power
      elif "kin" in spect_field.lower(): data_power = data_power / 2
      elif "mag" in spect_field.lower(): data_power = data_power / (8 * np.pi)
      else: raise Exception(f"Error: {spect_field} is an invalid spectra field (should be vel, kin, or mag)")
    except: raise Exception("Error: failed to read spectra-file in:", filepath_file)
    return data_k, data_power

def loadAllSpectra(
    filepath, spect_field, outputs_per_t_turb,
    spect_comp      = "total",
    file_start_time = 2,
    file_end_time   = np.inf,
    read_every      = 1,
    bool_verbose    = True
  ):
  if ("vel" in spect_field.lower()) or ("kin" in spect_field.lower()):
                                      file_end_str = "spect_vels.dat"
  elif "mag" in spect_field.lower():  file_end_str = "spect_mags.dat"
  elif "cur" in spect_field.lower():  file_end_str = "spect_current.dat"
  else: raise Exception("Error: invalid spectra filename:", spect_field)
  ## get list of spect-filenames in directory
  list_spectra_filenames = WWFnF.getFilesFromFilepath(
    filepath           = filepath,
    filename_ends_with = file_end_str,
    loc_file_index     = -3,
    file_start_index   = outputs_per_t_turb * file_start_time,
    file_end_index     = outputs_per_t_turb * file_end_time
  )
  ## initialise list of spectra data
  list_k_group_t     = []
  list_power_group_t = []
  list_turb_times    = []
  ## loop over each of the spectra file names
  for filename, _ in WWLists.loopListWithUpdates(list_spectra_filenames[::read_every], bool_verbose):
    ## convert file index to simulation time
    turb_time = float(filename.split("_")[-3]) / outputs_per_t_turb
    ## load data
    list_k, list_power = loadSpectrum(
      filepath_file = f"{filepath}/{filename}",
      spect_field   = spect_field,
      spect_comp    = spect_comp
    )
    ## store data
    list_k_group_t.append(list_k)
    list_power_group_t.append(list_power)
    list_turb_times.append(turb_time)
  ## return spectra data
  return {
    "list_k_group_t"     : list_k_group_t,
    "list_power_group_t" : list_power_group_t,
    "list_turb_times"    : list_turb_times
  }

def getPlotsPerEddy_fromFlashLog(
    filepath, num_t_turb,
    bool_verbose = True
  ):
  ## helper functions
  def getName(line):
    return line.split("=")[0].lower()
  def getValue(line):
    return line.split("=")[1].split("[")[0]
  ## search routine
  bool_tmax_found          = False
  bool_plot_interval_found = None
  with open(f"{filepath}/{FileNames.FILENAME_FLASH_LOG}", "r") as fp:
    for line in fp.readlines():
      if ("tmax" in getName(line)) and ("dtmax" not in getName(line)):
        tmax = float(getValue(line))
        bool_tmax_found = True
      elif "plotfileintervaltime" in getName(line):
        plot_file_interval = float(getValue(line))
        bool_plot_interval_found = True
      if bool_tmax_found and bool_plot_interval_found:
        outputs_per_t_turb = tmax / plot_file_interval / num_t_turb
        ## TODO: if not integer(?) then error
        if bool_verbose:
          print(f"The following has been read from {FileNames.FILENAME_FLASH_LOG}:")
          print("\t> 'tmax'".ljust(25),                 "=", tmax)
          print("\t> 'plotFileIntervalTime'".ljust(25), "=", plot_file_interval)
          print("\t> # plt-files / t_turb".ljust(25),   "=", outputs_per_t_turb)
          print(f"\tAssuming the simulation has been setup to run for a max of {num_t_turb} t/t_turb.")
          print(" ")
        return outputs_per_t_turb
  ## failed to read quantity
  raise Exception(f"ERROR: failed to read outputs_per_t_turb from {FileNames.FILENAME_FLASH_LOG}")

def computePlasmaConstants(Mach, k_turb, Re=None, Rm=None, Pm=None):
  ## Re and Pm have been defined
  if (Re is not None) and (Pm is not None):
    Re  = float(Re)
    Pm  = float(Pm)
    Rm  = Re * Pm
    nu  = round(Mach / (k_turb * Re), 5)
    eta = round(nu / Pm, 5)
  ## Rm and Pm have been defined
  elif (Rm is not None) and (Pm is not None):
    Rm  = float(Rm)
    Pm  = float(Pm)
    Re  = Rm / Pm
    eta = round(Mach / (k_turb * Rm), 5)
    nu  = round(eta * Pm, 5)
  ## error
  else: raise Exception(f"ERROR: insufficient plasma Reynolds numbers provided: Re = {Re}, Rm = {Rm}, Pm = {Rm}")
  return {
    "nu"  : nu,
    "eta" : eta,
    "Re"  : Re,
    "Rm"  : Rm,
    "Pm"  : Pm
  }

def computePlasmaNumbers(Re=None, Rm=None, Pm=None):
  ## Re and Pm have been defined
  if (Re is not None) and (Pm is not None):
    Rm = Re * Pm
  ## Rm and Pm have been defined
  elif (Rm is not None) and (Pm is not None):
    Re = Rm / Pm
  elif (Re is not None) and (Rm is not None):
    Pm = Rm / Re
  ## error
  else: raise Exception(f"ERROR: insufficient plasma Reynolds numbers provided: Re = {Re}, Rm = {Rm}, Pm = {Rm}")
  return {
    "Re"  : Re,
    "Rm"  : Rm,
    "Pm"  : Pm
  }

def getNumberFromString(string, ref):
  string_lower = string.lower()
  ref_lower    = ref.lower()
  return float(string_lower.replace(ref_lower, "")) if ref_lower in string_lower else None


## END OF LIBRARY