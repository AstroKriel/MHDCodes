## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import h5py
import numpy as np

## load user defined modules
from TheUsefulModule import WWFnF, WWLists
from TheLoadingModule import FileNames


## ###############################################################
## FUNCTIONS FOR LOADING FLASH DATA
## ###############################################################
def reformatFlashData(field, num_blocks, num_procs):
  """ reformatField
  PURPOSE:
    This code reformats FLASH (HDF5) simulation data from:
      [iprocs*jprocs*kprocs, nzb, nxb, nyb] -> [simulation sub-domain number, sub-domain z-data, sub-domain x-data, sub-domain y-data]
    to:
      [kprocs*nzb, iprocs*nxb, jprocs*nyb] -> 3D domain [z-data, x-data, y-data]
    for processing / visualization.
  INPUTS:
    > field      : the FLASH field.
    > num_blocks : number of pixels in each spatial direction (i, j, k) of the simulation sub-domain that is being simulated by each processor.
    > num_procs  : number of processors between which each spatial direction [i, j, k] of the total-simulation domain is divided.
  """
  ## extract number of blocks the simulation was divided into in each [i, j, k] direction
  nxb = num_blocks[0]
  nyb = num_blocks[1]
  nzb = num_blocks[2]
  ## extract number of pixels each block had in each [i, j, k] direction
  iprocs = num_procs[0]
  jprocs = num_procs[1]
  kprocs = num_procs[2]
  ## initialise the output, organised field
  field_sorted = np.zeros([nxb*iprocs, nyb*jprocs, nzb*kprocs])
  ## sort and store the unsorted field into the output field
  for k in range(kprocs):
    for j in range(jprocs):
      for i in range(iprocs):
        field_sorted[
          k * nzb : (k+1) * nzb,
          i * nxb : (i+1) * nxb,
          j * nyb : (j+1) * nyb
        ] = field[j + (i * iprocs) + (k * jprocs * jprocs)]
  return field_sorted

def loadPltData_3D(
    filepath_file,
    num_blocks      = [ 36, 36, 48 ],
    num_procs       = [ 8,  8,  6  ],
    str_field       = "mag",
    bool_print_info = False
  ):
  ## open hdf5 file stream: [iProc*jProc*kProc, nzb, nyb, nxb]
  with h5py.File(filepath_file, "r") as h5file:
    list_keys = [
      key
      for key in list(h5file.keys())
      if key.startswith(str_field)
    ]
    data_x, data_y, data_z = np.array([ h5file[index_key] for index_key in list_keys ])
    if bool_print_info: 
      print("--------- All the keys stored in the FLASH file:\n\t" + "\n\t".join(list(h5file.keys())))
      print("--------- All the keys that were used: " + str(list_keys))
    h5file.close() # close the file stream
    ## reformat data
    data_sorted_x = reformatFlashData(data_x, num_blocks, num_procs)
    data_sorted_y = reformatFlashData(data_y, num_blocks, num_procs)
    data_sorted_z = reformatFlashData(data_z, num_blocks, num_procs)
  ## return data
  return data_sorted_x, data_sorted_y, data_sorted_z

def loadPltData_slice_field(filepath_file, num_blocks, num_procs, str_field):
  ## open hdf5 file stream: [iProc*jProc*kProc, nzb, nyb, nxb]
  with h5py.File(filepath_file, "r") as flash_file:
    ## collect all variables to combine
    list_keys = [
      key
      for key in list(flash_file.keys())
      if key.startswith(str_field)
    ]
    ## calculate the magnitude of the field
    list_data = []
    for data in [
        np.array(flash_file[key])
        for key in list_keys[:2]
      ]:
      ## reformat data
      data_sorted = reformatFlashData(data, num_blocks, num_procs)
      data_slice = data_sorted[ :, :, len(data_sorted[0,0,:])//2 ]
      list_data.append(data_slice)
    return list_data

def loadPltData_slice_magnitude(
    filepath_file, num_blocks, num_procs, str_field,
    bool_norm       = False,
    bool_print_info = False
  ):
  ## open hdf5 file stream: [iProc*jProc*kProc, nzb, nyb, nxb]
  with h5py.File(filepath_file, "r") as flash_file:
    ## collect all variables to combine
    list_keys = [
      key
      for key in list(flash_file.keys())
      if key.startswith(str_field)
    ]
    ## calculate the magnitude of the field
    data_magnitude = np.sqrt(sum(
      np.array(flash_file[key])**2
      for key in list_keys
    ))
    if bool_print_info: 
      print("--------- All the keys stored in the FLASH file:\n\t" + "\n\t".join(list(flash_file.keys())))
      print("--------- All the keys that were used: " + str(list_keys))
    flash_file.close() # close the file stream
    ## reformat data
    data_sorted = reformatFlashData(data_magnitude, num_blocks, num_procs)
    data_slice = data_sorted[ :, :, len(data_sorted[0,0,:])//2 ]
    ## return rms-normalised data
    if bool_norm: return data_slice / np.sqrt(np.mean(data_slice**2))**2
    return data_slice

def loadAllPltData_slice(
    filepath,
    start_time     = 0,
    end_time       = np.inf,
    str_field      = "mag",
    num_blocks     = [36, 36, 48],
    num_procs      = [8, 8, 6],
    plots_per_eddy = 10,
    bool_debug     = False
  ):
  ## get all plt files in the directory
  filenames = WWFnF.getFilesFromFilepath(
    filepath              = filepath,
    filename_contains     = FileNames.FILENAME_FLASH_PLT,
    filename_not_contains = "spect",
    loc_file_index        = -1,
    file_start_index      = plots_per_eddy * start_time,
    file_end_index        = plots_per_eddy * end_time
  )
  ## find min and max colorbar limits
  col_min_val = np.nan
  col_max_val = np.nan
  ## save field slices and simulation times
  list_field_mags = []
  list_sim_times  = []
  for filename, _ in WWLists.loopListWithUpdates(filenames):
    ## load dataset
    field_magnitude = loadPltData_slice_magnitude(
      filepath_file = f"{filepath}/{filename}",
      num_blocks    = num_blocks,
      num_procs     = num_procs,
      str_field     = str_field
    )
    ## check the dimensions of the 2D slice
    if bool_debug:
      print( len(field_magnitude) )
      print( len(field_magnitude[0]) )
    ## append slice of field magnitude
    list_field_mags.append( field_magnitude[:,:] )
    ## append the simulation time
    list_sim_times.append( float(filename.split("_")[-1]) / plots_per_eddy )
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
  return col_min_val, col_max_val, list_field_mags, list_sim_times

def loadTurbData(
    filepath, t_turb,
    var_x      = 0, # time
    quantity   = None,
    var_y      = None,
    time_start = 1,
    time_end   = np.inf,
    bool_debug = False
  ):
  ## define which quantities to read in
  if var_y is None:
    if quantity is None:
      raise Exception("ERROR: neither a quantity index or name have been provided")
    with open(f"{filepath}/{FileNames.FILENAME_FLASH_VOL}", "r") as fp:
      file_first_line = fp.readline()
      bool_format_new = "#01_time" in file_first_line.split() # new if #01_time else #00_time
    if "mach" in quantity.lower():
      var_y = 13 if bool_format_new else 8
    elif "kin" in quantity.lower():
      var_y = 9 if bool_format_new else 6
    elif "mag" in quantity.lower():
      var_y = 11 if bool_format_new else 29
    else: raise Exception("ERROR: reading in ")
  ## initialise quantity to track data traversal
  data_x = []
  data_y = []
  prev_time = np.inf
  ## read data backwards
  with open(f"{filepath}/{FileNames.FILENAME_FLASH_VOL}", "r") as fp:
    num_data_columns = len(fp.readline().split())
    for line in reversed(fp.readlines()):
      data_split = line.replace("\n", "").split()
      ## only look at lines where there is data is defined for every quantity
      if len(data_split) == num_data_columns:
        if not("#" in data_split[var_x][0]) and not("#" in data_split[var_y][0]):
          ## calculate the simulation time
          cur_time = float(data_split[var_x]) / t_turb # normalise by eddy turnover time
          ## if the simulation has been restarted, only read the progressed data
          if cur_time < prev_time: # walk backwards
            cur_val = float(data_split[var_y])
            if cur_val == 0.0:
              if bool_debug: raise Exception(f"Error: encountered 0-value in quantity index {var_y} in {FileNames.FILENAME_FLASH_VOL} at time = {cur_time}")
              continue
            data_x.append(cur_time)
            data_y.append(cur_val)
            prev_time = cur_time
  ## re-order the data
  data_x = data_x[::-1]
  data_y = data_y[::-1]
  ## subset data based on time bounds
  index_start = WWLists.getIndexClosestValue(data_x, time_start)
  index_end   = WWLists.getIndexClosestValue(data_x, time_end)
  data_x_sub = data_x[index_start : index_end]
  data_y_sub = data_y[index_start : index_end]
  return data_x_sub, data_y_sub

def loadSpectra(filepath_file, spect_field, spect_quantity="total"):
  with open(filepath_file, "r") as fp:
    data_file = fp.readlines()
    ## store data in [row, col] (only read in the main dataset: line 6 onwards)
    data  = np.array([x.strip().split() for x in data_file[6:]])
    var_x = 1
    if   "tot" in spect_quantity.lower(): var_y = 15 # total
    elif "lgt" in spect_quantity.lower(): var_y = 11 # longitudinal
    elif "trv" in spect_quantity.lower(): var_y = 13 # transverse
    else: raise Exception(f"Error: You have passed an invalid spectra quantity: '{spect_quantity}'.")
    try:
      data_x = np.array(list(map(float, data[:, var_x]))) 
      data_y = np.array(list(map(float, data[:, var_y])))
      if   "vel" in spect_field.lower(): data_y = data_y
      elif "kin" in spect_field.lower(): data_y = data_y / 2
      elif "mag" in spect_field.lower(): data_y = data_y / (8 * np.pi)
      else: raise Exception(f"Error: You have passed an invalid spectra field: '{spect_field}'.")
    except: raise Exception("Error: Failed to read spectra-file:", filepath_file)
    return data_x, data_y

def loadAllSpectra(
    filepath, spect_field, plots_per_eddy,
    spect_quantity  = "total",
    file_start_time = 2,
    file_end_time   = np.inf,
    read_every      = 1,
    bool_verbose    = True
  ):
  ## get list of spect-filenames in directory
  if ("vel" in spect_field.lower()) or ("kin" in spect_field.lower()):
    file_end_str = "spect_vels.dat"
  elif "mag" in spect_field.lower():
    file_end_str = "spect_mags.dat"
  else: raise Exception("Error: received an unexpected spectra filename:", spect_field)
  list_spectra_filenames = WWFnF.getFilesFromFilepath(
    filepath          = filepath,
    filename_endswith = file_end_str,
    loc_file_index    = -3,
    file_start_index  = plots_per_eddy * file_start_time,
    file_end_index    = plots_per_eddy * file_end_time
  )
  ## initialise list of spectra data
  list_k_group_t     = []
  list_power_group_t = []
  list_sim_times     = []
  ## loop over each of the spectra file names
  for filename, _ in WWLists.loopListWithUpdates(
      list_spectra_filenames[::read_every],
      bool_verbose
    ):
    ## convert file index to simulation time
    file_sim_time = float(filename.split("_")[-3]) / plots_per_eddy
    ## load data
    list_k, list_power = loadSpectra(
      filepath_file  = f"{filepath}/{filename}",
      spect_field    = spect_field,
      spect_quantity = spect_quantity
    )
    ## store data
    list_k_group_t.append(list_k)
    list_power_group_t.append(list_power)
    list_sim_times.append(file_sim_time)
  ## return spectra data
  return {
    "list_k_group_t"     : list_k_group_t,
    "list_power_group_t" : list_power_group_t,
    "list_sim_times"     : list_sim_times
  }

def getPlotsPerEddy_fromFlashLog(
    filepath,
    num_t_turb   = 100,
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
        plots_per_eddy = tmax / plot_file_interval / num_t_turb
        if bool_verbose:
          print(f"The following has been read from {FileNames.FILENAME_FLASH_LOG}:")
          print("\t> 'tmax'".ljust(25),                 "=", tmax)
          print("\t> 'plotFileIntervalTime'".ljust(25), "=", plot_file_interval)
          print("\t> # plt-files / t_turb".ljust(25),   "=", plots_per_eddy)
          print(f"\tAssuming the simulation has been setup to run for a max of {num_t_turb} t/t_turb.")
          print(" ")
        return plots_per_eddy
  ## failed to read quantity
  raise Exception(f"ERROR: failed to read plots_per_eddy from {FileNames.FILENAME_FLASH_LOG}")

def getPlasmaConstants_fromFlashInput(filepath, rms_Mach, k_turb):
    bool_found_nu  = False
    bool_found_eta = False
    ## search through file for parameters
    with open(f"{filepath}/{FileNames.FILENAME_FLASH_INPUT}") as file_lines:
      for line in file_lines:
        list_line_elems = line.split()
        ## ignore empty lines
        if len(list_line_elems) == 0: continue
        ## read value for 'diff_visc_nu'
        if list_line_elems[0] == "diff_visc_nu":
          nu = float(list_line_elems[2])
          bool_found_nu = True
        ## read value for 'resistivity'
        if list_line_elems[0] == "resistivity":
          eta = float(list_line_elems[2])
          bool_found_eta = True
        ## stop searching if both parameters have been identified
        if bool_found_nu and bool_found_eta: break
    ## compute plasma numbers
    if bool_found_nu and bool_found_eta:
      nu  = nu
      eta = eta
      Re  = int(rms_Mach / (k_turb * nu))
      Rm  = int(rms_Mach / (k_turb * eta))
      Pm  = int(nu / eta)
      print(f"\t> Re = {Re}, Rm = {Rm}, Pm = {Pm}, nu = {nu:0.2e}, eta = {eta:0.2e},")
      return {
        "nu"  : nu,
        "eta" : eta,
        "Re"  : Re,
        "Rm"  : Rm,
        "Pm"  : Pm
      }
    else:
      bool_found_neither = (not bool_found_nu) and (not bool_found_eta)
      raise Exception("ERROR: could not compute {}{}{}.".format(
        "nu"   if not bool_found_nu  else "",
        " or " if bool_found_neither else "",
        "eta"  if not bool_found_eta else ""
      ))

def computeDissipationConstants(Mach, k_turb, Re=None, Rm=None, Pm=None):
  ## Re and Pm have been defined
  if (Re is not None) and (Pm is not None):
    nu  = round(Mach / (k_turb * Re), 5)
    eta = round(nu / Pm, 5)
    Rm  = round(Mach / (k_turb * eta))
  ## Rm and Pm have been defined
  elif (Rm is not None) and (Pm is not None):
    eta = round(Mach / (k_turb * Rm), 5)
    nu  = round(eta * Pm, 5)
    Re  = round(Mach / (k_turb * nu))
  ## error
  else: raise Exception(f"ERROR: insufficient plasma Reynolds numbers defined: Re = {Re}, Rm = {Rm}, Pm = {Rm}")
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
  else: raise Exception(f"ERROR: insufficient plasma Reynolds numbers defined: Re = {Re}, Rm = {Rm}, Pm = {Rm}")
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