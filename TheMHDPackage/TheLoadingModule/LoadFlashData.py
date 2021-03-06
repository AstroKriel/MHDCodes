## START OF MODULE


## ###############################################################
## MODULES
## ###############################################################
import h5py
import numpy as np

## relative import of 
from TheUsefulModule import WWFnF, WWLists


## ###############################################################
## FUNCTIONS FOR LOADING FLASH DATA
## ###############################################################
def reformatFLASHField(field, num_blocks, num_procs):
  """ reformatField
  PURPOSE:
    This code reformats FLASH (HDF5) simulation data from:
      [iprocs*jprocs*kprocs, nzb, nxb, nyb] -> [simulation sub-domain number, sub-domain z-data, sub-domain x-data, sub-domain y-data]
    format to:
      [kprocs*nzb, iprocs*nxb, jprocs*nyb] -> full domain [z-data, x-data, y-data]
    for processing / visualization.
  INPUTS:
    > field      : the FLASH field.
    > num_blocks : number of pixels in each spatial direction [i, j, k] of the simulation sub-domain being simulated by each processor.
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
          k*nzb:(k+1)*nzb,
          i*nxb:(i+1)*nxb,
          j*nyb:(j+1)*nyb
        ] = field[j + i*iprocs + k*jprocs*jprocs]
  return field_sorted


def loadFLASHFieldSlice(
    filepath_data,
    num_blocks      = [ 36, 36, 48 ],
    num_procs       = [ 8,  8,  6  ],
    str_field       = "mag",
    bool_rms_norm   = False,
    bool_print_info = False
  ):
  ## open hdf5 file stream: [iProc*jProc*kProc, nzb, nyb, nxb]
  flash_file = h5py.File(filepath_data, "r")
  ## collect all variables to combine
  names = [s for s in list(flash_file.keys()) if s.startswith(str_field)]
  ## calculate the magnitude of the field
  data = np.sqrt(sum(np.array(flash_file[i])**2 for i in names))
  if bool_print_info: 
    print("--------- All the keys stored in the FLASH file:\n\t" + "\n\t".join(list(flash_file.keys())))
    print("--------- All the keys that were used: " + str(names))
  flash_file.close() # close the file stream
  ## reformat data
  data_sorted = reformatFLASHField(data, num_blocks, num_procs)
  data_slice = data_sorted[ :, :, len(data_sorted[0,0,:])//2 ]
  ## normalise data by rms
  if bool_rms_norm:
    return data_slice**2 / np.sqrt(np.mean(data_slice**2))**2
  ## return data
  else: return data_slice


def loadListFLASHFieldSlice(
    filepath_data,
    file_start_index  = 2,
    file_end_index    = np.inf,
    num_blocks        = [ 36, 36, 48 ],
    num_procs         = [ 8,  8,  6  ],
    str_field         = "mag",
    bool_rms_norm     = False,
    bool_hide_updates = False
  ):
  ## initialise list of cube data
  list_data_mag_sorted = []
  ## filter for datacube files
  flash_filenames = WWFnF.getFilesFromFolder(
    filepath           = filepath_data, 
    str_contains       = "Turb_hdf5_plt_cnt_",
    str_not_contains   = "spect",
    file_index_placing = -1,
    file_start_index   = file_start_index,
    file_end_index     = file_end_index
  )
  ## loop over each of the datacube file names
  for filename, _ in WWLists.loopListWithUpdates(flash_filenames, bool_hide_updates):
    ## load all datacube files in folder
    list_data_mag_sorted.append(
      loadFLASHFieldSlice(
        filepath_data = WWFnF.createFilepath([ filepath_data, filename ]),
        num_blocks    = num_blocks,
        num_procs     = num_procs,
        str_field     = str_field,
        bool_rms_norm = bool_rms_norm
      )
    )
  if not len(list_data_mag_sorted) > 0:
    raise Exception("Could not load any data in:", filepath_data)
  ## get colorbar limits
  list_col_range = [
    np.min(list_data_mag_sorted),
    np.max(list_data_mag_sorted)
  ]
  ## return data
  return list_data_mag_sorted, list_col_range


def loadFLASH3DField(
    filepath_data,
    num_blocks      = [ 36, 36, 48 ],
    num_procs       = [ 8,  8,  6  ],
    str_field       = "mag",
    bool_print_info = False
  ):
  flash_file = h5py.File(filepath_data, "r") # open hdf5 file stream: [iProc*jProc*kProc, nzb, nyb, nxb]
  names = [s for s in list(flash_file.keys()) if s.startswith(str_field)]
  data_x, data_y, data_z = np.array([flash_file[i] for i in names])
  if bool_print_info: 
    print("--------- All the keys stored in the FLASH file:\n\t" + "\n\t".join(list(flash_file.keys()))) # print keys
    print("--------- All the keys that were used: " + str(names))
  flash_file.close() # close the file stream
  ## reformat data
  data_sorted_x = reformatFLASHField(data_x, num_blocks, num_procs)
  data_sorted_y = reformatFLASHField(data_y, num_blocks, num_procs)
  data_sorted_z = reformatFLASHField(data_z, num_blocks, num_procs)
  return data_sorted_x, data_sorted_y, data_sorted_z


def loadFLASHFieldDataList(
    filepath_data,
    start_time = 0,
    end_time   = np.inf,
    str_field  = "mag",
    num_blocks = [36, 36, 48],
    num_procs  = [8, 8, 6],
    plots_per_eddy   = 10,
    plot_every_index = 1
  ):
  ## get all plt files in the directory
  filenames = WWFnF.getFilesFromFolder(
    filepath           = filepath_data,
    str_contains       = "Turb_hdf5_plt_cnt_",
    str_not_contains   = "spect",
    file_index_placing = -1,
    file_start_index   = (start_time * plots_per_eddy),
    file_end_index     = (end_time * plots_per_eddy)
  )
  ## find min and max colorbar limits
  col_min_val = np.nan
  col_max_val = np.nan
  ## save field slices and simulation times
  list_field_mags = []
  list_sim_times  = []
  plot_file_indices = range(len(filenames))[::plot_every_index]
  for filename, file_index in WWLists.loopListWithUpdates(filenames):
    ## load dataset
    field_mag = loadFLASHFieldSlice(
      WWFnF.createFilepath([filepath_data, filename]),
      num_blocks,
      num_procs,
      str_field
    )
    # ## check the dimensions of the 2D slice
    # print( len(field_mag) )
    # print( len(field_mag[0]) )
    ## append slice of field magnitude
    list_field_mags.append( field_mag[:,:] )
    ## append the simulation timepoint
    list_sim_times.append( float(filename.split("_")[-1]) / plots_per_eddy )
    ## find min and max value
    col_min_val = np.nanmin([
      np.nanmin(field_mag[:,:]),
      col_min_val
    ])
    col_max_val = np.nanmax([
      np.nanmax(field_mag[:,:]),
      col_max_val
    ])
  print(" ")
  return col_min_val, col_max_val, list_field_mags, list_sim_times


def loadTurbData(
    filepath_data,
    var_y,
    t_turb, # ell_turb / (c_s * Mach)
    time_start = 1,
    time_end   = np.inf
  ):
  ## load data
  filepath_turb    = WWFnF.createFilepath([ filepath_data, "Turb.dat" ])
  first_line       = open(filepath_turb).readline().split()
  num_data_columns = len(first_line)
  ## initialise x and y data
  var_x  = 0 # time
  data_x = []
  data_y = []
  ## initialise data-traversal variable
  prev_time = np.inf
  ## read data backwards
  for line in reversed( open(filepath_turb).readlines() ):
    data_split = line.split()
    ## only look at lines where there is data for each tracked quantity
    if len(data_split) == num_data_columns:
      ## don't look at the labels
      if not("#" in data_split[var_x][0]) and not("#" in data_split[var_y][0]):
        ## calculate the normalised time
        cur_time = float(data_split[var_x]) / t_turb # normalise by eddy turnover time
        ## if the simulation has been restarted, only read the progressed data
        if cur_time < prev_time: # walking backwards
          data_x.append(cur_time)
          data_y.append(float(data_split[var_y]))
          prev_time = cur_time
  ## reverse the reversed lists
  data_x = data_x[::-1]
  data_y = data_y[::-1]
  ## subset data
  index_start = WWLists.getIndexClosestValue(data_x, time_start)
  index_end   = WWLists.getIndexClosestValue(data_x, time_end)
  return data_x[index_start : index_end], data_y[index_start : index_end]


def getPlotsPerEddy(filepath, num_t_turb=100, bool_hide_updates=False):
  def getName(line):
    return line.split("=")[0].lower()
  def getValue(line):
    return line.split("=")[1].split("[")[0]
  tmax = None
  plot_file_interval = None
  for line in open(filepath + "/Turb.log").readlines():
    if ("tmax" in getName(line)) and ("dtmax" not in getName(line)):
      tmax = float(getValue(line))
    elif "plotfileintervaltime" in getName(line):
      plot_file_interval = float(getValue(line))
    if (tmax is not None) and (plot_file_interval is not None):
      plots_per_eddy = tmax / plot_file_interval / num_t_turb
      if not(bool_hide_updates):
        print("Read from 'Turb.log':")
        print("\t> 'tmax'".ljust(25),                 "=", tmax)
        print("\t> 'plotFileIntervalTime'".ljust(25), "=", plot_file_interval)
        print("\t> # plt-files / t_turb".ljust(25),   "=", plots_per_eddy)
        print(f"\tAssumed the simulation ran for {num_t_turb} t/t_turb")
        print(" ")
      return plots_per_eddy
  return None


def loadSpectra(filepath_data, str_spectra_type):
  data_file = open(filepath_data).readlines() # load in data
  data      = np.array([x.strip().split() for x in data_file[6:]]) # store all data: [row, col]
  try:
    data_x = np.array(list(map(float, data[:, 1])))  # variable: wave number (k)
    data_y = np.array(list(map(float, data[:, 15]))) # variable: power spectrum
    if "vel" in str_spectra_type:
      data_y = data_y / 2
    elif "mag" in str_spectra_type:
      data_y = data_y / (8 * np.pi)
    else: raise Exception("You have passed an invalid spectra type to 'loadSpectra()'.")
    bool_failed_to_read = False
  except:
    bool_failed_to_read = True
    data_x = []
    data_y = []
  return data_x, data_y, bool_failed_to_read


def loadListSpectra(
    filepath_data,
    str_spectra_type  = "mag",
    plots_per_eddy    = 10,
    file_start_time   = 2,
    file_end_time     = np.inf,
    read_every        = 1,
    bool_hide_updates = False
  ):
  ## initialise list of spectra data
  k_group_t       = []
  power_group_t   = []
  list_sim_times      = []
  list_failed_to_load = []
  ## filter for spectra data-files
  spectra_filenames = WWFnF.getFilesFromFolder(
    filepath           = filepath_data, 
    str_contains       = "hdf5_plt_cnt",
    str_endswith       = "spect_" + str_spectra_type + "s.dat",
    file_index_placing = -3,
    file_start_index   = plots_per_eddy * file_start_time,
    file_end_index     = plots_per_eddy * file_end_time
  )
  ## loop over each of the spectra file names
  for filename, _ in WWLists.loopListWithUpdates(spectra_filenames[::read_every], bool_hide_updates):
    ## load data
    spectra_k, spectra_power, bool_failed_to_read = loadSpectra(
      filepath_data    = WWFnF.createFilepath([ filepath_data, filename ]),
      str_spectra_type = str_spectra_type
    )
    ## check if the data was read successfully
    if bool_failed_to_read:
      list_failed_to_load.append(filename)
      continue
    ## store data
    k_group_t.append(spectra_k)
    power_group_t.append(spectra_power)
    list_sim_times.append(
      float(filename.split("_")[-3]) / plots_per_eddy
    )
  ## list those files that failed to load
  if len(list_failed_to_load) > 0:
    print("\tFailed to read in the following files:", "\n\t\t> ".join(
      [" "] + list_failed_to_load
    ))
  ## return spectra data
  return k_group_t, power_group_t, list_sim_times


## END OF MODULE