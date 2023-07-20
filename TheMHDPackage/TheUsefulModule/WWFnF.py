## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import os, re, shutil
import numpy as np


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def createFolder(directory, bool_verbose=True):
  if not(os.path.exists(directory)):
    os.makedirs(directory)
    if bool_verbose: print("Success: Created folder:\n\t" + directory + "\n")
  elif bool_verbose: print("Warning: Folder already exists:\n\t" + directory + "\n")

def createFilepath(list_filepath_folders):
  return re.sub("/+", "/", "/".join([
    folder
    for folder in list_filepath_folders
    if not(folder == "")
  ]))


## ###############################################################
## FILTERING FILES IN DIRECTORY
## ###############################################################
def makeFilter(
    filename_contains     = None,
    filename_not_contains = None,
    filename_starts_with  = None,
    filename_ends_with    = None,
    loc_file_index        = None,
    file_start_index      = 0,
    file_end_index        = np.inf,
    filename_split_by     = "_"
  ):
  """ makeFilter
    PURPOSE: Create a filter condition for files that look a particular way.
  """
  def meetsCondition(element):
    bool_contains     = filename_contains     is None or element.__contains__(filename_contains)
    bool_not_contains = filename_not_contains is None or not(element.__contains__(filename_not_contains))
    bool_starts_with  = filename_starts_with  is None or element.startswith(filename_starts_with)
    bool_ends_with    = filename_ends_with    is None or element.endswith(filename_ends_with)
    if all([ bool_contains, bool_not_contains, bool_starts_with, bool_ends_with ]):
      if loc_file_index is not None:
        ## check that the file index falls within the specified range
        if len(element.split(filename_split_by)) > abs(loc_file_index):
          file_index = int(element.split(filename_split_by)[loc_file_index])
          bool_after_start = file_index >= file_start_index
          bool_before_end  = file_index <= file_end_index
          ## if the file meets all the required conditions
          if (bool_after_start and bool_before_end): return True
      ## all specified conditions have been met
      else: return True
    ## file doesn't meet conditions
    else: return False
  return meetsCondition

def getFilesInDirectory(
    directory, 
    filename_contains     = None,
    filename_starts_with  = None,
    filename_ends_with    = None,
    filename_not_contains = None,
    loc_file_index        = None,
    file_start_index      = 0,
    file_end_index        = np.inf
  ):
  myFilter = makeFilter(
    filename_contains     = filename_contains,
    filename_not_contains = filename_not_contains,
    filename_starts_with  = filename_starts_with,
    filename_ends_with    = filename_ends_with,
    loc_file_index        = loc_file_index,
    file_start_index      = file_start_index,
    file_end_index        = file_end_index
  )
  return list(filter(myFilter, sorted(os.listdir(directory))))


## ###############################################################
## WORKING WITH FILES
## ###############################################################
def copyFile(directory_from, directory_to, filename, bool_verbose=True):
  ## copy the file and it's permissions
  shutil.copy(
    f"{directory_from}/{filename}",
    f"{directory_to}/{filename}"
  )
  if bool_verbose:
    print(f"Coppied:")
    print(f"\t> File: {filename}")
    print(f"\t> From: {directory_from}")
    print(f"\t> To:   {directory_to}")


## END OF LIBRARY