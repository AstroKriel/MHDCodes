## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import os, re, shutil
import numpy as np


## ###############################################################
## WORKING WITH FILES / FOLDERS
## ###############################################################
def makeFilter(
    str_contains       = None,
    str_not_contains   = None,
    str_startswith     = None,
    str_endswith       = None,
    file_index_placing = None,
    file_start_index   = 0,
    file_end_index     = np.inf,
    str_split_by       = "_"
  ):
  """ makeFilter
    PURPOSE: Create a filter condition for files that look a particular way.
  """
  def meetsCondition(element):
    if str_contains is not None:
      bool_contains = element.__contains__(str_contains)
    else: bool_contains = True
    if str_not_contains is not None:
      bool_not_contains = not(element.__contains__(str_not_contains))
    else: bool_not_contains = True
    if str_startswith is not None:
      bool_startswith = element.startswith(str_startswith)
    else: bool_startswith = True
    if str_endswith is not None:
      bool_endswith = element.endswith(str_endswith)
    else: bool_endswith = True
    if (bool_contains and 
        bool_not_contains and 
        bool_startswith and 
        bool_endswith):
      if file_index_placing is not None:
        ## check that the file index falls within the specified range
        if len(element.split(str_split_by)) > abs(file_index_placing):
          bool_time_after  = (
            int(element.split(str_split_by)[file_index_placing]) >= file_start_index
          )
          bool_time_before = (
            int(element.split(str_split_by)[file_index_placing]) <= file_end_index
          )
          ## if the file meets all the required conditions
          if (bool_time_after and bool_time_before): return True
      ## otherwise, all specified conditions have been met
      else: return True
    ## otherwise, don't look at the file
    else: return False
  return meetsCondition


def getFilesFromFolder(
    filepath, 
    str_contains       = None,
    str_startswith     = None,
    str_endswith       = None,
    str_not_contains   = None,
    file_index_placing = None,
    file_start_index   = 0,
    file_end_index     = np.inf
  ):
  myFilter = makeFilter(
    str_contains,
    str_not_contains,
    str_startswith,
    str_endswith,
    file_index_placing,
    file_start_index,
    file_end_index
  )
  return list(filter(myFilter, sorted(os.listdir(filepath))))


def readLineFromFile(filepath, des_str, bool_case_sensitive=True):
  for line in open(filepath).readlines():
    if bool_case_sensitive:
      if des_str in line:
        return line
    else:
      if des_str.lower() in line.lower():
        return line
  return None


def createFolder(filepath, bool_hide_updates=False):
  if not(os.path.exists(filepath)):
    os.makedirs(filepath)
    if not(bool_hide_updates):
      print("SUCCESS: Folder created. \n\t" + filepath + "\n")
  elif not(bool_hide_updates):
    print("WARNING: Folder already exists (folder not created). \n\t" + filepath + "\n")


def createFilepath(list_filepath_folders):
  return re.sub( '/+', '/', "/".join([
    folder for folder in list_filepath_folders if not(folder == "")
  ]) )


def createName(list_name_elems):
  return re.sub( '_+', '_', "_".join([
    elems for elems in list_name_elems if not(elems == "")
  ]) )


def copyFileFromNTo(directory_from, directory_to, filename):
  shutil.copy( # copy the file and it's permissions (i.e. executable)
    f"{directory_from}/{filename}",
    f"{directory_to}/{filename}"
  )
  print(f"\t> Successfully coppied: {filename}")
  print(f"\t\t From: {directory_from}")
  print(f"\t\t To: {directory_to}")


## END OF LIBRARY