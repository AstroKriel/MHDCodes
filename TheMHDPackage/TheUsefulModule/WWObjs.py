## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import os
import json
import numpy as np

## relative import of 'createFilepath'
from TheUsefulModule import WWFnF

## always import the c-version of pickle
try: import cPickle as pickle
except ModuleNotFoundError: import pickle


## ###############################################################
## WORKING WITH PICKLE-FILES
## ###############################################################
def saveObj2Pickle(
    obj,
    filepath,
    filename,
    bool_hide_updates = False
  ):
  ## create filepath where object is to be saved
  filepath_file = WWFnF.createFilepath([filepath, filename])
  ## save file
  with open(filepath_file, "wb") as output:
    pickle.dump(obj, output, -1) # -1 specifies highest binary protocol
  ## indicate success
  if not(bool_hide_updates):
    print("Saved pickle-file:", filepath_file)

def loadPickle2Obj(
    filepath,
    filename,
    bool_raise_error  = False,
    bool_hide_updates = False
  ):
  ## create filepath where object is stored
  filepath_file = WWFnF.createFilepath([filepath, filename])
  ## read file if it exists
  if os.path.isfile(filepath_file):
    if not(bool_hide_updates):
      print("Reading in pickle-file:", filepath_file)
    with open(filepath_file, "rb") as input:
      return pickle.load(input)
  ## indicate the file was not found
  else:
    ## raise exception
    if bool_raise_error:
      Exception("No pickle-file '{:s}' found in '{:s}'.".format(
        filename, filepath
      ))
    ## return flag
    return -1


## ###############################################################
## WORKING WITH JSON-FILES
## ###############################################################
class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def saveObj2Json(
    obj,
    filepath,
    filename,
    bool_hide_updates = False
  ):
  ## create filepath where object is to be saved
  filepath_file = WWFnF.createFilepath([filepath, filename])
  ## save object to file
  with open(filepath_file, "w") as file_pointer:
    json.dump(
      obj       = vars(obj), # store member variables in a dictionary
      fp        = file_pointer,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  ## indicate success
  if not(bool_hide_updates):
    print("Saved json-file:", filepath_file)

def loadJson2Dict(
    filepath,
    filename,
    bool_raise_error  = False,
    bool_hide_updates = False
  ):
  ## create filepath where object is stored
  filepath_file = WWFnF.createFilepath([filepath, filename])
  ## read file if it exists
  if os.path.isfile(filepath_file):
    if not(bool_hide_updates):
      print("Reading in json-file:", filepath_file)
    with open(filepath_file, "r") as input:
      return json.load(input)
  ## indicate the file was not found
  else:
    ## raise exception
    if bool_raise_error:
      Exception("No json-file '{:s}' found in '{:s}'.".format(
        filename, filepath
      ))
    ## return flag
    return -1


## ###############################################################
## WORKING WITH OBJECTS
## ###############################################################
def updateObjAttr(obj, attr, desired_val):
  ## check that the new attribute value is not None
  if desired_val is not None:
    ## check that the new value is not the same as the old value
    if not(getattr(obj, attr) == desired_val):
      ## change the attribute value
      setattr(obj, attr, desired_val)
      return True
  ## don't change the attribute value
  return False

def printObjAttrNames(obj):
  ## loop over all the attribute-names in the object
  for attr in vars(obj):
    print(attr)


## END OF LIBRARY