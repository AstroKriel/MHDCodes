## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import os, json
import numpy as np

## import user defined modules
from TheUsefulModule import WWFnF


## ###############################################################
## WORKING WITH JSON-FILES
## ###############################################################
class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):    return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray):  return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def saveObj2JsonFile(obj, filepath, filename, bool_verbose=True):
  ## create filepath where object will be saved
  filepath_file = f"{filepath}/{filename}"
  ## save object to file
  with open(filepath_file, "w") as fp:
    json.dump(
      obj       = vars(obj), # store obj member-variables in a dictionary
      fp        = fp,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  ## indicate success
  if bool_verbose: print("Saved json-file:", filepath_file)

## ###############################################################
## WORKING WITH DICTIONARIES
## ###############################################################
def getDictWithoutKeys(input_dict, list_keys):
  return {
    k : v
    for k, v in input_dict.items()
    if k not in list_keys
  }

def saveDict2JsonFile(filepath_file, input_dict, bool_verbose=True):
  ## if json-file already exists, then append dictionary
  if os.path.isfile(filepath_file):
    appendDict2JsonFile(filepath_file, input_dict, bool_verbose)
  ## create json-file with dictionary
  else: createJsonFile(filepath_file, input_dict, bool_verbose)

def createJsonFile(filepath_file, dict2save, bool_verbose=True):
  filepath_file = filepath_file.replace("//", "/")
  with open(filepath_file, "w") as fp:
    json.dump(
      obj       = dict2save,
      fp        = fp,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  if bool_verbose: print("Saved json-file:", filepath_file)

def appendDict2JsonFile(filepath_file, dict2append, bool_verbose=True):
  ## read json-file into dict
  with open(filepath_file, "r") as fp_r:
    dict_old = json.load(fp_r)
  ## append extra contents to dict
  dict_old.update(dict2append)
  ## update (overwrite) json-file
  with open(filepath_file, "w+") as fp_w:
    json.dump(
      obj       = dict_old,
      fp        = fp_w,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  if bool_verbose: print("Updated json-file:", filepath_file)

def readJsonFile2Dict(filepath, filename, bool_verbose=True):
  filepath_file = f"{filepath}/{filename}"
  ## read file if it exists
  if os.path.isfile(filepath_file):
    if bool_verbose: print("Reading in json-file:", filepath_file)
    with open(filepath_file, "r") as input:
      return json.load(input)
  ## indicate the file was not found
  else: raise Exception(f"Error: No json-file found: {filepath_file}")


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