## START OF MODULE


## ###############################################################
## DEPENDENCIES: REQUIRED MODULES
## ###############################################################
import os, io

## relative import of 'createFilepath'
from TheUsefulModule import WWFnF

## always import the c-version of pickle
try: import cPickle as pickle
except ModuleNotFoundError: import pickle


## ###############################################################
## WORKING WITH OBJECTS
## ###############################################################
def savePickleObject(obj, filepath_folder, obj_filename):
  ## create filepath where object will be saved
  obj_filepath = WWFnF.createFilepath([filepath_folder, obj_filename])
  ## if the file exists, then delete it
  if os.path.isfile(obj_filepath):
    os.remove(obj_filepath)
  ## save new object
  with open(obj_filepath, "wb") as output:
    pickle.dump(obj, output, -1)
  ## print success to terminal
  print("\t> Object saved: " + obj_filepath)

def loadPickleObject(
    filepath_folder,
    obj_filename,
    bool_check = False,
    bool_hide_updates = False
  ):
  ## create filepath where object will be loaded from
  obj_filepath = WWFnF.createFilepath([filepath_folder, obj_filename])
  ## if the file exists, then read it in
  if os.path.isfile(obj_filepath):
    if not bool_hide_updates: print("\t> Loading: " + obj_filepath)
    with open(obj_filepath, "rb") as input:
      return pickle.load(input)
  else:
    if bool_check: return -1
    else: raise Exception("No object '{:s}' found in '{:s}'.".format(
      obj_filename,
      filepath_folder
    ))

def updateAttr(obj, attr, desired_val):
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
  ## loop over all the attribute variable names in the object
  for attr in vars(obj):
    print(attr)


## END OF MODULE