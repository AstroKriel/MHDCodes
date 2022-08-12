## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np


## ###############################################################
## PRINTING TO THE TERMINAL
## ###############################################################
def printInfo(str_justified, input_info, num_char_spacing=15):
  ## make sure that the second input won't overlap with the first
  if len(str_justified) > num_char_spacing:
    num_char_spacing = len(str_justified) + 1
  ## print string
  if isinstance(input_info, str):
    print(str_justified.ljust(num_char_spacing+1) + input_info)
  ## print number (i.e. int, float)
  elif isinstance(input_info, (int, float)):
    print(str_justified.ljust(num_char_spacing+1) + str(input_info))
  ## print list / array
  elif isinstance(input_info, (list, np.ndarray)):
    ## if the list is a list of strings
    if isinstance(input_info[0], str):
      print(str_justified.ljust(num_char_spacing), input_info)
    ## otherwise assume its a list of numbers (i.e. int, float)
    else: print(str_justified.ljust(num_char_spacing), [ str(elem) for elem in input_info ])
  else: Exception("ERROR: Handling of variable type '{}' has not been implemented yet.".format(
    type(input_info)
  ))


## END OF LIBRARY