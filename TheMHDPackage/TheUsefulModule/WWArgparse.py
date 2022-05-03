## START OF MODULE


## ###############################################################
## DEPENDENCIES: REQUIRED MODULES
## ###############################################################
import sys, argparse


## ###############################################################
## WORKING WITH ARGUMENT INPUTS
## ###############################################################
def str2bool(v):
  """ str2bool
  BASED ON: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
  """
  if isinstance(v, bool): return v
  if v.lower() in ("yes", "true", "t", "y", "1"): return True
  elif v.lower() in ("no", "false", "f", "n", "0"): return False
  else: raise argparse.ArgumentTypeError("Boolean value expected.")

class MyParser(argparse.ArgumentParser):
  def error(self, message):
    self.print_help()
    sys.exit(1)


## END OF MODULE