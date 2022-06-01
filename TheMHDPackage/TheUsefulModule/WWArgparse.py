## START OF MODULE


## ###############################################################
## MODULES
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
  elif v.lower() in ("yes", "true", "t", "y", "1"): return True
  elif v.lower() in ("no", "false", "f", "n", "0"): return False
  else: raise argparse.ArgumentTypeError("Boolean value expected.")

class MyHelpFormatter(argparse.RawDescriptionHelpFormatter):
  def _format_action(self, action):
    parts = super(argparse.RawDescriptionHelpFormatter, self)._format_action(action)
    if action.nargs == argparse.PARSER:
      parts = "\n".join(parts.split("\n")[1:])
    return parts

class MyParser(argparse.ArgumentParser):
  def __init__(self, description):
    super(MyParser, self).__init__(
      description     = description,
      formatter_class = lambda prog: MyHelpFormatter(prog, max_help_position=50),
    )
  def error(self, message):
    sys.stderr.write("ERROR: {}\n\n".format(message))
    self.print_help()
    sys.exit(2)


## END OF MODULE