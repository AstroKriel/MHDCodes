## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import subprocess


## ###############################################################
## HELPFUL FUNCTIONS
## ###############################################################
def printLine(mssg):
  print(mssg, flush=True)

def runCommand(
    command,
    directory          = None,
    bool_debug         = False
  ):
  if bool_debug:
    print(command)
  else:
    p = subprocess.Popen(command, shell=True, cwd=directory)
    p.wait()


## END OF LIBRARY