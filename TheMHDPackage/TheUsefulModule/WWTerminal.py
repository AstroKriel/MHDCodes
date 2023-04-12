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
    bool_print_command = True,
    bool_debug         = False
  ):
  if bool_print_command: print(command)
  if not bool_debug:
    p = subprocess.Popen(command, shell=True, cwd=directory)
    p.wait()


## END OF LIBRARY