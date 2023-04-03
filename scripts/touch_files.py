#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys, time, glob, random, datetime
from numba import jit


## ###############################################################
## HELPER FUNCTION: connvert time in seconds to date
## ###############################################################
def getDate(time_sec):
  return datetime.datetime.fromtimestamp(time_sec).strftime("%d/%m/%Y %H:%M:%S")


## ###############################################################
## MAIN PROGRAM FUNCTIONS
## ###############################################################
def touch(sub_directory):
  if BOOL_VERBOSE: print("T-ing:", sub_directory)
  if BOOL_TOUCH:
    if BOOL_RANDOM:
      time_in_a_week = 60 * 60 * 24 * 7
      time_ago = time_in_a_week * random.random()
    else: time_ago = 0
    new_time_access   = CURRENT_TIME - time_ago
    new_time_modified = CURRENT_TIME - time_ago
    try: os.utime(sub_directory, (new_time_access, new_time_modified))
    except: print("ERROR: couldn't modify directory, probably due to file permission issues:", sub_directory)

def lookAtSubDirectory(directory, file_type, count):
  print("Looking at:", directory)
  for sub_directory in glob.glob(f"{directory}/{file_type}"):
    if os.path.isfile(sub_directory):
      touch(sub_directory)
      count += 1
    if os.path.isdir(sub_directory):
      touch(sub_directory)
      count += 1
      count = lookAtSubDirectory(sub_directory, file_type, count)
  return count


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
CURRENT_TIME = time.mktime(time.localtime())
BOOL_TOUCH   = 0
BOOL_VERBOSE = 0
BOOL_RANDOM  = 0


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  start_time_sec = time.time()
  count = lookAtSubDirectory(directory=".", file_type="*", count=0)
  print(f"Number of files processed: {count:.3E}")
  print(f"Execution time: {time.time() - start_time_sec:.2f} seconds")
  sys.exit()


## END OF PROGRAM