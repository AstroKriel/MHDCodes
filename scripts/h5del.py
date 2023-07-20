#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys, h5py

from TheUsefulModule import WWTerminal

## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def helpMessage():
    print("""
This program deletes one or more datasets from a HDF5 file.

Usage:
    h5del.py <hdf5 filename> <dataset1> [<dataset2> <dataset3> ...]

Arguments:
    <hdf5 filename>   Name of the HDF5 file to modify
    <dataset1>        Name of the first dataset to delete
    <dataset2>        (Optional) Name of a second dataset to delete
    <dataset3>        (Optional) Name of a third dataset to delete
    ...               (Optional) Additional datasets to delete

Example:
    h5del file.h5 dataset1 dataset2

Notes:
    - Warning: this program won't ask for confirmation before deleting the requested datasets.
    - The original HDF5 file will be modified, so make a backup copy before running this program.
    """)

def h5del(filename, list_dsets, directory):
  def _runCommand(command):
    WWTerminal.runCommand(command, directory)
  if not os.path.isfile(filename):
    raise Exception(f"Error: hdf5-file '{filename}' does not exist")
  with h5py.File(filename, "a") as fp:
    ## check that the datasets exist within the hdf5 file
    list_dsets_to_delete = []
    for dset in list_dsets:
      if dset not in fp:
        print(f"Warning: dataset '{dset}' does not exist in: {filename}")
      else: list_dsets_to_delete.append(dset)
    ## delete all the requested datasets
    for dset in list_dsets_to_delete:
      del fp[dset]
      print(f"Deleted '{dset}' from: {filename}")
  ## adjust hdf5-file size to only what is needed
  if len(list_dsets_to_delete) > 0:
    _runCommand(f"h5repack -i {filename} -o {filename}_tmp")
    _runCommand(f"mv {filename}_tmp {filename}")

## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## check that sufficient amount of input arguments have been provided
  if len(sys.argv) < 3:
    helpMessage()
    sys.exit(1)
  ## extract input arguments
  filename = sys.argv[1]
  list_dsets = sys.argv[2:]
  h5del(filename, list_dsets, os.getcwd())

## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit(0)

## END OF PROGRAM