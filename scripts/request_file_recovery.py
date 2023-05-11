#!/usr/bin/env python3
import os
from datetime import date
from tqdm.auto import tqdm


## ###############################################################
## CLASS HANDINGLING REQUESTS
## ###############################################################
class FileRecovery():
  def __init__(self):
    ## create filenames
    self.filename_quarantined    = f"list_quarantined_files_{date.today()}.txt"
    self.filename_batch_recovery = f"list_quarantined_files_{date.today()}_batch_request.txt"

  def fetchQuarantinedFiles(self):
    ## pipe the list of quarantined files into a text-file
    print("Fetching list of quarantined files...")
    os.system(f"nci-file-expiry list-quarantined > {self.filename_quarantined}")
  
  def requestIndividualRecovery(self):
    print("Requesting file recovery (one-at-a-time)...")
    ## request quarantined files be put back one-at-a-time
    with open(self.filename_quarantined, "r") as file:
      list_files = file.readlines()
      print(f"There are {len(list_files)} quarantined files.")
      file_count = 1
      for line in list_files:
        file_id = line.split(" ")[0]
        file_destination = line.split(" ")[-1].replace("\n", "")
        if not("PATH".lower() in file_destination.lower()):
          ## if the file has not already been returned
          if not os.path.isfile(file_destination):
            print(f"Processing request {file_count}: {file_id} {file_destination}")
            os.system(f"nci-file-expiry recover {file_id} {file_destination}")
          ## increment file
          file_count += 1

  def createBatchFile(self):
    ## pipe all quarantined file information into a file
    self.fetchQuarantinedFiles()
    ## filter information for ID and PATH
    print("Creating file for batch recovery...")
    with open(self.filename_batch_recovery, "w") as file_batch:
      with open(self.filename_quarantined, "r") as file_full:
        list_files = file_full.readlines()
        num_quarantined_files = len(list_files)
        print(f"There are {num_quarantined_files} quarantined files.")
        num_files_already_put_back = 1
        for line in tqdm(list_files):
          file_id = line.split(" ")[0]
          file_destination = line.split(" ")[-1].replace("\n", "")
          if not("PATH".lower() in file_destination.lower()):
            ## if the file has not already been returned
            if not os.path.isfile(file_destination):
              file_batch.write(f"{file_id} {file_destination}\n")
            else: num_files_already_put_back += 1
        print(f"There are {num_files_already_put_back} quarantined files that have already been put back.")
    return (num_quarantined_files - num_files_already_put_back) > 0

  def requestBatchRecovery(self):
    ## create file for batch recovery
    if self.createBatchFile():
      ## request batch file-recovery
      print("Requesting batch file-recovery...")
      os.system(f"nci-file-expiry batch-recover {self.filename_batch_recovery}")
    else: print("All quarantined files have already been put back.")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  obj = FileRecovery()
  obj.requestBatchRecovery()


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


# END OF PROGRAM
