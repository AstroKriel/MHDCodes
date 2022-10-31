#!/bin/env python3

## author: Christoph Federrath, 2014-2018
## modified: Neco Kriel, 2022

## ###############################################################
## MODULES
## ###############################################################
import os, sys, subprocess, fnmatch
from tempfile import mkstemp
from shutil import move, copyfile


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def readFromFile_chk(chkfilename, dset, param):
  return_val = 0
  fh, tempfile = mkstemp()
  ftemp = open(tempfile, 'w')
  shellcmd = "h5dump -d \""+dset+"\" "+chkfilename+" |grep '\""+param+"' -A 1"
  subprocess.call(shellcmd, shell=True, stdout=ftemp, stderr=ftemp)
  ftemp.close()
  os.close(fh)
  ftemp = open(tempfile, 'r')
  for line in ftemp:
    if line.find(param)!=-1: continue
    if (dset=="integer scalars") or (dset=="integer runtime parameters"):
      return_val = int(line.lstrip().rstrip())
    if (dset=="real scalars") or (dset=="real runtime parameters"):
      return_val = float(line.lstrip().rstrip())
  os.remove(tempfile)
  if BOOL_DEBUG==True: print(param+" = ", return_val)
  return return_val

def getFileNumber_chk(chkfilename):
  if BOOL_DEBUG==True: print("reading checkpointfilenumber from "+chkfilename+"...")
  return_val = readFromFile_chk(chkfilename, "integer scalars", "checkpointfilenumber")
  if BOOL_DEBUG==True: print("checkpointfilenumber = ", return_val)
  return return_val

def getNextFileNumber_plt(chkfilename):
  if BOOL_DEBUG==True: print("reading plotfilenumber from "+chkfilename+"...")
  return_val = readFromFile_chk(chkfilename, "integer scalars", "plotfilenumber")
  if BOOL_DEBUG==True: print("next plotfilenumber = ", return_val)
  return return_val

def getNextFileNumber_particle(chkfilename):
  if BOOL_DEBUG==True: print("reading particlefilenumber from "+chkfilename+"...")
  return_val = readFromFile_chk(chkfilename, "integer scalars", "particlefilenumber")
  if BOOL_DEBUG==True: print("next particlefilenumber = ", return_val)
  return return_val

def getNextFileNumber_movie(chkfilename):
  if BOOL_DEBUG==True: print("computing next movie file number from "+chkfilename+"...")
  try:
    movie_dump_num = readFromFile_chk(chkfilename, "integer runtime parameters", "movie_dump_num")
  except ValueError:
    print("Movie module was not compiled in. Proceeding without...")
    return -1
  movie_dstep_dump = readFromFile_chk(chkfilename, "integer runtime parameters", "movie_dstep_dump")
  movie_dt_dump = readFromFile_chk(chkfilename, "real runtime parameters", "movie_dt_dump")
  simtime = readFromFile_chk(chkfilename, "real scalars", "time")
  simstep = readFromFile_chk(chkfilename, "integer scalars", "nstep")
  if BOOL_DEBUG==True: print("movie_dstep_dump, movie_dt_dump = ", movie_dstep_dump, movie_dt_dump)
  if movie_dstep_dump==0 and movie_dt_dump==0.0:
    return -1
  if movie_dstep_dump==0:
    return_val = int(simtime/movie_dt_dump)+1
  if movie_dt_dump==0.0:
    return_val = int(simstep/movie_dstep_dump)+1
  if movie_dump_num!=return_val:
    print("CAUTION: movie_dump_num in chk file is NOT equal to computed next movie_dump_num! Using chk movie_dump_num anyway.")
  return_val = movie_dump_num
  if BOOL_DEBUG==True: print("next movie_dump_num = ", return_val)
  return return_val

def inplaceChangeFlashParamFile(filename, restart, chknum, pltnum, partnum, movnum):
  fh, tempfile = mkstemp()
  ftemp = open(tempfile, 'w')
  f = open(filename, 'r')
  for line in f:
    # replace restart
    if line.lower().find("restart")==0:
      if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
      i = line.find("=")
      newline = line[0:i+1]+" "+restart+"\n"
      line = newline
      if BOOL_DEBUG==True: print(filename+": replaced with: "+line.rstrip())
    # replace checkpointfilenumber
    if line.lower().find("checkpointfilenumber")==0:
      if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
      i = line.find("=")
      newline = line[0:i+1]+" %(#)d\n" % {"#":chknum}
      line = newline
      if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
    # replace plotfilenumber
    if line.lower().find("plotfilenumber")==0:
      if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
      i = line.find("=")
      newline = line[0:i+1]+" %(#)d\n" % {"#":pltnum}
      line = newline
      if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
    # replace particlefilenumber
    if line.lower().find("particlefilenumber")==0:
      if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
      i = line.find("=")
      newline = line[0:i+1]+" %(#)d\n" % {"#":partnum}
      line = newline
      if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
    # replace movie_dump_num
    if line.lower().find("movie_dump_num")==0:
      if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
      i = line.find("=")
      newline = line[0:i+1]+" %(#)d\n" % {"#":movnum}
      line = newline
      if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
      # replace init_restart with .false. (if present)
      if line.lower().find("init_restart")==0:
          if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
          i = line.find("=")
          newline = line[0:i+1]+".false.\n"
          line = newline
          if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
    # add lines to temporary output file
    ftemp.write(line)
  ftemp.close()
  os.close(fh)
  f.close()
  os.remove(filename)
  move(tempfile, filename)
  os.chmod(filename, 0o644)

def getJobFileName():
  if os.path.isfile("job.cmd"):
    print("Found job.cmd. Replacing shell.out...")
    return "job.cmd"
  if os.path.isfile("job.sh"):
    print("Found job.sh. Replacing shell.out...")
    return "job.sh"

def inplaceChangeJobFile(filename, newflag):
  fh, tempfile = mkstemp()
  ftemp = open(tempfile, 'w')
  f = open(filename, 'r')
  for line in f:
    # increment shell*.out
    if line.find("shell.out")>0:
      if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
      ibeg = line.find("shell.out")+9
      iend = line.find(" 2>&1")
      count_string = line[ibeg:iend].rstrip()
      if count_string == "":
        count = 0
      else:
        count = int(count_string)
      count += 1
      if newflag==True: count = 0
      new_count_string = "%(#)02d" % {"#":count}
      newline = line[0:ibeg]+new_count_string+line[iend:len(line)]
      line = newline
      if BOOL_DEBUG==True: print(filename+": found line   : "+line.rstrip())
    # add lines to temporary output file
    ftemp.write(line)
  ftemp.close()
  os.close(fh)
  f.close()
  os.remove(filename)
  move(tempfile, filename)
  os.chmod(filename, 0o644)

def helpMe(args, nargs):
  for arg in args:
    if ((nargs < 2) or arg.find("-help")!=-1) or (arg.find("--help")!=-1):
      print(" ")
      print("USAGE options for "+args[0]+":")
      print(" "+args[0]+" <filename> (filename must be a FLASH chk file for restart)")
      print(" "+args[0]+" -auto      (uses last available chk file in current directory)")
      print(" "+args[0]+" -new       (prepares new simulation: restart = .false.)")
      print(" ")
      quit()


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  args = sys.argv
  nargs = len(sys.argv)
  helpMe(args, nargs)
  ## make a backup copy of flash.par
  print("Copying 'flash.par' to 'flash.par_restart_backup' as backup...")
  copyfile("flash.par","flash.par_restart_backup")
  print(" ")
  ## reset flash.par for new simulations (restart = .false.)
  if args[1] == "-new":
    print("Resetting flash.par for starting FLASH from scratch (restart = .false.):")
    inplaceChangeFlashParamFile("flash.par", ".false.", 0, 0, 0, 0)
    job_file = getJobFileName()
    if job_file: inplaceChangeJobFile(job_file, True)
    quit()
  ## automatically determine last chk file and prepare restart = .true.
  if args[1] == "-auto":
    chkfiles = []
    for file in os.listdir('.'):
      if fnmatch.fnmatch(file, '*_hdf5_chk_*'):
        chkfiles.append(file)
    chkfiles.sort()
    chkfilename = chkfiles[len(chkfiles)-1]
    print(f"Found the following last FLASH checkpoint file: '{chkfilename}'.")
    print("...and using it to prepare restart:")
    chknum = getFileNumber_chk(chkfilename)
    pltnum = getNextFileNumber_plt(chkfilename)
    partnum = getNextFileNumber_particle(chkfilename)
    movnum = getNextFileNumber_movie(chkfilename)
    inplaceChangeFlashParamFile("flash.par", ".true.", chknum, pltnum, partnum, movnum)
    job_file = getJobFileName()
    if job_file: inplaceChangeJobFile(job_file, False)
    quit()
  ## assume that user supplied the name of a valid chk file
  chkfilename = args[1]
  print("Using FLASH checkpoint file '"+chkfilename+"' to prepare restart:")
  chknum = getFileNumber_chk(chkfilename)
  pltnum = getNextFileNumber_plt(chkfilename)
  partnum = getNextFileNumber_particle(chkfilename)
  movnum = getNextFileNumber_movie(chkfilename)
  inplaceChangeFlashParamFile("flash.par", ".true.", chknum, pltnum, partnum, movnum)
  job_file = getJobFileName()
  if job_file: inplaceChangeJobFile(job_file, False)


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BOOL_DEBUG = False


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM