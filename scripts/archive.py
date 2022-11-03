#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by Christoph Federrath, 2021
#  based on a script by Mark Krumholz

# Archiving script
import os
import os.path as osp
import getpass
import subprocess as sp
import glob
import argparse
import copy

# Read arguments
parser = argparse.ArgumentParser(description='Data archiving script.')
parser.add_argument("-i", dest='inputfiles', nargs='*', help="File(s) to archive (sub-directories are ignored).")
parser.add_argument('-a', '--archivedir', default=None, help='Name of archive directory; default is all of target directory after username.')
parser.add_argument('-d', '--delete', default=False, action='store_true', help='Delete files after archiving.')
parser.add_argument('-c', '--cmd', default='mdss', help='Archiving command (default: mdss)')
parser.add_argument('-o', '--overwrite', default=False, action='store_true', help='If set, archive files are overwritten; '+
          'default is that files whose names and sizes match files already in the archive are skipped.')
parser.add_argument('-s', '--skip_user_confirmation', default=False, action='store_true',
          help='Do not ask user to confirm skipping of sub-directories / unreadable files.')
args = parser.parse_args()

if args.inputfiles is None:
  print("Error -- inputfiles for archiving must be specified; use option -i")
  exit()

# make list of files to transfer to archive
check_transferfiles = sorted([x for x in list(args.inputfiles)])
# check that we are running the command from within the folder where the files sit;
# otherwise paths handling gets more complicated
for f in check_transferfiles:
  if '/' in f:
    print("Error -- only files allowed as input, no directories; need to run from within archiving dir")
    exit()
# further check for read permission and ignore sub-directories
transferfiles = []
for f in check_transferfiles:
  if '/' in f:
    print("Error -- only files allowed as input, no directories; need to run from within archiving dir")
    exit()
  if not os.access(f, os.R_OK):
    error_msg = "Ignoring '"+f+"', which does not have read permission; press Enter to continue..."
    if args.skip_user_confirmation: print(error_msg)
    else: input(error_msg)
    continue
  if os.path.isdir(f):
    error_msg = "Ignoring '"+f+"', which is a directory; press Enter to continue..."
    if args.skip_user_confirmation: print(error_msg)
    else: input(error_msg)
    continue
  # append to actual transfer list
  transferfiles.append(f)

# get user name
uname = getpass.getuser()

# Get name of directory on archive
if args.archivedir is not None:
  archivedir = args.archivedir
else:
  wd = os.getcwd()
  archivedir = ''
  while True:
    wd, dirname = osp.split(wd)
    if dirname != uname:
      archivedir = osp.join(dirname, archivedir)
    else:
      break

print("---\nUsing archive directory path: ", uname+'/'+archivedir)

# Get currently stored file list; create archive directory if necessary
print("---\nChecking files already in archive...")
try:
  curfiles = sp.check_output(args.cmd + " ls -l " + uname+'/'+archivedir, shell=True, stderr=sp.STDOUT).decode().split('\n')[1:-1]
except sp.CalledProcessError:
  # If here, we need to create the directory
  pathcomp = []
  dircpy = copy.deepcopy(archivedir)
  while dircpy and dircpy != osp.sep:
    dircpy, d = osp.split(dircpy)
    if d:
      pathcomp.append(d)
  curpath = pathcomp.pop()
  while True:
    try:
      sp.check_output(args.cmd + " ls " + uname+'/'+curpath, shell=True, stderr=sp.STDOUT)
    except sp.CalledProcessError:
      break
    curpath = osp.join(curpath, pathcomp.pop())
  sp.call(args.cmd + " mkdir " + uname+'/'+curpath, shell=True)
  for p in pathcomp[::-1]:
    curpath = osp.join(curpath, p)
    sp.call(args.cmd + " mkdir " + uname+'/'+curpath, shell=True)
  curfiles = []

# Cull existing files
if not args.overwrite:
  tfiles = []
  cname = []
  csize = []
  for c in curfiles:
    cname.append(c.split()[8])
    csize.append(int(c.split()[4]))
  for t in transferfiles:
    if osp.basename(t) not in cname:
      tfiles.append(t)
    else:
      # Compare sizes
      fsize = osp.getsize(t)
      archsize = csize[cname.index(t)]
      if fsize != archsize:
        tfiles.append(t)
      else:
        print("Skipping file already in archive: ", t)
else:
  tfiles = transferfiles

# Now transfer
print("---\nNow starting to transfer files...")
if len(tfiles) > 0:
  filestr = ''
  for t in tfiles:
    filestr = filestr + ' ' + t
  cmd = args.cmd + ' put' + filestr + ' ' + uname+'/'+archivedir
  print(cmd)
  sp.call(cmd, shell=True)


# If requested, delete file that transferred ok
if args.delete:
  print("---\nNow deleting local files...")
  curfiles = sp.check_output(args.cmd + " ls -l " + uname+'/'+archivedir, shell=True, stderr=sp.STDOUT).decode().split('\n')[1:-1]
  for c in curfiles:
    cname = c.split()[8]
    csize = int(c.split()[4])
    if osp.isfile(cname):
      if osp.getsize(cname) == csize:
        print("Deleting file: ", cname)
        os.remove(cname)