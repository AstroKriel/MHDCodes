## ARCHIVING SCRIPT
## author: Mark Krumholz (6 Dec 2022)
## edited: Neco Kriel (6 Dec 2022)


## import modules
import os, getpass, glob, argparse, copy
import subprocess as sp

## read arguments
parser = argparse.ArgumentParser(description="Data archiving script")
parser.add_argument("-t", "--target",
                    default = None,
                    help    = "name of directory to archive; default is .")
parser.add_argument("-a", "--archivedir",
                    default = None,
                    help    = "name of archive directory; default is " +
                              "all of target directory after username")
parser.add_argument("-d", "--delete",
                    default = False, 
                    action  = "store_true",
                    help    = "delete files after archiving")
parser.add_argument("-c", "--cmd",
                    default = "mdss",
                    help    = "archiving command")
parser.add_argument("-o", "--overwrite",
                    default = False,
                    action  = "store_true",
                    help    = "if set, archive files are overwritten; "+
                              "default is that files whose names and sizes "+
                              "match files already in the archive are skipped")
args = parser.parse_args()

## grab the list of HDF5 and sink files from the target directory
if args.target is not None:
  target = args.target
else: target = ""
hdf5files = glob.glob(os.path.join(target, "*.hdf5"))
sinkfiles = glob.glob(os.path.join(target, "*.sink"))
hdf5files.sort()
sinkfiles.sort()
transferfiles = hdf5files + sinkfiles + [ "orion2.ini" ]

## get name of directory on archive
if args.archivedir is not None:
  archivedir = args.archivedir
else:
  if args.target is not None:
    wd = args.target
  else: wd = os.getcwd()
  uname = getpass.getuser()
  archivedir = ""
  while True:
    wd, dirname = os.path.split(wd)
    if dirname != uname:
      archivedir = os.path.join(dirname, archivedir)
    else: break

## get currently stored file list; create archive directory if necessary
try: curfiles = sp.check_output(
    f"{args.cmd} ls -l {archivedir}",
    shell  = True,
    stderr = sp.STDOUT
  ).split("\n")[1:-1]
except sp.CalledProcessError:
  ## if here: we need to create the directory
  pathcomp = []
  dircpy = copy.deepcopy(archivedir)
  while dircpy and dircpy != os.path.sep:
    dircpy, d = os.path.split(dircpy)
    if d: pathcomp.append(d)
  curpath = pathcomp.pop()
  while True:
    try: sp.check_output(
        f"{args.cmd} ls {curpath}",
        shell  = True,
        stderr = sp.STDOUT
      )
    except sp.CalledProcessError: break
    curpath = os.path.join(curpath, pathcomp.pop())
  sp.call(f"{args.cmd} mkdir {curpath}", shell=True)
  for p in pathcomp[::-1]:
    curpath = os.path.join(curpath, p)
    sp.call(f"{args.cmd} mkdir {curpath}", shell=True)
  curfiles = []

## cull existing files
if not args.overwrite:
  tfiles = []
  cname = []
  csize = []
  for c in curfiles:
    cname.append(c.split()[8])
    csize.append(int(c.split()[4]))
  for t in transferfiles:
    if os.path.basename(t) not in cname:
      tfiles.append(t)
    else:
      # Compare sizes
      fsize = os.path.getsize(os.path.join(target, t))
      archsize = csize[cname.index(t)]
      if fsize != archsize:
        tfiles.append(t)
else:
  tfiles = transferfiles

## now transfer
if len(tfiles) > 0:
  filestr = ""
  for t in tfiles:
    filestr = filestr + " " + t
  cmd = f"{args.cmd} put {filestr} {archivedir}"
  sp.call(cmd, shell=True)

## if requested: delete file that transferred ok
if args.delete:
  curfiles = sp.check_output(
    f"{args.cmd} ls -l {archivedir}",
    shell  = True,
    stderr = sp.STDOUT
  ).split("\n")[1:-1]
  for c in curfiles:
    cname = c.split()[8]
    csize = int(c.split()[4])
    if os.path.isfile(os.path.join(target, cname)):
      if os.path.getsize(os.path.join(target, cname)) == csize:
        os.remove(os.path.join(target, cname))


## END OF ARCHIVER SCRIPT