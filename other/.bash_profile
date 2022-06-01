## .bash_profile

##  Commands in this file are executed when a bash user first logs in
##
##  This file may alternatively be called $HOME/.bash_profile,
##  $HOME/.bash_login or $HOME/.profile - the first one of these
##  found is used.
##
##  See http://nf.nci.org.au/facilities/software/modules.php
##  for instructions on how to set your environment to use specific
##  software packages.
##
##  Note that the .bashrc file is NOT executed at login time but 
##  instead every time a sh or bash script is run. By default .bashrc
##  file is NOT sourced at login time but is in subshells. Source here
##  to pick up aliases etc.

[ -f $HOME/.bashrc ] && . $HOME/.bashrc

umask 027

## module load dot adds the current directory to the end of your commands search path
module load dot
module load pbs
module load intel-compiler/2019.3.199
module load intel-python3/2019.3.075
module load hdf5/1.10.5p
module load szip/2.1.1
module load fftw3/3.3.8
module load openmpi/4.0.2
module load ffmpeg


## END OF FILE
