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

## minimal display name
export PS1="\n\u: \W$ "

# ## Add extra line after executing command
# PS1="\n$PS1"

## home
export HOME=/home/586/$USER
alias dhome="cd $HOME"

## bin
export BIN=$HOME/bin
alias dbin="cd $BIN"

## SCRATCH directories
export SCRATCH_ek9=/scratch/ek9/$USER
export SCRATCH_jh2=/scratch/jh2/$USER

alias dscrek9="cd $SCRATCH_ek9"
alias dscrjh2="cd $SCRATCH_jh2"

# GDATA
export GDATA_ek9=/g/data1b/ek9/$USER
export GDATA_jh2=/g/data1b/jh2/$USER

alias dgdek9="cd $GDATA_ek9"
alias dgdjh2="cd $GDATA_jh2"

## analysis codes / MHD packaage
export MHDCodes=$HOME/MHDCodes
export MHDPackage=$MHDCodes/TheMHDPackage

alias dmhd="cd $MHDCodes"

export PATH=$PATH:$MHDCodes/analysis
export PATH=$PATH:$MHDCodes/scripts
export PYTHONPATH=$PYTHONPATH:$MHDPackage

## FLASH directories
export FLASH=$HOME/nk-flash
export FORCE=$FLASH/source/physics/sourceTerms/Stir/StirFromFileMain/forcing_generator
export STIR=$FLASH/source/Simulation/SimulationMain/StirFromFile

alias dflash="cd $FLASH"
alias dforce="cd $FORCE"
alias dstir="cd $STIR"

## useful 'tool' directories
export TOOLS=$HOME/nk-tools
export SPECTRA=$TOOLS/spectra

alias dtools="cd $TOOLS"

## END OF FILE
