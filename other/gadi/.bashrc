## ~/.bashrc


## ======== SOURCE GLOBAL .BASHRC ========
## =======================================
## Source global definitions (Required for modules)
source /etc/bashrc


## ======== SOURCE ~/.ALIAS ========
## =================================
source ~/.alias


## ======== CONFIGURE WORKING SHELL ========
## =========================================

# ## load modules
# source ~/modules_flash
# source ~/modules_quokka

## enable group read-ownership of files
umask 027

## up/down arrows trace command history and ignore repeated entries
HISTCONTROL=ignoredups


## ======== CHANGE DISPLAY NAME ========
## =====================================
lightblue="\033[38;5;232;48;5;45m"
green="\e[32m"
yellow="\033[38;5;232;48;5;220m"
white_on_red="\033[38;5;255;48;5;160m"
white="\e[00m"
get_git_branch() {
  git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}
PS1="\n\[${lightblue}\] \w \[${white}\]"                 # full filepath
PS1+="\n\[${green}\]\u\[${white}\]: "                    # username
PS1+="\[${yellow}\] \W \[${white}\]"                     # current folder
PS1+="\[${white_on_red}\]\$(get_git_branch)\[${white}\]" # current branch
PS1+="\[${white}\] -> "                                  # sign-off
export PS1;


## ======== APPEND BINS TO PATH ========
## =====================================
export PATH=$HOME/bin:$PATH
export PATH=$PATH:/bin/gnuplot


## ======== SET PYTHON ENVIRONMENT VARIABLES ========
## ==================================================
export PYTHONPATH=$HOME/PYTHON
export PYTHONSTARTUP=$PYTHONPATH/python_startup.py
export PATH=$PATH:$PYTHONPATH
export PATH=$PATH:$HOME/.local/bin
export MATPLOTLIBRC=$PYTHONPATH/matplotlib/matplotlibrc
export MPLCONFIGDIR=$PYTHONPATH/matplotlib


## ======== DEFINE USER ENVIRONMENT VARIABLES ========
## ===================================================

## user directories: MHD analysis codes
export PATH=$PATH:$MHDCodes/analysis
export PATH=$PATH:$MHDCodes/scripts
export PATH=$PATH:$MHDPackage
export PYTHONPATH=$PYTHONPATH:$MHDPackage

## for Portable, Extensible Toolkit for Scientific Computation (PETSC)
export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH


## END OF BASHRC