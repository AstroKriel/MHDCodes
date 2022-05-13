## .bashrc

## Source global definitions (Required for modules)
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

##if in_interactive_shell; then

    ## echo 'in interactive shell'
    
    ## This is where you put settings that you'd like in
    ## interactive shells. E.g. prompts, or aliases
    ## The 'module' command offers path manipulation that
    ## will only modify the path if the entry to be added
    ## is not already present. Use these functions instead of e.g.
    ## PATH=${HOME}/bin:$PATH

    ##prepend_path PATH ${HOME}/bin
    ##prepend_path PATH ${HOME}/.local/bin
    
##    if in_login_shell; then
	## This is where you place things that should only
	## run when you login. If you'd like to run a
	## command that displays the status of something, or
	## load a module, or change directory, this is the
	## place to put it
	## module load pbs
	## cd /scratch/${PROJECT}/${USER}
##    fi

##fi

## Anything here will run whenever a new shell is launched, which
## includes when running commands like 'less'. Commands that
## produce output should not be placed in this section.
##
## If you need different behaviour depending on what machine you're
## using to connect to Gadi, you can use the following test:
##
## if [[ $SSH_CLIENT =~ 11.22.33.44 ]]; then
##     Do something when I connect from the IP 11.22.33.44
## fi
##
## If you want different behaviour when entering a PBS job (e.g.
## a default set of modules), test on the $in_pbs_job variable.
## This will run when any new shell is launched in a PBS job,
## so it should not produce output
##
## if in_pbs_job; then
##      module load openmpi/4.0.1
## fi

## source .alias if exists
test -s ~/.alias && . ~/.alias || true

## add $HOME/bin to PATH, so own programs (e.g., prep_restart.py, etc.) should go in there
export PATH=$HOME/bin:$PATH

## ------------ set Python envirnoment vars ------------
export PYTHONPATH=$HOME/PYTHON
export PYTHONSTARTUP=$PYTHONPATH/python_startup.py
export PATH=$PATH:$PYTHONPATH
export PATH=$PATH:$HOME/.local/bin/
export MATPLOTLIBRC=$PYTHONPATH/matplotlib/matplotlibrc
export MPLCONFIGDIR=$PYTHONPATH/matplotlib

## for IDL environment
export IDL_STARTUP=${HOME}/IDL/idlstartup

# for PETSC
export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH

## end of file