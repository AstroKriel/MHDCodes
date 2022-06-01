## ~/.bashrc
eval "echo 'Source-ing: .bashrc...'"


## ======== SOURCE ~/.ALIAS ========
## ==================================
if [ -f ~/.alias ]; then
        source ~/.alias
fi


## ======== CONFIGURE WORKING SHELL ========
## =========================================

## hide default shell warning in terminal
export BASH_SILENCE_DEPRECATION_WARNING=1

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


## ======== SET PYTHON ENVIRONMENT VARIABLES ========
## ==================================================
export PYENV_ROOT="$HOME/.pyenv"
export PYTHONPATH="$MHDPackage"


## ======== SET GLOBAL PATH VARIABLE ========
## ==========================================
export PATH=/usr/local/bin:/usr/bin:/bin
# export PATH="$PATH:/Library/TeX/texbin"
export PATH="$PATH:/opt/homebrew/bin"
export PATH="$PATH:$PYENV_ROOT/bin"
export PATH="$PATH:AMReX_ROOT"
export PATH="$PATH:/Applications/VisIt.app/Contents/Resources/bin"
export PATH="$PATH:/Applications/Blender.app/Contents/MacOS"


## ======== ENABLE PYENV ENVIRONMENT ========
## ==========================================
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"


## END OF BASHRC