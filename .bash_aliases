# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias lltr='ls -lhtr'

# Function for checking whether or not an alias already exists
# and if it doesn't it makes the alias
alias_func() {
  aname=$1
  acmd=$2
  if type $aname >/dev/null 2>&1; then
    echo "$aname is a command or alias already! Not aliasing"
  else
    alias $aname="$acmd"
  fi
}

# To unset functions
unsetfunc() {
   if type $1 >/dev/null 2>&1; then
      unset -f "$@"
   else
      echo "$@ is not a function."
   fi
}

# The if statement below is 
# the equivalent of alias_func for functions!
# Copy the if statement guard to all functions!
if [[ -z $(type -t contains) ]]; then
  contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
  }
else
   echo "contains already defined."
fi


# Check out: http://www.catonmat.net/blog/wp-content/uploads/2008/09/sed1line.txt
# for awesome sed oneliners to make into functions (or aliases)!

# Check out: http://www.catonmat.net/blog/wp-content/uploads/2008/09/awk1line.txt
# for awesome AWK oneliners to make into functions (or aliases)!

alias_func back "cd $OLDPWD"

alias_func basehostname "hostname -s"
alias_func bhname "basehostname"

if [[ -z $(type -t k9) ]]; then
   k9() {
     kill -9 "$@"
   }
else
   echo "k9 already defined."
fi

#if [[ -z $(type -t whereis) ]]; then
#   whereis() {
#      find . -name "$1*";
#   }
#else
#   echo "whereis already defined."
#fi

# Put a symbolic link called (linkname) to PWD in ~
if [[ -z $(type -t lnpwd) ]]; then
   lnpwd() {
     linkname=$1
     ln -s "$PWD" ~/"${linkname}"
   }
else
   echo "lnpwd already defined."
fi

# Dump the output of a man page
# to another (cmd).MAN.OUT without the annoying
# extra characters
if [[ -z $(type -t dump_man) ]]; then
   dump_man() {
      man $1 | col -b > $1.MAN.OUT
   }
else
   echo "dump_man already defined."
fi

if [[ -z $(type -t c) ]]; then
   c() {
      if [ -z "$*" ]; then
         destination=~
      else
         destination=$*
      fi
      builtin cd "${destination}" > /dev/null
      ls -alhtr
      pwd
   }
else
   echo "c already defined."
fi

#l() {
#   ls "$@"
#}

if [[ -z $(type -t mkdircd) ]]; then
   mkdircd() {
      mkdir $1;
      cd $1;
      pwd;
      ls -alhtr;
   }
   alias_func mkc "mkdircd"
else
   echo "mkdircd already defined."
fi

if [[ -z $(type -t cpdir) ]]; then
   cpdir() {
      cp -r $1 $2
   }
else
   echo "cpdir already defined."
fi

alias_func srcit "source ~/.bashrc"
alias_func srcali "source ~/.bash_aliases"

alias_func chali "gvim ~/.bash_aliases &"
alias_func charc "gvim ~/.bashrc &"
alias_func refali "exec bash"

alias_func chvimrc "gvim ~/.vimrc &"
alias_func chcrc "gvim ~/c_code.vim"

if [[ -z $(type -t jason) ]]; then
   jason() {
      skey=$1;
      rep=$2;
      mkdir BACKUP;
      cp *.c BACKUP;
      cp *.h BACKUP;
      perl -p -i -e "s/$1/$2/g" *.c;
      perl -p -i -e "s/$1/$2/g" *.h;
   }
   alias_func undo_jason "cp BACKUP/*.c .;cp BACKUP/*.h .;"
   alias_func jason_undo "undo_jason;"

   alias_func clean_jason "rm -rf BACKUP;"
   alias_func jason_clean "clean_jason;"

else
   echo "jason already defined."
fi

if [[ -z $(type -t v) ]]; then
   v() {
      vim $1;
   }
else
   echo "v already defined."
fi

if [[ -z $(type -t g) ]]; then
   g() {
      gvim "$@" &
   }
else
   echo "g already defined."
fi

if [[ -z $(type -t e) ]]; then
   e() {
      evince "$@" &
   }
else
   echo "e already defined."
fi

alias_func u "cd ..; ls -lhtr; pwd"
alias_func uu "cd ..; ls -lhtr; pwd"

alias_func gurl "git remote -v"
alias_func gba "git branch -a"

# Always override gs. I don't normally use Ghostscript
#alias gs="git status"
gs() {
   if [[ "$#" -eq 1 ]]; then
      num_lines="$1"
   else
      num_lines=10
   fi
   git status | head -n "${num_lines}"
}


alias_func replace_spaces "for f in *\ *; do mv \"\$f\" \"\${f// /_}\"; done"
alias_func repspaces "replace_spaces"

if [[ -z $(type -t gdiff) ]]; then
   gdiff() {
      git diff "$@"
   }
else
   echo "gdiff already defined."
fi

if [[ -z $(type -t gdiffs) ]]; then
   gdiffs() {
      git diff --staged "$@"
   }
else
   echo "gdiffs already defined."
fi

if [[ -z $(type -t gclone) ]]; then
   gclone() {
      url="git@github.com:hazevt04/${1}.git"
      # Extract the directory name from the Git URL
      dirname=$1
      git clone "$url" --branch develop "$dirname"
   }
fi

if [[ -z $(type -t gcloneb) ]]; then
   gcloneb() {
      url="git@github.com:hazevt04/${1}.git"
      branch=$2
      dirname=$1
      git clone "$url" --branch "$branch" "${dirname}_${branch}"
   }
else
   echo "glconeb already defined."
fi

if [[ -z $(type -t gfetch) ]]; then
   gfetch() {
      url="git@github.com:hazevt04/${1}.git"
      branch=$2
      git fetch "$url" "$branch"
   }
else
   echo "gfetch already defined."
fi

if [[ -z $(type -t gpull) ]]; then
   gpull() {
      url="git@github.com:hazevt04/${1}.git"
      branch=$2
      git pull "$url" "$branch"
   }
else
   echo "gpull already defined."
fi

if [[ -z $(type -t gcommit) ]]; then
   gcommit() {
      git commit -m "$@"
   }
else
   echo "gcommit already defined."
fi

alias_func gcomm "gcommit"
   
if [[ -z $(type -t gpo) ]]; then
   gpo() {
      current_branch=$(git status -v | grep "On branch" | cut -d ' ' -f 3)
      git push origin $current_branch
   }
else
   echo "gpo already defined."
fi

#if [[ -z $(type -t gpo) ]]; then
#   gpob() {
#      git push origin $1
#   }
#else
#   echo "gpob already defined."
#fi


if [[ -z $(type -t gadd) ]]; then
   gadd() {
      git add "$@"
   }
else
   echo "gadd already defined."
fi

alias_func gaddu "git add -u"

if [[ -z $(type -t gacp) ]]; then
   gacp() {
      gaddu
      gcomm "$@"
      gpo
   }
else
   echo "gacp already defined."
fi

if [[ -z $(type -t mmm) ]]; then
   mmm() {
      /home/glenn/Programming/Ruby/mad_men_money/mad_men_money "$@"
   }
else
   echo "mmm already defined. Maybe it's already a command or function"
fi

alias_func gpu_info "/usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery"
alias_func gpuinfo "gpu_info"
alias_func gpu_query "gpu_info"
alias_func gpuquery "gpu_info"
alias_func lsgpu "gpu_info"

#alias_func nvvpvm "sudo /usr/local/cuda/bin/nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java"
alias_func nvvps "sudo /usr/local/cuda/bin/nvvp"

# If your GPU app keeps giving memory allocation errors:
alias_func gpu_reset "sudo rmmod nvidia_uvm ; sudo modprobe nvidia_uvm"

alias_func vsc "code"

alias_func inflation_calculator "/home/glenn/Programming/Python/inflation_calculator"

alias_func nvidia_driver_check "sudo ubuntu-drivers devices"

alias_func chrome "google-chrome &"

alias_func gtweaks "gnome-tweaks &"

if [[ -z $(type -t grc) ]]; then
   grc() {
      gnuradio-companion "$@" &
   }
else
   echo "grc already defined. Maybe it's already a command or function"
fi


if [[ -z $(type -t srcfind) ]]; then
   code_find() {
      find . \( -type f \( \( -name "*.cpp" \) -o \( -name "*.hpp" \) -o \( -name "*.cuh" \) -o \( -name "*.cu" \) \) \) -exec grep -Hn $1 {} \;
   }
   alias_func codefind "code_find"
   alias_func src_find "code_find"
   alias_func srcfind "code_find"
else
   echo "multi_find already defined. Maybe it's already a command or function"
fi

