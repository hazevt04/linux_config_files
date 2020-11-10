# BASH Aliases

# Enable color support of ls and also add handy aliases
if [[ -x /usr/bin/dircolors ]]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
#alias ll='ls -alFh'
#alias la='ls -A'
#alias l='ls -CF'
#alias lltr='ls -ltrh'

# Now using exa instead of ls

if [[ -z $(type -t exa) ]]; then
  echo "exa, the modern replacement for ls not found. (Google it). Using default ls for now"
  alias l='ls'
  alias ll='ls -l'
  alias la='ls -a'
  alias lla='ls -la'
  alias ltr='ls -ltr'
  alias ltree='ls -R'
else
  alias l='exa --git'
  alias ll='exa --long --header --git'
  alias la='exa --all --git'
  alias lla='exa --long --all --git'
  alias ltr='exa --long --time=modified --reverse --git'
  alias ltree='exa --long --tree --git'
fi


alias myprocs="ps aux | grep ghazelw"

# Safety aliases
alias rm='rm -i'

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

#alias_func u "cd ..; ls -lhtr; pwd"
#alias_func uu "cd ../..; ls -lhtr; pwd"
#alias_func uuu "cd ../../..; ls -lhtr; pwd"
alias_func u "cd ..; ltr; pwd"
alias_func uu "cd ../..; ltr; pwd"
alias_func uuu "cd ../../..; ltr; pwd"

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
fi


# Check out: http://www.catonmat.net/blog/wp-content/uploads/2008/09/sed1line.txt
# for awesome sed oneliners to make into functions (or aliases)!

# Check out: http://www.catonmat.net/blog/wp-content/uploads/2008/09/awk1line.txt
# for awesome AWK oneliners to make into functions (or aliases)!

alias_func back "cd $OLDPWD"

alias_func basehostname "hostname -s"
alias_func bhname "basehostname"

if [[ -z $(type -t disk_usage) ]]; then
   disk_usage() {
      du -sh -- "$@" | sort -h
   }
   alias_func dush "disk_usage"
   # quota_check already a command
   alias_func mydu "disk_usage ~"
   alias_func duall "disk_usage *"
fi

alias_func diskcheck "df -h ~${USER}"
alias_func disk_check "diskcheck"
alias_func myspace "diskcheck"
alias_func room "diskcheck"
#alias_func dc "diskcheck"



if [[ -z $(type -t k9) ]]; then
   k9() {
     kill -9 "$@"
   }
fi

if [[ -z $(type -t whereis) ]]; then
   whereis() {
      find . -name "$1*";
   }
fi

if [[ -z $(type -t searchcode) ]]; then
   searchcode() {
      find . -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" -o -name "*.cuh" \) -exec grep -Hn "$@" {} \;
   }

   alias_func search_code "searchcode"
   alias_func find_in_code "searchcode"
   alias_func fic "searchcode"
fi


# Put a symbolic link called (linkname) to PWD in ~
if [[ -z $(type -t lnpwd) ]]; then
   lnpwd() {
     linkname=$1
     ln -s "$PWD" ~/"${linkname}"
   }
fi

# Dump the output of a man page
# to another (cmd).MAN.OUT without the annoying
# extra characters
if [[ -z $(type -t dump_man) ]]; then
   dump_man() {
      man $1 | col -b > $1.MAN.OUT
   }
fi

if [[ -z $(type -t c) ]]; then
   c() {
      if [ -z "$*" ]; then
         destination=~
      else
         destination=$*
      fi
      builtin cd "${destination}" > /dev/null;
      ltr;
      pwd;
   }
fi

if [[ -z $(type -t mkdircd) ]]; then
   mkdircd() {
      mkdir $1;
      cd $1;
      pwd;
      ltr;
   }
   alias_func mkc "mkdircd"
fi

alias_func srcit "source ~/.bashrc"
alias_func srcali "source ~/.bash_aliases"

alias_func chali "vim ~/.bash_aliases"
alias_func chalig "gvim ~/.bash_aliases &"
alias_func charc "vim ~/.bashrc.$USER"
alias_func charcg "gvim ~/.bashrc.$USER &"
# Since I keep doing this typo
alias_func chargc "gvim ~/.bashrc.$USER &"
alias_func refali "exec bash"
alias_func reafli "exec bash"

alias_func chvimrc "vim ~/.vimrc"
alias_func chvimrcg "gvim ~/.vimrc &"
alias_func chavimrc "chvimrc"
alias_func chavimrcg "chvimrcg"
alias_func chcrc "vim ~/c_code.vim"
alias_func chcrcg "gvim ~/c_code.vim &"

if [[ -z $(type -t jason) ]]; then
   jason() {
      skey=$1;
      rep=$2;
      mkdir BACKUP;
      cp *.c BACKUP;
      cp *.cpp BACKUP;
      cp *.h BACKUP;
      cp *.cu BACKUP;
      cp *.cuh BACKUP;
      perl -p -i -e "s/$1/$2/g" *.c;
      perl -p -i -e "s/$1/$2/g" *.cpp;
      perl -p -i -e "s/$1/$2/g" *.h;
      perl -p -i -e "s/$1/$2/g" *.cu;
      perl -p -i -e "s/$1/$2/g" *.cuh;
   }
fi

alias_func undo_jason "cp BACKUP/*.c .;cp BACKUP/*.cpp .;cp BACKUP/*.h .;cp BACKUP/*.cu .;cp BACKUP/*.cuh ."
alias_func jason_undo "undo_jason;"

alias_func clean_jason "rm -rf BACKUP"
alias_func jason_clean "clean_jason"

if [[ -z $(type -t v) ]]; then
   v() {
      vim $1;
   }
fi

if [[ -z $(type -t g) ]]; then
   g() {
      gvim "$@" &
   }
fi

if [[ -z $(type -t gro) ]]; then
   gro() {
      gvim -RO "$@" &
   }
fi

if [[ -z $(type -t e) ]]; then
   e() {
      evince "$@" &
   }
fi

alias_func chrome "google-chrome &"

###########################
## Libre Office Aliases
###########################
if [[ -z $(type -t libre_calc) ]]; then
   libre_calc() {
      scalc --nologo "$@" &
   }   

   alias_func librecalc "libre_calc"
   alias_func lcalc "libre_calc"
   alias_func spreadsheet "libre_calc"
   alias_func excel "libre_calc"
   
fi


if [[ -z $(type -t libre_writer) ]]; then
   libre_writer() {
      swriter --nologo "$@" &
   }   
   
   alias_func librewriter "libre_swriter"
   alias_func lwriter "libre_writer"
   alias_func word "libre_writer"
   alias_func office "libre_writer"
fi

if [[ -z $(type -t libre_impress) ]]; then
   libre_impress() {
      simpress --nologo "$@" &                                                                                                                                                                     
   }   
   
   alias_func libreimpress "libre_impress"
   alias_func limpress "libre_impress"
   alias_func powerpoint "libre_impress"
fi



###########################
## Git Aliases
###########################

alias_func gurl "git remote -v"
alias_func ginit "git init; git config user.name \"hazevt04\""

alias_func ginitwork "git init; git config user.name ${USER}"

alias_func gba "git branch -a"

# Always override gs. I don't normally use Ghostscript
#alias gs="git status | head"
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

alias_func glog "git log"

if [[ -z $(type -t gdiff) ]]; then
   gdiff() {
      git diff "$@"
   }
fi

if [[ -z $(type -t gdiffs) ]]; then
   gdiffs() {
      git diff --staged "$@"
   }
fi

if [[ -z $(type -t gclone) ]]; then
   gclone() {
      url="${MY_GIT_SERVER}/${1}.git"
      # Extract the directory name from the Git URL
      dirname=$1
      git clone "$url" --branch develop "$dirname"
   }
fi

if [[ -z $(type -t gcloneb) ]]; then
   gcloneb() {
      url="${MY_GIT_SERVER}/${1}.git"
      branch=$2
      dirname=$1
      git clone "$url" --branch "$branch" "${dirname}_${branch}"
   }
fi

if [[ -z $(type -t gfetch) ]]; then
   gfetch() {
      url="${MY_GIT_SERVER}/${1}.git"
      branch=$2
      git fetch "$url" "$branch"
   }
fi

if [[ -z $(type -t gpull) ]]; then
   gpull() {
      url="${MY_GIT_SERVER}/${1}.git"
      branch=$2
      git pull "$url" "$branch"
   }
fi

if [[ -z $(type -t gcommit) ]]; then
   gcommit() {
      git commit -m "$@"
   }
   alias_func gcomm "gcommit"
fi

   
if [[ -z $(type -t gpo) ]]; then
   gpo() {
      current_branch=$(git status -v | grep "On branch" | cut -d ' ' -f 3)
      git push -u origin $current_branch
   }
fi

if [[ -z $(type -t gpo) ]]; then
   gpob() {
      git push origin $1
   }
fi


if [[ -z $(type -t gadd) ]]; then
   gadd() {
      git add "$@"
   }
fi

alias_func gaddu "git add -u"

if [[ -z $(type -t gacp) ]]; then
   gacp() {
      gaddu
      gcomm "$@"
      gpo
   }
fi

if [[ -z $(type -t gcheck) ]]; then
    gcheck() {
      git checkout "$@"
    }
    
    alias_func gch "gcheck"
    alias_func gchk "gcheck"
    alias_func gcheckout "gcheck"
fi


if [[ -z $(type -t gstash) ]]; then
    gstash() {
      git stash
    }

    alias_func gst "gstash"
fi

###################################
# tmux aliases
###################################
alias_func tmux_conf "vim ~/.tmux.conf"
alias_func chtmuxconf "tmux_conf"
alias_func chtconf "tmux_conf"
alias_func tconf "tmux_conf"

# list exisiting sessions
alias_func tmux_list_sessions "tmux list-sessions"
alias_func tmuxlistsessions "tmux_list_sessions"
alias_func tls "tmux_list_sessions"
alias_func tl "tmux_list_sessions"


# create a new session
if [[ -z $(type -t tmux_new_session) ]]; then
   tmux_new_session() {
      tmux new -s "$@"
   }
   alias_func tmux_newsession "tmux_new_session"
   alias_func tmuxnewsession "tmux_new_session"
   alias_func tnewsession "tmux_new_session"
   alias_func tns "tmux_new_session"
fi

# reattach session
if [[ -z $(type -t tmux_attach_session) ]]; then
   tmux_attach_session() {
      tmux attach-session -t "$@"
   }
   alias_func tmux_attachsession "tmux_attach_session"
   alias_func tmuxattachsession "tmux_attach_session"
   alias_func tas "tmux_attach_session"
   alias_func tra "tmux_attach_session"
fi

# kill a session
if [[ -z $(type -t tmux_kill_session) ]]; then
   tmux_kill_session() {
      tmux kill-session -t "$@"
   }
   alias_func tmux_killsession "tmux_kill_session"
   alias_func tmuxkillsession "tmux_kill_session"
   alias_func tkillsession "tmux_kill_session"
   alias_func tks "tmux_kill_session"
fi

#############################
# SCP Aliases and functions
#############################
if [[ -z $(type -t sendit) ]]; then
   sendit() {
      # Use all but the last argument
      # as filenames
      files="${@:1:$(($#-1))}"
      # Assume the last arg is the remote hostname
      remote_hostname="${@:$(($#))}"
      #echo "files are ${files}"
      #echo "remote_hostname is ${remote_hostname}"
      #scp_command="scp ${files} ${USER}@${remote_hostname}:/home/${USER}"
      #echo "scp_command will be ${scp_command}"
      scp -rp ${files} ${USER}@${remote_hostname}:/home/${USER}
   }
   alias_func send_it "sendit"
   alias_func send_files "sendit"
   alias_func sendfiles "sendit"
   alias_func send_file "sendit"
   alias_func sendfile "sendit"
   
fi

# Send dotfiles and other setup files to other machine
if [[ -z $(type -t sendsetupfiles) ]]; then
   sendsetupfiles() {
      scp -rp .bashrc .bashrc.${USER} .bash_aliases .vimrc gen_proto vim_files "$1"@"$2":
   }
   alias_func sendsetup "sendsetupfiles"
   alias_func setitup "sendsetupfiles"
fi



###############################
# CUDA Documentation
###############################
alias_func cuda_doc_dir "c /usr/local/cuda/doc/pdf"
alias_func cuda_docs "cuda_doc_dir"
alias_func cuda_doc "cuda_doc_dir"
alias_func cudadocs "cuda_doc_dir"
alias_func cudadoc "cuda_doc_dir"
alias_func cudocs "cuda_doc_dir"
alias_func cudoc "cuda_doc_dir"


################################
# Main Work directory aliases
################################
alias_func sandboxcuda "c ${HOME}/Sandbox/CUDA/cuda-example"
alias_func sandboxcu "sandboxcuda"
alias_func sbc "sandboxcuda"




if [[ -z $(type -t nvprofit) ]]; then
   nvprofit() {
      # Use all but the last argument
      # as options
      options="${@:1:$(($#-1))}"
      # Assume the last arg is the execfile
      execfile="${@:$(($#))}"
      sudo /usr/local/cuda/bin/nvprof ${options} "${PWD}/${execfile}"
   }
fi

# BROKEN
# If your execfile has multiple options, split the nvprof options and the 
# 'execfile execfile_options' with an 'X'
#if [[ -z $(type -t nvprofx) ]]; then
#   nvprofx() {
#      str="'$*'"
#      delimiter=X
#      s=$str$delimiter
#      array=()
#      while [[ $s ]]; do
#         array+=( "${s%%"$delimiter"*}" )
#         s=${s#*"$delimiter"}
#      done;
#      # Shhhh
#      declare -p array > /dev/null;

#      # Clean up the strings by removing leading whitespace and extra 's(ticks) at the end
#      nvprof_options="$(echo -e "${array[0]}" | sed -e "s/^[']*//")" 
#      exec_and_options="$(echo -e "${array[1]}" | sed -e "s/^[[:space:]]*//" | sed -e "s/[']*$//")"

#      printf "nvprof options is ${nvprof_options}\n"
#      printf "command going into nvprof will be ${PWD}/${exec_and_options}\n"
#      printf "Will try to do:\n sudo /usr/local/cuda/bin/nvprof $nvprof_options $PWD/${exec_and_options}\n"
#      sudo /usr/local/cuda/bin/nvprof ${nvprof_options} "$PWD\/${exec_and_options}"
#   }
#fi

if [[ -z $(type -t nvprofo) ]]; then
   nvprofo() {
      sudo /usr/local/cuda/bin/nvprof --cpu-profiling on --cpu-profiling-mode top-down -o out.$(date +"%Y%m%d_%H%M%S").prof "${PWD}/${@}"
   }
fi

if [[ -z $(type -t cusrc) ]]; then
   cusrc() {
      if [[ "$#" -gt 0 ]]; then
         c ${1}/CUDA/src;
      else
         c CUDA/src;
      fi
   }
fi

if [[ -z $(type -t grc) ]]; then
   grc() {
      if [[ "$#" -gt 0 ]]; then
         gnuradio-companion ${1} &
      else
         gnuradio-companion &
      fi
   }
   alias_func gnrad "grc"
   alias_func gnurad "grc"
fi

alias_func grcincludespmt "c ${HOME}/gnuradio/gnuradio-runtime/include/pmt"
alias_func grcincpmt "grcincludespmt"

alias_func grcincludes "c ${HOME}/gnuradio/gnuradio-runtime/include/gnuradio"
alias_func grcinc "grcincludes"


alias_func device_query "/usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery"
alias_func device_info "device_query"
alias_func gpu_info "device_query"
alias_func gpuinfo "device_query"
alias_func lsgpu "device_query"


alias_func ncmk "rm -rf build && mkdircd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make"
alias_func cmk "find . -delete && cmake -DCMAKE_BUILD_TYPE=Release .. && make VERBOSE=1"
alias_func cmkdbg "find . -delete && cmake -DCMAKE_BUILD_TYPE=Debug .. && make VERBOSE=1"
alias_func mk "make VERBOSE=1"

if [[ -f ~/.work_aliases ]]; then
  source ~/.work_aliases
fi
