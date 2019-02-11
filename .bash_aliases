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
fi

if [[ -z $(type -t whereis) ]]; then
   whereis() {
      find . -name "$1*";
   }
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
      builtin cd "${destination}" > /dev/null
      ls -altr
      pwd
   }
fi

#l() {
#   ls "$@"
#}

if [[ -z $(type -t mkdircd) ]]; then
   mkdircd() {
      mkdir $1;
      cd $1;
      pwd;
      ls -altr;
   }
   alias_func mkc "mkdircd"
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
fi

alias_func undo_jason "cp BACKUP/*.c .;cp BACKUP/*.h .;"
alias_func jason_undo "undo_jason;"

alias_func clean_jason "rm -rf BACKUP;"
alias_func jason_clean "clean_jason;"

if [[ -z $(type -t v) ]]; then
   v() {
      vim $1;
   }
fi

if [[ -z $(type -t g) ]]; then
   g() {
      gvim "$@";
   }
fi

if [[ -z $(type -t e) ]]; then
   e() {
      evince "$@";
   }
fi

alias_func u "cd ..; ls -ltr; pwd"
alias_func uu "cd ..; ls -ltr; pwd"

alias_func gurl "git remote -v"
alias_func gba "git branch -a"

# Always override gs. I don't normally use Ghostscript
alias gs="git status"

alias_func replace_spaces "for f in *\ *; do mv \"\$f\" \"\${f// /_}\"; done"
alias_func repspaces "replace_spaces"

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
fi

if [[ -z $(type -t gfetch) ]]; then
   gfetch() {
      url="git@github.com:hazevt04/${1}.git"
      branch=$2
      git fetch "$url" "$branch"
   }
fi

if [[ -z $(type -t gpull) ]]; then
   gpull() {
      url="git@github.com:hazevt04/${1}.git"
      branch=$2
      git pull "$url" "$branch"
   }
fi

if [[ -z $(type -t gcommit) ]]; then
   gcommit() {
      git commit -m "$@"
   }
fi

alias_func gcomm "gcommit"
   
if [[ -z $(type -t gpo) ]]; then
   gpo() {
      current_branch=$(git status -v | grep "On branch" | cut -d ' ' -f 3)
      git push origin $current_branch
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




if [[ -z $(type -t mmm) ]]; then
   mmm() {
      /home/glenn/Programming/mad_men_money/mad_men_money "$@"
   }
else
   echo "mmm already defined. Maybe it's already a command or function"
fi
