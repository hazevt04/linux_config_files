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

# The if statement below is 
# the equivalent of alias_func for functions!
# Copy the if statement guard to all functions!
if [[ -z $(type -t contains) ]]; then
  function contains() {
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

k9() {
  kill -9 "$@"
}

function whereis (){
  find . -name "$1*";
}

# Put a symbolic link called (linkname) to PWD in ~
lnpwd() {
  linkname=$1
  ln -s "$PWD" ~/"${linkname}"
}

# Dump the output of a man page
# to another (cmd).MAN.OUT without the annoying
# extra characters
dump_man() {
   man $1 | col -b > $1.MAN.OUT
}

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

#l() {
#   ls "$@"
#}

mkdircd() {
   mkdir $1;
   cd $1;
   pwd;
   ls -altr;
}

alias_func srcit "source ~/.bashrc"
alias_func srcali "source ~/.bash_aliases"

alias_func chali "gvim ~/.bash_aliases &"
alias_func charc "gvim ~/.bashrc &"

alias_func chvimrc "gvim ~/.vimrc &"
alias_func chcrc "gvim ~/c_code.vim"

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

v() {
   vim $1;
}


alias_func u "cd ..; ls -ltr; pwd"
alias_func uu "cd ..; ls -ltr; pwd"

alias_func gurl "git remote -v"
alias_func gba "git branch -a"

# Always override gs. I don't normally use Ghostscript
alias gs="git status"

gdiff() {
   git diff "$@"
}

gdiffs() {
   git diff --staged "$@"
}

gclone() {
   url="git@github.com:hazevt04/${1}.git"
   # Extract the directory name from the Git URL
   dirname=$1
   git clone "$url" --branch develop "$dirname"
}

gcloneb() {
   url="git@github.com:hazevt04/${1}.git"
   branch=$2
   dirname=$1
   git clone "$url" --branch "$branch" "${dirname}_${branch}"
}

gfetch() {
   url="git@github.com:hazevt04/${1}.git"
   branch=$2
   git fetch "$url" "$branch"
}


gpull() {
   url="git@github.com:hazevt04/${1}.git"
   branch=$2
   git pull "$url" "$branch"
}


gcommit() {
   git commit -m "$@"
}

gpo() {
   git push origin $1
}




