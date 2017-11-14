alias_func() {
  aname=$1
  acmd=$2
  if type $aname 2>/dev/null; then
    echo "$aname is a command or alias already! Not aliasing"
  else
    alias $aname="$acmd"
  fi
}

k9() {
  kill -9 "$@"
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
}

#l() {
#   ls "$@"
#}

mkdircd() {
   mkdir $1;
   cd $1;
   pwd;
   ls -ltr;
}

alias_func srcit "source ~/.bashrc"
alias_func srcali "source ~/.bash_aliases"

alias_func chali "gvim ~/.bash_aliases &"
alias_func charc "gvim ~/.bashrc &"

alias_func chvimrc "gvim ~/.vimrc &"
alias_func chcrc "gvim ~/c_code.vim"

alias_func g "gvim -o $@"

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

alias_func gstat "git status"

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




