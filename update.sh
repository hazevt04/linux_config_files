#!/bin/bash

die () {
    echo >&2 "$@"
    exit 1
}

[ "$#" -eq 1 ] || die "1 argument required, $# provided"

cp ~/.bash_aliases .
cp ~/.bashrc.${USER} .bashrc.username
cp ~/.bashrc .

cp -r ~/vim_files .
cp ~/.vimrc .

cp ~/.gdbinit .
cp ~/.tmux.conf .

gacp "$@"
