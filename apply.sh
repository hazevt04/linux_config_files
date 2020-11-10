#!/bin/bash

cp .bash_aliases ~/
cp .bashrc.username ~/.bashrc.${USER}
cp .bashrc ~/

cp -r vim_files ~/
cp .vimrc ~/

cp .gdbinit ~/
cp .tmux.conf ~/

if [[ -z $(type -t ruby) ]];
    echo "My .vimrc uses a Ruby script. I need Ruby installed or comment out the 'gen_proto' line in the .vimrc.";
    exit 1;
else
    git clone git@github.com:hazevt04/gen_proto.git && mv gen_proto ~/
fi

source .bashrc
