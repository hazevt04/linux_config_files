set nocompatible              " be iMproved, required

filetype plugin on

" Enable Syntax processing"
syntax enable

" Allow backspacing over everything in insert mode
set bs=2

" Set number of visual spaces"
set tabstop=3

" Number of spaces in tab"
set softtabstop=3

" Set shiftwidth for >> and <<"
set shiftwidth=3

" Tabs are spaces"
set expandtab

" Show line numbers"
set number 

" Highlight current line"
set cursorline

" load filetype specific indent files
filetype indent on

" visual autocomplete on"
set wildmenu

" redraw only when we need to"
set lazyredraw

" highlight matching [({})]
set showmatch

" search as characters are entered"
set incsearch

" highlight matches"
set hlsearch

" Load C Code abbreviations and settings for *.c/*.h files"
au BufReadPost,BufNewFile *.c,*.cpp source ~/vim_files/c_code.vim
au BufReadPost,BufNewFile *.h source ~/vim_files/c_code.vim

if has("autocmd")
   " Set Makefiles to NOT use spaces for tabs
   autocmd FileType make setlocal noexpandtab

   " run a script for every new file
   autocmd BufNewFile * execute "0, 0 !/usr/bin/gen_proto/gen_proto.rb <afile>"

   " For Vim change line highlighting for insert mode
   " vs normal mode
   set cursorline

   "Default color for cursorLine
   highlight CursorLine ctermbg=None ctermfg=None

   "Change color when entering Insert mode
   autocmd InsertEnter * highlight CursorLine ctermbg=White ctermfg=Black

   "Revert to default when leaving Insert mode
   autocmd InsertLeave * highlight CursorLine ctermbg=None ctermfg=None

endif


noremap <F3> :set invpaste<CR>

noremap <F7> :set invnumber<CR>

noremap <F8> :set invrelativenumber<CR>
