set nocompatible              " be iMproved, required

filetype plugin on

" Enable Syntax processing"
syntax enable

set encoding=utf-8

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
au BufReadPost,BufNewFile *.c source ~/vim_files/my_c.vim
au BufReadPost,BufNewFile *.h source ~/vim_files/my_c.vim

au BufReadPost,BufNewFile *.cpp source ~/vim_files/my_cpp.vim
au BufReadPost,BufNewFile *.hpp source ~/vim_files/my_cpp.vim

au BufReadPost,BufNewFile *.cu source ~/vim_files/my_cuda.vim
au BufReadPost,BufNewFile *.cuh source ~/vim_files/my_cuda.vim

if has("autocmd")
   " Set Makefiles to NOT use spaces for tabs
   autocmd FileType make setlocal noexpandtab

   " run a script for every new file
   autocmd BufNewFile * execute "0, 0 !/usr/bin/gen_proto <afile>"

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

let mapleader = ","

nnoremap <leader>W :%s/\s\+$//<cr>:let @/=''<CR>11
" Trying out vundle
" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'

""""""""""""""""""""""""""""""""""""""
" NERDCOMMENTER SECTION START
""""""""""""""""""""""""""""""""""""""
Plugin 'scrooloose/nerdcommenter'
" Add spaces after comment delimiters by default
let g:NERDSpaceDelims = 1

" Use compact syntax for prettified multi-line comments
let g:NERDCompactSexyComs = 1

" Align line-wise nt delimiters flush left instead of following code indentation
let g:NERDDefaultAlign = 'left'

" Set a language to use its alternate delimiters by default
let g:NERDAltDelims_java = 1

" Add your own custom formats or override the defaults
"let g:NERDCustomDelimiters = { 'c': { 'left': '/*','right': '*/' } }
let g:NERDCustomDelimiters = { 'C': { 'left': '//','right': '' } }
let g:NERDCustomDelimiters = { 'cuda': { 'left': '//','right': '' } }
let g:NERDCustomDelimiters = { 'c++': { 'left': '//','right': '' } }

" Allow commenting and inverting empty lines (useful when commenting a region)
let g:NERDCommentEmptyLines = 1

" Enable trimming of trailing whitespace when uncommenting
let g:NERDTrimTrailingWhitespace = 1

" Enable NERDCommenterToggle to check all selected lines is commented or not
let g:NERDToggleCheckAllLines = 1
""""""""""""""""""""""""""""""""""""""
" NERDCOMMENTER SECTION END
""""""""""""""""""""""""""""""""""""""

Plugin 'ycm-core/YouCompleteMe' 

"let g:ycm_clangd_binary_path = "/usr/bin/clangd"

" All of your Plugins must be added before the following line
call vundle#end()            " required
" " I use vim-plug:
" " https://github.com/junegunn/vim-plug
" "
" " Specify a directory for plugins
" " - For Neovim: stdpath('data') . '/plugged'
" " - Avoid using standard Vim directory names like 'plugin'
" call plug#begin('~/.vim/plugged')
"
" Plug 'scrooloose/nerdcommenter'
"
" " Add spaces after comment delimiters by default
" let g:NERDSpaceDelims = 1
"
" " Use compact syntax for prettified multi-line comments
" let g:NERDCompactSexyComs = 1
"
" " Align line-wise nt delimiters flush left instead of following code indentation
" let g:NERDDefaultAlign = 'left'
"
" " Set a language to use its alternate delimiters by default
" let g:NERDAltDelims_java = 1
"
" " Add your own custom formats or override the defaults
" "let g:NERDCustomDelimiters = { 'c': { 'left': '/*','right': '*/' } }
" let g:NERDCustomDelimiters = { 'C': { 'left': '//','right': '' } }
" let g:NERDCustomDelimiters = { 'cuda': { 'left': '//','right': '' } }
" let g:NERDCustomDelimiters = { 'c++': { 'left': '//','right': '' } }
"
" " Allow commenting and inverting empty lines (useful when commenting a region)
" let g:NERDCommentEmptyLines = 1
"
" " Enable trimming of trailing whitespace when uncommenting
" let g:NERDTrimTrailingWhitespace = 1
"
" " Enable NERDCommenterToggle to check all selected lines is commented or not
" let g:NERDToggleCheckAllLines = 1
"
" " Initialize plugin system
"
" " Not worth the hassle for me right now
" "Plug 'ycm-core/YouCompleteMe'
"
"
" call plug#end()
