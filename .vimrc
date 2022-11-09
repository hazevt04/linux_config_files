" My vimrc

"#############################################
" General Settings
"#############################################

" be improved, required
set nocompatible              

set guifont=Monospace\ 16

colorscheme desert

" TURN OFF THAT ANNOYING PC SPEAKER BEEP!!!!
set visualbell

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

" Folding. Use it for your sanity
set foldmethod=indent   
set foldnestmax=10
set nofoldenable
set foldlevel=2

if has ("autocmd")
   autocmd GUIEnter * set vb t_vb= " for your GUI
   autocmd VimEnter * set vb t_vb=
   
   " For Vim change line highlighting for insert mode
   " vs normal mode
   set cursorline

   " Default color for cursorLine
   highlight CursorLine ctermbg=None ctermfg=None

   " Change color when entering Insert mode
   autocmd InsertEnter * highlight CursorLine ctermbg=White ctermfg=Black

   " Revert to default when leaving Insert mode
   autocmd InsertLeave * highlight CursorLine ctermbg=None ctermfg=None


   " Restore the scroll position and folding state
   au BufWinLeave *.c mkview
   au BufWinEnter *.c silent loadview
   
   au BufWinLeave *.cu mkview
   au BufWinEnter *.cu silent loadview
endif

"#############################################
" File Type Stuff
"#############################################
filetype plugin on

" Enable Syntax processing"
syntax enable
" load filetype specific indent files
filetype indent on
"let hostname = substitute(system('hostname'), '\n', '', '')
if has("autocmd")
   " Set Makefiles to NOT use spaces for tabs
   autocmd FileType make setlocal noexpandtab
   
   " Load C Code abbreviations and settings for *.c/*.h files"
   au BufReadPost,BufNewFile *.c source ~/vim_files/my_c.vim
   au BufReadPost,BufNewFile *.h source ~/vim_files/my_c.vim

   " Load C++ Code abbreviations and settings for *.cpp and *.hpp (header) files
   au BufReadPost,BufNewFile *.cpp source ~/vim_files/my_cpp.vim
   au BufReadPost,BufNewFile *.hpp source ~/vim_files/my_cpp.vim

   " Load CUDA Code abbreviations and settings for *.cu/*.cuh files
   au BufReadPost,BufNewFile *.cu source ~/vim_files/my_cuda.vim
   au BufReadPost,BufNewFile *.cuh source ~/vim_files/my_cuda.vim
   
   au BufReadPost,BufNewFile *.py source ~/vim_files/my_python.vim
   au BufReadPost,BufNewFile *.m source ~/vim_files/my_matlab.vim
   
   au BufReadPost,BufNewFile *.sv source ~/vim_files/my_systemverilog.vim

   " My own custom file extension-- .dailylog for my daily log wiki code file
   au BufReadPost,BufNewFile *.dailylog source ~/vim_files/my_dailylog.vim
   
   " Needs Ruby to run. sharkarmy doesn't have ruby.
   " run a script for every new file
   if has("ruby")
      autocmd BufNewFile * execute "0, 0 !~/gen_proto/gen_proto <afile>"
   endif

endif

"#############################################
" Keyboard mappings
"#############################################
" Shows that the leader key has been entered in the bottom righthand corner
set showcmd
" Set the <leader> key to ','
let mapleader = ","
noremap <F3> :set invpaste<CR>
noremap <F7> :set invnumber<CR>
noremap <F8> :set invrelativenumber<CR>
" Strip all trailing whitespace in the current file
nnoremap <leader>W :%s/\s\+$//<cr>:let @/=''<CR>
" Open this .vimrc file in a vertical split window
" to edit it along with the current file open
nnoremap <leader>ev <C-w><C-v><C-l>:e ~/.vimrc<cr>
" ESC is too far for exiting insert mode
inoremap jj <ESC>

"#############################################
" Macros
"#############################################
" Macro for going from 'printf( \"Foo: %d\", foo )' to 'printf( "Foo: " << foo
" )'
" and then to 'std::cout << "Foo: " << foo'.
" Steps: 
" 1) 0f% Go to the start of the line, find the 
" first percentage sign (assumes % is not used in any other way, sadly)
" 2) 2xmc Delete the %d part, then store the current position in the 
" line on marker c.
" 3) f"f, Find the end of the string, then find the first comma.
" 4) dwdw Delete the comma, then delete the variable name. This will 
" allow the variable to be stored so it can be pasted later.
" 5) [backtick]c Go to the stored position in the line at mark c.
" 6) i" << ^[ Insert " << into the string and escape back to command mode.
" 7) pa << \"^[ Paste the stored variable name, then insert the string << ".

" Repeat the previous macro 100 times. Should never finish, but should
" be enough to convert all of the variables
" 100@a

" Convert the Start and End to C++ (std::cout)
"
" 1) 0tr Go to start of line, and find UNTIL first r. This is in case of different indentation levels. We will be positioned at the p in printf.
" 2) cwcout << ^[ Change printf to cout << and escape to command mode.
" 3) f(x Find the ( in printf( and delete it.
" 4) $F)x Go to the end of the line, find the last ) and delete it.

" Add the std::endl to the end (assuming you have "\n" after running the previous
" macros
"
" 1) $F\ Go to the end of the line and find the last \
" 2) h Move cursor back 1 character
" 3) 3dw Delete 3 'words'  (by now the "\n" should be gone)
" 4) std::endl; Add the std::endl
" $F\h3dwstd::endl;

"#############################################
" Plugins
"#############################################
if empty(glob('~/.vim/autoload/plug.vim'))
  silent !curl -fLo ~/.vim/autoload/plug.vim --create-dirs
   \ https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
  autocmd VimEnter * PlugInstall --sync | source ~/.vimrc
endif

" Specify a directory for plugins
call plug#begin('~/.vim/plugged')
Plug 'scrooloose/nerdcommenter'
Plug 'scrooloose/nerdtree'
Plug 'mileszs/ack.vim'
Plug 'easymotion/vim-easymotion'
Plug 'tpope/vim-sleuth'
" Plug 'rainbow_parenthsis'
" Initialize plugin system
call plug#end()

"--------------------------------
" NERDCommenter Stuff
"--------------------------------
" For commented out code, I don't want a space after
" the comment delimiter. For actual comments, I do
" want the space.
" Add spaces after comment delimiters by default
"let g:NERDSpaceDelims = 1

" Align line-wise comment delimiters flush left instead of following code indentation
let g:NERDDefaultAlign = 'left'

" Use compact syntax for prettified multi-line comments
let g:NERDCompactSexyComs = 1

" Add your own custom formats or override the defaults
"let g:NERDCustomDelimiters = { 'c': { 'left': '//','right': '' } }
let g:NERDCustomDelimiters = { 
\ 'c': { 'left': '//','right': '' },
\ 'h': { 'left': '//','right': '' },
\ 'CUDA': { 'left': '//','right': '' },
\ 'cpp': { 'left': '//','right': '' }
\ }

"--------------------------------
" NERDTree Stuff
"--------------------------------
" If opening a directory, NERDTREE opens automatically
autocmd StdinReadPre * let s:std_in=1
autocmd VimEnter * if argc() == 1 && isdirectory(argv()[0]) && !exists("s:std_in") | exe 'NERDTree' argv()[0] | wincmd p | ene | exe 'cd '.argv()[0] | endif

map <C-n> :NERDTreeToggle<CR>

"--------------------------------
" Ack Stuff
"--------------------------------
" For ack.vim which uses ack (like grep but better)
nnoremap <leader>a :Ack
