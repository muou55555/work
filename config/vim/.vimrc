" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim

call vundle#begin()
" let Vundle manage Vundle, required
Plugin 'gmarik/Vundle.vim'
Bundle 'Valloric/YouCompleteMe'
Plugin 'scrooloose/syntastic'
Plugin 'tell-k/vim-autopep8'"
Plugin 'nvie/vim-flake8'

call vundle#end()

"let g:syntastic_ignore_files=[".*\.py$"]
"autocmd FileType python map <F7>   :call flake8<CR>
let g:syntastic_python_checkers = ['pyflakes']
"set statusline=\ %<[%1*%*%n%R%H]\ %f%m\ %0(%{&fileformat}\ <%l/%L>%)\ 
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 0
let g:syntastic_check_on_wq = 0


let mapleader = ""
let g:ycm_autoclose_preview_window_after_completion=1
"map <leader>gl  :YcmCompleter :GoToDefinitionElseDeclaration<CR>
map gl  :YcmCompleter GoToDefinitionElseDeclaration<CR>

"默认配置文件路径"
"let g:ycm_global_ycm_extra_conf = '~/.ycm_extra_conf.py'

""打开vim时不再询问是否加载ycm_extra_conf.py配置"
"let g:ycm_confirm_extra_conf=1
set completeopt=longest,menu
""python解释器路径"
let g:ycm_path_to_python_interpreter='/home/work/anaconda2/envs/root-2.7/bin/python'
"let g:ycm_path_to_python_interpreter='/home/work/anaconda2/bin/python'
"按gb 会跳转到定义
"nnoremap <silent> gb :YcmCompleter GoToDefinitionElseDeclaration<CR>
"使用ctags生成的tags文件
let g:ycm_collect_identifiers_from_tag_files = 1
""是否开启语义补全"
let g:ycm_seed_identifiers_with_syntax=1
""是否在注释中也开启补全"
"let g:ycm_complete_in_comments=1
"let g:ycm_collect_identifiers_from_comments_and_strings = 0
""开始补全的字符数"
"let g:ycm_min_num_of_chars_for_completion=2
""补全后自动关机预览窗口"
let g:ycm_autoclose_preview_window_after_completion=1
""禁止缓存匹配项,每次都重新生成匹配项"
let g:ycm_cache_omnifunc=0
""字符串中也开启补全"
let g:ycm_complete_in_strings = 1
""离开插入模式后自动关闭预览窗口"
let g:ycm_err_symbol = '<<'
let g:ycm_warning_symbol = '<*'
" 设置在下面几种格式的文件上屏蔽ycm
let g:ycm_filetype_blacklist = {
  \ 'tagbar' : 1,
  \ 'nerdtree' : 1,
  \}

let g:indentLine_char='|'
let g:Autopep8_disable_show_diff = 1

let python_highlight_all=0

" disable or enable underline at cursor line 2011-11-04
set cursorline

" set extra space char at the end of th line to red color 2011-11-04
autocmd ColorScheme * highlight ExtraWhitespace ctermbg=red guibg=red
highlight ExtraWhitespace ctermbg=red guibg=red
match ExtraWhitespace /\s\+$/

" don't create .*.swp file 2011-11-04
setlocal noswapfile

" auto (缩进)
set autoindent

" show line number
set number

" support chinese 2011-11-04
set encoding=utf-8
set fileencodings=utf-8,ucs-bom,shift-jis,gb18030,gbk,gb2312,cp936
set ambiwidth=double
set guifont=Courier\ 10\ Pitch\ 12

" when open a file, cursor at the last time where you exit. 2011-11-04
autocmd BufReadPost *
			\ if line("'\"") > 0 && line("'\"") <= line("$") |
			\ exe "normal! g`\""|
			\ endif

" gap between line
"set linespace=2

set runtimepath+=~/vimfiles

" don't auto change line
"set nowrap
" auto change line
set wrap

" show the uncomplete command. 2011-11-04
set showcmd

" set the cursor line color. 2011-11-04
if has("gui_running")
	set cursorline
	hi cursorline guibg=#033333
endif

" search option. 2011-11-04
set ignorecase smartcase	"ignore upper or lower char
set incsearch				"high light char when search
set hlsearch				"set high light after search
"clear high light word. 2011-11-04
map <esc><esc><esc> :noh<cr>
" support mous
set mouse=n

" not compatible with vi. 2011-11-04
set nocompatible

set modelines=5
au GUIEnter * simalt ~x
set guioptions+=t
" remove the toolbar in vim of version GUI
set guioptions-=T

" hide the menu bar
set guioptions-=m

" set num=4 line to curors - when move vertical...
set so=8

" text, tab and indent related.
set tabstop=4
set softtabstop=4
set shiftwidth=4
set smarttab

set ai	"auto indent
set si	"smart indent

" set how many lines of history vim has to remember
set history=1000
" colorscheme ps_color

" auto backup the open file
set nowritebackup
" auto save
set autowrite

" enable filetype plugin
filetype plugin indent on

set backspace=indent,eol,start
set foldmethod=indent
set foldlevel=9999
"set list
"set listchars=tab:>-,trail:.,extends:>
set suffixes+=.pyc,.pyo

" show the match char(eg: {},()...)
set showmatch

" support c/c++ indent
set cin
set cino=:0,g0,u0,(0,W4

set fileformat=unix
" always show current position
set ruler
"set visualbell
"set iskeyword-=.

" the commandbar height
set cmdheight=1

" always show the status line
set laststatus=2

" enable magic
set magic

"set t_Co=256
"colorscheme jellybeans
"colorscheme guardian
"colorscheme molokai
"colorscheme wombat256mod

" cscope setting
if has ("cscope")
	set csprg=/usr/bin/cscope
	set csto=1
	set cst
	set nocsverb
	if filereadable ("cscope.out")
		cs add cscope.out
	endif
	if filereadable ("/home/work/anaconda2/envs/root-2.7/cscope.out")
		cs add /home/work/anaconda2/envs/root-2.7/cscope.out
	endif
	if filereadable ("/home/work/anaconda2/pkgs/cscope.out")
		cs add /home/work/anaconda2/pkgs/cscope.out
	endif
	set csverb
endif

" auto save
"set autowrite
set fo-=at
" 折叠
set fdm=syntax
syntax on
imap jj <esc>
map <space> :bn<CR>
map <tab> :bp<CR>
"map <c-j> :tj <c-r>=expand("<cword>")<cr><cr>
nmap <c-j> :tj <c-r>=expand("<cword>")<cr><cr>
nmap <c-j>v :vsp \| tj <c-r>=expand("<cword>")<cr><cr>
nmap <c-j>h :sp \| tj <c-r>=expand("<cword>")<cr><cr>

nmap <c-[>c :cs find c <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>d :cs find d <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>e :cs find e <c-r>=expand("<cword>")<cr>
nmap <c-[>f :cs find f <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>g :cs find g <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>i :cs find i <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>s :cs find s <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>t :cs find t <c-r>=expand("<cword>")<cr><cr>

nmap <c-[>hc :sp \| cs find c <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>hd :sp \| cs find d <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>he :sp \| cs find e <c-r>=expand("<cword>")<cr>
nmap <c-[>hf :sp \| cs find f <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>hg :sp \| cs find g <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>hi :sp \| cs find i <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>hs :sp \| cs find s <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>ht :sp \| cs find t <c-r>=expand("<cword>")<cr><cr>

nmap <c-[>vc :vsp \| cs find c <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>vd :vsp \| cs find d <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>ve :vsp \| cs find e <c-r>=expand("<cword>")<cr>
nmap <c-[>vf :vsp \| cs find f <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>vg :vsp \| cs find g <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>vi :vsp \| cs find i <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>vs :vsp \| cs find s <c-r>=expand("<cword>")<cr><cr>
nmap <c-[>vt :vsp \| cs find t <c-r>=expand("<cword>")<cr><cr>

" set the diferent color to statu line in diferent status.
function! InsertStatuslineColor(mode)
	if a:mode == 'i'
		hi statusline guibg=peru
	elseif a:mode == 'r'
		hi statusline guibg=blue
	endif
endfunction
au InsertEnter * call InsertStatuslineColor(v:insertmode)
au InsertLeave * hi statusline guibg=orange guifg=write
hi statusline guibg=green

" ab the mail
:ab mailbox panyingxiong@thtf.com.cn
" set the status line
"set statusline=\ %<[%1*%*%n%R%H]\ %f\ %m\ \ \ 0x%B%=\ %y\ %0(%{&fileformat}\ %{&encoding}\ <%c:%l/%L>\ %p%%%)\ 
set statusline=\ %<[%1*%*%n%R%H]\ %f%m\ %0(%{&fileformat}\ <%l/%L>%)\ 
"set statusline=\ %<%F[%1*%M%*%n%R%H]%=\ \ %y\ \ %0(%{&fileformat}\ \ %{&encoding}\ %c:%l/%L%)\ \ 

"autopen NERDTree and focus cursor in new document
"autocmd vimEnter * NERDTree
autocmd vimEnter * wincmd p

" NERD Tree
let NERDTreeWinSize=15
let NERDTreeWinPos='right'
"显示隐藏文件
let NERDTreeShowHidden=1

" taglist
let Tlist_WinWidth=15

map <F3> :NERDTreeToggle<CR>
map <F2> :TlistToggle<CR>
map <F4> :cwindow<CR>
let NERDTreeIgnore=['\.pyc$', '\.out$', '*cscope*']


" add main function.(添加main框架)
map main :call Addmain()<CR>
function Addmain()
	call append(line(".") + 0,"#include <stdio.h>")
	call append(line(".") + 1,"")
	call append(line(".") + 2,"int main(int argc, char * argv[])")
	call append(line(".") + 3,"{")
	call append(line(".") + 4,"")
	call append(line(".") + 5,"\treturn 0;")
	call append(line(".") + 6,"}")
	call append(line(".") + 7,"")
endfunction

map time :call Gettime()<cr>
function Gettime()
	call append(".","//=== HeroPan === ". strftime("%Y-%m-%d %H:%M:%S"))
endfunction

autocmd BufNewFile *py exec ":call SetPythonTitle()"

function SetPythonTitle()
    call setline(1,"#!/usr/bin/env python")
    call append( line("."),"#-*- coding: utf-8 -*-" )
    call append(line(".")+1,"")
    call append(line(".")+2,"\"\"\"")
    call append(line(".")+3, "\File Name: ".expand("%"))
    call append(line(".")+4, "\Created Time: ".strftime("%Y-%m-%d",localtime()))
    call append(line(".")+5,"\"\"\"")
    call append(line(".")+6,"")
    call append(line(".")+7,"")
	call append(line(".")+8,"if __name__ == '__main__':")
	call append(line(".")+9,"  #Todo Somethings")
endfunction

"按F5运行python"
"map <F5> :w<CR> :call RunPython()<CR>
map <F5> :call RunPython()<CR>
"map <F5> :Autopep8<CR> :w<CR> :call RunPython()<CR>
function RunPython()
	let mp = &makeprg
	let ef = &errorformat
	let exeFile = expand("%:t")
	setlocal makeprg=python\ -u
	set efm=%C\ %.%#,%A\ \ File\ \"%f\"\\,\ line\ %l%.%#,%Z%[%^\ ]%\\@=%m
	silent make %
	copen
	let &makeprg = mp
	let &errorformat = ef
endfunction

map <F6> :call FormartSrc()<CR>

function FormartSrc1()
	"exec "w"
	if &filetype == 'c'
		exec "!astyle --style=ansi -a --suffix=none %"
	elseif &filetype == 'cpp' || &filetype == 'hpp'
		exec "r !astyle --style=ansi --one-line=keep-statements -a --suffix=none %> /dev/null 2>&1"
	elseif &filetype == 'perl'
		exec "!astyle --style=gnu --suffix=none %"
	elseif &filetype == 'py'||&filetype == 'python'
		exec "r !autopep8 -in-place --aggressive --aggressive &lt; % &gt;"
		"exec "r !autopep8 --aggressive %"
	elseif &filetype == 'java'
		exec "!astyle --style=java --suffix=none %"
	elseif &filetype == 'jsp'
		exec "!astyle --style=gnu --suffix=none %"
	elseif &filetype == 'xml'
		exec "!astyle --style=gnu --suffix=none %"
	else
		exec "normal gg=G"
		return
	endif
																																					    exec "e! %"
endfunction


