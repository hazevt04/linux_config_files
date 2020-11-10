set tabstop=3
set softtabstop=3
set shiftwidth=3
set autoindent
set smartindent

ab aprt printf( " \n" );
ab aprn printf( "\n" );

ab acout std::cout << "";
ab acoutn std::cout << "\n";
ab acoutf std::cout << __func__ << "\n";
ab acerr std::cerr << __func__ << "() ERROR: " << "\n";
ab astds std::string{}<Esc>i

ab adout debug_cout( debug, "Val is ", val , "\n" );
ab adfout debug_cout( debug, __func__, "(): Val is ", val , "\n" );
ab adprt debug_printf( debug, "%s(): \n", __func__ );

ab aforl for( int index = 0; index < terminal; ++index ) {<CR>}
ab aforlv for( size_t index; index < vec.size(); ++index ) {<CR>}
ab aif if( condition ) {<CR>}
ab awhil while( !condition ) {<CR>}
ab acase switch( var ) {<CR>case( op1 ):<CR>break;<CR>case( op2 ):<CR>break;<CR>default:<CR>exit;<CR>}

