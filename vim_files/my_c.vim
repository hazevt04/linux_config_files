set tabstop=3
set softtabstop=3
set shiftwidth=3
set expandtab

ab aforl for( int index = 0; index < terminal; index++ ) {<CR>}
ab aif if( condition ) {<CR>}
ab awhil while( !condition ) {<CR>}
ab acase switch( var ) {<CR>case( op1 ):<CR>break;<CR>case( op2 ):<CR>break;<CR>default:<CR>exit;<CR>}
ab aprt printf( "" );
ab aprn printf( "\n" );
ab apre fprintf( stderr, "Inside %s() ERROR: \n", __func__ );
ab adprt debug_printf( debug, "" );
ab adprn debug_printf( debug, "\n" );
ab adpre debug_printf( debug, "Inside %s() ERROR: \n", __func__ );






