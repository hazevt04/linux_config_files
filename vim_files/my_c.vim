set tabstop=3
set softtabstop=3
set shiftwidth=3

ab aprt printf( "" );
ab aprn printf( "\n" );
ab apre fprintf( stderr, "Inside %s() ERROR: \n", __func__ );

ab ahprt HDEBUG_PRINTF( "" );
ab ahprn HDEBUG_PRINTF( "\n" );
ab ahpre HDEBUG_PRINTF( "Inside %s() ERROR: \n", __func__ );

ab adprt debug_printf( debug, "%s(): \n", __func__ );

ab aforl for( int index = 0; index < terminal; ++index ) {<CR>}
ab aif if( condition ) {<CR>}
ab awhil while( !condition ) {<CR>}
ab acase switch( var ) {<CR>case( op1 ):<CR>break;<CR>case( op2 ):<CR>break;<CR>default:<CR>exit;<CR>}

