source my_c.vim

ab acout std::cout << "";
ab acerr std::cerr << "";

ab acerrn std::cerr << "\n";
ab acerrfn std::cerr << __func__ << "(): ERROR: " << err_msg << "\n";

ab acoutn std::cout << "\n"; 
ab acoutf std::cout << __func__ << "(): ";
ab acoutfn std::cout << __func__ << "(): \n";
ab acoutfe std::cout << __func__ << "(): ERROR: " << err_msg << "\n"; 

ab atcatch try {<CR>} catch ( std::exception& ex ) {<CR>std::cout << __func__ << "() ERROR: " << ex.what() << "\n";<CR>}

ab aforl for( int index = 0; index < terminal; ++index ) {<CR>}
ab aforlv for ( std::vector<T>::size_t index = 0; index != vec.size(); ++index ) {<CR>}

