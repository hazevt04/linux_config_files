set tabstop=3
set softtabstop=3
set shiftwidth=3

ab aprt $display( "" );
ab aprn $display( "\n" );

ab algw logic [DATA_WIDTH - 1:0] sig;
ab aforl for( index = 0; index < terminal; ++index ) begin<CR>end
ab aif if( condition ) begin<CR>end
ab aiaff always_ff (@posedge clk) begin<CR>if (reset) begin<CR>end<CR>else begin<CR>end<CR>end<CR>end

