set filetype=cuda
set tabstop=3
set softtabstop=3
set shiftwidth=3
set autoindent
set smartindent

" Use same settings as my C++ settings
source ~/vim_files/my_cpp.vim

" Add abbreviations
ab acudatry try_cuda_func_throw( cerror, cudaFunc )
ab acudaha cudaHostAlloc( (void**)&var, num_bytes, cudaHostAllocMapped )
ab acudafh cudaFreeHost( var )
ab acudam cudaMalloc( (void**)&d_var, num_byes )
ab acudacpyhd cudaMemcpy( d_var, var, num_byes, cudaMemcpyHostToDevice )
ab acudacpydh cudaMemcpy( var, d_var, num_byes, cudaMemcpyDeviceToHost )
ab acudamcpyhd cudaMemcpyAsync( d_var, var, num_byes, cudaMemcpyHostToDevice, streams[stream_num] )
ab acudamcpydh cudaMemcpyAsync( var, d_var, num_byes, cudaMemcpyDeviceToHost, streams[stream_num] )
