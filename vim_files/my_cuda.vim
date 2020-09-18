source my_cpp.vim

ab aglobk __global__ void kernel( const int num_vals ) {<CR>int global_index = ( blockIdx.x * blockDim.x ) + threadIdx.x;<CR>int stride = blockDim.x * gridDim.x;<CR>for (int index = global_index; index < num_vals; index+=stride) {<CR>}<CR>}<Esc>kko

ab adevk __device__ __forceinline__ dev_kern( ) {<CR>}<Esc>ko
