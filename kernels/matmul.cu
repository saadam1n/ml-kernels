#include <torch/extension.h>

#if !defined(CPP_IMPL) && !defined(CUDA_IMPL)
#define CUDA_IMPL
#endif

#if defined(CPP_IMPL)

torch::Tensor kernel_binding(torch::Tensor a, torch::Tensor b);

#elif defined(CUDA_IMPL)

#include <cuda.h>
#include <cuda_runtime.h>

template<int TP, int TQ>
__device__ void populate_smem_tile(int p, int q, int row_offset, int col_offset, float* matrix, float (&tile)[TP][TQ]) {
    /*
    p, q - shape of matrix (p x q)
    row_offset
    col_offset
    */

    for(int i = threadIdx.y; i < TP; i += blockDim.y) {
        for(int j = threadIdx.x; j < TQ; j += blockDim.x) {
            int row_read = row_offset + i;
            int col_read = col_offset + j;

            tile[i][j] = row_read < p && col_read < q ? matrix[row_read * q + col_read] : 0.0;
        }
    }
}

template<int BLOCK_SIZE, int TILE_A, int TILE_B, int TILE_K>
__global__ void matmul_kernel(int n, int m, int k, float* a, float* b, float* c) {
    /*
    Thread assignment:
        x ----->
    y   0 1 2 3 
    |   4 5 6 7
    |   8 9 A B
    v   C D E F
    */

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    bool inside = (row < n && col < m);

    __shared__ float smem_tile_a[TILE_A][TILE_K];
    __shared__ float smem_tile_b[TILE_K][TILE_B];

    // positions within each tile
    int tc = col % TILE_B;
    int tr = row % TILE_A;

    // these stay the same
    int tile_a_row_offset = blockIdx.y * blockDim.y;
    int tile_b_col_offset = blockIdx.x * blockDim.x;

    // assume divisbility
    int num_tile_mm_iters = k / TILE_K; 

    float sum = 0.0;
    for(int mmi = 0; mmi < num_tile_mm_iters; mmi++) {

        // step 1: populate cache

        int k_offset = TILE_K * mmi;

        #if 0
        populate_smem_tile(n, k, tile_a_row_offset, k_offset, a, smem_tile_a);
        populate_smem_tile(k, m, k_offset, tile_b_col_offset, b, smem_tile_b);
        #else
        smem_tile_a[threadIdx.y][threadIdx.x] = a[row * k + k_offset + threadIdx.x];
        smem_tile_b[threadIdx.y][threadIdx.x] = b[(k_offset + threadIdx.y) * m + col];
        #endif

        __syncthreads();

        // step 2: run tile matmul


        for(int i = 0; i < TILE_K; i++) {
            // smem_tile_a has no bank conflicts
            // - tr same for warp
            // - i same for block

            // smem_tile_b has no bank conflicts
            // - i same for block
            // - tc spread over banks 
            sum += smem_tile_a[tr][i] * smem_tile_b[i][tc];
        }

        // step 3: sync to prevent overwrite
        __syncthreads();

    }


    if(inside) {
        c[row * m + col] = sum;
    }
}

torch::Tensor kernel_binding(torch::Tensor a, torch::Tensor b) {
    // check whether tensors are on same CUDA device
    TORCH_CHECK(a.device().is_cuda(), "Tensor a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Tensor b must be on CUDA");
    TORCH_CHECK(a.device() == b.device(), "Tensors must be on the same CUDA device");
    
    // check whether tensor is FP32
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Tensor b must be float32");

    // check whether sizes match
    int n = a.size(0);
    int k = a.size(1);
    int m = b.size(1);

    TORCH_CHECK(b.size(0) == k, "Matrix dimensions don't match: a is ", n, "x", k, " but b is ", b.size(0), "x", m);

    // ensure contiguity 
    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();

    // create space for c
    torch::Tensor c = torch::empty({n, m}, a.options());

    // launch kernel
    constexpr int BLOCK_SIZE = 32;
    dim3 grid_dim((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);

    matmul_kernel<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE><<<grid_dim, block_dim>>>(n, m, k, a_contig.data_ptr<float>(), b_contig.data_ptr<float>(), c.data_ptr<float>());

    // check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // return c
    return c;
}

#endif