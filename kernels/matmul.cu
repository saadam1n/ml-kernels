#include <torch/extension.h>

#if !defined(CPP_IMPL) && !defined(CUDA_IMPL)
#define CUDA_IMPL
#endif

#if defined(CPP_IMPL)

torch::Tensor kernel_binding(torch::Tensor a, torch::Tensor b);

#elif defined(CUDA_IMPL)

#include <cuda.h>
#include <cuda_runtime.h>

template<int BY, int BX, int TP, int TQ>
__device__ void populate_smem_tile(int p, int q, int row_offset, int col_offset, float* matrix, float (&tile)[TP][TQ]) {
    /*
    p, q - shape of matrix (p x q)
    row_offset
    col_offset
    */

    
    // for loop unrolling: avoid iterating by values dependent on threadIdx

    static_assert(TP % BY == 0);
    static_assert(TQ % BX == 0);

    constexpr int TP_LDS = TP / BY;
    constexpr int TQ_LDS = TQ / BX;

    #pragma unroll
    for(int i = 0; i < TP_LDS; i++) {
        int tile_row = BY * i + threadIdx.y;
        int row_read = row_offset + tile_row;

        #pragma unroll
        for(int j = 0; j < TQ_LDS; j++) {
            int tile_col = BX * j + threadIdx.x;
            int col_read = col_offset + tile_col;

            //tile[tile_row][tile_col] = row_read < p && col_read < q ? matrix[row_read * q + col_read] : 0;
            tile[tile_row][tile_col] = matrix[row_read * q + col_read];
        }
    }
}

template<int BY, int BX, int TILE_A, int TILE_B, int TILE_K, int REG_A, int REG_B>
__global__ void matmul_kernel(int n, int m, int k, float* a, float* b, float* c) {
    /*
    Thread assignment:
        x ----->
    y   0 1 2 3 
    |   4 5 6 7
    |   8 9 A B
    v   C D E F
    */

    __shared__ float smem_tile_a[TILE_A][TILE_K];
    __shared__ float smem_tile_b[TILE_K][TILE_B];

    // positions within each tile
    int tc_base = threadIdx.y * REG_B;
    int tr_base = threadIdx.x * REG_A;

    // note: we need to be careful to launch only enough threads such that there are no oob threads
    static_assert(BX * REG_B == TILE_B);
    static_assert(BY * REG_A == TILE_A);

    // these stay the same
    int tile_a_row_offset = blockIdx.y * blockDim.y * REG_B;
    int tile_b_col_offset = blockIdx.x * blockDim.x * REG_A;

    // assume divisbility
    int num_tile_mm_iters = k / TILE_K; 

    //static_assert(k % TILE_K == 0);


    float a_reg[REG_A];
    float b_reg[REG_B];
    float c_reg[REG_A * REG_B] = {0.0};

    for(int mmi = 0; mmi < num_tile_mm_iters; mmi++) {

        // step 1: populate cache

        int k_offset = TILE_K * mmi;

        // possibly inefficient when we want to calculate multiple results per thread
        populate_smem_tile<BY, BX>(n, k, tile_a_row_offset, k_offset, a, smem_tile_a);
        populate_smem_tile<BY, BX>(k, m, k_offset, tile_b_col_offset, b, smem_tile_b);


        __syncthreads();

        // step 2: run tile matmul

        for(int i = 0; i < TILE_K; i++) {
            // ld into reg memory

            // read col of a at a time (might transpose to avoid bank conflicts)
            #pragma unroll
            for(int j = 0; j < REG_A; j++) {
                a_reg[j] = smem_tile_a[tr_base + j][i];
            }

            // read row of b at a time
            #pragma unroll
            for(int j = 0; j < REG_B; j++) {
                b_reg[j] = smem_tile_b[i][tc_base + j];
            }

            #pragma unroll
            for(int r = 0; r < REG_A; r++) {
                #pragma unroll
                for(int c = 0; c < REG_B; c++) {
                    c_reg[r * REG_B + c] += a_reg[r] * b_reg[c];
                }
            }

        }


        // step 3: sync to prevent overwrite
        __syncthreads();

    }

    #pragma unroll
    for(int r = 0; r < REG_A; r++) {
        #pragma unroll
        for(int c_off = 0; c_off < REG_B; c_off++) {
            int res_r = tile_a_row_offset + tr_base + r;
            int res_c = tile_b_col_offset + tc_base + c_off;

            c[res_r * m + res_c] = c_reg[r * REG_B + c_off];
        }
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
    constexpr int BLOCK_SIZE = 16;
    constexpr int REG_TILING = 8;
    constexpr int SIDE_LEN = BLOCK_SIZE * REG_TILING;

    dim3 grid_dim((m + SIDE_LEN - 1) / SIDE_LEN, (n + SIDE_LEN - 1) / SIDE_LEN, 1);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);

    matmul_kernel<BLOCK_SIZE, BLOCK_SIZE, SIDE_LEN, SIDE_LEN, BLOCK_SIZE, REG_TILING, REG_TILING><<<grid_dim, block_dim>>>(n, m, k, a_contig.data_ptr<float>(), b_contig.data_ptr<float>(), c.data_ptr<float>());

    // check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // return c
    return c;
}

#endif