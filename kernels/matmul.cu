#include <torch/extension.h>

#if !defined(CPP_IMPL) && !defined(CUDA_IMPL)
#define CUDA_IMPL
#endif

#if defined(CPP_IMPL)

torch::Tensor kernel_binding(torch::Tensor a, torch::Tensor b);

#elif defined(CUDA_IMPL)

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(int n, int m, int k, float* a, float* b, float* c) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < n && col < m) {
        
        float sum = 0.0;

        for(int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * m + col];
        }

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
    int block_size = 32;
    dim3 grid_dim((n + block_size - 1) / block_size, (m + block_size - 1) / block_size, 1);
    dim3 block_dim(block_size, block_size, 1);

    matmul_kernel<<<grid_dim, block_dim>>>(n, m, k, a_contig.data_ptr<float>(), b_contig.data_ptr<float>(), c.data_ptr<float>());

    // check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // return c
    return c;
}

#endif