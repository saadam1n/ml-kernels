import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define your CUDA kernel as a string
cuda_source = """
__global__ void my_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}

torch::Tensor my_function(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    my_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel()
    );
     
    return output;
}
"""

cpp_source = """
torch::Tensor my_function(torch::Tensor input);
"""

print(f"Is cuda avail? {torch.cuda.is_available()}")

# JIT compile - only compiles once, then uses cached version
custom_ops = load_inline(
    name='custom_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['my_function'],
    verbose=True,
    extra_cuda_cflags=['-O3']
)

# Use it
x = torch.randn(1000000, device='cuda')
result = custom_ops.my_function(x)

ref = x * 2

print(f"Error: {F.l1_loss(result, ref)}")