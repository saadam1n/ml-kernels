import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import torch.utils.benchmark as benchmark

def compile_kernel_from_file(path):
    """
    Compile a CUDA kernel from a file using JIT compilation.
    
    :param path: Path to the .cu file containing both C++ binding and CUDA implementation
    :return: Compiled kernel function
    """
    with open(path, "r") as f:
        source = f.read()

    # cpp source contains signature for function. the kernel is always "kernel_binding"
    cpp_src = "#define CPP_IMPL\n" + source
    # cuda source contains actual kernel implementation
    cuda_src = "#define CUDA_IMPL\n" + source

    custom_ops = load_inline(
        name='custom_ops',
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["kernel_binding"],
        verbose=True,
        extra_cuda_cflags=['-O3']
    )

    return custom_ops.kernel_binding


def test_matmul():
    print(f"Running matmul test...")

    kernel = compile_kernel_from_file("kernels/matmul.cu")

    # Step 1: accuracy test
    print("\n=== Accuracy Test ===")
    
    # Test various matrix sizes
    test_sizes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 256, 128),
    ]
    
    for M, N, K in test_sizes:
        # Create random input matrices
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        # Compute with custom kernel
        C_custom = kernel(A, B)
        
        # Compute with PyTorch reference
        C_pytorch = torch.matmul(A, B)
        
        # Check correctness
        max_diff = (C_custom - C_pytorch).abs().max().item()
        rel_error = ((C_custom - C_pytorch).abs() / (C_pytorch.abs() + 1e-8)).mean().item()
        
        print(f"Size ({M}x{K}) @ ({K}x{N}): max_diff={max_diff:.6f}, rel_error={rel_error:.6f}")
        
        # Assert reasonable accuracy (adjust tolerance as needed)
        assert max_diff < 1e-3, f"Accuracy check failed for size {M}x{N}x{K}"
    
    print("✓ All accuracy tests passed!")

    # Step 2: performance test
    print("\n=== Performance Test ===")
    
    # Test on larger matrices for performance comparison
    perf_sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    
    for M, N, K in perf_sizes:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(10):
            _ = kernel(A, B)
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()
        
        # Benchmark custom kernel
        t_custom = benchmark.Timer(
            stmt='kernel(A, B)',
            globals={'kernel': kernel, 'A': A, 'B': B}
        ).blocked_autorange(min_run_time=1.0)
        
        # Benchmark PyTorch
        t_pytorch = benchmark.Timer(
            stmt='torch.matmul(A, B)',
            globals={'A': A, 'B': B}
        ).blocked_autorange(min_run_time=1.0)
        
        custom_time = t_custom.median * 1000  # Convert to ms
        pytorch_time = t_pytorch.median * 1000
        speedup = pytorch_time / custom_time
        
        # Calculate FLOPS (2*M*N*K operations)
        flops = 2 * M * N * K
        custom_tflops = (flops / (custom_time / 1000)) / 1e12
        pytorch_tflops = (flops / (pytorch_time / 1000)) / 1e12
        
        print(f"\nSize ({M}x{K}) @ ({K}x{N}):")
        print(f"  Custom kernel:  {custom_time:.3f} ms ({custom_tflops:.2f} TFLOPS)")
        print(f"  PyTorch:        {pytorch_time:.3f} ms ({pytorch_tflops:.2f} TFLOPS)")
        print(f"  Speedup:        {speedup:.2f}x")

if __name__ == "__main__":
    test_matmul()