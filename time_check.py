"""
Benchmark GPU vs CPU speed.
"""
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

matrix_size = 1000
cpu_matrix1 = torch.randn(matrix_size, matrix_size)
cpu_matrix2 = torch.randn(matrix_size, matrix_size)

# Measure time for CPU computation
start_time = time.perf_counter()
cpu_result = torch.matmul(cpu_matrix1, cpu_matrix2)
cpu_time = time.perf_counter() - start_time
print(f"CPU computation time: {cpu_time:.6f} seconds")

# Measure time for GPU computation
if torch.cuda.is_available():
    gpu_matrix1 = cpu_matrix1.to(device)
    gpu_matrix2 = cpu_matrix2.to(device)

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    result = torch.matmul(gpu_matrix1, gpu_matrix2)
    end_time.record()
    torch.cuda.synchronize()
    gpu_time = 0.001 * start_time.elapsed_time(end_time)
    print(f"GPU computation time: {gpu_time:.6f} seconds")
