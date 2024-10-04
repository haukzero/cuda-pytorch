import torch
from torch.utils.cpp_extension import load
from time import perf_counter_ns


def warmup(device="cuda"):
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    _ = a @ b
    print("Warmup done")


if __name__ == "__main__":
    warmup()

    device = "cuda"

    a = torch.randn(2, 128, 2048, device=device)
    b = torch.randn(a.shape, device=device)

    start = perf_counter_ns() / 1e6
    c1 = a + b
    end = perf_counter_ns() / 1e6
    print(f"PyTorch Add cost time: {end - start} ms")

    cuda_module = load(
        name="add2",
        extra_include_paths=["include"],
        sources=["kernel/add_kernel.cu", "kernel/add_ops.cpp"],
    )

    start = perf_counter_ns() / 1e6
    c2 = cuda_module.add2(a, b)
    end = perf_counter_ns() / 1e6
    print(f"My Add cost time: {end - start} ms")

    if torch.allclose(c1, c2):
        print("All close")
    else:
        print("Not all close")
        print("C1: ")
        print(c1)
        print("C2: ")
        print(c2)