import torch
from torch.utils.cpp_extension import load
from torch.autograd.profiler import profile

cuda_module = load(
        name="add2",
        extra_include_paths=["include"],
        sources=["kernel/add_kernel.cu", "kernel/add_ops.cpp"],
    )

def warmup(device="cuda"):
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    _ = a @ b
    print("Warmup done")

def pytorch_add(a, b):
    return a + b


def my_add(a, b):
    return cuda_module.add2(a, b)


if __name__ == "__main__":
    warmup()

    device = "cuda"

    a = torch.randn(2, 128, 2048, device=device)
    b = torch.randn(a.shape, device=device)

    with profile(use_cuda=True) as prof:
        c1 = pytorch_add(a, b)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with profile(use_cuda=True) as prof:
        c2 = my_add(a, b)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    if torch.allclose(c1, c2, rtol=0, atol=1e-2):
        print("All close")
    else:
        print("Not all close")
        print("C1: ")
        print(c1)
        print("C2: ")
        print(c2)