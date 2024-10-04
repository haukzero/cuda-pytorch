from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="add2",
    version="0.1",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            name="add2",
            sources=["kernel/add_kernel.cu", "kernel/add_ops.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
