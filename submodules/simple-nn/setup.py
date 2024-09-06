from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == "nt":
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_nn",
    ext_modules=[
        CUDAExtension(
            name="simple_nn._C",
            sources=["simple_nn.cu", "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
