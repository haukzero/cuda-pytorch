#include "my_add.h"
#include <torch/extension.h>

torch::Tensor torch_add2(const torch::Tensor &a, const torch::Tensor &b)
{
    auto c = torch::empty_like(a);
    add2(c.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), c.numel());
    return c;
}

// 在名为 add2_cmake 的模块注册一个名为 add2 的函数
TORCH_LIBRARY(add2_cmake, m)
{
    m.def("add2", &torch_add2);
}
