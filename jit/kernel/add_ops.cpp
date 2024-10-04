#include "my_add.h"
#include <torch/torch.h>
#include <pybind11/pybind11.h>

torch::Tensor torch_add2(const torch::Tensor &a, const torch::Tensor &b)
{
    auto c = torch::empty_like(a);
    add2(c.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), c.numel());
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add2", &torch_add2, "Add two tensors");
}
