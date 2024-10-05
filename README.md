# 在 PyTorch 中使用自定义的 CUDA 算子

> 注意: 在 C++ 代码声明中需要导入 `torch` 头文件, 若已安装 pytorch， 则不必重复安装 libtorch, 以免出现冲突问题

## 三种方法

- [即时编译 (JIT)](./jit/README.md)
- [使用 setup 打包](./setup/README.md)
- [使用 CMake](./cmake/README.md)

### 三种方法比较

- 速度:
    - 使用 CMake 方式会更慢, 其余二者速度相仿
- 更新维护:
    - 使用 JIT 和 CMake 的方式只需重新编译既可, 更为方便
    - 使用 setup 打包在更新时需要先卸载原先的包再重新打包
- python 导入方便程度:
    - setup 方式只需像其他包一样 import 即可
    - CMake 方式需要指明连接到的算子所在动态库位置
    - JIT 方式需要把 C/C++/CUDA 源代码位置显示指明

## 示例代码结构

三种方法均以一个简单的 tensor 加法为样例, 并在 `time.py` 文件中将其与 PyTorch 原生加法的运算速度作比较, 在 `model.py` 文件中将这个算子放入到神经网络中做反向传播

```
.
├── Makefile
├── include
│   └── my_add.h
├── kernel
│   ├── add_kernel.cu
│   └── add_ops.cpp
├── model.py
└── time.py
```
