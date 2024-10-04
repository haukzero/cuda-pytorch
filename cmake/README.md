# 使用 CMake 导入 CUDA 算子

- 前提:
    - 安装 cmake, cuda 版本的 pytorch
    - 将 python 头文件目录导入 CPATH 环境变量
        - 在 `~/.bashrc` 中添加: `export CPATH=/usr/include/python3.xx:$CPATH`, 其中 `3.xx` 代表自己 python 对应版本号
- 使用头文件 `<torch/extension.h>` 中的宏 `TORCH_LIBRARY` 来注册算子
- 运行命令可参考 Makefile
