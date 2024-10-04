import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Function


class Add2Function(Function):
    @staticmethod
    def forward(ctx, a, b):
        # ctx 用于保存信息，以便在 backward 时使用
        # ctx.save_for_backward(a, b)
        return torch.ops.add2_cmake.add2(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        # 可以在此处调用 ctx.saved_tensors 获取保存的信息
        # a, b = ctx.saved_tensors
        return grad_output, grad_output


# c = a^2 + b^2
class Net(nn.Module):
    def __init__(self, shape):
        super(Net, self).__init__()
        self.shape = shape
        self.a = nn.Parameter(torch.randn(shape))
        self.b = nn.Parameter(torch.randn(shape))

    def forward(self):
        a2 = self.a**2
        b2 = self.b**2
        return Add2Function.apply(a2, b2)


if __name__ == "__main__":
    torch.ops.load_library("build/libadd2.so")

    shape = (2, 128, 2048)
    device = "cuda"
    lr = 1e-2

    net = Net(shape).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    epoches = 1000
    loss_list = []
    for epoch in range(epoches):
        optimizer.zero_grad()
        loss = net().sum()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curve")
    plt.savefig("loss.png")
    print("Loss curve saved")
