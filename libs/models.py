""" Libraries """
import torch



""" Models """
class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_2d       = torch.nn.Conv2d()
        self.linear        = torch.nn.Linear()
        self.batch_norm_2d = torch.nn.BatchNorm2d()
        self.silu          = torch.nn.SiLU()

    def forward(self, input):
        x = input
        x = self.conv_2d(x)
        x = self.linear(x)
        x = self.batch_norm_2d(x)
        x = self.silu(x)
        return x


class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.my_module = MyModule()

    def forward(self, input):
        x = input
        x = self.my_module(x)
        return x