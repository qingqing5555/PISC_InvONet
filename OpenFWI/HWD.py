import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch, padding=(0, 0)):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar', dim=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)  # 16 512 32 35
        x = self.conv_bn_relu(x)
        return x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = Down_wt(5, 32, 2)  # 输入通道数，输出通道数
    input = torch.rand(16, 5, 1000, 70)
    output = block(input)   # 3 96 64 32
    print(output.size())
