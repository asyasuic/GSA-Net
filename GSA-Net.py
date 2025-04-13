import math
from typing import Dict
import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, ratio=2, dw_size=7, stride=1):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.LeakyReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels,   reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out1 = self.conv1(x_cat)
        out2 = self.bn1(out1)
        out3 = self.act1(out2)
        x_h, x_w = torch.split(out3, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h

class ConvAttension(nn.Module):
    def __init__(self, dim, dilation):
        super(ConvAttension, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim,
                                padding_mode='reflect')  # depthwise conv
        self.Conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, dilation=7)
        self.CA = CoordAttention(dim, dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.act1 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(dim)
        self.act2 = nn.GELU()
        self.ghostconv = GhostModule(dim, dim)
        self.attention = CoordAttention(dim, dim, dilation)

    def forward(self, x):
        residual = x
        x1 = self.Conv(x)
        x1 = self.CA(x1)
        x = self.dwconv(x)
        x = self.norm1(x)
        x = self.act1(x)
        x2 = x + residual
        x = self.ghostconv(x2)
        x = x + x2
        x = self.attention(x)
        return self.act2(x + x1)

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

class DWGA(nn.Sequential):
    def __init__(self, out_channels, dilation,  layer_num=1):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(ConvAttension(out_channels, dilation))
        super(DWGA, self).__init__(
            *layers
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, layer_num=1):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(ConvAttension(out_channels, dilation=1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv1x1(x)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class EC(nn.Sequential):
    def __init__(self, in_chn, out_channels, kernel_size, strides, padding):
        super(EC, self).__init__(
            nn.Conv2d(in_channels=in_chn, out_channels=out_channels, kernel_size=kernel_size, stride=strides, padding=padding),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*4, kernel_size=(1, 1),  stride=(1, 1), padding=0),
            nn.Conv2d(in_channels=out_channels*4, out_channels=in_chn, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(in_chn),
        )





class LMFA(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):  # 6 12 18 | 3 5 7 | 4 8 12
        super(LMFA, self).__init__()

        self.LMFA_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 1, stride=1, padding=0, dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.LMFA_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 1, stride=1, padding=0, dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.LMFA_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 1, stride=1, padding=0, dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.LMFA_block1(x)
        x2 = self.LMFA_block2(x)
        x3 = self.LMFA_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class GSANet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(GSANet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.in_conv = nn.Sequential(
            DWGA(in_channels, dilation=1),
            nn.Conv2d(in_channels, base_c, kernel_size=(5, 5), padding=2, padding_mode='reflect'),
            nn.BatchNorm2d(base_c),
            nn.GELU()
        )

        self.aspp = LMFA(16, 16)


        self.FirstFloor = DWGA(base_c, dilation=1)
        self.down1 = Down(base_c, base_c * 4)
        self.SecondFloor = DWGA(base_c * 4, dilation=1)
        self.down2 = Down(base_c * 4, base_c * 8)
        self.ThirdFloor = DWGA(base_c * 8, dilation=3)
        self.down3 = Down(base_c * 8, base_c * 16)
        self.FourthFloor = DWGA(base_c * 16, dilation=5, layer_num=3)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 16, base_c * 16)
        self.Transition = EC(base_c * 16, 1, 1, 1, 0)
        self.up1 = Up(base_c * 32, base_c * 8, bilinear, layer_num=1)
        self.up2 = Up(base_c * 16, base_c * 4, bilinear, layer_num=3)
        self.up3 = Up(base_c * 8, base_c * 2, bilinear)
        self.up4 = Up(base_c * 3, base_c, bilinear)
        self.out_conv = OutConv(16, num_classes)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.FirstFloor(x1)

        x3 = self.down1(x2)
        x4 = self.SecondFloor(x3)

        x5 = self.down2(x4)
        x6 = self.ThirdFloor(x5)

        x7 = self.down3(x6)
        x8 = self.FourthFloor(x7)

        x9 = self.down4(x8)
        x10 = self.Transition(x9)
        x11 = self.act(x10 + x9)

        x = self.up1(x11, x8)
        x = self.up2(x, x6)
        x = self.up3(x, x4)
        x = self.up4(x, x2)
        output = self.aspp(x)
        logits = self.out_conv(output)


        return {"out": logits}


if __name__ == '__main__':
    model = GSANet(in_channels=3, num_classes=2, base_c=16).to('cuda')
    summary(model, input_size=(3, 256, 256))

    # 计算 FLOPs
    dummy_input = torch.randn(1, 3, 256, 256)
    try:
        import thop

        # 计算 FLOPs 和 参数，返回一个元组 (flops, params)
        flops, params = thop.profile(model.to("cuda"), (dummy_input.to("cuda"),), verbose=False)
        # 将 flops 转换为 GFLOPS，params 转换为百万参数
        flops, params = flops / 1e9, params / 1e6
        print(f'Total GFLOPS: {flops:.3f}')
        print(f'Total params: {params:.3f} million')
    except ImportError:
        print("thop is not installed, cannot calculate FLOPs and params")