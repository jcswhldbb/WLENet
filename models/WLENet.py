import torch.nn as nn
import math
import torch
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
from contextlib import nullcontext


    
class DoubleConv(nn.Module):
    """(Conv -> BN -> GELU) x 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        wave="db2",
        smooth_descriptor=False,
    ):
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("通道数必须大于 0")

        self.in_channels = in_channels
        self.smooth_descriptor = smooth_descriptor

        self.pool = nn.MaxPool2d(2, 2)
        self.conv = DoubleConv(in_channels, out_channels)

        self.dwt = DWTForward(
            J=1,
            wave=wave,
            mode="periodization",
        )
        self.idwt = DWTInverse(
            wave=wave,
            mode="periodization",
        )

        self.min_input_size = max(
            4,
            self.dwt.h0_col.numel(),
            self.dwt.h0_row.numel(),
            self.idwt.g0_col.numel(),
            self.idwt.g0_row.numel(),
        )

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ) / 8.0

        sobel_y = sobel_x.t().contiguous()

        self.register_buffer(
            "sobel_x",
            sobel_x.view(1, 1, 3, 3).repeat(
                in_channels, 1, 1, 1
            ),
        )
        self.register_buffer(
            "sobel_y",
            sobel_y.view(1, 1, 3, 3).repeat(
                in_channels, 1, 1, 1
            ),
        )

        self.diff_gate = nn.Sequential(
    nn.Conv2d(
        in_channels*2,
        in_channels,
        kernel_size=3,
        padding=1,
        groups=in_channels,
        bias=False,
    ),
    nn.Conv2d(
        in_channels,
        in_channels,
        kernel_size=1,
        bias=True,
    ),
    nn.Sigmoid(),

)



    def _reconstruct_low(self, x):
        h, w = x.shape[-2:]

        ll, high = self.dwt(x)

        zero_high = [
            torch.zeros_like(hp) for hp in high
        ]

        x_low = self.idwt((ll, zero_high))

        return x_low[..., :h, :w]

    def _build_low_descriptor(self, low_ref):
        if self.smooth_descriptor:
            low_ref = F.avg_pool2d(
                F.pad(
                    low_ref,
                    (1, 1, 1, 1),
                    mode="replicate",
                ),
                kernel_size=3,
                stride=1,
            )

        padded = F.pad(
            low_ref,
            (1, 1, 1, 1),
            mode="replicate",
        )

        gx = F.conv2d(
            padded,
            self.sobel_x,
            groups=self.in_channels,
        )
        gy = F.conv2d(
            padded,
            self.sobel_y,
            groups=self.in_channels,
        )

        return torch.sqrt(
            gx.square() + gy.square() + 1e-6
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"输入应为 [B, {self.in_channels}, H, W]"
            )

        h, w = x.shape[-2:]

        if (
            h % 2 != 0
            or w % 2 != 0
            or min(h, w) < self.min_input_size
        ):
            raise ValueError(
                f"高宽必须为偶数且不小于 "
                f"{self.min_input_size}，实际为 {(h, w)}"
            )

        # 1. 低频重建
        x_low = self._reconstruct_low(x)
        # 2. 低频主线
        low_ref = self.pool(x_low)
        # 3. 原始特征下采样
        x_down = self.pool(x)
        # 4. 下采样差异
        detail = x_down - low_ref
        edge = self._build_low_descriptor(low_ref)       
        b, c, out_h, out_w = detail.shape
        
        
        context = torch.stack(
            [detail.detach(), edge],
            dim=2,
        ).reshape(b, 2 * c, out_h, out_w)

        # 5. 低频结构梯度

        gate = self.diff_gate(context)
        fused = low_ref + gate * detail

        return self.conv(fused)
        
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.relu = nn.ReLU(inplace=True)
        self.ChannelAttention = ChannelAttention(F_int)
        self.SpatialAttention = SpatialAttention()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        x = x * self.ChannelAttention(psi)  # 通道注意力
        x = x * self.SpatialAttention(psi)

        return x


class ChannelAttention(nn.Module):  # 通道注意力机制
    def __init__(self, in_planes, scaling=16):  # scaling为缩放比例，
        # 用来控制两个全连接层中间神经网络神经元的个数，一般设置为16，具体可以根据需要微调
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):  # 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class Att_up(nn.Module):
    def __init__(self, channels1, channels2):
        super().__init__()
        self.dwt = DWTForward(J=1, wave='db2', mode='periodization')
        self.attention = Attention_block(channels2, channels2, channels2)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels1, channels2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels2),
            nn.ReLU()
        )

        self.fusion = DoubleConv(channels2 + channels1, channels1)
        self.conv = DoubleConv(channels2 * 2, channels2)



    def forward(self, x1, x2):
        x2_ll,_ = self.dwt(x2)
        x1 = torch.cat([x2_ll, x1], dim=1)
        x1 = self.fusion(x1)
        x1_up = self.up(x1)
        x2 = self.attention(x1_up, x2)

        x = torch.cat([x2, x1_up], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class wlenet(nn.Module):
    def __init__(self, n_channels, num_classes, bilinear=False):
        super(wlenet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512))



        self.up1 = Att_up(512, 256)
        self.up2 = Att_up(256, 128)
        self.up3 = Att_up(128, 64)
        self.up4 = Att_up(64, 32)
        self.outc = (OutConv(32, num_classes))



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out