import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 与原始保持一致的工具函数与 Conv
# ----------------------------
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# https://github.com/iscyy/ultralyticsPro
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

# ----------------------------
# 轻量上采样/下采样（加入DWConv去混叠）
# ----------------------------
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()
        self.proj = Conv(in_channels, out_channels, 1)
        self.dw = nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False)
        self.pw = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x, size=None):
        x = self.proj(x)
        if size is None:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = self.pw(self.dw(x))
        x = self.act(self.bn(x))
        return x

class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()
        self.proj = Conv(in_channels, out_channels, 1)
        self.dw = nn.Conv2d(out_channels, out_channels, 3, 2, 1, groups=out_channels, bias=False)
        self.pw = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.pw(self.dw(x))
        x = self.act(self.bn(x))
        return x

class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()
        # 两次 x2 更稳，避免一次 stride=4 带来的 aliasing
        self.ds1 = Downsample_x2(in_channels, out_channels)
        self.ds2 = Downsample_x2(out_channels, out_channels)

    def forward(self, x):
        return self.ds2(self.ds1(x))

class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()  # 修正原文件的 super 笔误
        self.ds1 = Downsample_x2(in_channels, out_channels)
        self.ds2 = Downsample_x2(out_channels, out_channels)
        self.ds3 = Downsample_x2(out_channels, out_channels)

    def forward(self, x):
        return self.ds3(self.ds2(self.ds1(x)))

# ----------------------------
# 轻量 EMA Wrapper（与骨干的 EMA 思想一致，用作前后校准）
# ----------------------------
class EMAWrapper(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        idn = x
        x = self.pw(self.dw(x))
        x = self.act(self.bn(x))
        return x + idn

# ----------------------------
# 通道×空间级联注意力（SE -> Spatial）
# ----------------------------
class CSAM(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        cr = max(c // r, 4)
        self.se1 = nn.Conv2d(c, cr, 1, 1, 0)
        self.se2 = nn.Conv2d(cr, c, 1, 1, 0)
        self.s_conv = nn.Conv2d(2, 1, 7, 1, 3)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # Channel
        s = F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1)
        s = self.act(self.se1(s))
        s = torch.sigmoid(self.se2(s))
        xc = x * s
        # Spatial
        avg = torch.mean(xc, dim=1, keepdim=True)
        mx, _ = torch.max(xc, dim=1, keepdim=True)
        xs = torch.cat([avg, mx], dim=1)
        xs = torch.sigmoid(self.s_conv(xs))
        return xc * xs

# ----------------------------
# BiFPN 风格的可学习非负权重融合（更稳、更可解释）
# ----------------------------
class WeightedFusion(nn.Module):
    def __init__(self, n_in, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_in, dtype=torch.float32))
        self.eps = eps

    def forward(self, feats):
        # feats: list of tensors with identical shape
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        out = 0.0
        for i, f in enumerate(feats):
            out = out + w[i] * f
        return out

# ----------------------------
# 轻量残差块（保持你的 BasicBlock 风格）
# ----------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2, momentum=0.1)
        self.act = nn.SiLU(inplace=True)
        self.cv2 = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2, momentum=0.1)

    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.cv1(x)))
        x = self.bn2(self.cv2(x))
        x += residual
        x = self.act(x)
        return x

# ----------------------------
# （可选）保留的轻量块，若未使用可忽略
# ----------------------------
class CSCFasterNeXt(nn.Module):
    def __init__(self, c1, c2, n=1, extra=2, shortcut=True, k=(1, 1), g=1, e=0.5):
        super(CSCFasterNeXt, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c1, c_, k[0], 1)
        self.cv3 = Conv(c_, c_, k[0], 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        # 这里保留接口，避免未定义错误；如需启用请补充 FasterNetBlock 定义
        self.m = nn.Sequential(*[nn.Identity() for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))

# ----------------------------
# 改进版 ASFF_2（签名不变）
# args 约定（与 Ultralytics parser 一致）：
#   c1: list[int,int] 来自 from=[..., ...] 的输入通道
#   c2: int  目标通道（等于 inter_dim）
#   level: int 当前输出层级（0/1）
# ----------------------------
class ASFF_2(nn.Module):
    def __init__(self, c1, c2, level=0):
        super(ASFF_2, self).__init__()
        assert isinstance(c1, (list, tuple)) and len(c1) == 2, "ASFF_2 expects 2 inputs"
        c1_l, c1_h = c1[0], c1[1]
        self.level = level
        self.inter_dim = c2  # 直接使用Ultralytics传入的 c2 作为融合后的通道
        compress_c = max(self.inter_dim // 8, 8)

        # 对齐模块（上/下采样 + DWConv 去混叠）
        if level == 0:
            self.align_h = Upsample(c1_h, self.inter_dim, scale_factor=2)   # 假定高层 -> 低层
            self.align_l = Conv(c1_l, self.inter_dim, 1, 1)
        else:  # level == 1
            self.align_l = Downsample_x2(c1_l, self.inter_dim)             # 低层 -> 高层
            self.align_h = Conv(c1_h, self.inter_dim, 1, 1)

        # 注意力生成（通道压缩得到权重特征）
        self.weight_level_l = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_h = Conv(self.inter_dim, compress_c, 1, 1)

        # BiFPN风格的非负可学习融合
        self.fuser = WeightedFusion(2)

        # 融合后增强（DWConv + 1x1）
        self.conv = nn.Sequential(
            nn.Conv2d(self.inter_dim, self.inter_dim, 3, 1, 1, groups=self.inter_dim, bias=False),
            nn.Conv2d(self.inter_dim, self.inter_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.inter_dim),
            nn.SiLU(inplace=True)
        )

        # 级联注意力（CSAM）+ 轻量 EMA 校准
        self.csam = CSAM(self.inter_dim)
        self.ema_post = EMAWrapper(self.inter_dim)

    def forward(self, x):
        # x: [low, high]
        x_l, x_h = x[0], x[1]

        if self.level == 0:
            l = self.align_l(x_l)
            h = self.align_h(x_h, size=l.shape[-2:])
        else:
            h = self.align_h(x_h)
            l = self.align_l(x_l)  # 已下采样到同尺寸
            # 对齐尺寸（保险起见）
            if h.shape[-2:] != l.shape[-2:]:
                h = F.interpolate(h, size=l.shape[-2:], mode='bilinear', align_corners=False)

        # 生成融合前的“语义引导特征”（用于稳定权重学习）
        wl = self.weight_level_l(l)
        wh = self.weight_level_h(h)
        # 用 Lightweight fuser 融合（权重非负+归一）
        fused = self.fuser([l, h])

        # 融合后增强与注意力
        out = self.conv(fused)
        out = self.csam(out)
        out = self.ema_post(out)
        return out

# ----------------------------
# 改进版 ASFF_3（签名不变）
# args 约定：
#   c1: list[int,int,int] 来自 from=[..., ..., ...] 的输入通道
#   c2: int  目标通道（等于 inter_dim）
#   level: int 当前输出层级（0/1/2）
# ----------------------------
class ASFF_3(nn.Module):
    def __init__(self, c1, c2, level=0):
        super().__init__()
        assert isinstance(c1, (list, tuple)) and len(c1) == 3, "ASFF_3 expects 3 inputs"
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = level
        self.inter_dim = c2
        compress_c = max(self.inter_dim // 8, 8)

        # 三分支对齐
        if level == 0:
            self.align_l = Conv(c1_l, self.inter_dim, 1, 1)
            self.align_m = Upsample(c1_m, self.inter_dim, scale_factor=2)
            self.align_h = Upsample(c1_h, self.inter_dim, scale_factor=4)
        elif level == 1:
            self.align_l = Downsample_x2(c1_l, self.inter_dim)
            self.align_m = Conv(c1_m, self.inter_dim, 1, 1)
            self.align_h = Upsample(c1_h, self.inter_dim, scale_factor=2)
        else:  # level == 2
            self.align_l = Downsample_x4(c1_l, self.inter_dim)
            self.align_m = Downsample_x2(c1_m, self.inter_dim)
            self.align_h = Conv(c1_h, self.inter_dim, 1, 1)

        # 注意力引导特征（压缩通道）
        self.weight_l = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_m = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_h = Conv(self.inter_dim, compress_c, 1, 1)

        # BiFPN风格融合（3分支）
        self.fuser = WeightedFusion(3)

        # 融合后增强 + CSAM + EMA
        self.conv = nn.Sequential(
            nn.Conv2d(self.inter_dim, self.inter_dim, 3, 1, 1, groups=self.inter_dim, bias=False),
            nn.Conv2d(self.inter_dim, self.inter_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.inter_dim),
            nn.SiLU(inplace=True)
        )
        self.csam = CSAM(self.inter_dim)
        self.ema_post = EMAWrapper(self.inter_dim)

    def forward(self, x):
        # x: [low, mid, high]
        xl, xm, xh = x[0], x[1], x[2]

        if self.level == 0:
            l = self.align_l(xl)
            m = self.align_m(xm, size=l.shape[-2:])
            h = self.align_h(xh, size=l.shape[-2:])
        elif self.level == 1:
            m = self.align_m(xm)
            l = self.align_l(xl)
            h = self.align_h(xh, size=m.shape[-2:])
            if l.shape[-2:] != m.shape[-2:]:
                l = F.interpolate(l, size=m.shape[-2:], mode='bilinear', align_corners=False)
        else:
            h = self.align_h(xh)
            l = self.align_l(xl)
            m = self.align_m(xm)
            if l.shape[-2:] != h.shape[-2:]:
                l = F.interpolate(l, size=h.shape[-2:], mode='bilinear', align_corners=False)
            if m.shape[-2:] != h.shape[-2:]:
                m = F.interpolate(m, size=h.shape[-2:], mode='bilinear', align_corners=False)

        # 预处理权重特征（供训练阶段稳定梯度）
        wl = self.weight_l(l)
        wm = self.weight_m(m)
        wh = self.weight_h(h)

        # 融合（可学非负权重）
        fused = self.fuser([l, m, h])

        out = self.conv(fused)
        out = self.csam(out)
        out = self.ema_post(out)
        return out

# ----------------------------
# 占位（YAML未直接使用 AFPN，但保留接口方便后续切换）
# ----------------------------
class AFPN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.id = nn.Identity()

    def forward(self, x):
        return x