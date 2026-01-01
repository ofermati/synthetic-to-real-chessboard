from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


# -------------------------
# Utils / init
# -------------------------

def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialize network weights (common for GANs)."""

    def _init(m: nn.Module):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname or "InstanceNorm2d" in classname:
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(_init)


def get_norm(norm: Literal["instance", "batch", "none"] = "instance"):
    if norm == "instance":
        return lambda c: nn.InstanceNorm2d(c, affine=False, track_running_stats=False)
    if norm == "batch":
        return lambda c: nn.BatchNorm2d(c)
    if norm == "none":
        return lambda c: nn.Identity()
    raise ValueError(f"Unknown norm: {norm}")


def get_act(act: Literal["relu", "lrelu"] = "relu"):
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "lrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    raise ValueError(f"Unknown act: {act}")


# -------------------------
# Building blocks
# -------------------------


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        norm: Literal["instance", "batch", "none"] = "instance",
        act: Literal["relu", "lrelu"] = "relu",
        use_act: bool = True,
    ):
        super().__init__()
        Norm = get_norm(norm)
        layers = [nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=(norm == "none"))]
        layers.append(Norm(out_c))
        if use_act:
            layers.append(get_act(act))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResnetBlock(nn.Module):
    """ResNet block used in CycleGAN-style generators."""

    def __init__(self, dim: int, norm: Literal["instance", "batch", "none"] = "instance"):
        super().__init__()
        Norm = get_norm(norm)
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=(norm == "none")),
            Norm(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=(norm == "none")),
            Norm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# -------------------------
# Generators
# -------------------------


class ResnetGenerator(nn.Module):
    """ResNet generator (CycleGAN classic)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        norm: Literal["instance", "batch", "none"] = "instance",
    ):
        super().__init__()
        Norm = get_norm(norm)

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=0, bias=(norm == "none")),
            Norm(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsample
        c = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=(norm == "none")),
                Norm(c * 2),
                nn.ReLU(inplace=True),
            ]
            c *= 2

        # Res blocks
        for _ in range(n_blocks):
            layers.append(ResnetBlock(c, norm=norm))

        # Upsample
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    c,
                    c // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=(norm == "none"),
                ),
                Norm(c // 2),
                nn.ReLU(inplace=True),
            ]
            c //= 2

        # Output
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),  # assumes images normalized to [-1, 1]
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSkipBlock(nn.Module):
    """U-Net skip connection block."""

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        in_nc: Optional[int] = None,
        submodule: Optional[nn.Module] = None,
        outermost: bool = False,
        innermost: bool = False,
        norm: Literal["instance", "batch", "none"] = "batch",
        use_dropout: bool = False,
    ):
        super().__init__()
        if in_nc is None:
            in_nc = outer_nc
        Norm = get_norm(norm)

        downconv = nn.Conv2d(in_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=(norm == "none"))
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = Norm(inner_nc)

        uprelu = nn.ReLU(inplace=True)
        upnorm = Norm(outer_nc)

        if outermost:
            assert submodule is not None
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=(norm == "none"))
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            assert submodule is not None
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=(norm == "none")
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        self.outermost = outermost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class UNetGenerator(nn.Module):
    """Pix2Pix U-Net generator."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        num_downs: int = 8,
        norm: Literal["instance", "batch", "none"] = "batch",
        use_dropout: bool = False,
    ):
        super().__init__()

        # Build innermost
        unet_block: nn.Module = UNetSkipBlock(ngf * 8, ngf * 8, innermost=True, norm=norm)

        # Add intermediate blocks with ngf*8
        for _ in range(num_downs - 5):
            unet_block = UNetSkipBlock(ngf * 8, ngf * 8, submodule=unet_block, norm=norm, use_dropout=use_dropout)

        # Gradually reduce channels
        unet_block = UNetSkipBlock(ngf * 4, ngf * 8, submodule=unet_block, norm=norm)
        unet_block = UNetSkipBlock(ngf * 2, ngf * 4, submodule=unet_block, norm=norm)
        unet_block = UNetSkipBlock(ngf, ngf * 2, submodule=unet_block, norm=norm)

        # Outermost
        self.model = UNetSkipBlock(out_channels, ngf, in_nc=in_channels, submodule=unet_block, outermost=True, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# -------------------------
# Discriminators
# -------------------------


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm: Literal["instance", "batch", "none"] = "instance",
    ):
        super().__init__()
        Norm = get_norm(norm)

        kw = 4
        padw = 1
        sequence: list[nn.Module] = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=(norm == "none"),
                ),
                Norm(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # last conv stride=1
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=(norm == "none"),
            ),
            Norm(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # output 1-channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# -------------------------
# Losses
# -------------------------


class GANLoss(nn.Module):
    """Generic GAN loss."""

    def __init__(self, gan_mode: Literal["lsgan", "bce"] = "lsgan"):
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.criterion = nn.MSELoss()
        elif gan_mode == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown gan_mode: {gan_mode}")

    def _target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        return torch.ones_like(pred) if is_real else torch.zeros_like(pred)

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = self._target(pred, is_real)
        return self.criterion(pred, target)


# -------------------------
# Convenience factory
# -------------------------


@dataclass
class NetConfig:
    img_channels: int = 3
    norm_g: Literal["instance", "batch", "none"] = "instance"
    norm_d: Literal["instance", "batch", "none"] = "instance"
    gan_mode: Literal["lsgan", "bce"] = "lsgan"


def build_generator(kind: Literal["resnet", "unet"], cfg: NetConfig, **kwargs) -> nn.Module:
    if kind == "resnet":
        return ResnetGenerator(in_channels=cfg.img_channels, out_channels=cfg.img_channels, norm=cfg.norm_g, **kwargs)
    if kind == "unet":
        return UNetGenerator(in_channels=cfg.img_channels, out_channels=cfg.img_channels, norm=cfg.norm_g, **kwargs)
    raise ValueError(f"Unknown generator kind: {kind}")


def build_discriminator(in_channels: int, cfg: NetConfig, **kwargs) -> nn.Module:
    return PatchDiscriminator(in_channels=in_channels, norm=cfg.norm_d, **kwargs)
