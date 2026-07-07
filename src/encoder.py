from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        padding = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = padding // 2
        pad_right = padding - pad_left
        return self.conv(F.pad(x, (pad_left, pad_right), "constant", 0))


class SamePadMaxPool1d(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        padding = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = padding // 2
        pad_right = padding - pad_left
        return self.max_pool(F.pad(x, (pad_left, pad_right), "constant", 0))


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        downsample: bool,
        use_bn: bool,
        use_dropout: bool,
        is_first_block: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.stride = stride if downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)
        self.conv1 = SamePadConv1d(in_channels, out_channels, kernel_size, self.stride, groups)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.5)
        self.conv2 = SamePadConv1d(out_channels, out_channels, kernel_size, 1, groups)
        self.max_pool = SamePadMaxPool1d(kernel_size=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_dropout:
                out = self.drop1(out)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_dropout:
            out = self.drop2(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)
        return out + identity


class PulsePPGResNet1D(nn.Module):
    """Pulse-PPG-style 1D ResNet backbone for PPG embeddings."""

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 128,
        kernel_size: int = 11,
        stride: int = 2,
        groups: int = 1,
        n_block: int = 12,
        final_pool: str = "max",
        downsample_gap: int = 2,
        increasefilter_gap: int = 4,
        use_bn: bool = True,
        use_dropout: bool = True,
        input_instance_norm: bool = True,
    ):
        super().__init__()
        if final_pool not in {"max", "avg"}:
            raise ValueError("final_pool must be 'max' or 'avg'")

        self.final_pool = final_pool
        self.input_norm = nn.InstanceNorm1d(in_channels) if input_instance_norm else nn.Identity()
        self.first_block_conv = SamePadConv1d(in_channels, base_filters, kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()

        out_channels = base_filters
        blocks: list[nn.Module] = []
        for i_block in range(n_block):
            is_first_block = i_block == 0
            downsample = i_block % downsample_gap == 1
            if is_first_block:
                block_in = base_filters
                block_out = block_in
            else:
                block_in = int(base_filters * 2 ** ((i_block - 1) // increasefilter_gap))
                block_out = block_in * 2 if (i_block % increasefilter_gap == 0 and i_block != 0) else block_in

            blocks.append(
                BasicBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=groups,
                    downsample=downsample,
                    use_bn=use_bn,
                    use_dropout=use_dropout,
                    is_first_block=is_first_block,
                )
            )
            out_channels = block_out

        self.basicblock_list = nn.ModuleList(blocks)
        self.output_dim = out_channels

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_norm(x)
        out = self.first_block_conv(out)
        out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        for block in self.basicblock_list:
            out = block(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        if self.final_pool == "avg":
            return torch.mean(features, dim=-1)
        return torch.max(features, dim=-1).values


def build_pulseppg_encoder(**kwargs) -> PulsePPGResNet1D:
    return PulsePPGResNet1D(**kwargs)
