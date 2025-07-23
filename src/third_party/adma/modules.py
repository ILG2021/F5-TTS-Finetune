from typing import Tuple

import torch.nn.functional as F
from torch import nn

from f5_tts.model.utils import lens_to_mask


class SpeechAlignMLP(nn.Module):
    def __init__(
        self,
        sampling_ratios: Tuple,
        in_channels: int,
        channels: int,
        out_channels: int,
        groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for i, _ in enumerate(sampling_ratios):
                module = nn.Conv1d(in_channels if i == 0 else channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x, ylens):
        # x in (B, T, D)
        mask = lens_to_mask(ylens).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode="linear")
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        return out * mask, olens


class TextAlignMLP(nn.Module):
    def __init__(
        self,
        reduce_ratios: Tuple,
        in_channels: int,
        channels: int,
        out_channels: int,
        groups: int = 1,
    ):
        super().__init__()
        self.reduce_ratios = reduce_ratios
        model = nn.ModuleList([])
        if len(reduce_ratios) > 0:
            for i, r in enumerate(reduce_ratios):
                module = nn.Conv1d(in_channels if i == 0 else channels, channels, 3, r, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x, ylens):
        # x in (B, T, D)
        for r in self.reduce_ratios:
            ylens = (ylens + 2 * 1 - 3) // r + 1
        mask = lens_to_mask(ylens).unsqueeze(-1)
        x = x.transpose(1, 2).contiguous()
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        return out * mask, olens