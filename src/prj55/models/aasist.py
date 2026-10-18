"""AASIST and AASIST-L Baseline Neural Architecture.

Derived from official AASIST implementation:
Copyright (c) 2021-present NAVER Corp.
MIT License (Jung et al., ICASSP 2022)
Reference: https://github.com/clovaai/aasist
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AASISTConfig:
    """Configuration dataclass for AASIST and AASIST-L models."""

    nb_samp: int = 64600
    first_conv: int = 128
    filts: list[Any] = field(
        default_factory=lambda: [70, [1, 32], [32, 32], [32, 24], [24, 24]]
    )
    gat_dims: list[int] = field(default_factory=lambda: [24, 32])
    pool_ratios: list[float] = field(default_factory=lambda: [0.4, 0.5, 0.7, 0.5])
    temperatures: list[float] = field(default_factory=lambda: [2.0, 2.0, 100.0, 100.0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "nb_samp": self.nb_samp,
            "first_conv": self.first_conv,
            "filts": self.filts,
            "gat_dims": self.gat_dims,
            "pool_ratios": self.pool_ratios,
            "temperatures": self.temperatures,
        }


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_bn(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x: Tensor) -> Tensor:
        nb_nodes = x.size(1)
        x_exp = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x_exp.transpose(1, 2)
        return x_exp * x_mirror

    def _derive_att_map(self, x: Tensor) -> Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_bn(self, x: Tensor) -> Tensor:
        org_size = x.size()
        x_flat = x.view(-1, org_size[-1])
        x_norm = self.bn(x_flat)
        return x_norm.view(org_size)

    def _init_new_params(self, *size: int) -> nn.Parameter:
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_proj_m = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weight_m = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_att_m = nn.Linear(in_dim, out_dim)
        self.proj_without_att_m = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(
        self, x1: Tensor, x2: Tensor, master: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1_proj = self.proj_type1(x1)
        x2_proj = self.proj_type2(x2)
        x = torch.cat([x1_proj, x2_proj], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_bn(x)
        x = self.act(x)

        x1_out = x.narrow(1, 0, num_type1)
        x2_out = x.narrow(1, num_type1, num_type2)
        return x1_out, x2_out, master

    def _update_master(self, x: Tensor, master: Tensor) -> Tensor:
        att_map = self._derive_att_map_master(x, master)
        return self._project_master(x, master, att_map)

    def _pairwise_mul_nodes(self, x: Tensor) -> Tensor:
        nb_nodes = x.size(1)
        x_exp = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x_exp.transpose(1, 2)
        return x_exp * x_mirror

    def _derive_att_map_master(self, x: Tensor, master: Tensor) -> Tensor:
        att_map = x * master
        att_map = torch.tanh(self.att_proj_m(att_map))
        att_map = torch.matmul(att_map, self.att_weight_m)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _derive_att_map(self, x: Tensor, num_type1: int, num_type2: int) -> Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11
        )
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22
        )
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12
        )
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12
        )

        att_board = att_board / self.temp
        return F.softmax(att_board, dim=-2)

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x: Tensor, master: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att_m(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_att_m(master)
        return x1 + x2

    def _apply_bn(self, x: Tensor) -> Tensor:
        org_size = x.size()
        x_flat = x.view(-1, org_size[-1])
        x_norm = self.bn(x_flat)
        return x_norm.view(org_size)

    def _init_new_params(self, *size: int) -> nn.Parameter:
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: float = 0.3) -> None:
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h: Tensor) -> Tensor:
        z = self.drop(h)
        weights = self.proj(z)
        scores = self.sigmoid(weights)
        return self.top_k_graph(scores, h, self.k)

    @staticmethod
    def top_k_graph(scores: Tensor, h: Tensor, k: float) -> Tensor:
        _, n_nodes, n_feat = h.size()
        nodes_keep = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, nodes_keep, dim=1)
        idx_exp = idx.expand(-1, -1, n_feat)
        h_weighted = h * scores
        return torch.gather(h_weighted, 1, idx_exp)


class SincConv(nn.Module):
    """SincNet bandpass filterbank convolution."""

    band_pass: Tensor

    @staticmethod
    def to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(f"SincConv only supports in_channels=1, got {in_channels}")

        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        nfft = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(nfft / 2) + 1)
        fmel = self.to_mel(f)
        filbandwidthsf = self.to_hz(np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 1))

        hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        band_pass = torch.zeros(self.out_channels, self.kernel_size)
        window = torch.tensor(np.hamming(self.kernel_size), dtype=torch.float32)

        for i in range(len(filbandwidthsf) - 1):
            fmin = filbandwidthsf[i]
            fmax = filbandwidthsf[i + 1]
            sinc_hi = np.sinc(2 * fmax * hsupp.numpy() / self.sample_rate)
            sinc_lo = np.sinc(2 * fmin * hsupp.numpy() / self.sample_rate)
            h_high = (2 * fmax / self.sample_rate) * sinc_hi
            h_low = (2 * fmin / self.sample_rate) * sinc_lo
            h_ideal = torch.tensor(h_high - h_low, dtype=torch.float32)
            band_pass[i, :] = window * h_ideal

        self.register_buffer("band_pass", band_pass)

    def forward(self, x: Tensor) -> Tensor:
        filters = self.band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(
            x,
            filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class ResidualBlock(nn.Module):
    def __init__(self, nb_filts: list[int], first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )
        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        return self.mp(out)


class AASIST(nn.Module):
    """AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention)."""

    def __init__(self, config: AASISTConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            cfg = AASISTConfig()
        elif isinstance(config, dict):
            cfg = AASISTConfig(**config)
        else:
            cfg = config
        self.config = cfg

        filts = cfg.filts
        gat_dims = cfg.gat_dims
        pool_ratios = cfg.pool_ratios
        temperatures = cfg.temperatures

        self.conv_time = SincConv(
            out_channels=filts[0], kernel_size=cfg.first_conv, in_channels=1
        )
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(ResidualBlock(nb_filts=filts[1], first=True)),
            nn.Sequential(ResidualBlock(nb_filts=filts[2])),
            nn.Sequential(ResidualBlock(nb_filts=filts[3])),
            nn.Sequential(ResidualBlock(nb_filts=filts[4])),
            nn.Sequential(ResidualBlock(nb_filts=filts[4])),
            nn.Sequential(ResidualBlock(nb_filts=filts[4])),
        )

        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.gat_layer_s = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[0]
        )
        self.gat_layer_t = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[1]
        )

        self.htrg_gat_st11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.htrg_gat_st12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.htrg_gat_st21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.htrg_gat_st22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        self.pool_s = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_t = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hs1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_ht1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hs2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_ht2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Raw audio waveform tensor of shape `[batch_size, 64600]`.

        Returns:
            tuple of:
              - `last_hidden`: Latent embedding tensor of shape `[batch_size, 160]`.
              - `output`: Logit predictions of shape `[batch_size, 2]`.
                Index 0 = spoof logit, Index 1 = bonafide logit.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x_in = x.unsqueeze(1)  # [bs, 1, seq_len]
        conv_out = self.conv_time(x_in)
        conv_out = conv_out.unsqueeze(dim=1)
        conv_pool = F.max_pool2d(torch.abs(conv_out), (3, 3))
        conv_bn = self.first_bn(conv_pool)
        x_enc = self.selu(conv_bn)

        e = self.encoder(x_enc)

        # Spectral GAT
        e_s, _ = torch.max(torch.abs(e), dim=3)
        e_s = e_s.transpose(1, 2) + self.pos_S
        gat_s = self.gat_layer_s(e_s)
        out_s = self.pool_s(gat_s)

        # Temporal GAT
        e_t, _ = torch.max(torch.abs(e), dim=2)
        e_t = e_t.transpose(1, 2)
        gat_t = self.gat_layer_t(e_t)
        out_t = self.pool_t(gat_t)

        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # Branch 1
        out_t1, out_s1, master1 = self.htrg_gat_st11(out_t, out_s, master=self.master1)
        out_s1 = self.pool_hs1(out_s1)
        out_t1 = self.pool_ht1(out_t1)
        out_t_aug1, out_s_aug1, master_aug1 = self.htrg_gat_st12(out_t1, out_s1, master=master1)
        out_t1 = out_t1 + out_t_aug1
        out_s1 = out_s1 + out_s_aug1
        master1 = master1 + master_aug1

        # Branch 2
        out_t2, out_s2, master2 = self.htrg_gat_st21(out_t, out_s, master=self.master2)
        out_s2 = self.pool_hs2(out_s2)
        out_t2 = self.pool_ht2(out_t2)
        out_t_aug2, out_s_aug2, master_aug2 = self.htrg_gat_st22(out_t2, out_s2, master=master2)
        out_t2 = out_t2 + out_t_aug2
        out_s2 = out_s2 + out_s_aug2
        master2 = master2 + master_aug2

        out_t1 = self.drop_way(out_t1)
        out_t2 = self.drop_way(out_t2)
        out_s1 = self.drop_way(out_s1)
        out_s2 = self.drop_way(out_s2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_t = torch.max(out_t1, out_t2)
        out_s = torch.max(out_s1, out_s2)
        master = torch.max(master1, master2)

        t_max, _ = torch.max(torch.abs(out_t), dim=1)
        t_avg = torch.mean(out_t, dim=1)
        s_max, _ = torch.max(torch.abs(out_s), dim=1)
        s_avg = torch.mean(out_s, dim=1)

        last_hidden = torch.cat([t_max, t_avg, s_max, s_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden, output


def load_aasist_l(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> AASIST:
    """Instantiate and optionally load verified AASIST-L pretrained weights."""
    model = AASIST(AASISTConfig())
    if checkpoint_path is not None:
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"AASIST checkpoint file not found: {checkpoint_path}")
        state_dict = torch.load(ckpt_file, map_location=device)
        # Remap state dict keys if needed to match clean pythonic attribute names
        remapped: dict[str, Tensor] = {}
        key_alias = {
            "GAT_layer_S": "gat_layer_s",
            "GAT_layer_T": "gat_layer_t",
            "HtrgGAT_layer_ST11": "htrg_gat_st11",
            "HtrgGAT_layer_ST12": "htrg_gat_st12",
            "HtrgGAT_layer_ST21": "htrg_gat_st21",
            "HtrgGAT_layer_ST22": "htrg_gat_st22",
            "pool_S": "pool_s",
            "pool_T": "pool_t",
            "pool_hS1": "pool_hs1",
            "pool_hT1": "pool_ht1",
            "pool_hS2": "pool_hs2",
            "pool_hT2": "pool_ht2",
            "conv_time.band_pass": "conv_time.band_pass",
            "att_projM": "att_proj_m",
            "att_weightM": "att_weight_m",
            "proj_with_attM": "proj_with_att_m",
            "proj_without_attM": "proj_without_att_m",
        }
        for k, v in state_dict.items():
            mapped_k = k
            for orig_k, alias_k in key_alias.items():
                if orig_k in mapped_k:
                    mapped_k = mapped_k.replace(orig_k, alias_k)
            remapped[mapped_k] = v

        model.load_state_dict(remapped, strict=False)

    model.to(device)
    model.eval()
    return model
