import math
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from src.utils import get_frame, cheb_approx, sinusoidal_posemb, CSiLU

def build_knn_graph(feat: torch.Tensor, k: int):
    N = feat.shape[0]
    k = max(int(k), 1)
    MAX_EXACT = 5000

    if N > MAX_EXACT:
        perm = torch.randperm(N, device=feat.device)[:MAX_EXACT]
        sub = feat[perm]
        sq_all = (feat * feat).sum(dim=1)
        sq_sub = (sub * sub).sum(dim=1)
        dist2 = (sq_all.unsqueeze(1) + sq_sub.unsqueeze(0) - 2.0 * feat @ sub.t())
        _, local_idx = dist2.topk(k, dim=1, largest=False)
        idx = perm[local_idx]
    else:
        sq = (feat * feat).sum(dim=1)
        dist2 = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (feat @ feat.t())
        dist2.fill_diagonal_(float('inf'))
        _, idx = dist2.topk(k, dim=1, largest=False)

    src = torch.arange(N, device=feat.device).unsqueeze(1).expand(N, k).reshape(-1)
    dst = idx.reshape(-1)

    r = torch.cat([src, dst])
    c = torch.cat([dst, src])

    ones = torch.ones(r.shape[0], device=feat.device)
    deg = torch.zeros(N, device=feat.device).scatter_add_(0, r, ones)
    deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
    edge_weight = deg_inv_sqrt[r] * deg_inv_sqrt[c]
    return torch.stack([r, c], dim=0), edge_weight

def cheb_propagate(x, edge_index, edge_weight, coeffs, num_nodes, s=1.0):
    if x.is_complex():
        return torch.complex(
            cheb_propagate(x.real, edge_index, edge_weight, coeffs, num_nodes, s),
            cheb_propagate(x.imag, edge_index, edge_weight, coeffs, num_nodes, s),
        )
    K = len(coeffs)
    row = edge_index[0]
    col = edge_index[1]

    def A_norm(h):
        msg = (edge_weight / s).unsqueeze(-1) * h[col]
        out = torch.zeros_like(h)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)
        return out

    Tx0 = x
    out = float(coeffs[0]) * Tx0
    if K < 2:
        return out
    Tx1 = -A_norm(Tx0)
    out = out + float(coeffs[1]) * Tx1
    for ki in range(2, K):
        Tx2 = -2.0 * A_norm(Tx1) - Tx0
        out = out + float(coeffs[ki]) * Tx2
        Tx0, Tx1 = Tx1, Tx2
    return out

class CLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        scale = (2.0 / in_features) ** 0.5
        self.W_r = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.W_i = nn.Parameter(torch.randn(out_features, in_features) * scale)

    def forward(self, x):
        xr, xi = x.real, x.imag
        return torch.complex(
            xr @ self.W_r.t() - xi @ self.W_i.t(),
            xr @ self.W_i.t() + xi @ self.W_r.t(),
        )

class FrameletLayer(nn.Module):
    def __init__(self, channels: int, approx: np.ndarray, s: float = 1.0):
        super().__init__()
        F = approx.shape[0]
        self.register_buffer('approx', torch.from_numpy(approx.copy()))
        self.s = s
        self.theta = nn.Parameter(torch.ones(F))
        self.W = CLinear(channels, channels)
        self.act = CSiLU()

    def forward(self, x, edge_index, edge_weight, num_nodes):
        out = torch.zeros_like(x)
        for i in range(self.approx.shape[0]):
            filtered = cheb_propagate(
                x, edge_index, edge_weight,
                self.approx[i], num_nodes, s=self.s)
            out = out + self.theta[i] * filtered
        return self.act(self.W(out))

class HSG(nn.Module):
    def __init__(
        self,
        seq_len: int = 12,
        pred_len: int = 12,
        signal_len: int = 16,
        num_nodes: int = 140,
        embed_size: int = 128,
        hidden_size: int = 64,
        k: int = 2,
        s: float = 2.0,
        frame_type: str = "Haar",
        cheb_order: int = 2,
        lev: int = 1,
        ma_kernel: int = 3,
        dropout: float = 0.0,
        learnable_filters: bool = False,
        phase_knn: bool = False,
        adaptive_k: bool = False,
        k_min: int = 1,
        multi_scale: bool = False,
        low_freq_ratio: float = 0.25,
        graph_depth: int = 1,
        dynamic_graph: bool = False,
        phase_alpha_init: float = 0.5,
        trend_graph: bool = False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.k = k

        if ma_kernel % 2 == 0:
            ma_kernel += 1
        self.ma_kernel = min(ma_kernel, max(3, seq_len - 2))

        C = signal_len // 2 + 1
        freq_idx = np.arange(C, dtype=np.float32) * float(s)
        freq_emb = sinusoidal_posemb(freq_idx, embed_size)
        self.register_buffer('freq_emb', torch.from_numpy(freq_emb))

        filters = get_frame(frame_type)
        approx = np.array([cheb_approx(f, cheb_order) for f in filters], dtype=np.float32)
        self.framelet = FrameletLayer(embed_size, approx, s=s)
        self.embed_out = CLinear(embed_size, 1)

        self.norm1 = nn.InstanceNorm1d(num_nodes, affine=True)
        self.W1 = nn.Linear(seq_len, hidden_size)
        self.norm2 = nn.InstanceNorm1d(num_nodes, affine=True)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.W_trend = nn.Linear(seq_len, hidden_size)
        self.W3 = nn.Linear(hidden_size, pred_len)

    def _moving_avg(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        k = self.ma_kernel
        pad = k // 2
        front = x[:, :1, :].expand(B, pad, N)
        back = x[:, -1:, :].expand(B, k - 1 - pad, N)
        x_pad = torch.cat([front, x, back], dim=1)
        return x_pad.unfold(1, k, 1).mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape

        trend = self._moving_avg(x)
        seasonal = x - trend

        S = torch.fft.rfft(seasonal, n=self.signal_len, dim=1, norm='ortho')
        C = S.shape[1]
        S_perm = S.permute(0, 2, 1)

        node_feat = S_perm.unsqueeze(-1) * self.freq_emb

        all_x, all_ei, all_ew = [], [], []
        offset = 0
        for b in range(B):
            nf = node_feat[b].reshape(N * C, -1)
            sg = S_perm[b].reshape(N * C, 1)
            feat_knn = torch.cat([sg.real, sg.imag], dim=-1)
            ei, ew = build_knn_graph(feat_knn, self.k)
            all_x.append(nf)
            all_ei.append(ei + offset)
            all_ew.append(ew)
            offset += N * C

        batch_x = torch.cat(all_x, dim=0)
        batch_ei = torch.cat(all_ei, dim=1)
        batch_ew = torch.cat(all_ew, dim=0)

        batch_out = self.framelet(batch_x, batch_ei, batch_ew, B * N * C)
        S_out = self.embed_out(batch_out).reshape(B, N, C)

        H = torch.fft.irfft(S_out.permute(0, 2, 1), n=self.seq_len, dim=1, norm='ortho')
        H = H.permute(0, 2, 1)

        H = self.act(self.norm1(H))
        H = self.W1(H)
        H = self.act(self.norm2(H))
        H = self.W2(H)

        trend_emb = self.W_trend(trend.permute(0, 2, 1))

        return self.W3(H + trend_emb)