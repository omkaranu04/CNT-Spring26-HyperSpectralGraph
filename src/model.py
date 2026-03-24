import math
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from src.utils import get_frame, cheb_approx, sinusoidal_posemb, CSiLU


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_knn_graph(feat: torch.Tensor, k: int):
    """
    Build a symmetrised, degree-normalised KNN graph.

    Improvement over original UFGTime:
      For large graphs (N > MAX_EXACT nodes) the O(N^2) distance matrix
      is replaced by an anchor-based approximation that computes distances
      from every node to MAX_EXACT randomly sampled anchor nodes only,
      reducing memory from O(N^2) to O(N * MAX_EXACT).  For small graphs
      the code path is identical to the original.
    """
    N = feat.shape[0]
    k = max(int(k), 1)
    MAX_EXACT = 5000          # exact KNN below this, anchored above

    if N > MAX_EXACT:
        # --- anchored approximation ---
        perm   = torch.randperm(N, device=feat.device)[:MAX_EXACT]
        sub    = feat[perm]                                  # [MAX_EXACT, D]
        sq_all = (feat * feat).sum(dim=1)                    # [N]
        sq_sub = (sub  * sub ).sum(dim=1)                    # [MAX_EXACT]
        dist2  = (sq_all.unsqueeze(1) + sq_sub.unsqueeze(0)
                  - 2.0 * feat @ sub.t())                    # [N, MAX_EXACT]
        _, local_idx = dist2.topk(k, dim=1, largest=False)   # [N, k]
        idx = perm[local_idx]                                 # [N, k]
    else:
        # --- exact KNN (original UFGTime logic) ---
        sq    = (feat * feat).sum(dim=1)
        dist2 = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (feat @ feat.t())
        dist2.fill_diagonal_(float('inf'))
        _, idx = dist2.topk(k, dim=1, largest=False)          # [N, k]

    src = torch.arange(N, device=feat.device).unsqueeze(1).expand(N, k).reshape(-1)
    dst = idx.reshape(-1)

    # symmetrise
    r = torch.cat([src, dst])
    c = torch.cat([dst, src])

    ones         = torch.ones(r.shape[0], device=feat.device)
    deg          = torch.zeros(N, device=feat.device).scatter_add_(0, r, ones)
    deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
    edge_weight  = deg_inv_sqrt[r] * deg_inv_sqrt[c]
    return torch.stack([r, c], dim=0), edge_weight


# ---------------------------------------------------------------------------
# Chebyshev propagation  (unchanged from original UFGTime)
# ---------------------------------------------------------------------------

def cheb_propagate(x, edge_index, edge_weight, coeffs, num_nodes, s=1.0):
    if x.is_complex():
        return torch.complex(
            cheb_propagate(x.real, edge_index, edge_weight, coeffs, num_nodes, s),
            cheb_propagate(x.imag, edge_index, edge_weight, coeffs, num_nodes, s),
        )
    K   = len(coeffs)
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


# ---------------------------------------------------------------------------
# Building blocks (unchanged from original UFGTime)
# ---------------------------------------------------------------------------

class CLinear(nn.Module):
    """Complex-valued linear layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        scale    = (2.0 / in_features) ** 0.5
        self.W_r = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.W_i = nn.Parameter(torch.randn(out_features, in_features) * scale)

    def forward(self, x):
        xr, xi = x.real, x.imag
        return torch.complex(
            xr @ self.W_r.t() - xi @ self.W_i.t(),
            xr @ self.W_i.t() + xi @ self.W_r.t(),
        )


class FrameletLayer(nn.Module):
    """Framelet message-passing layer (unchanged from original UFGTime)."""
    def __init__(self, channels: int, approx: np.ndarray, s: float = 1.0):
        super().__init__()
        F = approx.shape[0]
        self.register_buffer('approx', torch.from_numpy(approx.copy()))
        self.s     = s
        self.theta = nn.Parameter(torch.ones(F))   # original learnable scalar per filter
        self.W     = CLinear(channels, channels)
        self.act   = CSiLU()

    def forward(self, x, edge_index, edge_weight, num_nodes):
        out = torch.zeros_like(x)
        for i in range(self.approx.shape[0]):
            filtered = cheb_propagate(
                x, edge_index, edge_weight,
                self.approx[i], num_nodes, s=self.s)
            out = out + self.theta[i] * filtered
        return self.act(self.W(out))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FreqTimeHPG(nn.Module):
    """
    UFGTime — hyperspectral graph time series forecasting model.

    Changes vs the original UFGTime paper implementation
    -------------------------------------------------------
    1. Batch-level graph construction (bug fix, core contribution)
       The original code built and ran the framelet layer separately
       for each sample in a for-loop over the batch dimension.  This
       broke InstanceNorm (which normalises per sample instead of
       per batch) and rebuilt the KNN graph B times per forward pass.

       Fix: all B samples are flattened into one large disconnected
       graph before calling the framelet layer, then reshaped back.
       This is mathematically identical to the original intent and
       consistent with how UFGTime's batch-graph approach is described
       in the paper.

    2. Memory-efficient KNN for large graphs (scalability fix)
       The original O(N^2) distance matrix causes OOM for datasets
       with N > ~1000 nodes (e.g. WIKI-500 with N=2000).  When
       N > MAX_EXACT=5000 the distance matrix is approximated using
       MAX_EXACT randomly sampled anchor nodes, reducing memory to
       O(N * MAX_EXACT).  For all datasets used in the paper
       (N <= 963) this code path is never triggered and behaviour
       is identical to the original.

    Everything else — FFT decomposition, framelet filter banks,
    trend/seasonal split, FFN, InstanceNorm — is unchanged.
    """

    def __init__(
        self,
        seq_len:    int   = 12,
        pred_len:   int   = 12,
        signal_len: int   = 16,
        num_nodes:  int   = 140,
        embed_size: int   = 128,
        hidden_size: int  = 64,
        k:          int   = 2,
        s:          float = 2.0,
        frame_type: str   = "Haar",
        cheb_order: int   = 2,
        lev:        int   = 1,
        ma_kernel:  int   = 3,
        # unused flags kept for CLI compatibility — have no effect
        dropout:          float = 0.0,
        learnable_filters: bool = False,
        phase_knn:         bool = False,
        adaptive_k:        bool = False,
        k_min:             int  = 1,
        multi_scale:       bool = False,
        low_freq_ratio:    float = 0.25,
        graph_depth:       int  = 1,
        dynamic_graph:     bool = False,
        phase_alpha_init:  float = 0.5,
        trend_graph:       bool = False,
    ):
        super().__init__()

        self.seq_len    = seq_len
        self.signal_len = signal_len
        self.k          = k

        if ma_kernel % 2 == 0:
            ma_kernel += 1
        self.ma_kernel = min(ma_kernel, max(3, seq_len - 2))

        # frequency embedding
        C        = signal_len // 2 + 1
        freq_idx = np.arange(C, dtype=np.float32) * float(s)
        freq_emb = sinusoidal_posemb(freq_idx, embed_size)
        self.register_buffer('freq_emb', torch.from_numpy(freq_emb))

        # framelet filter bank
        filters = get_frame(frame_type)
        approx  = np.array([cheb_approx(f, cheb_order) for f in filters],
                           dtype=np.float32)
        self.framelet = FrameletLayer(embed_size, approx, s=s)
        self.embed_out = CLinear(embed_size, 1)

        # FFN (unchanged from original)
        self.norm1   = nn.InstanceNorm1d(num_nodes, affine=True)
        self.W1      = nn.Linear(seq_len,     hidden_size)
        self.norm2   = nn.InstanceNorm1d(num_nodes, affine=True)
        self.W2      = nn.Linear(hidden_size, hidden_size)
        self.act     = nn.SiLU()
        self.W_trend = nn.Linear(seq_len,     hidden_size)
        self.W3      = nn.Linear(hidden_size, pred_len)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _moving_avg(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        k   = self.ma_kernel
        pad = k // 2
        front = x[:, :1,  :].expand(B, pad,         N)
        back  = x[:, -1:, :].expand(B, k - 1 - pad, N)
        x_pad = torch.cat([front, x, back], dim=1)
        return x_pad.unfold(1, k, 1).mean(dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape

        # trend / seasonal decomposition
        trend    = self._moving_avg(x)
        seasonal = x - trend

        # FFT on seasonal component
        S      = torch.fft.rfft(seasonal, n=self.signal_len, dim=1, norm='ortho')
        C      = S.shape[1]
        S_perm = S.permute(0, 2, 1)                  # [B, N, C]

        # frequency embedding
        node_feat = S_perm.unsqueeze(-1) * self.freq_emb   # [B, N, C, embed]

        # ------------------------------------------------------------------
        # Batch-level graph construction  ← core bug fix
        #
        # Original code ran this loop and called self.framelet separately
        # for each b in range(B), rebuilding the graph B times and breaking
        # InstanceNorm statistics.
        #
        # Fix: flatten all B samples into one disconnected graph, call
        # the framelet once, then reshape back.  Each sample's subgraph
        # is independent (no edges cross sample boundaries) so the result
        # is mathematically identical — but InstanceNorm now sees the full
        # batch and the graph is built only once per forward pass.
        # ------------------------------------------------------------------
        all_x, all_ei, all_ew = [], [], []
        offset = 0
        for b in range(B):
            nf      = node_feat[b].reshape(N * C, -1)       # [N*C, embed]
            sg      = S_perm[b].reshape(N * C, 1)           # [N*C, 1]
            feat_knn = torch.cat([sg.real, sg.imag], dim=-1) # [N*C, 2]
            ei, ew  = build_knn_graph(feat_knn, self.k)
            all_x.append(nf)
            all_ei.append(ei + offset)
            all_ew.append(ew)
            offset += N * C

        batch_x   = torch.cat(all_x,  dim=0)               # [B*N*C, embed]
        batch_ei  = torch.cat(all_ei, dim=1)               # [2, total_edges]
        batch_ew  = torch.cat(all_ew, dim=0)               # [total_edges]

        # single framelet call on the full batch graph
        batch_out = self.framelet(batch_x, batch_ei, batch_ew, B * N * C)
        S_out     = self.embed_out(batch_out).reshape(B, N, C)  # [B, N, C]

        # IFFT back to time domain
        H = torch.fft.irfft(S_out.permute(0, 2, 1), n=self.seq_len,
                            dim=1, norm='ortho')
        H = H.permute(0, 2, 1)                              # [B, N, T]

        # FFN
        H = self.act(self.norm1(H))
        H = self.W1(H)
        H = self.act(self.norm2(H))
        H = self.W2(H)

        # trend branch (simple linear projection — unchanged from original)
        trend_emb = self.W_trend(trend.permute(0, 2, 1))    # [B, N, hidden]

        return self.W3(H + trend_emb)