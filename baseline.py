"""
baseline.py
===========
Baseline attention models for benchmarking against TreeAttention.

Two baselines are provided so we can measure two distinct axes of improvement:

1. StandardAttention
   - Classic O(N²) softmax self-attention with NO tree structure.
   - Answers: "does incorporating tree distance help at all?"

2. NaiveTreeAttention
   - Exact O(N²) tree attention:
       A_{i,j} = exp(q_i^T k_j / sqrt(d_qk)) * f(d(i,j))
       O_i     = Σ_j A_{i,j} * v_j     (un-normalised kernel attention)
   - This is the *exact reference* for what TreeAttention/FFTTreeAttention
     approximate sub-quadratically.
   - Answers: "how accurate is the linearisation?"
     and "what quality is achievable with O(N²) tree attention?"

Both models expose the same forward(x, tree) API as TreeAttentionModel so
they can be plugged into the same training loop without modification.
"""

import math
from collections import defaultdict, deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _tree_attention import Tree


# ============================================================================
# Utility: all-pairs tree distance
# ============================================================================

def compute_tree_distances(tree: Tree) -> torch.Tensor:
    """
    Compute all-pairs shortest-path distances on the tree using BFS.
    Edges are treated as *undirected* with the stored edge weight.
    Time complexity: O(N²).

    Returns
    -------
    dist : FloatTensor [N, N]
        dist[i, j] = weighted tree distance from node i to node j.
        Self-distances are 0.  Unreachable pairs have value inf
        (should never occur in a connected tree).
    """
    N = tree.num_nodes

    # Build undirected adjacency list from the directed edge_weights dict
    adj: defaultdict = defaultdict(list)
    for (u, v), w in tree.edge_weights.items():
        adj[u].append((v, float(w)))
        adj[v].append((u, float(w)))

    dist = torch.full((N, N), float("inf"), dtype=torch.float32)

    for src in range(N):
        dist[src, src] = 0.0
        queue: deque = deque([(src, 0.0)])
        visited = {src}
        while queue:
            u, d_u = queue.popleft()
            for v, w in adj[u]:
                if v not in visited:
                    visited.add(v)
                    dist[src, v] = d_u + w
                    queue.append((v, d_u + w))

    return dist


# ============================================================================
# Baseline 1 – Standard O(N²) softmax attention  (no tree structure)
# ============================================================================

class StandardAttention(nn.Module):
    """
    Vanilla multi-query self-attention:

        A      = softmax(Q K^T / sqrt(d_qk))   [N, N]
        output = A V                             [N, d_v]

    The tree argument is accepted but completely ignored.
    This model represents the standard Transformer baseline.
    """

    def __init__(self, d_model: int, d_qk: int, d_v: int):
        super().__init__()
        self.d_model = d_model
        self.d_qk    = d_qk
        self.d_v     = d_v
        self.scale   = math.sqrt(d_qk)

        self.Wq = nn.Linear(d_model, d_qk)
        self.Wk = nn.Linear(d_model, d_qk)
        self.Wv = nn.Linear(d_model, d_v)
        self.Wo = nn.Linear(d_v,     d_model)

    def forward(
        self,
        x: torch.Tensor,
        tree: Optional[Tree] = None,   # accepted for API compatibility; ignored
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : [N, d_model]
        tree : ignored

        Returns
        -------
        output : [N, d_model]
        """
        q = self.Wq(x)                                    # [N, d_qk]
        k = self.Wk(x)                                    # [N, d_qk]
        v = self.Wv(x)                                    # [N, d_v]

        scores = torch.matmul(q, k.T) / self.scale        # [N, N]
        attn   = F.softmax(scores, dim=-1)                # [N, N]
        out    = torch.matmul(attn, v)                    # [N, d_v]
        output = self.Wo(out)                             # [N, d_model]

        if return_attention:
            return output, attn
        return output


# ============================================================================
# Baseline 2 – Naive O(N²) tree attention  (exact reference implementation)
# ============================================================================

class NaiveTreeAttention(nn.Module):
    """
    Exact O(N²) tree attention with learnable polynomial distance weighting.

    This is the *precise* computation that TreeAttention/FFTTreeAttention
    approximate in sub-quadratic time.

    Math
    ----
        f(d)     = Σ_{m=0}^{D} c_m * d^m          (learnable polynomial)
        A_{i,j}  = exp(q_i^T k_j / √d_qk) * f(d̃(i,j))
        O_i      = Σ_j A_{i,j} * v_j               (un-normalised, row-normalised for stability)

    where d̃(i,j) = d(i,j) / max_dist is the distance normalised to [0,1]
    for numerical stability with higher polynomial degrees.

    The learnable poly_coeff is initialised identically to TreeAttention so
    that a direct quality comparison is meaningful.
    """

    def __init__(
        self,
        d_model:     int,
        d_qk:        int,
        d_v:         int,
        poly_degree: int = 2,
    ):
        super().__init__()
        self.d_model     = d_model
        self.d_qk        = d_qk
        self.d_v         = d_v
        self.poly_degree = poly_degree
        self.scale       = math.sqrt(d_qk)

        self.Wq = nn.Linear(d_model, d_qk)
        self.Wk = nn.Linear(d_model, d_qk)
        self.Wv = nn.Linear(d_model, d_v)
        self.Wo = nn.Linear(d_v,     d_model)

        # Same initialisation as TreeAttention.poly_coeff for a fair comparison
        self.poly_coeff = nn.Parameter(torch.randn(poly_degree + 1) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        tree: Tree,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : [N, d_model]
        tree : Tree structure used to compute pairwise distances

        Returns
        -------
        output : [N, d_model]
        """
        N      = x.size(0)
        device = x.device
        dtype  = x.dtype

        q = self.Wq(x)   # [N, d_qk]
        k = self.Wk(x)   # [N, d_qk]
        v = self.Wv(x)   # [N, d_v]

        # ── Attention scores (numerically stable) ────────────────────────────
        scores = torch.matmul(q, k.T) / self.scale                # [N, N]
        scores = scores - scores.max(dim=-1, keepdim=True).values  # stability
        exp_scores = torch.exp(scores)                             # [N, N]

        # ── Pairwise tree distances ──────────────────────────────────────────
        dist = compute_tree_distances(tree).to(device=device, dtype=dtype)  # [N, N]

        # Normalise distances to [0, 1] for numerical stability with d^m
        finite_mask = dist < float("inf")
        max_dist = dist[finite_mask].max().clamp(min=1.0)
        dist_norm = (dist / max_dist).clamp(max=1.0)   # [N, N]

        # ── Polynomial weighting  f(d̃) = Σ_m c_m * d̃^m ─────────────────────
        f_dist = torch.zeros(N, N, device=device, dtype=dtype)
        d_pow  = torch.ones(N, N, device=device, dtype=dtype)   # d̃^0
        for m in range(self.poly_degree + 1):
            f_dist = f_dist + self.poly_coeff[m] * d_pow
            if m < self.poly_degree:
                d_pow = d_pow * dist_norm

        # ── Weighted & row-normalised attention ──────────────────────────────
        A     = exp_scores * f_dist                              # [N, N]
        denom = A.sum(dim=-1, keepdim=True).abs() + 1e-6
        A_norm = A / denom                                       # [N, N]

        out    = torch.matmul(A_norm, v)   # [N, d_v]
        output = self.Wo(out)              # [N, d_model]

        if return_attention:
            return output, A
        return output


# ============================================================================
# Full model wrappers  (encoder → attention → mean-pool → classifier)
# ============================================================================

class StandardAttentionModel(nn.Module):
    """Complete model wrapping StandardAttention."""

    def __init__(
        self,
        d_input:     int,
        d_model:     int,
        d_qk:        int,
        d_v:         int,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.attention = StandardAttention(d_model, d_qk, d_v)
        self.task_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, tree: Optional[Tree] = None) -> torch.Tensor:
        """
        x : [N, d_input]
        Returns logits : [1, num_classes]
        """
        h      = self.encoder(x)          # [N, d_model]
        o      = self.attention(h, tree)  # [N, d_model]
        agg    = o.mean(dim=0)            # [d_model]
        logits = self.task_head(agg.unsqueeze(0))
        return logits


class NaiveTreeAttentionModel(nn.Module):
    """Complete model wrapping NaiveTreeAttention."""

    def __init__(
        self,
        d_input:     int,
        d_model:     int,
        d_qk:        int,
        d_v:         int,
        num_classes: int,
        poly_degree: int = 2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.attention = NaiveTreeAttention(d_model, d_qk, d_v, poly_degree)
        self.task_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, tree: Tree) -> torch.Tensor:
        """
        x : [N, d_input]
        Returns logits : [1, num_classes]
        """
        h      = self.encoder(x)          # [N, d_model]
        o      = self.attention(h, tree)  # [N, d_model]
        agg    = o.mean(dim=0)            # [d_model]
        logits = self.task_head(agg.unsqueeze(0))
        return logits
