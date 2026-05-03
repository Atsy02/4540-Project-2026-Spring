import math
from typing import Dict

import torch
import torch.nn as nn

from _tree_attention import (
    RandomFeatureMap,
    TaylorFeatureMap,
    Tree,
    TreeAttention,
    TreeAttentionModel,
    TreeGraphDataset,
    SyntheticTreeDataset,
    compute_f1,
    plot_metrics,
    train,
)


class FFTTreeAttention(TreeAttention):
    """
    FFT-accelerated version of TreeAttention.
    Replaces binomial O(d^2) polynomial shift with O(d log d) convolution.
    """

    def _tree_dp(
        self,
        M: torch.Tensor,
        tree: Tree,
        device: torch.device,
    ) -> Dict[int, torch.Tensor]:
        N = M.size(0)
        d = self.poly_degree
        feat_dim, d_v = M.size(1), M.size(2)

        factorial = torch.tensor(
            [math.factorial(i) for i in range(d + 1)],
            dtype=M.dtype,
            device=device,
        )
        inv_factorial = 1.0 / factorial

        inS = {
            m: torch.zeros(N, feat_dim, d_v, dtype=M.dtype, device=device)
            for m in range(d + 1)
        }

        order_up = tree.bottom_up_order()
        for i in order_up:
            inS[0][i] = M[i]
            for c in tree.children[i]:
                w_ic = float(tree.edge_weights.get((i, c), 1.0))
                child_stack = torch.stack([inS[k][c] for k in range(d + 1)], dim=0)
                shifted = self._shift_poly_fft(
                    poly_stack=child_stack,
                    edge_weight=w_ic,
                    factorial=factorial,
                    inv_factorial=inv_factorial,
                )
                for m in range(d + 1):
                    inS[m][i] = inS[m][i] + shifted[m]

        outS = {
            m: torch.zeros(N, feat_dim, d_v, dtype=M.dtype, device=device)
            for m in range(d + 1)
        }
        S = {
            m: torch.zeros(N, feat_dim, d_v, dtype=M.dtype, device=device)
            for m in range(d + 1)
        }

        for m in range(d + 1):
            S[m][tree.root] = inS[m][tree.root]

        order_down = tree.top_down_order()
        for i in order_down:
            for c in tree.children[i]:
                w_ic = float(tree.edge_weights.get((i, c), 1.0))

                child_in_stack = torch.stack([inS[k][c] for k in range(d + 1)], dim=0)
                child_shifted_to_parent = self._shift_poly_fft(
                    poly_stack=child_in_stack,
                    edge_weight=w_ic,
                    factorial=factorial,
                    inv_factorial=inv_factorial,
                )

                parent_stack = torch.stack([S[k][i] for k in range(d + 1)], dim=0)
                p_stack = parent_stack - child_shifted_to_parent
                out_stack = self._shift_poly_fft(
                    poly_stack=p_stack,
                    edge_weight=w_ic,
                    factorial=factorial,
                    inv_factorial=inv_factorial,
                )

                for m in range(d + 1):
                    outS[m][c] = out_stack[m]
                    S[m][c] = inS[m][c] + outS[m][c]

        return S

    def _shift_poly_fft(
        self,
        poly_stack: torch.Tensor,
        edge_weight: float,
        factorial: torch.Tensor,
        inv_factorial: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute:
            T_m = sum_{k=0}^m C(m,k) * w^(m-k) * S_k
        via FFT convolution in factorial-normalized domain.

        Args:
            poly_stack: [d+1, feat_dim, d_v], where poly_stack[k] = S_k
        Returns:
            shifted: [d+1, feat_dim, d_v], where shifted[m] = T_m
        """
        d = poly_stack.size(0) - 1
        feat_dim = poly_stack.size(1)
        d_v = poly_stack.size(2)

        L = 1 << ((2 * d + 1).bit_length())

        A = poly_stack * inv_factorial.view(-1, 1, 1)  # S_k / k!
        powers = torch.tensor(
            [edge_weight ** t for t in range(d + 1)],
            dtype=poly_stack.dtype,
            device=poly_stack.device,
        )
        B = powers * inv_factorial  # w^t / t!
        B = B.view(d + 1, 1, 1)

        A_padded = torch.zeros(L, feat_dim, d_v, dtype=poly_stack.dtype, device=poly_stack.device)
        B_padded = torch.zeros(L, 1, 1, dtype=poly_stack.dtype, device=poly_stack.device)
        A_padded[: d + 1] = A
        B_padded[: d + 1] = B

        A_fft = torch.fft.rfft(A_padded, n=L, dim=0)
        B_fft = torch.fft.rfft(B_padded, n=L, dim=0)
        C_fft = A_fft * B_fft
        C = torch.fft.irfft(C_fft, n=L, dim=0)[: d + 1]

        shifted = C * factorial.view(-1, 1, 1)  # T_m = m! * C_m
        return shifted


class FFTTreeAttentionModel(TreeAttentionModel):
    """TreeAttentionModel variant that uses FFTTreeAttention."""

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_qk: int,
        d_v: int,
        num_classes: int,
        poly_degree: int = 2,
        feature_map_type: str = "random",
        feature_dim: int = 128,
    ):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        if feature_map_type not in {"random", "taylor"}:
            raise ValueError("feature_map_type must be 'random' or 'taylor'")

        self.tree_attention = FFTTreeAttention(
            d_model=d_model,
            d_qk=d_qk,
            d_v=d_v,
            poly_degree=poly_degree,
            feature_map_type=feature_map_type,
            feature_dim=feature_dim,
        )

        self.task_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
