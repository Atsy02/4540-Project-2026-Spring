import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ============================================================================
# 1. FEATURE MAP (Linearization of Attention)
# ============================================================================

class RandomFeatureMap(nn.Module):
    """Approximate exp(q^T k / sqrt(d)) using random features (Performer-style)"""

    def __init__(self, d_in: int, d_out: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.d_out = d_out
        # Random matrix for projection
        self.register_buffer('projection', torch.randn(d_in, d_out) / np.sqrt(d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_in] or [N, d_in]
        Returns:
            phi(x): [batch_size, seq_len, d_out] or [N, d_out]
        """
        projected = x @ self.projection  # [N, d_out]
        # Normalize for stability
        return torch.exp(projected) / (torch.norm(projected, dim=-1, keepdim=True) + 1e-6)


class TaylorFeatureMap(nn.Module):
    """Approximate exp using Taylor expansion: sum_{k=0}^{K} x^k / k!"""

    def __init__(self, d_in: int, d_out: int, num_terms: int = 4):
        super().__init__()
        self.num_terms = num_terms
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Taylor approximation of exp"""
        result = torch.ones_like(x[..., :1])  # First term (0! = 1)
        term = self.linear(x)

        for k in range(1, self.num_terms):
            result = result + term / np.math.factorial(k)
            term = term * self.linear(x) / (k + 1)

        return torch.relu(result) + 1e-6  # Ensure positivity


# ============================================================================
# 2. TREE DATA STRUCTURES & UTILITIES
# ============================================================================

class Tree:
    """Representation of a rooted tree with weighted edges"""

    def __init__(self, num_nodes: int, root: int = 0):
        self.num_nodes = num_nodes
        self.root = root
        self.adj = defaultdict(list)  # adjacency: node -> [(child, weight), ...]
        self.parent = [-1] * num_nodes
        self.children = [[] for _ in range(num_nodes)]
        self.edge_weights = {}  # (u, v) -> weight

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        """Add directed edge u -> v (parent -> child)"""
        self.adj[u].append((v, weight))
        self.children[u].append(v)
        self.parent[v] = u
        self.edge_weights[(u, v)] = weight

    def get_subtree_nodes(self, node: int) -> List[int]:
        """BFS to get all nodes in subtree rooted at node"""
        nodes = [node]
        queue = [node]
        while queue:
            u = queue.pop(0)
            for v, _ in self.adj[u]:
                nodes.append(v)
                queue.append(v)
        return nodes

    def bottom_up_order(self) -> List[int]:
        """Topological sort: leaves before parents"""
        visited = [False] * self.num_nodes
        order = []

        def dfs(u):
            visited[u] = True
            for v, _ in self.adj[u]:
                if not visited[v]:
                    dfs(v)
            order.append(u)

        dfs(self.root)
        return order

    def top_down_order(self) -> List[int]:
        """Level-order traversal: root first"""
        order = [self.root]
        queue = [self.root]
        idx = 0
        while idx < len(queue):
            u = queue[idx]
            idx += 1
            for v, _ in self.adj[u]:
                order.append(v)
                queue.append(v)
        return order


# ============================================================================
# 3. TREE ATTENTION MECHANISM
# ============================================================================

class TreeAttention(nn.Module):
    """
    Tree-based attention with polynomial distance weighting.

    Computes: O_i = phi(q_i)^T sum_{m=0}^d c_m S_i^(m)
    where S_i^(m) = sum_{j=1}^N d(i,j)^m M_j
    """

    def __init__(
            self,
            d_model: int,
            d_qk: int,
            d_v: int,
            poly_degree: int = 2,
            feature_map_type: str = "random",
            feature_dim: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        self.poly_degree = poly_degree
        self.feature_dim = feature_dim

        # Projections
        self.Wq = nn.Linear(d_model, d_qk)
        self.Wk = nn.Linear(d_model, d_qk)
        self.Wv = nn.Linear(d_model, d_v)
        # o_i has shape [d_v], so Wo should map d_v -> d_model.
        self.Wo = nn.Linear(d_v, d_model)

        # Feature map for linearization
        if feature_map_type == "random":
            self.phi = RandomFeatureMap(d_qk, feature_dim)
        else:
            self.phi = TaylorFeatureMap(d_qk, feature_dim)

        # Polynomial coefficients (learnable or fixed)
        self.register_parameter(
            "poly_coeff",
            nn.Parameter(torch.randn(poly_degree + 1) * 0.1)
        )

    def forward(
            self,
            x: torch.Tensor,
            tree: Tree,
            return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [N, d_model] - node features
            tree: Tree object with structure
            return_attention: if True, also return S_i^(m) tensors for analysis

        Returns:
            output: [N, d_model]
        """
        N = x.size(0)
        device = x.device

        # Step 1: Project to Q, K, V
        q = self.Wq(x)  # [N, d_qk]
        k = self.Wk(x)  # [N, d_qk]
        v = self.Wv(x)  # [N, d_v]

        # Step 2: Linearize via feature map
        phi_q = self.phi(q)  # [N, feature_dim]
        phi_k = self.phi(k)  # [N, feature_dim]

        # M_j = phi(k_j) ⊗ v_j^T -> [N, feature_dim, d_v]
        M = torch.einsum('nd,nv->ndv', phi_k, v)

        # Step 3: Two-pass tree DP
        S = self._tree_dp(M, tree, device)  # Dict: m -> [N, feature_dim, d_v]

        # Step 4: Compute attention output
        output = torch.zeros(N, self.d_model, device=device)

        for i in range(N):
            Y_i = torch.zeros(self.feature_dim, self.d_v, device=device)
            for m in range(self.poly_degree + 1):
                Y_i += self.poly_coeff[m] * S[m][i]  # [feature_dim, d_v]

            # O_i = phi(q_i)^T @ Y_i -> [d_v]
            o_i = phi_q[i] @ Y_i  # [feature_dim] @ [feature_dim, d_v] -> [d_v]

            # Project back to d_model
            output[i] = self.Wo(o_i.view(-1))

        if return_attention:
            return output, S
        return output

    def _tree_dp(
            self,
            M: torch.Tensor,
            tree: Tree,
            device: torch.device
    ) -> Dict[int, torch.Tensor]:
        """
        Execute two-pass DP on tree to compute S_i^(m) for all nodes and degrees.

        Returns:
            S: Dict where S[m][i] is the aggregated sum for degree m at node i
        """
        N = M.size(0)
        d = self.poly_degree
        feat_dim, d_v = M.size(1), M.size(2)

        # Initialize in-subtree sums
        inS = {}
        for m in range(d + 1):
            inS[m] = torch.zeros(N, feat_dim, d_v, device=device)

        # Bottom-up pass
        order_up = tree.bottom_up_order()
        for i in order_up:
            # Base case: M_i contributes to inS[0]
            inS[0][i] = M[i]

            # Add children contributions using binomial expansion
            for c in tree.children[i]:
                w_ic = tree.edge_weights.get((i, c), 1.0)

                for m in range(d + 1):
                    for k in range(m + 1):
                        coeff = torch.tensor(
                            self._binomial(m, k) * (w_ic ** (m - k)),
                            dtype=torch.float32,
                            device=device
                        )
                        inS[m][i] = inS[m][i] + coeff * inS[k][c]

        # Top-down pass
        outS = {}
        S = {}
        for m in range(d + 1):
            outS[m] = torch.zeros(N, feat_dim, d_v, device=device)
            S[m] = torch.zeros(N, feat_dim, d_v, device=device)

        # Root initialization
        for m in range(d + 1):
            S[m][tree.root] = inS[m][tree.root]

        order_down = tree.top_down_order()
        for i in order_down:
            for c in tree.children[i]:
                w_ic = tree.edge_weights.get((i, c), 1.0)

                # Compute P_c^(k): global sum at parent minus c-subtree
                P = {}
                for kdeg in range(d + 1):
                    P[kdeg] = S[kdeg][i].clone()
                    for ldeg in range(kdeg + 1):
                        coeff = torch.tensor(
                            self._binomial(kdeg, ldeg) * (w_ic ** (kdeg - ldeg)),
                            dtype=torch.float32,
                            device=device
                        )
                        P[kdeg] = P[kdeg] - coeff * inS[ldeg][c]

                # Compute outS and S for child
                for m in range(d + 1):
                    for kdeg in range(m + 1):
                        coeff = torch.tensor(
                            self._binomial(m, kdeg) * (w_ic ** (m - kdeg)),
                            dtype=torch.float32,
                            device=device
                        )
                        outS[m][c] = outS[m][c] + coeff * P[kdeg]
                    S[m][c] = inS[m][c] + outS[m][c]

        return S

    @staticmethod
    def _binomial(n: int, k: int) -> float:
        """Compute binomial coefficient C(n, k)"""
        if k > n or k < 0:
            return 0.0
        if k == 0 or k == n:
            return 1.0
        from math import comb
        return float(comb(n, k))


# ============================================================================
# 4. TREE-STRUCTURED DATASET
# ============================================================================

class TreeGraphDataset(Dataset):
    """
    Base class for tree-structured graph datasets.
    Each sample is (features, tree, label).
    """

    def __init__(self, num_samples: int, num_nodes: int, d_feat: int, num_classes: int):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.d_feat = d_feat
        self.num_classes = num_classes
        self.samples = []
        self._generate_samples()

    def _generate_samples(self):
        """Override in subclass"""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tree, int]:
        return self.samples[idx]


class SyntheticTreeDataset(TreeGraphDataset):
    """Generate synthetic tree-structured data for testing"""

    def _generate_samples(self):
        for _ in range(self.num_samples):
            # Random tree structure
            tree = Tree(self.num_nodes, root=0)
            for i in range(1, self.num_nodes):
                parent = np.random.randint(0, i)
                weight = np.random.uniform(0.5, 2.0)
                tree.add_edge(parent, i, weight=weight)

            # Random node features
            features = torch.randn(self.num_nodes, self.d_feat)

            # Random label
            label = np.random.randint(0, self.num_classes)

            self.samples.append((features, tree, label))


# ============================================================================
# 5. FULL MODEL
# ============================================================================

class TreeAttentionModel(nn.Module):
    """Complete model: encoder -> tree attention -> classifier"""

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
        super().__init__()
        # Simple encoder (identity or small MLP)
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.tree_attention = TreeAttention(
            d_model=d_model,
            d_qk=d_qk,
            d_v=d_v,
            poly_degree=poly_degree,
            feature_map_type=feature_map_type,
            feature_dim=feature_dim,
        )

        # Task head: aggregate node representations and classify
        self.task_head = nn.Sequential(
            nn.Linear(d_model, 256),  # mean pooled node representation
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, tree: Tree) -> torch.Tensor:
        """
        Args:
            x: [N, d_input]
            tree: Tree object

        Returns:
            logits: [num_classes]
        """
        # Encode
        h = self.encoder(x)  # [N, d_model]

        # Tree attention
        o = self.tree_attention(h, tree)  # [N, d_model]

        # Aggregate (mean pooling) and classify
        agg = o.mean(dim=0)  # [d_model]
        logits = self.task_head(agg.unsqueeze(0))

        return logits


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================
def compute_f1(model, dataset, device):
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for features, tree, label in dataset:
            features = features.to(device)
            logits = model(features, tree)
            pred = logits.argmax(dim=1).item()
            preds.append(pred)
            targets.append(label)
    # Macro F1 for multi-class
    return f1_score(targets, preds, average='macro')

def train(
    model,
    train_dataset,
    val_dataset=None,
    num_epochs=20,
    batch_size=1,
    learning_rate=1e-3,
    device="cpu"
):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = 0.0
            for features, tree, label in batch:
                features = features.to(device)
                label = torch.as_tensor(label, device=device).long()

                logits = model(features, tree)
                loss = criterion(logits, label.unsqueeze(0))

                batch_loss += loss

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item()

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # Validation phase
        if val_dataset is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for features, tree, label in val_dataset:
                    features = features.to(device)
                    label = torch.tensor(label, device=device).long()

                    logits = model(features, tree)
                    loss = criterion(logits, label.unsqueeze(0))
                    val_loss += loss.item()

                    if logits.argmax(dim=1) == label:
                        correct += 1
                    total += 1

            val_loss /= len(val_dataset)
            val_acc = correct / total
            # Compute F1 score
            val_f1 = compute_f1(model, val_dataset, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            print(
                f"Epoch {epoch + 1:3d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        else:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_loss:.4f}")

    return history

def plot_metrics(metric_histories, poly_degrees, metric_key, ylabel, title):
    plt.figure(figsize=(8, 5))
    for deg, hist in zip(poly_degrees, metric_histories):
        plt.plot(hist[metric_key], label=f"Poly Degree {deg}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test example
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create synthetic dataset
    train_dataset = SyntheticTreeDataset(
        num_samples=100,
        num_nodes=20,
        d_feat=8,
        num_classes=4
    )
    val_dataset = SyntheticTreeDataset(
        num_samples=20,
        num_nodes=20,
        d_feat=8,
        num_classes=4
    )

    # save metrics history for each poly_degree
    metric_histories = []
    poly_degrees = [1, 2, 4, 8]

    for poly_degree in poly_degrees:
        print(f"\n{'=' * 60}")
        print(f"Training with polynomial degree: {poly_degree}")
        print(f"{'=' * 60}")

        model = TreeAttentionModel(
            d_input=8,
            d_model=64,
            d_qk=32,
            d_v=32,
            num_classes=4,
            poly_degree=poly_degree,
            feature_map_type="random",
            feature_dim=128,
        )

        history = train(
            model,
            train_dataset,
            val_dataset,
            num_epochs=200,
            batch_size=8,
            learning_rate=1e-3,
            device=device,
        )
        metric_histories.append(history)

    # visulization of the results
    plot_metrics(metric_histories, poly_degrees, "val_acc", "Validation Accuracy", "Validation Accuracy by Polynomial Degree")
    plot_metrics(metric_histories, poly_degrees, "val_f1", "Validation F1 Score", "Validation F1 Score by Polynomial Degree")
    plot_metrics(metric_histories, poly_degrees, "val_loss", "Validation Loss", "Validation Loss by Polynomial Degree")