'''
This part is for the trial of higher order term effect using real data
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Optional
import ast
import re
import numpy as np
from collections import defaultdict
import math

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
        # Stabilize exp: subtract row-wise max before exp to avoid overflow -> inf/nan logits.
        m = projected.amax(dim=-1, keepdim=True)
        z = projected - m
        phi = torch.exp(z.clamp(max=80.0))
        return phi / (phi.sum(dim=-1, keepdim=True) + 1e-8)


class TaylorFeatureMapFixed(nn.Module):
    def __init__(self, d_in, d_out, num_terms=4):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.num_terms = num_terms

    def forward(self, x):
        z = self.linear(x)

        result = torch.ones_like(z)
        term = torch.ones_like(z)

        for k in range(1, self.num_terms):
            term = term * z / k
            result = result + term

        return result.clamp(min=1e-6)

class FavorFeatureMap(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

    def forward(self, x):
        # x: [N, d]
        proj = x @ self.proj  # [N, d_out]

        # ||x||^2 / 2
        norm = (x ** 2).sum(dim=-1, keepdim=True) / 2

        # exp(w^T x - ||x||^2/2)
        z = proj - norm

        return torch.exp(z.clamp(max=50))

class ReluFeatureMap(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        return F.relu(self.linear(x)) + 1e-6


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
    def __init__(
        self,
        d_model,
        d_qk,
        d_v,
        poly_degree=2,
        feature_map_type="favor",
        feature_dim=128,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        self.poly_degree = poly_degree
        self.feature_dim = feature_dim

        self.Wq = nn.Linear(d_model, d_qk)
        self.Wk = nn.Linear(d_model, d_qk)
        self.Wv = nn.Linear(d_model, d_v)
        self.Wo = nn.Linear(d_v, d_model)

        # new feature map
        if feature_map_type == "favor":
            self.phi = FavorFeatureMap(d_qk, feature_dim)
        elif feature_map_type == "relu":
            self.phi = ReluFeatureMap(d_qk, feature_dim)
        else:
            self.phi = TaylorFeatureMapFixed(d_qk, feature_dim)

        self.poly_coeff = nn.Parameter(torch.ones(poly_degree + 1))

    def forward(self, x, tree):
        N = x.size(0)
        device = x.device

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        phi_q = self.phi(q)          # [N, F]
        phi_k = self.phi(k)          # [N, F]

        # =========================
        # Create two DP input
        # =========================

        # value tensor
        M = torch.einsum('nd,nv->ndv', phi_k, v)   # [N, F, dv]

        # normalizer tensor
        Z = phi_k                                  # [N, F]

        # =========================
        # Tree DP
        # =========================

        S = self._tree_dp(M, tree, device)         # value
        ZS = self._tree_dp(Z.unsqueeze(-1), tree, device)  # normalizer

        # =========================
        # 聚合 polynomial
        # =========================

        # [N, F, dv]
        Y = torch.zeros(N, self.feature_dim, self.d_v, device=device)
        ZY = torch.zeros(N, self.feature_dim, 1, device=device)

        for m in range(self.poly_degree + 1):
            Y += self.poly_coeff[m] * S[m]
            ZY += self.poly_coeff[m] * ZS[m]

        # =========================
        # kernel attention（向量化）
        # =========================

        # numerator: [N, dv]
        num = (phi_q.unsqueeze(-1) * Y).sum(dim=1)

        # denominator: [N, 1]
        den = (phi_q.unsqueeze(-1) * ZY).sum(dim=1)

        out = num / (den + 1e-6)

        return self.Wo(out)

    # =========================
    # Tree DP
    # =========================
    def _tree_dp(self, M, tree, device):
        """
        M: [N, F, dv] or [N, F, 1]
        """
        N = M.size(0)
        d = self.poly_degree
        shape = M.shape[1:]

        inS = {m: torch.zeros(N, *shape, device=device) for m in range(d + 1)}

        # bottom-up
        for i in tree.bottom_up_order():
            inS[0][i] = M[i]

            for c in tree.children[i]:
                w = tree.edge_weights.get((i, c), 1.0)

                for m in range(d + 1):
                    for k in range(m + 1):
                        coeff = math.comb(m, k) * (w ** (m - k))
                        inS[m][i] += coeff * inS[k][c]

        outS = {m: torch.zeros(N, *shape, device=device) for m in range(d + 1)}
        S = {m: torch.zeros(N, *shape, device=device) for m in range(d + 1)}

        for m in range(d + 1):
            S[m][tree.root] = inS[m][tree.root]

        # top-down
        for i in tree.top_down_order():
            for c in tree.children[i]:
                w = tree.edge_weights.get((i, c), 1.0)

                P = {}
                for kdeg in range(d + 1):
                    P[kdeg] = S[kdeg][i].clone()
                    for ldeg in range(kdeg + 1):
                        coeff = math.comb(kdeg, ldeg) * (w ** (kdeg - ldeg))
                        P[kdeg] -= coeff * inS[ldeg][c]

                for m in range(d + 1):
                    for kdeg in range(m + 1):
                        coeff = math.comb(m, kdeg) * (w ** (m - kdeg))
                        outS[m][c] += coeff * P[kdeg]

                    S[m][c] = inS[m][c] + outS[m][c]

        return S

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
    def __init__(
        self,
        vocab_size,
        d_input,
        d_model,
        d_qk,
        d_v,
        num_classes,
        poly_degree=2,
        feature_map_type="favor",
        feature_dim=128,
    ):
        super().__init__()

        # embedding process
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
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

        # attention pooling
        self.pool_q = nn.Parameter(torch.randn(d_model))

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward_hidden(self, x, tree):
        x = x.long().squeeze(-1)

        h = self.embedding(x)
        h = self.encoder(h)

        o = self.tree_attention(h, tree)

        # attention pooling
        scores = (o @ self.pool_q) / math.sqrt(o.size(-1))
        weights = torch.softmax(scores, dim=0)
        agg = (weights.unsqueeze(-1) * o).sum(dim=0)

        return self.cls_head(agg)

    def forward(self, x, tree):
        hid = self.forward_hidden(x, tree)  # [hidden]
        return self.classifier(hid).unsqueeze(0)  # [1, num_classes]


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================


import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# import processed_data
import pickle
import os

def load_pickle(filename):
    path = os.path.join('processed_data', filename)
    with open(path, 'rb') as f:
        return pickle.load(f)

train_dataset = load_pickle('train_data.pkl')
val_dataset = load_pickle('val_data.pkl')
test_dataset = load_pickle('test_data.pkl')
token_to_label = load_pickle('token_to_label.pkl')
input_token_to_id = load_pickle('input_token_to_id.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_tree_from_sample(sample):
    # 优先使用AST，如果无效可以简单构建单链（只有根，常用作跳过异常样本）Use AST first; if it's invalid then use the single link
    code = sample['full_code']
    if sample.get('has_valid_ast', False):
        try:
            parsed = ast.parse(code)
            return _tree_from_python_ast(parsed)
        except Exception:
            pass
    # simple root node
    tree = Tree(num_nodes=1, root=0)
    return tree

class CodeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = np.array(item['input_ids'], dtype=np.int64)  # 🔥 改成 int64
        y = int(item['gt_label'])
        tree = build_tree_from_sample(item)
        return torch.from_numpy(x), y, tree

def collate_fn(batch):
    xs, ys, trees = zip(*batch)

    max_len = max(x.shape[0] for x in xs)

    xs_padded = []
    for x in xs:
        pad_len = max_len - x.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, dtype=torch.long)
            x = torch.cat([x, pad])
        xs_padded.append(x)

    xs_padded = torch.stack(xs_padded)  # [B, L]
    ys = torch.tensor(ys, dtype=torch.long)

    return xs_padded, ys, list(trees)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []

    criterion = nn.CrossEntropyLoss() #use cross entropy loss now

    with torch.no_grad():
        for x, y, trees in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = torch.stack(
                [model(x[i], trees[i]).squeeze(0) for i in range(x.size(0))],
                dim=0
            )

            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds += logits.argmax(dim=1).cpu().tolist()
            targets += y.cpu().tolist()

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')

    return avg_loss, acc, f1

def train_model(
    poly_degree,
    train_dataset,
    val_dataset,
    test_dataset,
    device,
    batch_size=16,
):
    d_model = 64
    d_qk = 32
    d_v = 32

    num_classes = max(token_to_label.values()) + 1
    vocab_size = len(input_token_to_id)

    model = TreeAttentionModel(
        vocab_size=vocab_size,
        d_input=1,
        d_model=d_model,
        d_qk=d_qk,
        d_v=d_v,
        num_classes=num_classes,
        poly_degree=poly_degree,
        feature_map_type="favor",
        feature_dim=128,
    ).to(device)

    train_loader = DataLoader(CodeDataset(train_dataset), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(CodeDataset(val_dataset), batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(CodeDataset(test_dataset), batch_size=batch_size, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for x, y, trees in tqdm(train_loader, desc=f"Epoch {epoch}, degree={poly_degree}"):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits_list = []
            for i in range(x.size(0)):
                logits = model(x[i], trees[i])
                logits_list.append(logits)

            logits = torch.cat(logits_list, dim=0)

            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}")

    return history

def plot_metric(histories, poly_degrees, metric, ylabel, title):
    plt.figure(figsize=(8, 5))
    for hist, deg in zip(histories, poly_degrees):
        plt.plot(hist[metric], label=f'degree={deg}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    poly_degrees = [1, 2, 4]
    all_histories = []
    for deg in poly_degrees:
        print(f"\n--- Polynomial Degree: {deg} ---")
        history = train_model(
            deg,
            train_dataset,
            val_dataset,
            test_dataset,
            device
        )
        all_histories.append(history)
    # visualization
    plot_metric(all_histories, poly_degrees, "val_acc", "Validation Accuracy", "Validation Accuracy by Polynomial Degree")
    plot_metric(all_histories, poly_degrees, "val_f1", "Validation F1 Score", "Validation F1 Score by Polynomial Degree")
    plot_metric(all_histories, poly_degrees, "val_loss", "Validation Loss", "Validation Loss by Polynomial Degree")
    plot_metric(all_histories, poly_degrees, "test_loss", "Test Loss", "Test Loss by Polynomial Degree")

def _tree_from_python_ast(parsed: ast.AST) -> Tree:
    nodes = []
    edges = []

    def visit(node: ast.AST, parent_idx: Optional[int] = None):
        idx = len(nodes)
        nodes.append(node)
        if parent_idx is not None:
            edges.append((parent_idx, idx))
        for child in ast.iter_child_nodes(node):
            visit(child, idx)

    visit(parsed)
    tree = Tree(num_nodes=max(len(nodes), 1), root=0)
    for u, v in edges:
        tree.add_edge(u, v, weight=1.0)
    return tree