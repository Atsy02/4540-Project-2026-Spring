"""
This part generally use the tree structure, but we add FFT optimization here
"""

import math
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from _tree_attention_high_order import (
    RandomFeatureMap,
    TaylorFeatureMapFixed,
    FavorFeatureMap,
    ReluFeatureMap,
    build_tree_from_sample,
    Dataset,
_tree_from_python_ast
)

class Tree:
    def __init__(self, num_nodes: int, root: int = 0):
        self.num_nodes = num_nodes
        self.root = root

        self.adj = defaultdict(list)
        self.parent = [-1] * num_nodes
        self.children = [[] for _ in range(num_nodes)]
        self.edge_weights = {}

        #  NEW (FFT support)
        self.depth = [-1] * num_nodes
        self.levels = []
        self.max_depth = 0
        self._levels_built = False

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        self.adj[u].append((v, weight))
        self.children[u].append(v)
        self.parent[v] = u
        self.edge_weights[(u, v)] = weight

    # ============================
    #  FFT REQUIRED
    # ============================
    def build_levels(self):
        from collections import deque

        self.depth = [-1] * self.num_nodes
        self.levels = []

        queue = deque([self.root])
        self.depth[self.root] = 0

        while queue:
            u = queue.popleft()
            d = self.depth[u]

            if d >= len(self.levels):
                self.levels.append([])

            self.levels[d].append(u)

            for v, _ in self.adj[u]:
                self.depth[v] = d + 1
                queue.append(v)

        self.max_depth = len(self.levels)
        self._levels_built = True

    def ensure_levels(self):
        if not self._levels_built:
            self.build_levels()

    # ============================
    # existing methods unchanged
    # ============================

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

        self.poly_coeff = nn.Parameter(
            torch.randn(poly_degree + 1) * 0.1
        ) # * 0.1 in order to avoid explosion caused by higher degree term
    def forward(self, x, tree):
        N = x.size(0)
        device = x.device

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        phi_q = self.phi(q) / (self.phi(q).sum(dim=-1, keepdim=True) + 1e-6)         # [N, F]
        phi_k = self.phi(k) / (self.phi(k).sum(dim=-1, keepdim=True) + 1e-6)        # [N, F]


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
        # Aggregate polynomial
        # =========================

        # [N, F, dv]
        Y = torch.zeros(N, self.feature_dim, self.d_v, device=device)
        ZY = torch.zeros(N, self.feature_dim, 1, device=device)

        for m in range(self.poly_degree + 1):
            Y += self.poly_coeff[m] * S[m]
            ZY += self.poly_coeff[m] * ZS[m]

        # =========================
        # kernel attention (vectorized)
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

        # embedding
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0
        )

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

        # more stable attention pooling
        self.pool_q = nn.Parameter(torch.randn(d_model) * 0.02)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward_hidden(self, x, tree):
        # x: [N, 1] → token ids
        x = x.long().squeeze(-1)

        # embedding
        h = self.embedding(x)

        # encoder
        h = self.encoder(h)

        # tree attention
        o = self.tree_attention(h, tree)

        # attention pooling
        scores = (o @ self.pool_q) / math.sqrt(o.size(-1))
        scores = scores - scores.max()   # 数值稳定
        weights = torch.softmax(scores, dim=0)

        agg = (weights.unsqueeze(-1) * o).sum(dim=0)

        return self.cls_head(agg)

    def forward(self, x, tree):
        hid = self.forward_hidden(x, tree)   # [hidden]
        return self.classifier(hid)          # [num_classes]

# =============
#      Main
# =============
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
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

class CodeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        #  int64 for embedding
        x = np.array(item['input_ids'], dtype=np.int64)

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
            pad = torch.zeros(pad_len, dtype=torch.long)  # pad=0
            x = torch.cat([x, pad])
        xs_padded.append(x)

    xs_padded = torch.stack(xs_padded)  # [B, L]
    ys = torch.tensor(ys, dtype=torch.long)

    return xs_padded, ys, list(trees)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y, trees in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = torch.stack(
                [model(x[i], trees[i]) for i in range(x.size(0))],
                dim=0
            )  # [B, num_classes]

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

    train_loader = DataLoader(
        CodeDataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        CodeDataset(val_dataset),
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        CodeDataset(test_dataset),
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for x, y, trees in tqdm(train_loader, desc=f"Epoch {epoch}, degree={poly_degree}"):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            #  forward
            logits = torch.stack(
                [model(x[i], trees[i]) for i in range(x.size(0))],
                dim=0
            )  # [B, num_classes]

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
    poly_degrees = [1,2,4]
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