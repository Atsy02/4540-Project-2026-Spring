"""
load_python150.py
=================
Utilities for loading the PY150k dataset and converting Python source code
into (features, tree, label) triples compatible with the tree-attention models.

Pipeline
--------
  PY150k row
      │  _detokenize_code()
      ▼
  Python source string
      │  ast.parse()
      ▼
  ast.AST object
      │  ast_to_tree_and_features()
      ▼
  (Tree, node_features [N, D_NODE_FEAT])   + label (int)

Label definition
----------------
  We predict the *gt* (ground-truth next token) field.
  Tokens are mapped to integer class IDs via a vocabulary built from the
  most-frequent tokens in the dataset.  Rare / unseen tokens map to the
  last class (OOV bucket).

Node features
-------------
  Each AST node is represented by a one-hot vector over Python's built-in
  AST node types (~100 types).  The encoder in the model projects this to
  D_MODEL via a learned linear layer, so no extra embedding table is needed.

Exported constants
------------------
  D_NODE_FEAT : int   – feature dimension (== number of AST node types)

Exported functions
------------------
  load_py150k_dataset()
  process_py150k(split, max_samples, num_classes, max_nodes) -> (data, D_NODE_FEAT)
  train_val_test_split(data, train_r, val_r)
"""

import ast
import re
import pickle
import os
from collections import Counter
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

from _tree_attention import Tree


# ============================================================================
# 1. AST node-type vocabulary  (fixed; built from Python's ast module)
# ============================================================================

# Collect all concrete AST node types defined in the ast module
_AST_NODE_TYPES: List[str] = sorted(
    name
    for name in dir(ast)
    if isinstance(getattr(ast, name), type)
    and issubclass(getattr(ast, name), ast.AST)
    and name != "AST"
)

_NODE_TYPE_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(_AST_NODE_TYPES)}

# Feature dimension exported for use by callers
D_NODE_FEAT: int = len(_AST_NODE_TYPES)   # typically ~100


def _node_feature(node: ast.AST) -> np.ndarray:
    """Return a one-hot feature vector for a single AST node."""
    feat = np.zeros(D_NODE_FEAT, dtype=np.float32)
    idx = _NODE_TYPE_TO_IDX.get(type(node).__name__, -1)
    if idx >= 0:
        feat[idx] = 1.0
    return feat


# ============================================================================
# 2. Source-code pre-processing
# ============================================================================

def _detokenize_code(text: str) -> str:
    """Restore PY150k tokenised code to parseable Python source."""
    if text is None:
        return ""
    s = str(text).strip()
    # Strip outer quotes if the whole string was stored as a quoted literal
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    s = s.replace("</s>", "")
    s = s.replace("<EOL>", "\n")
    s = s.replace("<STR_LIT>", "x")
    s = s.replace("<NUM_LIT>", "0")
    s = s.replace("<CHAR_LIT>", "c")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _parse_code(sample: dict) -> Optional[ast.AST]:
    """
    Try to parse Python source from a PY150k row dict.
    Returns the root ast.AST node, or None on failure.
    """
    for field in ("full_line", "gt", "input"):
        raw = sample.get(field, "")
        if not raw:
            continue
        code = _detokenize_code(raw)
        for mode in ("exec", "eval"):
            try:
                return ast.parse(code, mode=mode)
            except SyntaxError:
                continue
    return None


# ============================================================================
# 3. AST → Tree conversion
# ============================================================================

def ast_to_tree_and_features(
    root: ast.AST,
    max_nodes: int = 150,
) -> Tuple[Optional[Tree], Optional[torch.Tensor]]:
    """
    Convert a Python AST to a Tree object and a node-feature matrix.

    Parameters
    ----------
    root      : root ast.AST node
    max_nodes : cap the tree size (large ASTs are truncated by BFS order)

    Returns
    -------
    tree     : Tree object with unit edge weights, or None if tree too small
    features : FloatTensor [N, D_NODE_FEAT], one-hot node-type features
    """
    node_list: List[ast.AST] = []
    edge_list: List[Tuple[int, int]] = []

    # BFS traversal so truncation at max_nodes is level-fair
    queue = [(root, -1)]
    head = 0
    while head < len(queue) and len(node_list) < max_nodes:
        node, parent_id = queue[head]
        head += 1
        curr_id = len(node_list)
        node_list.append(node)
        if parent_id >= 0:
            edge_list.append((parent_id, curr_id))
        for child in ast.iter_child_nodes(node):
            if len(node_list) < max_nodes:
                queue.append((child, curr_id))

    N = len(node_list)
    if N < 2:
        return None, None

    tree = Tree(N, root=0)
    for parent, child in edge_list:
        tree.add_edge(parent, child, weight=1.0)

    features = torch.tensor(
        np.stack([_node_feature(n) for n in node_list]),
        dtype=torch.float32,
    )   # [N, D_NODE_FEAT]

    return tree, features


# ============================================================================
# 4. Label vocabulary
# ============================================================================

def build_label_vocab(
    samples,
    num_classes: int,
) -> Dict[str, int]:
    """
    Build a token→class-ID mapping from the most frequent gt tokens.
    The last class ID (num_classes - 1) is the OOV bucket.
    """
    counter: Counter = Counter()
    for s in samples:
        gt = s.get("gt", "")
        if gt:
            counter[gt] += 1
    top = [tok for tok, _ in counter.most_common(num_classes - 1)]
    return {tok: i for i, tok in enumerate(top)}


# ============================================================================
# 5. Main dataset builder
# ============================================================================

def load_py150k_dataset():
    """Load the PY150k dataset from HuggingFace (requires datasets library)."""
    from datasets import load_dataset
    return load_dataset("AISE-TUDelft/PY150k")


def process_py150k(
    split: str = "train",
    max_samples: int = 600,
    num_classes: int = 50,
    max_nodes: int = 150,
    vocab_build_size: int = 5_000,
    cache_path: Optional[str] = None,
) -> Tuple[List[Tuple[torch.Tensor, Tree, int]], int]:
    """
    Load PY150k, parse code → AST → Tree, return list of samples.

    Parameters
    ----------
    split           : "train" | "validation" | "test"
    max_samples     : maximum number of valid samples to return
    num_classes     : vocabulary size (last class = OOV)
    max_nodes       : maximum AST nodes per tree
    vocab_build_size: rows used to build the token vocabulary
    cache_path      : if given, save/load processed data as a pickle file

    Returns
    -------
    data        : list of (features [N, D_NODE_FEAT], Tree, label) tuples
    D_NODE_FEAT : feature dimension  ← pass this to the model as d_input
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path} …")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data, D_NODE_FEAT

    print(f"Loading PY150k [{split}] …")
    ds = load_py150k_dataset()
    raw_split = ds[split]

    # Build label vocabulary from the first vocab_build_size rows
    vocab_rows = raw_split.select(range(min(vocab_build_size, len(raw_split))))
    vocab = build_label_vocab(vocab_rows, num_classes)
    oov_label = num_classes - 1
    print(f"Vocabulary: {len(vocab)} tokens + 1 OOV  (total {num_classes} classes)")

    # Process rows one by one
    data: List[Tuple[torch.Tensor, Tree, int]] = []
    skipped = 0

    for i, sample in enumerate(raw_split):
        if len(data) >= max_samples:
            break

        root_ast = _parse_code(sample)
        if root_ast is None:
            skipped += 1
            continue

        tree, features = ast_to_tree_and_features(root_ast, max_nodes)
        if tree is None:
            skipped += 1
            continue

        gt    = sample.get("gt", "")
        label = vocab.get(gt, oov_label)

        data.append((features, tree, label))

        if len(data) % 100 == 0:
            print(f"  Processed {len(data)}/{max_samples} "
                  f"(scanned {i+1}, skipped {skipped})")

    print(f"Done: {len(data)} samples  (skipped {skipped})")

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Cached to {cache_path}")

    return data, D_NODE_FEAT


# ============================================================================
# 6. Train / val / test split helper
# ============================================================================

def train_val_test_split(
    data: list,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
) -> Tuple[list, list, list]:
    """Split a list of samples into train / val / test."""
    n          = len(data)
    n_train    = int(n * train_ratio)
    n_val      = int(n * val_ratio)
    train_data = data[:n_train]
    val_data   = data[n_train : n_train + n_val]
    test_data  = data[n_train + n_val :]
    return train_data, val_data, test_data


# ============================================================================
# 7. Quick sanity check
# ============================================================================

if __name__ == "__main__":
    print(f"AST node types: {D_NODE_FEAT}")
    print(f"Examples: {_AST_NODE_TYPES[:10]}")

    data, feat_dim = process_py150k(
        split="train",
        max_samples=50,
        num_classes=20,
        max_nodes=80,
    )
    print(f"\nSample 0:")
    features, tree, label = data[0]
    print(f"  features : {features.shape}  dtype={features.dtype}")
    print(f"  tree     : {tree.num_nodes} nodes, root={tree.root}")
    print(f"  label    : {label}")

    train_data, val_data, test_data = train_val_test_split(data)
    print(f"\nSplit: train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")
