import os
import re
from datasets import load_dataset
import torch
import pickle
import ast
import numpy as np
from collections import defaultdict
from collections import Counter
from _tree_attention import Tree


def load_py150k_dataset():
    """load PY150k"""
    ds = load_dataset("AISE-TUDelft/PY150k")
    return ds

def _detokenize_code(text: str) -> str:
    """将数据集里的 token 化代码恢复成可被 ast.parse 解析的 Python 源码。"""
    if text is None:
        return ""
    s = str(text).strip()
    # 去掉首尾引号包裹（常见于整段被当成字符串存储）
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        s = s[1:-1].strip()
    # 常见特殊 token 处理
    s = s.replace("</s>", "")
    s = s.replace("<EOL>", "\n")
    # 占位符替换为合法 Python 字面量
    s = s.replace("<STR_LIT>", "x")     # 用标识符 x，避免引号嵌套问题
    s = s.replace("<NUM_LIT>", "0")
    s = s.replace("<CHAR_LIT>", "c")
    # 清理多余空白
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_ast_from_code(row: dict[str, any]) -> dict[str, any]:
    """
    row: data_split 的一行（dict 或 pandas.Series）
    返回:
      {
        "ast_dump": ...,
        "code_used": ...,
        "source_col": "full_line"|"gt"|"input"
      }
    解析失败返回 None
    """
    candidates = [
        ("full_line", row.get("full_line", "")),
        ("gt", row.get("gt", "")),
        ("input", row.get("input", "")),
    ]
    for source_col, raw in candidates:
        code = _detokenize_code(raw)
        if not code:
            continue
        # 先按完整代码解析
        try:
            tree = ast.parse(code, mode="exec")
            return {
                "ast_dump": ast.dump(tree, include_attributes=False),
                "code_used": code,
                "source_col": source_col,
            }
        except SyntaxError:
            pass
        # 若是单表达式，尝试 eval
        try:
            tree = ast.parse(code, mode="eval")
            return {
                "ast_dump": ast.dump(tree, include_attributes=False),
                "code_used": code,
                "source_col": source_col,
            }
        except SyntaxError:
            continue
    return None

def convert_ast_to_tree_structure(ast_node):
    """Transfer the AST to Tree Class"""
    if ast_node is None:
        return None

    tree = Tree(num_nodes=100)  # pre-distribute the nodes
    node_id_map = {}
    counter = [0]

    def traverse(node, parent_id=None):
        current_id = counter[0]
        counter[0] += 1
        node_id_map[id(node)] = current_id

        if parent_id is not None:
            tree.add_edge(parent_id, current_id, weight=1.0)

        for child in ast.iter_child_nodes(node):
            traverse(child, current_id)

    traverse(ast_node)
    tree.num_nodes = counter[0]
    return tree


def create_node_features(code_string):
    """Create eigenvector for every node"""
    # token embedding
    tokens = code_string.split()
    features = torch.randn(len(tokens), 32)
    return features


def process_py150k_dataset(split='train'):
    """Change the dataset to the format of tree attention"""
    ds = load_py150k_dataset()
    data_split = ds[split]

    processed_data = []

    for idx, sample in enumerate(data_split):
        try:
            # get the output
            full_code = sample['full_line']

            # get AST
            ast_node = extract_ast_from_code(full_code)
            if ast_node is None:
                continue

            # Change it to Tree structure
            tree = convert_ast_to_tree_structure(ast_node)
            if tree is None:
                continue

            # create node features
            features = create_node_features(full_code)

            # Build real token vocabulary on first pass
            # Then map tokens to labels
            token = sample['gt']
            label = token_to_label.get(token, oov_label)  # Map to real token ID


            processed_data.append((features, tree, label))

        except Exception as e:
            continue

    return processed_data


def split_data(data, train_ratio=3 / 5, val_ratio=1 / 15, test_ratio=1 / 3):
    """
    按指定比例划分数据
    train: 3/5 (60%)
    val:   1/15 (~6.7%)
    test:  1/3 (~33.3%)
    """
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


def save_processed_splits(train_data, val_data, test_data, output_dir='./processed_data'):
    """保存处理后的数据为pickle文件"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    with open(os.path.join(output_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    print(f"Data saved to {output_dir}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


def load_processed_splits(input_dir='./processed_data'):
    """从pickle文件加载处理后的数据"""
    with open(os.path.join(input_dir, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(input_dir, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(input_dir, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    return train_data, val_data, test_data


if __name__ == "__main__":
    # 处理数据集
    print("Loading and processing PY150k dataset...")
    processed_data = process_py150k_dataset(split='train')

    # 划分数据
    train_data, val_data, test_data = split_data(processed_data)

    # 保存数据
    save_processed_splits(train_data, val_data, test_data)