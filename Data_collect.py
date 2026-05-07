'''
This is the revised edition of importing the python150 dataset
'''

import os
import re
from datasets import load_dataset
import torch
import pickle
import ast
import numpy as np
from collections import defaultdict, Counter
from _tree_attention import Tree


def load_py150k_dataset():
    """Load PY150k dataset from Hugging Face"""
    ds = load_dataset("AISE-TUDelft/PY150k")
    return ds


def _detokenize_code(text: str) -> str:
    """
    restore the tokenized code to Python source code processed by ast.parse
    """
    if text is None:
        return ""

    s = str(text).strip()

    # strip the ""
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        s = s[1:-1].strip()

    # special tokens
    s = s.replace("</s>", "")
    s = s.replace("<EOL>", "\n")

    # change the placeholder to string
    s = s.replace("<STR_LIT>", "'x'")  
    s = s.replace("<NUM_LIT>", "0")
    s = s.replace("<CHAR_LIT>", "'c'")

    # clear all the spaces
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def extract_ast_from_code(code: str) -> ast.AST:
    """
    generate AST Nodes
    code: the code string to be processed
    return: AST, or None if there's error
    """
    if not code or not isinstance(code, str):
        return None

    code = _detokenize_code(code)
    if not code:
        return None

    # parse it as the complete code first
    try:
        tree = ast.parse(code, mode="exec")
        return tree
    except SyntaxError:
        pass

    # if it's a single expression: change it to eval
    try:
        tree = ast.parse(code, mode="eval")
        return tree
    except SyntaxError:
        pass

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


def tokenize_input_code(code_string: str, max_length: int = 150) -> np.ndarray:
    """
    transfer the input code token sequence
    split the code by space and special symbles
    """
    tokens = code_string.split()
    # concatenate or fill to max_length
    if len(tokens) >= max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + ['<PAD>'] * (max_length - len(tokens))

    return np.array(tokens, dtype=object)


def build_vocab_from_labels(data_split, max_vocab_size: int = 10000):
    """
    build the vocabulary and map it into labels
    return: (token_to_label dict, oov_label int)
    """
    token_counter = Counter()

    # get the frequency of all gt tokens
    for sample in data_split:
        token = sample.get('gt', '')
        if token and isinstance(token, str):
            token_counter[token] += 1

    # sort the vocabulary and keep the most frequent max_vocab_size
    token_to_label = {}
    for idx, (token, _) in enumerate(token_counter.most_common(max_vocab_size)):
        token_to_label[token] = idx

    oov_label = len(token_to_label)  # Out-of-vocabulary label
    token_to_label['<OOV>'] = oov_label
    token_to_label['<PAD>'] = oov_label + 1

    return token_to_label, oov_label


def build_vocab_from_input_tokens(data_split, max_vocab_size: int = 50000):
    """
    Build vocabulary for input token sequences.
    return: (token_to_id dict, unk_id int)
    """
    token_counter = Counter()

    for sample in data_split:
        input_code = sample.get('input', '')
        if not input_code or not isinstance(input_code, str):
            continue
        input_tokens = tokenize_input_code(input_code)
        token_counter.update(input_tokens.tolist())

    token_to_id = {'<PAD>': 0, '<UNK>': 1}
    for token, _ in token_counter.most_common(max_vocab_size):
        if token in token_to_id:
            continue
        token_to_id[token] = len(token_to_id)

    return token_to_id, token_to_id['<UNK>']


def process_py150k_dataset(split='train', debug=False, max_vocab_size=10000):
    """
    process PY150k dataset：
    - input: the prefix token series
    - gt: next token (target

    return: [(input_tokens, gt_label), ...]
    """
    print(f"Loading PY150k dataset ({split} split)...")
    ds = load_py150k_dataset()
    data_split = ds[split]

    # build vocabulary
    print("Building vocabulary from 'gt' field...")
    token_to_label, oov_label = build_vocab_from_labels(data_split, max_vocab_size)
    print("Building vocabulary from 'input' field...")
    input_token_to_id, input_unk_id = build_vocab_from_input_tokens(data_split, max_vocab_size=max_vocab_size)
    print(f"Vocabulary size: {len(token_to_label)}")
    print(f"Input vocabulary size: {len(input_token_to_id)}")

    processed_data = []
    failed_count = 0
    success_count = 0

    total = len(data_split)

    for idx, sample in enumerate(data_split):
        if (idx + 1) % 1000 == 0:
            print(f"Processing... {idx + 1}/{total} (Success: {success_count}, Failed: {failed_count})")

        try:
            # get the input and output (gt)
            input_code = sample.get('input', '')
            gt_token = sample.get('gt', '')

            # check whether the data is valid
            if not input_code or not isinstance(input_code, str):
                if debug and failed_count < 10:
                    print(f"[{idx}] invalid input ")
                failed_count += 1
                continue

            if not gt_token or not isinstance(gt_token, str):
                if debug and failed_count < 10:
                    print(f"[{idx}] invalid gt ")
                failed_count += 1
                continue

            # tokenize the input into series
            input_tokens = tokenize_input_code(input_code)
            input_ids = np.array(
                [input_token_to_id.get(tok, input_unk_id) for tok in input_tokens],
                dtype=np.int64
            )

            # transfer the gt to label
            gt_label = token_to_label.get(gt_token, oov_label)

            # parse the AST
            full_code = sample.get('full_line', '')
            ast_node = None
            if full_code:
                ast_node = extract_ast_from_code(full_code)

            # add samples
            processed_data.append({
                'input_tokens': input_tokens,
                'input_ids': input_ids,
                'gt_label': gt_label,
                'gt_token': gt_token,
                'has_valid_ast': ast_node is not None,
                'full_code': full_code
            })

            success_count += 1

        except Exception as e:
            if debug and failed_count < 10:
                print(f"[{idx}] error occured: {str(e)}")
            failed_count += 1
            continue

    print(f"\nProcessing completed!")
    print(f"  Success: {success_count}")
    print(f"  Fail: {failed_count}")
    print(f"  Success rate: {success_count / total * 100:.2f}%")

    return processed_data, token_to_label, input_token_to_id


def split_data(data, train_ratio=0.6, val_ratio=0.067, test_ratio=0.333):
    """
    train: 60%
    val:   ~6.7%
    test:  ~33.3%
    """
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


def save_processed_data(train_data, val_data, test_data, token_to_label,
                        input_token_to_id=None,
                        output_dir='./processed_data'):
    """save the data into pickles"""
    os.makedirs(output_dir, exist_ok=True)

    # save the data
    with open(os.path.join(output_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    with open(os.path.join(output_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    # save the vocabulary
    with open(os.path.join(output_dir, 'token_to_label.pkl'), 'wb') as f:
        pickle.dump(token_to_label, f)
    if input_token_to_id is not None:
        with open(os.path.join(output_dir, 'input_token_to_id.pkl'), 'wb') as f:
            pickle.dump(input_token_to_id, f)

    print(f"\n Data saved to {output_dir}")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")
    print(f"  Vocabulary size: {len(token_to_label)}")


def load_processed_data(input_dir='./processed_data'):
    """load the data from pickles"""
    with open(os.path.join(input_dir, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(input_dir, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(input_dir, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    with open(os.path.join(input_dir, 'token_to_label.pkl'), 'rb') as f:
        token_to_label = pickle.load(f)
    input_vocab_path = os.path.join(input_dir, 'input_token_to_id.pkl')
    if os.path.exists(input_vocab_path):
        with open(input_vocab_path, 'rb') as f:
            input_token_to_id = pickle.load(f)
    else:
        input_token_to_id = None

    return train_data, val_data, test_data, token_to_label, input_token_to_id


def process_py150k_dataset_pipeline():
    print("=" * 60)
    print("PY150k Dataset Processing Pipeline")
    print("=" * 60)

    processed_data, token_to_label, input_token_to_id = process_py150k_dataset(
        split='train',
        debug=False,
        max_vocab_size=1000
    )

    # divide the data into train/validation/test set
    print("\nSplitting data into train/val/test...")
    train_data, val_data, test_data = split_data(processed_data)

    # save the data
    save_processed_data(train_data, val_data, test_data, token_to_label, input_token_to_id=input_token_to_id)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

process_py150k_dataset_pipeline()

