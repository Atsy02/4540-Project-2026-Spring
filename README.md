# 4540-Project-2026-Spring
4540 Project 2026 Spring

In this project, we propose a tree attention model in which the tokens are the tree nodes and the distances of the nodes d are expressed as polynomials. We use tree dynamic programming to build the tree and reached time complexity of O(N·d^2) where N is the number of treenodes and d is the degree of the polynomial. In order to train the model in a reasonable amount of time when d is large, we add FFT optimization in the model and reduce the time complexity of O(N·dlogd).

We had three stages of experiment: in the first stage, we tried the model on synthetic datasets and used scaling analysis see the computational efficiency. Then we used python150k dataset to predict the next token of Python code sentences, testing the performance of tree models. We furthered our study by removing the out-of-language (OOV) tokens expecting to let the higher order terms make difference in training performance.

======Codes======
1. Loading the data
load_python150.py

2. Draft - trial on low order with OOV tokens
Data_collect.py - a revised version of data collection which has been replaced in the latest stage of the experiment
_tree_attention_high_order.py - tree model
_tree_FFT_revised.py - FFT tree model

3. OOV removal analysis
_tree_attention.py - tree model
baseline.py - building the tree attention
run_benchmark - main program of traning (can use both synthetic and real data)
