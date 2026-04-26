import torch
from torch.utils.data import Dataset
from _tree_attention import TreeAttentionModel
from load_python150 import load_data

# ListDataset class to convert list of (features, tree, label) tuples into a PyTorch Dataset
class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features, tree, label = self.data[idx]
        return features, tree, label

# Function to train a single tree attention model
def train_single_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        for features, tree, label in train_loader:
            optimizer.zero_grad()
            outputs = model(features, tree)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

    # Evaluate on validation and test sets
    evaluate_model(model, val_loader, test_loader)

# Function to train models for multiple polynomial degrees
def train_with_splits(data, degrees=[1, 2, 4]):
    results = {}
    for degree in degrees:
        model = TreeAttentionModel(degree=degree)
        train_loader, val_loader, test_loader = create_data_loaders(data, degree)
        results[degree] = train_single_model(model, train_loader, val_loader, test_loader)
    return results

# Main entry point to load processed data and train all models
def load_and_train():
    data = load_data()  # Load data using the provided utility
    train_with_splits(data)

# Note: The evaluate_model and create_data_loaders functions should be implemented as needed.