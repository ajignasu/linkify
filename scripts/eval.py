import os
import sys
import torch
import shutil
import argparse
import numpy as np
import logging
import random
import json
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LinearRegression
from data_generation.graph_data import AssemblyGraphDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_utils import visualize_graph
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def calculate_topk_accuracy(y_true, y_pred_proba, classes, ks=None):
    """Calculates Top-K accuracy for a set of predictions."""
    if ks is None:
        ks = [1, 3, 5, 10, 50]
    
    topk_correct = {k: 0 for k in ks}
    num_samples = len(y_true)

    # Get the indices of the top k probabilities for all samples at once
    max_k = max(ks)
    # Argsort returns ascending order, so we slice from the end
    top_k_indices = np.argsort(y_pred_proba, axis=1)[:, -max_k:][:, ::-1]
    
    for i in range(num_samples):
        true_label = y_true[i]
        top_k_labels = classes[top_k_indices[i]]
        
        for k in ks:
            if true_label in top_k_labels[:k]:
                topk_correct[k] += 1
    
    topk_accuracies = {f'top{k}_acc': correct / num_samples for k, correct in topk_correct.items()}
    return topk_accuracies


def load_indices_from_file(root_dir, suffix=""):
    logging.info(f"Loading indices with suffix: {suffix}")
    train_indices = np.load(os.path.join(root_dir, f'train_indices{suffix}.npy'), allow_pickle=True)
    val_indices   = np.load(os.path.join(root_dir, f'val_indices{suffix}.npy'),   allow_pickle=True)
    test_indices  = np.load(os.path.join(root_dir, f'test_indices{suffix}.npy'),  allow_pickle=True)
    return train_indices, val_indices, test_indices

def average_embedding_baseline(dataset, device):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False)
    total_loss = 0.0
    num_graphs = 0
    for data in tqdm(dataloader, desc="Average Embedding Baseline Evaluation"):
        data = data.to(device)
        masked_node_idx = data.masked_node_idx.item()
        mask = torch.ones(data.x.size(0), dtype=torch.bool)
        mask[masked_node_idx] = False
        avg_embedding = torch.mean(data.x[mask], dim=0)
        ground_truth = data.y.squeeze(0)
        loss = torch.nn.functional.mse_loss(avg_embedding, ground_truth)
        total_loss += loss.item()
        num_graphs += 1
    return total_loss / num_graphs

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = torch.mean(x, dim=0)
        return self.linear(x)
    
def pytorch_linear_regression_baseline(train_dataset, test_dataset, device, node_feature_size):
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False)
    total_loss = 0.0
    num_graphs = 0
    input_dim = node_feature_size
    output_dim = node_feature_size
    model = LinearRegressionModel(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    num_epochs = 100

    logging.info("--- PyTorch Linear Baseline Hyperparameters ---")
    logging.info(f"  Input Dim: {input_dim}")
    logging.info(f"  Output Dim: {output_dim}")
    logging.info(f"  Optimizer: Adam (lr=0.001)")
    logging.info(f"  Criterion: MSELoss")
    logging.info(f"  Num Epochs: {num_epochs}")
    logging.info("---------------------------------------------")

    for epoch in tqdm(range(num_epochs), desc="Pytorch Linear Regression Training Epoch"):
        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()

            # Exclude the masked node's features to prevent data leakage
            masked_node_idx = data.masked_node_idx.item()
            mask = torch.ones(data.x.size(0), dtype=torch.bool, device=data.x.device)
            mask[masked_node_idx] = False
            features_without_masked_node = data.x[mask]

            output = model(features_without_masked_node)
            loss = criterion(output, data.y.squeeze(0))
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="PyTorch Linear Regression Evaluation"):
            data = data.to(device)

            # Also apply masking here for evaluation
            masked_node_idx = data.masked_node_idx.item()
            mask = torch.ones(data.x.size(0), dtype=torch.bool, device=data.x.device)
            mask[masked_node_idx] = False
            features_without_masked_node = data.x[mask]

            output = model(features_without_masked_node)
            loss = criterion(output, data.y.squeeze(0))
            total_loss += loss.item()
            num_graphs += 1
    return total_loss / num_graphs

def sklearn_linear_regression_baseline(train_dataset, test_dataset):
    # This baseline runs on the CPU. Data is moved from device if necessary.
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False)

    # --- Prepare training data ---
    X_train_list, y_train_list = [], []
    for data in tqdm(train_dataloader, desc="Scikit-learn Preparing Train Data"):
        data = data.to('cpu')
        masked_node_idx = data.masked_node_idx.item()
        mask = torch.ones(data.x.size(0), dtype=torch.bool)
        mask[masked_node_idx] = False
        features_without_masked_node = data.x[mask]
        
        mean_embedding = torch.mean(features_without_masked_node, dim=0)
        X_train_list.append(mean_embedding.numpy())
        y_train_list.append(data.y.squeeze(0).numpy())

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    # --- Train the model ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Prepare test data ---
    X_test_list, y_test_list = [], []
    for data in tqdm(test_dataloader, desc="Scikit-learn Preparing Test Data"):
        data = data.to('cpu')
        masked_node_idx = data.masked_node_idx.item()
        mask = torch.ones(data.x.size(0), dtype=torch.bool)
        mask[masked_node_idx] = False
        features_without_masked_node = data.x[mask]
        
        mean_embedding = torch.mean(features_without_masked_node, dim=0)
        X_test_list.append(mean_embedding.numpy())
        y_test_list.append(data.y.squeeze(0).numpy())

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    # --- Predict and calculate loss ---
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    
    return mse

# def majority_class_baseline(train_dataset, test_dataset):
#     """Predict the most common class from training set for all test samples."""
#     from collections import Counter
    
#     # Collect all training labels
#     train_labels = []
#     for data in train_dataset:
#         if hasattr(data, 'y_cls'):  # Classification task
#             train_labels.append(data.y_cls.item())
    
#     if not train_labels:
#         logging.warning("No classification labels found in training set")
#         return 0.0
    
#     # Find majority class
#     majority_class = Counter(train_labels).most_common(1)[0][0]
#     logging.info(f"Majority class baseline: predicting class {majority_class} for all samples")
    
#     # Evaluate on test set
#     correct = 0
#     total = 0
#     for data in test_dataset:
#         if hasattr(data, 'y_cls'):
#             true_label = data.y_cls.item()
#             if true_label == majority_class:
#                 correct += 1
#             total += 1
    
#     accuracy = correct / total if total > 0 else 0.0
#     logging.info(f"Majority class baseline accuracy: {accuracy:.4f}")
#     return accuracy


def majority_class_baseline(train_dataset, test_dataset):
    """Predict the most common class from training set for all test samples."""
    from collections import Counter
    
    # Collect all training labels
    train_labels = []
    for data in train_dataset:
        if hasattr(data, 'y_cls'):  # Classification task
            train_labels.append(data.y_cls.item())
    
    logging.info(f"Training labels collected: {len(train_labels)} samples")
    
    if not train_labels:
        logging.warning("No classification labels found in training set")
        return {'majority_acc': 0, 'majority_precision': 0, 'majority_recall': 0, 'majority_f1': 0}
    
    # Find majority class
    majority_class = Counter(train_labels).most_common(1)[0][0]
    logging.info(f"Majority class: {majority_class}")
    
    # Collect test labels
    test_labels = [data.y_cls.item() for data in test_dataset if hasattr(data, 'y_cls')]
    logging.info(f"Test labels collected: {len(test_labels)} samples")
    
    # Log frequency of the majority class in the test dataset
    majority_class_count = test_labels.count(majority_class)
    logging.info(f"Majority class {majority_class} appears {majority_class_count} times in the test dataset.")
    
    # Log class distributions
    train_class_distribution = Counter(train_labels)
    test_class_distribution = Counter(test_labels)
    logging.info(f"Training class distribution: {train_class_distribution}")
    logging.info(f"Test class distribution: {test_class_distribution}")
    
    if majority_class not in test_labels:
        logging.warning(f"Majority class {majority_class} not found in test labels.")
    
    # Evaluate on test set
    y_true = []
    y_pred = []
    for data in test_dataset:
        if hasattr(data, 'y_cls'):
            y_true.append(data.y_cls.item())
            y_pred.append(majority_class)
            
    if not y_true:
        accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    else:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    logging.info(f"Majority class baseline accuracy: {accuracy:.4f}")
    logging.info(f"Majority class baseline precision (weighted): {precision:.4f}")
    logging.info(f"Majority class baseline recall (weighted): {recall:.4f}")
    logging.info(f"Majority class baseline F1 (weighted): {f1:.4f}")
    
    return {
        'majority_acc': accuracy,
        'majority_precision': precision,
        'majority_recall': recall,
        'majority_f1': f1,
    }

def logistic_regression_baseline(train_dataset, test_dataset):
    """Train logistic regression on individual node features (ignoring graph structure)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    # Prepare training data
    X_train_list = []
    y_train_list = []
    
    for data in tqdm(train_dataset, desc="Preparing logistic regression training data"):
        if hasattr(data, 'y_cls') and hasattr(data, 'masked_node_idx'):
            masked_idx = data.masked_node_idx.item()
            # Use the mean of neighbor features as a proxy for the masked node
            mask = torch.ones(data.x.size(0), dtype=torch.bool)
            mask[masked_idx] = False
            if mask.sum() > 0:  # If there are neighbors
                neighbor_mean = torch.mean(data.x[mask], dim=0)
                X_train_list.append(neighbor_mean.numpy())
                y_train_list.append(data.y_cls.item())
    
    if not X_train_list:
        logging.warning("No valid training samples for logistic regression")
        ks = [1, 3, 5, 10, 50]
        results = {'logreg_acc': 0, 'logreg_precision': 0, 'logreg_recall': 0, 'logreg_f1': 0}
        for k in ks:
            results[f'logreg_top{k}_acc'] = 0
        return results
    
    X_train = np.vstack(X_train_list)
    y_train = np.array(y_train_list)
    
    # Train model
    model = LogisticRegression(max_iter=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prepare test data
    X_test_list = []
    y_test_list = []
    
    for data in tqdm(test_dataset, desc="Preparing logistic regression test data"):
        if hasattr(data, 'y_cls') and hasattr(data, 'masked_node_idx'):
            masked_idx = data.masked_node_idx.item()
            mask = torch.ones(data.x.size(0), dtype=torch.bool)
            mask[masked_idx] = False
            if mask.sum() > 0:
                neighbor_mean = torch.mean(data.x[mask], dim=0)
                X_test_list.append(neighbor_mean.numpy())
                y_test_list.append(data.y_cls.item())
    
    if not X_test_list:
        logging.warning("No valid test samples for logistic regression")
        ks = [1, 3, 5, 10, 50]
        results = {'logreg_acc': 0, 'logreg_precision': 0, 'logreg_recall': 0, 'logreg_f1': 0}
        for k in ks:
            results[f'logreg_top{k}_acc'] = 0
        return results
    
    X_test = np.vstack(X_test_list)
    y_test = np.array(y_test_list)
    
    # Evaluate - calculate Top-K accuracies first to ensure consistency
    y_pred_proba = model.predict_proba(X_test)
    topk_results = calculate_topk_accuracy(y_test, y_pred_proba, model.classes_)
    
    # Use the same prediction method as Top-K for consistency
    y_pred_top1 = model.classes_[np.argsort(y_pred_proba, axis=1)[:, -1]]
    
    accuracy = topk_results['top1_acc']  # Use Top-K result for consistency
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_top1, average='weighted', zero_division=0)

    logging.info(f"Logistic regression baseline accuracy: {accuracy:.4f}")
    logging.info(f"Logistic regression baseline precision (weighted): {precision:.4f}")
    logging.info(f"Logistic regression baseline recall (weighted): {recall:.4f}")
    logging.info(f"Logistic regression baseline F1 (weighted): {f1:.4f}")

    # Log Top-K accuracies
    for key, value in topk_results.items():
        logging.info(f"Logistic regression baseline {key.replace('_', ' ').title()}: {value:.4f}")
    
    results = {
        'logreg_acc': accuracy,
        'logreg_precision': precision,
        'logreg_recall': recall,
        'logreg_f1': f1
    }
    # Add top-k results, renaming keys to be specific to this baseline
    for k, v in topk_results.items():
        results[f'logreg_{k}'] = v
        
    return results

def knn_baseline(train_dataset, test_dataset, k=5):
    """k-Nearest Neighbors baseline using masked node's neighbor features."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import precision_recall_fscore_support
    
    # Prepare training data
    X_train_list = []
    y_train_list = []
    
    for data in tqdm(train_dataset, desc="Preparing k-NN training data"):
        if hasattr(data, 'y_cls') and hasattr(data, 'masked_node_idx'):
            masked_idx = data.masked_node_idx.item()
            mask = torch.ones(data.x.size(0), dtype=torch.bool)
            mask[masked_idx] = False
            if mask.sum() > 0:
                neighbor_mean = torch.mean(data.x[mask], dim=0)
                X_train_list.append(neighbor_mean.numpy())
                y_train_list.append(data.y_cls.item())
    
    if not X_train_list:
        ks = [1, 3, 5, 10, 50]
        results = {'knn_acc': 0, 'knn_precision': 0, 'knn_recall': 0, 'knn_f1': 0}
        for k_val in ks:
            results[f'knn_top{k_val}_acc'] = 0
        return results
    
    X_train = np.vstack(X_train_list)
    y_train = np.array(y_train_list)
    
    # Train model
    model = KNeighborsClassifier(n_neighbors=min(k, len(X_train)), metric='cosine')
    model.fit(X_train, y_train)
    
    # Prepare test data and evaluate
    X_test_list = []
    y_test_list = []
    
    for data in tqdm(test_dataset, desc="Preparing k-NN test data"):
        if hasattr(data, 'y_cls') and hasattr(data, 'masked_node_idx'):
            masked_idx = data.masked_node_idx.item()
            mask = torch.ones(data.x.size(0), dtype=torch.bool)
            mask[masked_idx] = False
            if mask.sum() > 0:
                neighbor_mean = torch.mean(data.x[mask], dim=0)
                X_test_list.append(neighbor_mean.numpy())
                y_test_list.append(data.y_cls.item())
    
    if not X_test_list:
        ks = [1, 3, 5, 10, 50]
        results = {'knn_acc': 0, 'knn_precision': 0, 'knn_recall': 0, 'knn_f1': 0}
        for k_val in ks:
            results[f'knn_top{k_val}_acc'] = 0
        return results
    
    X_test = np.vstack(X_test_list)
    y_test = np.array(y_test_list)
    
    # Evaluate - calculate Top-K accuracies first to ensure consistency
    y_pred_proba = model.predict_proba(X_test)
    topk_results = calculate_topk_accuracy(y_test, y_pred_proba, model.classes_)
    
    # Use the same prediction method as Top-K for consistency
    y_pred_top1 = model.classes_[np.argsort(y_pred_proba, axis=1)[:, -1]]
    
    accuracy = topk_results['top1_acc']  # Use Top-K result for consistency
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_top1, average='weighted', zero_division=0)

    logging.info(f"k-NN (k={k}) baseline accuracy: {accuracy:.4f}")
    logging.info(f"k-NN (k={k}) baseline precision (weighted): {precision:.4f}")
    logging.info(f"k-NN (k={k}) baseline recall (weighted): {recall:.4f}")
    logging.info(f"k-NN (k={k}) baseline F1 (weighted): {f1:.4f}")
    for key, value in topk_results.items():
        logging.info(f"k-NN (k={k}) baseline {key.replace('_', ' ').title()}: {value:.4f}")

    results = {
        'knn_acc': accuracy,
        'knn_precision': precision,
        'knn_recall': recall,
        'knn_f1': f1,
    }
    # Add top-k results, renaming keys to be specific to this baseline
    for key_k, v in topk_results.items():
        results[f'knn_{key_k}'] = v
        
    return results

def random_baseline(train_dataset, test_dataset):
    """Random baseline - randomly guess from all possible classes found in the training set."""
    # This function uses python's `random` module, which should be seeded for reproducibility.
    from sklearn.metrics import precision_recall_fscore_support
    
    # Collect all unique training labels
    train_labels = set()
    for data in train_dataset:
        if hasattr(data, 'y_cls'):  # Classification task
            train_labels.add(data.y_cls.item())
    
    if not train_labels:
        logging.warning("No classification labels found in training set for random baseline.")
        return {'random_acc': 0, 'random_precision': 0, 'random_recall': 0, 'random_f1': 0}
    
    possible_classes = list(train_labels)
    num_classes = len(possible_classes)
    logging.info(f"Random baseline: found {num_classes} unique classes in the training set.")

    y_true = [data.y_cls.item() for data in test_dataset if hasattr(data, 'y_cls')]
    if not y_true:
        logging.warning("No classification labels found in the test set.")
        # Return a dictionary with all metrics to avoid downstream errors
        ks = [1, 3, 5, 10, 50]
        results = {'random_acc': 0, 'random_precision': 0, 'random_recall': 0, 'random_f1': 0}
        for k in ks:
            results[f'random_top{k}_acc'] = 0
        return results

    # --- Top-K and other metrics calculation ---
    ks = [1, 3, 5, 10, 50]
    topk_correct = {k: 0 for k in ks}
    y_pred_top1 = []
    num_samples = len(y_true)
    max_k = max(ks)
    
    # Ensure we don't try to sample more classes than exist
    sample_size = min(max_k, len(possible_classes))

    for i in range(num_samples):
        true_label = y_true[i]
        # Randomly sample `sample_size` unique classes as the top predictions for this sample
        top_n_predictions = random.sample(possible_classes, sample_size)
        y_pred_top1.append(top_n_predictions[0]) # For precision/recall/F1
        
        for k in ks:
            if k <= sample_size and true_label in top_n_predictions[:k]:
                topk_correct[k] += 1

    # --- Finalize metrics ---
    topk_accuracies = {f'random_top{k}_acc': correct / num_samples for k, correct in topk_correct.items()}
    accuracy = topk_accuracies['random_top1_acc']
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_top1, average='weighted', zero_division=0)
    
    # Log results
    logging.info(f"Random baseline accuracy: {accuracy:.4f}")
    logging.info(f"Random baseline precision (weighted): {precision:.4f}")
    logging.info(f"Random baseline recall (weighted): {recall:.4f}")
    logging.info(f"Random baseline F1 (weighted): {f1:.4f}")
    for key, value in topk_accuracies.items():
        if key != 'random_top1_acc': # Top-1 is already logged as 'accuracy'
             logging.info(f"Random baseline {key.replace('_', ' ').title()}: {value:.4f}")

    # --- Consolidate results into a single dictionary ---
    results = {
        'random_acc': accuracy,
        'random_precision': precision,
        'random_recall': recall,
        'random_f1': f1,
    }
    results.update(topk_accuracies)
    
    return results


def create_splits(n, root_dir, suffix):
    """Creates and saves train/val/test splits to disk."""
    logging.info(f"Creating new splits for suffix '{suffix}' and saving to {root_dir}.")
    n_train = max(1, int(0.7 * n))
    n_val   = max(1, int(0.1 * n))
    n_test  = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)
    perm = np.random.permutation(n)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train+n_val]
    test_idx  = perm[n_train+n_val:]
    np.save(os.path.join(root_dir, f"train_indices{suffix}.npy"), train_idx)
    np.save(os.path.join(root_dir, f"val_indices{suffix}.npy"),   val_idx)
    np.save(os.path.join(root_dir, f"test_indices{suffix}.npy"),  test_idx)
    logging.info(f"Splits saved for suffix '{suffix}'. Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx

def get_splits(args, full_dataset):
    """Loads existing splits or exits if they don't exist."""
    suffix = {"base": "", "RE": "_RE", "PARCON": "_PARCON", "FULCON": "_FULCON"}[args.aug_type]
    idx_file = os.path.join(args.root, f"train_indices{suffix}.npy")

    if not os.path.isfile(idx_file):
        logging.error(f"Split file not found: {idx_file}")
        logging.error("Please run the script with the --generate_splits flag first to create the data splits.")
        sys.exit(1)
    
    logging.info(f"Loading splits from {args.root} with suffix '{suffix}'")
    return load_indices_from_file(args.root, suffix)

def run_single_baseline_trial(args, train_dataset, test_dataset, device, node_feature_size):
    """
    Runs a single trial of baseline evaluations on a given data split.
    """
    # Determine if we're dealing with classification or regression
    sample_data = train_dataset[0]
    is_classification = hasattr(sample_data, 'y_cls')
    is_regression = hasattr(sample_data, 'y') and hasattr(sample_data, 'masked_node_idx')
    
    results = {}

    if args.baseline_type in ['classification', 'all'] and is_classification:
        logging.info("--- Running Classification Baselines ---")
        
        random_metrics = random_baseline(train_dataset, test_dataset)
        results.update(random_metrics)
        
        majority_metrics = majority_class_baseline(train_dataset, test_dataset)
        results.update(majority_metrics)
        
        logreg_metrics = logistic_regression_baseline(train_dataset, test_dataset)
        results.update(logreg_metrics)
        
        knn_metrics = knn_baseline(train_dataset, test_dataset, k=5)
        results.update(knn_metrics)

    if args.baseline_type in ['regression', 'all'] and is_regression:
        logging.info("--- Running Regression Baselines ---")
        avg_loss = average_embedding_baseline(test_dataset, device)
        results['average_embedding_mse'] = avg_loss

        if args.linear_baseline_type in ['pytorch', 'all']:
            pytorch_loss = pytorch_linear_regression_baseline(train_dataset, test_dataset, device, node_feature_size)
            results['pytorch_linear_mse'] = pytorch_loss

        if args.linear_baseline_type in ['sklearn', 'all']:
            sklearn_loss = sklearn_linear_regression_baseline(train_dataset, test_dataset)
            results['sklearn_linear_mse'] = sklearn_loss

    return results


def create_trial_summary_plots(all_trial_results, logging_dir):
    """Create summary plots showing distribution of metrics across trials."""
    import matplotlib.pyplot as plt
    
    metrics_to_plot = list(all_trial_results.keys())
    if not metrics_to_plot:
        return

    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots((num_metrics + 2) // 3, 3, figsize=(15, 5 * ((num_metrics + 2) // 3)))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        values = all_trial_results[metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        axes[i].hist(values, bins=min(10, len(values)), alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        axes[i].set_title(f'{metric.replace("_", " ").title()}\nMean: {mean_val:.4f} ± {std_val:.4f}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].remove()

    plt.tight_layout()
    plot_path = os.path.join(logging_dir, 'baseline_trial_summary_statistics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Trial summary plots saved to: {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate baselines on base or augmented datasets.")
    parser.add_argument('--root', type=str, required=True, help="Root directory for the dataset")
    parser.add_argument('--model_name', type=str, default='GATv2Classification')
    parser.add_argument('--valid_assemblies_path', type=str, default=None)
    parser.add_argument('--embeddings_path', type=str, required=True, help="Path to the node embeddings parquet file.")
    parser.add_argument('--contacts_path', type=str, default=None)
    parser.add_argument('--contact_embeddings_path', type=str, default=None, help='Path to contact embeddings parquet file')
    parser.add_argument('--edge_feature_type', type=str, default='scalar', choices=['scalar', 'embedding'], help='Type of edge features to use')
    parser.add_argument('--num_clusters', type=int, default=500, help='Number of clusters for classification task')
    parser.add_argument('--aug_type', type=str, default='base',
                        choices=['base', 'RE', 'PARCON', 'FULCON'],
                        help='Which version of the dataset to evaluate')
    parser.add_argument('--baseline_type', type=str, default='all',
                        choices=['regression', 'classification', 'all'],
                        help='Which type of baselines to run')
    parser.add_argument('--linear_baseline_type', type=str, default='all',
                        choices=['pytorch', 'sklearn', 'all'],
                        help='Which linear regression baseline to run.')
    parser.add_argument('--generate_splits', action='store_true', help='Generate new train/val/test splits')
    parser.add_argument('--visualize_aug',  action='store_true', help='Visualise one graph across ALL augmentations')
    parser.add_argument('--vis_index',      type=int, default=0,  help='Graph index to visualise')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of trials to run for statistical significance.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    args = parser.parse_args()

    # example use: python eval.py --root /mnt/drive/linkify-data/newContacts_filteringRound1Augmented/ --embeddings_path /mnt/drive/linkify-data/parts_pointcloud_embeddings_384_v2.parquet --baseline_type classification
    # example use with 10 trials: python eval.py --root /mnt/drive/linkify-data/newContacts_filteringRound1Augmented/ --embeddings_path /mnt/drive/linkify-data/parts_pointcloud_embeddings_384_v2.parquet --baseline classification --num_trials 10

    # --- Set up Logging ---
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "baseline_eval_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    run_prefix = f"multi_trial_{args.num_trials}_" if args.num_trials > 1 else ""
    log_filename = f"eval_{run_prefix}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    
    # Use a unique logging directory for each run to store plots and stats
    logging_dir = os.path.join(log_dir, os.path.splitext(log_filename)[0])
    os.makedirs(logging_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(logging_dir, 'eval.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logs and artifacts will be saved to: {logging_dir}")
    logging.info(f"Command: {' '.join(sys.argv)}")
    logging.info("========================================")
    logging.info("CONFIGURATION PARAMETERS:")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"  {arg}: {value}")
    logging.info("========================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set default paths if not provided
    valid_assemblies_path = args.valid_assemblies_path or os.path.join(args.root, 'valid_assemblies.parquet')
    embeddings_path = args.embeddings_path
    
    # Determine contact paths based on edge feature type
    contacts_path = None
    contact_embeddings_path = None
    edge_feature_mode = 'scalar'
    model_name = args.model_name

    if args.contact_embeddings_path is not None:
        # Only use edge features when contact embeddings are explicitly provided
        if args.edge_feature_type == 'scalar':
            contacts_path = args.contacts_path or os.path.join(args.root, 'contacts/contacts.parquet')
            edge_feature_mode = 'scalar'
        elif args.edge_feature_type == 'embedding':
            contact_embeddings_path = args.contact_embeddings_path
            edge_feature_mode = 'embedding'
    else:
        # No contact embeddings provided - use no edge attributes at all
        contacts_path = None
        contact_embeddings_path = None
        edge_feature_mode = 'scalar'  # This won't matter since no paths are provided
        
        # Switch to NoEdgeAttr model variant
        if model_name == 'GATv2Classification':
            model_name = 'GATv2ClassificationNoEdgeAttr'
        elif model_name == 'GATv2':
            model_name = 'GATv2NoEdgeAttr'
        
        logging.info(f"No contact embeddings provided. Switching to model: {model_name}")

    full_dataset = AssemblyGraphDataset(
        root=args.root,
        model_type=model_name,
        valid_assemblies_path=valid_assemblies_path,
        embeddings_path=embeddings_path,
        contacts_path=contacts_path,
        contact_embeddings_path=contact_embeddings_path,
        edge_feature_mode=edge_feature_mode,
        aug_type=args.aug_type,
        random_edges=False,
        num_clusters=args.num_clusters
    )

    logging.info(f"Dataset loaded with {len(full_dataset)} graphs")
    logging.info(f"Node feature size: {full_dataset.num_node_features}")
    
    # Log first few assembly IDs to check if ordering is consistent
    sample_ids = []
    for i in range(min(5, len(full_dataset))):
        sample_ids.append(str(full_dataset[i].assembly_id))
    logging.info(f"First 5 assembly IDs: {sample_ids}")

    # ------------------------------------------------------------------
    # Handle --generate_splits separately and exit
    # ------------------------------------------------------------------
    suffix_map = {"base": "", "RE": "_RE", "PARCON": "_PARCON", "FULCON": "_FULCON"}
    if args.generate_splits:
        suffix = suffix_map[args.aug_type]
        create_splits(len(full_dataset), args.root, suffix)
        logging.info("Splits generated successfully. You can now run the evaluation.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Optional visualisation of the SAME assembly across augmentations
    # ------------------------------------------------------------------
    if args.visualize_aug:
        target_idx = args.vis_index

        # --- load base graph and remember its assembly_id -------------
        base_ds = AssemblyGraphDataset(
            root=args.root,
            model_type=args.model_name,
            valid_assemblies_path=valid_assemblies_path,
            embeddings_path=embeddings_path,
            contacts_path=contacts_path,
            aug_type="base",
            random_edges=False
        )
        base_graph = base_ds[target_idx]
        aid = str(base_graph.assembly_id)

        def _save_vis(graph, tag):
            visualize_graph(graph)
            src = os.path.join(os.getcwd(), "logs", "visualize_graphs", "test.png")
            dst = os.path.join(os.getcwd(), "logs", "visualize_graphs", f"{aid}_{tag}.png")
            shutil.move(src, dst)

        logging.info(f"[Vis] Base graph (assembly_id={aid})")
        _save_vis(base_graph, "BASE")

        for tag in ["RE", "PARCON", "FULCON"]:
            ds = AssemblyGraphDataset(
                root=args.root,
                model_type=args.model_name,
                valid_assemblies_path=valid_assemblies_path,
                embeddings_path=embeddings_path,
                contacts_path=contacts_path,
                aug_type=tag,
                random_edges=False
            )
            # find the same assembly in this dataset
            g = next((d for d in ds if str(d.assembly_id) == aid), None)
            if g is None:
                logging.warning(f"[Vis] Assembly {aid} missing in {tag} dataset. Skipped")
                continue
            logging.info(f"[Vis] {tag} graph")
            _save_vis(g, tag)

        logging.info("[Vis] Graphs saved under logs/visualize_graphs/*.png")

    # ------------------------------------------------------------------
    # Main Evaluation Logic
    # ------------------------------------------------------------------
    
    # Load the fixed splits to be used for all trials
    train_indices, val_indices, test_indices = get_splits(args, full_dataset)
    train_dataset = full_dataset.index_select(train_indices)
    test_dataset  = full_dataset.index_select(test_indices)
    logging.info(f"Using fixed splits - Train: {len(train_dataset)}, Val: {len(val_indices)}, Test: {len(test_dataset)}")


    if args.num_trials > 1:
        logging.info(f"\n{'='*80}")
        logging.info(f"RUNNING {args.num_trials} TRIALS ON FIXED SPLIT")
        logging.info(f"{'='*80}")
        
        all_trial_results = defaultdict(list)

        for trial in range(args.num_trials):
            logging.info(f"--- Starting Trial {trial + 1}/{args.num_trials} ---")
            # Set seed for reproducibility for each trial
            if args.seed is not None:
                trial_seed = args.seed + trial
                np.random.seed(trial_seed)
                random.seed(trial_seed)
                torch.manual_seed(trial_seed)
                logging.info(f"Set random seed for trial to: {trial_seed}")
            
            trial_results = run_single_baseline_trial(args, train_dataset, test_dataset, device, full_dataset.num_node_features)
            
            for key, value in trial_results.items():
                all_trial_results[key].append(value)
            
            logging.info(f"Trial {trial + 1} Results: {json.dumps(trial_results, indent=2)}")

        # Calculate and report statistics
        logging.info(f"\n{'='*80}")
        logging.info(f"AGGREGATED RESULTS ACROSS {args.num_trials} TRIALS")
        logging.info(f"{'='*80}")
        
        stats_file = os.path.join(logging_dir, "aggregated_baseline_statistics.json")
        aggregated_stats = {}
        
        for metric, values in all_trial_results.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                aggregated_stats[metric] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'values': [float(v) for v in values]
                }
                logging.info(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        with open(stats_file, 'w') as f:
            json.dump(aggregated_stats, f, indent=4)
        logging.info(f"\nAggregated statistics saved to: {stats_file}")
        
        create_trial_summary_plots(all_trial_results, logging_dir)
        
    else: # Single trial run
        logging.info("--- Running Single Evaluation Trial ---")
        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            logging.info(f"Set random seed to: {args.seed}")
            
        results = run_single_baseline_trial(args, train_dataset, test_dataset, device, full_dataset.num_node_features)

        logging.info("\n=== BASELINE SUMMARY ===")
        for metric, value in results.items():
            logging.info(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        logging.info("========================")