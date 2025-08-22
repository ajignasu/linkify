import os
import sys
import json
import torch
import joblib
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data, DataLoader
from model import GATv2Classification, GATv2ClassificationNoEdgeAttr, GATClassification
from data_generation.graph_data import AssemblyGraphDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import make_directory, save_config, plot_train_val_loss, plot_train_val_accuracy, get_dataloader, \
    log_loss, create_log_loss_file, compute_distance_matrix, visualize_topk_parts, visualize_graph
import logging
from datetime import datetime
import random
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import warnings
import heapq

VISUALIZATION_CANDIDATE_POOL_SIZE = 5  # multiplier for the number of candidates to store

def extract_part_uuid(compound_uuid):
    """
    Input format:  occurrenceID_partUUID
    Return        : partUUID  (token AFTER the first underscore)
    """
    return str(compound_uuid).split('_', 1)[-1]

def build_part_uuid_to_assembly_id_map(root, embeddings_df):
    """Builds a map from part UUID to assembly ID."""
    part_uuid_to_assembly_id = {}
    
    # Method 1: Use explicit 'assembly_id' column if available
    if 'assembly_id' in embeddings_df.columns:
        mapping_df = embeddings_df.dropna(subset=['uuid', 'assembly_id'])
        part_uuids = mapping_df['uuid'].apply(extract_part_uuid)
        part_uuid_to_assembly_id.update(dict(zip(part_uuids, mapping_df['assembly_id'].astype(str))))
    
    # Method 2: Fallback to scanning directories
    unmapped_uuids = set(embeddings_df['uuid']) - set(part_uuid_to_assembly_id.keys())
    if unmapped_uuids:
        print(f"{len(unmapped_uuids)} part UUIDs still unmapped. Scanning directories...")
        for assembly_id in tqdm(os.listdir(root), desc="Scanning assemblies"):
            assembly_dir = os.path.join(root, assembly_id)
            if os.path.isdir(assembly_dir):
                for fname in os.listdir(assembly_dir):
                    if fname.endswith('.png'):
                        part_uuid = fname[:-4]
                        if part_uuid in unmapped_uuids:
                            part_uuid_to_assembly_id[part_uuid] = assembly_id
    
    return part_uuid_to_assembly_id

def apply_augmentations(data, mask_prob, edge_dropout_p, feature_noise):
    """Apply data augmentations during training"""
    # Node feature masking
    if mask_prob > 0:
        mask = torch.rand(data.x.size(0)) < mask_prob
        data.x[mask] = 0
    
    # Edge dropout
    if edge_dropout_p > 0 and data.edge_index.size(1) > 0:
        edge_mask = torch.rand(data.edge_index.size(1)) > edge_dropout_p
        data.edge_index = data.edge_index[:, edge_mask]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr[edge_mask]
    
    # Feature noise
    if feature_noise > 0:
        noise = torch.randn_like(data.x) * feature_noise
        data.x = data.x + noise
    
    return data

def train(model, dataloader, optimizer, criterion, device, mask_node=False, mask_prob=0.0, edge_dropout_p=0.0, feature_noise=0.0, clip_grad=0.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for data in dataloader:
        data = data.to(device)
        data = apply_augmentations(data, mask_prob, edge_dropout_p, feature_noise)
        # visualize_graph(data)
        optimizer.zero_grad()
        output, _ = model(data)
        masked_node_indices = data.ptr[:-1] + data.masked_node_idx.squeeze(-1)
        masked_output = output[masked_node_indices]
        target_labels = data.y_cls.squeeze(-1)
        if target_labels.dim() == 0:
            target_labels = target_labels.unsqueeze(0)
        loss = criterion(masked_output, target_labels)
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item()
        preds = masked_output.argmax(dim=1)
        correct += (preds == target_labels).sum().item()
        total += target_labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return total_loss / len(dataloader), accuracy


def test(model, dataloader, criterion, device, cluster_centers, k, data_root, part_uuid_to_assembly_id, logging_dir, uuid_to_embedding, cluster_to_parts=None, visualize_topk=False, visualize_best_predictions=0, visualize_worst_predictions=0):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    correct_top10 = 0
    correct_top20 = 0
    correct_top50 = 0
    correct_top100 = 0
    total = 0
    all_preds = []
    all_labels = []
    top_correct_predictions = []
    top_incorrect_predictions = []

    if cluster_to_parts is None and visualize_topk:
        print("WARNING: cluster_to_parts not provided, falling back to embedding-based visualization")
        valid_uuids = list(uuid_to_embedding.keys())
        valid_embeddings = [uuid_to_embedding[u] for u in valid_uuids]
        if len(valid_embeddings) == 0:
            raise ValueError("uuid_to_embedding is empty or contains no valid embeddings")
        valid_embeddings_array = np.vstack(valid_embeddings).astype(np.float32)  # (N, D)
        all_embeddings_norm = F.normalize(torch.tensor(valid_embeddings_array, dtype=torch.float32, device=device), p=2, dim=1)

    samples_to_visualize = random.sample(range(len(dataloader.dataset)), min(3, len(dataloader.dataset)))
    visualizations_created = 0
    max_visualizations = len(samples_to_visualize) * len(k)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(device)
            output, node_embeddings = model(data)

            # masked node evaluation
            masked_node_indices = data.ptr[:-1] + data.masked_node_idx.squeeze(-1)
            masked_output = output[masked_node_indices]
            cluster_labels = data.y_cls.squeeze(-1)
            if cluster_labels.dim() == 0:
                cluster_labels = cluster_labels.unsqueeze(0)

            loss = criterion(masked_output, cluster_labels)
            total_loss += loss.item()

            # confidence for top prediction
            softmax_output = F.softmax(masked_output, dim=1)
            top1_confidence, preds = torch.max(softmax_output, dim=1)

            # balanced accuracy
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(cluster_labels.cpu().numpy())

            # top-k compute
            _, top_preds = torch.topk(masked_output, k=max(k), dim=1)
            true_labels_reshaped = cluster_labels.view(-1, 1)
            correct = top_preds == true_labels_reshaped

            correct_top1 += correct[:, :1].sum().item()
            correct_top3 += correct[:, :3].sum().item()
            correct_top5 += correct[:, :5].sum().item()
            if max(k) >= 10:
                correct_top10 += correct[:, :10].sum().item()
            if max(k) >= 20:
                correct_top20 += correct[:, :20].sum().item()
            if max(k) >= 50:
                correct_top50 += correct[:, :50].sum().item()
            if max(k) >= 100:
                correct_top100 += correct[:, :100].sum().item()
            total += cluster_labels.size(0)

            # track best preds
            if visualize_best_predictions > 0:
                pool_size = visualize_best_predictions * VISUALIZATION_CANDIDATE_POOL_SIZE
                correct_mask = (preds == cluster_labels)
                for j in range(correct_mask.size(0)):
                    if correct_mask[j]:
                        confidence = top1_confidence[j].item()
                        sample_key = (i, j)  # numeric tiebreaker
                        
                        # Use a min-heap to keep track of a pool of top N predictions
                        if len(top_correct_predictions) < pool_size:
                            heapq.heappush(top_correct_predictions, (confidence, sample_key, data.get_example(j), top_preds[j].cpu().numpy()))
                        else:
                            heapq.heappushpop(top_correct_predictions, (confidence, sample_key, data.get_example(j), top_preds[j].cpu().numpy()))

            # track bad preds
            if visualize_worst_predictions > 0:
                pool_size = visualize_worst_predictions * VISUALIZATION_CANDIDATE_POOL_SIZE
                incorrect_mask = (preds != cluster_labels)
                for j in range(incorrect_mask.size(0)):
                    if incorrect_mask[j]:
                        confidence = top1_confidence[j].item()
                        sample_key = (i, j)  # numeric tiebreaker
                        
                        # Use a min-heap to keep track of a pool of top N most confident incorrect predictions
                        if len(top_incorrect_predictions) < pool_size:
                            heapq.heappush(top_incorrect_predictions, (confidence, sample_key, data.get_example(j), top_preds[j].cpu().numpy()))
                        else:
                            heapq.heappushpop(top_incorrect_predictions, (confidence, sample_key, data.get_example(j), top_preds[j].cpu().numpy()))

            # visualize
            if visualize_topk and i in samples_to_visualize:
                for j in range(len(data.y_cls)):
                    if visualizations_created >= max_visualizations:
                        break
                    single_graph_data = data.get_example(j)
                    test_assembly_id = single_graph_data.assembly_id
                    test_masked_idx = int(single_graph_data.masked_node_idx.item())
                    if test_masked_idx < len(single_graph_data.node_uuids):
                        test_part_uuid = extract_part_uuid(single_graph_data.node_uuids[test_masked_idx])
                    else:
                        continue
                    reference_part = (test_assembly_id, test_part_uuid)
                    sample_predictions = top_preds[j]
                    for kk in k:
                        if cluster_to_parts is not None:
                            topk_cluster_ids = sample_predictions[:kk].cpu().numpy()
                            topk_pairs = []
                            parts_per_cluster = max(1, min(3, 15 // kk))
                            cluster_info = {}
                            for cluster_id in topk_cluster_ids:
                                cluster_id = int(cluster_id)
                                if cluster_id in cluster_to_parts:
                                    cluster_parts = cluster_to_parts[cluster_id]
                                    # randomly sample parts from this cluster
                                    sampled_parts = random.sample(cluster_parts, min(parts_per_cluster, len(cluster_parts)))
                                    for part_uuid in sampled_parts:
                                        if part_uuid in part_uuid_to_assembly_id:
                                            assembly_id = part_uuid_to_assembly_id[part_uuid]
                                            topk_pairs.append((assembly_id, part_uuid))
                                            cluster_info[part_uuid] = cluster_id                            
                            true_cluster_id = cluster_labels[j].item()
                            cluster_info[test_part_uuid] = true_cluster_id
                            if logging_dir is not None:
                                os.makedirs(logging_dir, exist_ok=True)
                                save_path = os.path.join(logging_dir, f"cluster_based_sample{i}_top{kk}_{test_assembly_id}_{test_part_uuid}.png")
                            else:
                                save_path = f"cluster_based_sample{i}_top{kk}_{test_assembly_id}_{test_part_uuid}.png"
                            print(f"Visualizing {len(topk_pairs)} parts from top-{kk} clusters: {topk_cluster_ids.tolist()}")
                            print(f"True cluster: {true_cluster_id}, Predicted clusters: {topk_cluster_ids.tolist()}")
                            
                        else:
                            global_masked_node_idx = data.ptr[j] + test_masked_idx
                            if node_embeddings.shape[1] == 256:
                                pred_embedding = node_embeddings[global_masked_node_idx].unsqueeze(0)
                            else:
                                center_np = cluster_centers[preds[j].item()].astype(np.float32)
                                pred_embedding = torch.tensor(center_np, dtype=torch.float32, device=device).unsqueeze(0)
                            pred_embedding_norm = F.normalize(pred_embedding, p=2, dim=1)
                            cos_sim = torch.mm(all_embeddings_norm, pred_embedding_norm.t()).squeeze(1)
                            topk_scores, topk_indices = torch.topk(cos_sim, k=kk, largest=True)
                            topk_pairs = []
                            cluster_info = None
                            if part_uuid_to_assembly_id:
                                for idx in topk_indices:
                                    retrieved_uuid = valid_uuids[idx.item()]
                                    retrieved_assembly = part_uuid_to_assembly_id.get(retrieved_uuid, "UNKNOWN_ASSEMBLY")
                                    
                                    if retrieved_assembly != "UNKNOWN_ASSEMBLY":
                                        topk_pairs.append((retrieved_assembly, retrieved_uuid))
                            if logging_dir is not None:
                                os.makedirs(logging_dir, exist_ok=True)
                                save_path = os.path.join(logging_dir, f"embedding_based_sample{i}_top{kk}_{test_assembly_id}_{test_part_uuid}.png")
                            else:
                                save_path = f"embedding_based_sample{i}_top{kk}_{test_assembly_id}_{test_part_uuid}.png"
                        if topk_pairs:
                            visualize_topk_parts(topk_pairs, data_root, images_subdir="", cols=min(kk, 10), save_path=save_path, reference_part=reference_part, cluster_info=cluster_info)
                        visualizations_created += 1

    if visualize_best_predictions > 0 and top_correct_predictions:
        print(f"\nVisualizing the top {visualize_best_predictions} correct predictions from a pool of {len(top_correct_predictions)} candidates...")
        num_to_visualize = min(visualize_best_predictions, len(top_correct_predictions))
        predictions_to_visualize = random.sample(top_correct_predictions, num_to_visualize)
        predictions_to_visualize.sort(key=lambda x: x[0], reverse=True)

        for idx, (confidence, sample_key, single_graph_data, sample_predictions) in enumerate(predictions_to_visualize):
            test_assembly_id = single_graph_data.assembly_id
            test_masked_idx = int(single_graph_data.masked_node_idx.item())
            if test_masked_idx < len(single_graph_data.node_uuids):
                test_part_uuid = extract_part_uuid(single_graph_data.node_uuids[test_masked_idx])
            else:
                continue
            reference_part = (test_assembly_id, test_part_uuid)
            true_cluster_id = single_graph_data.y_cls.item()
            kk = 5
            topk_cluster_ids = sample_predictions[:kk]
            topk_pairs = []
            cluster_info = {test_part_uuid: true_cluster_id}
            if cluster_to_parts:
                for cluster_id in topk_cluster_ids:
                    cluster_id = int(cluster_id)
                    if cluster_id in cluster_to_parts:
                        cluster_parts = cluster_to_parts[cluster_id]
                        sampled_parts = random.sample(cluster_parts, min(3, len(cluster_parts)))
                        for part_uuid in sampled_parts:
                            if part_uuid in part_uuid_to_assembly_id:
                                assembly_id = part_uuid_to_assembly_id[part_uuid]
                                if (assembly_id, part_uuid) not in topk_pairs:
                                    topk_pairs.append((assembly_id, part_uuid))
                                    cluster_info[part_uuid] = cluster_id

            save_path = os.path.join(logging_dir, f"best_prediction_{idx+1}_conf{confidence:.4f}_{test_assembly_id}_{test_part_uuid}.png")
            print(f"  {idx+1}) Confidence: {confidence:.4f}, True Cluster: {true_cluster_id}, Predicted Clusters: {topk_cluster_ids.tolist()}")
            if topk_pairs:
                visualize_topk_parts(topk_pairs, data_root, images_subdir="", cols=min(kk, 5), save_path=save_path, reference_part=reference_part, cluster_info=cluster_info)
            else:
                print(f"    - Could not generate visualization for best prediction {idx+1} (no parts found for predicted clusters).")

    if visualize_worst_predictions > 0 and top_incorrect_predictions:
        print(f"\nVisualizing the top {visualize_worst_predictions} most confident incorrect predictions from a pool of {len(top_incorrect_predictions)} candidates...")
        num_to_visualize = min(visualize_worst_predictions, len(top_incorrect_predictions))
        predictions_to_visualize = random.sample(top_incorrect_predictions, num_to_visualize)
        predictions_to_visualize.sort(key=lambda x: x[0], reverse=True)

        for idx, (confidence, sample_key, single_graph_data, sample_predictions) in enumerate(predictions_to_visualize):
            test_assembly_id = single_graph_data.assembly_id
            test_masked_idx = int(single_graph_data.masked_node_idx.item())
            if test_masked_idx < len(single_graph_data.node_uuids):
                test_part_uuid = extract_part_uuid(single_graph_data.node_uuids[test_masked_idx])
            else:
                continue
            reference_part = (test_assembly_id, test_part_uuid)
            true_cluster_id = single_graph_data.y_cls.item()
            predicted_cluster_id = sample_predictions[0]
            kk = 5
            topk_cluster_ids = sample_predictions[:kk]
            topk_pairs = []
            cluster_info = {test_part_uuid: f"True: {true_cluster_id}"}
            if cluster_to_parts:
                for cluster_id in topk_cluster_ids:
                    cluster_id = int(cluster_id)
                    if cluster_id in cluster_to_parts:
                        cluster_parts = cluster_to_parts[cluster_id]
                        sampled_parts = random.sample(cluster_parts, min(3, len(cluster_parts)))
                        for part_uuid in sampled_parts:
                            if part_uuid in part_uuid_to_assembly_id:
                                assembly_id = part_uuid_to_assembly_id[part_uuid]
                                if (assembly_id, part_uuid) not in topk_pairs:
                                    topk_pairs.append((assembly_id, part_uuid))
                                    cluster_info[part_uuid] = f"Pred: {cluster_id}"
            
            save_path = os.path.join(logging_dir, f"worst_prediction_{idx+1}_conf{confidence:.4f}_{test_assembly_id}_{test_part_uuid}.png")
            print(f"  {idx+1}) Confidence: {confidence:.4f}, True Cluster: {true_cluster_id}, Predicted Cluster: {predicted_cluster_id}")
            if topk_pairs:
                visualize_topk_parts(topk_pairs, data_root, images_subdir="", cols=min(kk, 5), save_path=save_path, reference_part=reference_part, cluster_info=cluster_info)
            else:
                print(f"    - Could not generate visualization for worst prediction {idx+1} (no parts found for predicted clusters).")

    accuracy_top1 = correct_top1 / total if total > 0 else 0.0
    accuracy_top3 = correct_top3 / total if total > 0 else 0.0
    accuracy_top5 = correct_top5 / total if total > 0 else 0.0
    accuracy_top10 = correct_top10 / total if total > 0 else 0.0
    accuracy_top20 = correct_top20 / total if total > 0 else 0.0
    accuracy_top50 = correct_top50 / total if total > 0 else 0.0
    accuracy_top100 = correct_top100 / total if total > 0 else 0.0
    
    if total > 0:
        per_class_recalls = np.zeros(cluster_centers.shape[0])
        unique_labels = np.unique(all_labels)
        for class_id in unique_labels:
            class_mask = np.array(all_labels) == class_id
            class_preds = np.array(all_preds)[class_mask]
            class_correct = np.sum(class_preds == class_id)
            per_class_recalls[class_id] = class_correct / np.sum(class_mask)
        balanced_acc = np.mean(per_class_recalls)

        # calculate precision, recall, f1-score
        labels = np.unique(np.concatenate((all_labels, all_preds)))
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0, labels=labels
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0, labels=labels
        )
    else:
        balanced_acc = 0.0
        precision_macro, recall_macro, f1_macro = 0.0, 0.0, 0.0
        precision_weighted, recall_weighted, f1_weighted = 0.0, 0.0, 0.0
    
    return total_loss / len(dataloader), accuracy_top1, accuracy_top3, accuracy_top5, accuracy_top10, accuracy_top20, accuracy_top50, accuracy_top100, balanced_acc, precision_macro, recall_macro, f1_macro, precision_weighted, recall_weighted, f1_weighted


def validate(model, dataloader, criterion, device, num_clusters=1000):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    correct_top10 = 0
    correct_top20 = 0
    correct_top50 = 0
    correct_top100 = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output, _ = model(data)

            # masked node evaluation
            masked_node_indices = data.ptr[:-1] + data.masked_node_idx.squeeze(-1)
            masked_output = output[masked_node_indices]
            cluster_labels = data.y_cls.squeeze(-1)
            if cluster_labels.dim() == 0:
                cluster_labels = cluster_labels.unsqueeze(0)

            loss = criterion(masked_output, cluster_labels)
            total_loss += loss.item()

            # balanced accuracy
            preds = masked_output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(cluster_labels.cpu().numpy())

            # top-k compute
            _, top_preds = torch.topk(masked_output, k=100, dim=1)

            true_labels_reshaped = cluster_labels.view(-1, 1)
            correct = top_preds == true_labels_reshaped

            # get hits
            correct_top1 += correct[:, :1].sum().item()
            correct_top3 += correct[:, :3].sum().item()
            correct_top5 += correct[:, :5].sum().item()
            correct_top10 += correct[:, :10].sum().item()
            correct_top20 += correct[:, :20].sum().item()
            correct_top50 += correct[:, :50].sum().item()
            correct_top100 += correct[:, :100].sum().item()
    
            total += cluster_labels.size(0)
    
    accuracy_top1 = correct_top1 / total if total > 0 else 0.0
    accuracy_top3 = correct_top3 / total if total > 0 else 0.0
    accuracy_top5 = correct_top5 / total if total > 0 else 0.0
    accuracy_top10 = correct_top10 / total if total > 0 else 0.0
    accuracy_top20 = correct_top20 / total if total > 0 else 0.0
    accuracy_top50 = correct_top50 / total if total > 0 else 0.0
    accuracy_top100 = correct_top100 / total if total > 0 else 0.0
    
    # fixed balanced accuracy calculation
    if total > 0:
        # calculate per-class recall for all possible classes (0 to num_clusters-1)
        per_class_recalls = np.zeros(num_clusters)
        unique_labels = np.unique(all_labels)
        for class_id in unique_labels:
            class_mask = np.array(all_labels) == class_id
            class_preds = np.array(all_preds)[class_mask]
            class_correct = np.sum(class_preds == class_id)
            per_class_recalls[class_id] = class_correct / np.sum(class_mask)
        
        balanced_acc = np.mean(per_class_recalls)
        labels = np.unique(np.concatenate((all_labels, all_preds)))
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0, labels=labels
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0, labels=labels
        )
    else:
        balanced_acc = 0.0
        precision_macro, recall_macro, f1_macro = 0.0, 0.0, 0.0
        precision_weighted, recall_weighted, f1_weighted = 0.0, 0.0, 0.0
    
    return total_loss / len(dataloader), accuracy_top1, accuracy_top3, accuracy_top5, accuracy_top10, accuracy_top20, accuracy_top50, accuracy_top100, balanced_acc, precision_macro, recall_macro, f1_macro, precision_weighted, recall_weighted, f1_weighted

def main():
    parser = argparse.ArgumentParser(description='Train a GNN for classification on assembly graphs.')
    parser.add_argument('--root', type=str, required=True, help='Root directory for the dataset')
    parser.add_argument('--embeddings_path', type=str, required=True, help='Path to the embeddings parquet file (required)')
    parser.add_argument('--model', type=str, default='GATv2Classification', choices=['GATv2Classification', 'GATv2ClassificationNoEdgeAttr', 'GATClassification'], help='Model architecture to use')
    parser.add_argument('--node_attr_size', type=int, default=256, help='Size of node attributes')
    parser.add_argument('--edge_feature_type', type=str, default='scalar', choices=['scalar','embedding'], help='Type of edge features to use')
    parser.add_argument('--contact_embeddings_path', type=str, default=None, help='Path to contact embeddings parquet (required if edge_feature_type=embedding)')
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of hidden layers')
    parser.add_argument('--masknode', action='store_true', help='Whether to mask a node during training')
    parser.add_argument('--num_clusters', type=int, default=1000, help='Number of clusters for classification')
    parser.add_argument('--layers', type=int, default=2, help='Number of model layers')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer')
    parser.add_argument('--schedule', type=str, default='constant', choices=['constant', 'cosine', 'cosine_w10', 'step'], help='Learning rate schedule')
    parser.add_argument('--gat_heads', type=int, default=1, help='Number of GAT attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--attn_drop', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--clip_grad', type=float, default=0.0, help='Gradient clipping threshold (0 = no clipping)')
    parser.add_argument('--mask_prob', type=float, default=0.0, help='Probability of masking node features during training')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor (0.0 = no smoothing)')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leaky_relu', 'elu', 'gelu'], help='Activation function')
    parser.add_argument('--residual', type=str, default='none', choices=['none', 'add', 'concat'], help='Residual connection type')
    parser.add_argument('--edge_dropout_p', type=float, default=0.0, help='Edge dropout probability')
    parser.add_argument('--feature_noise', type=float, default=0.0, help='Standard deviation of noise to add to features during training')
    parser.add_argument('--aug_type', type=str, default='base', 
                    choices=['base', 'RE', 'MST', 'PARCON', 'FULCON'], 
                    help="Type of graph augmentation to use. RE=Random Edges, MST=Minimum Spanning Tree, PARCON=Edge Drop, FULCON=Fully Connected.")
    parser.add_argument('--aug_fraction', type=float, default=0.1, help='Fraction of edges to drop for PARCON augmentation.')
    parser.add_argument('--logdir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/node_classification'), help='Directory to save logs and model checkpoints')
    parser.add_argument('--savefreq', type=int, default=5, help='Frequency of saving model checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate a pre-trained model without training')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model for evaluation (required if --eval_only is used)')
    parser.add_argument('--visualize_topk', action='store_true', help='Enable top-k visualization')
    parser.add_argument('--visualize_best_predictions', type=int, default=0, help='Number of top correct predictions to visualize from the test set. (default: 0)')
    parser.add_argument('--visualize_worst_predictions', type=int, default=0, help='Number of top incorrect predictions to visualize from the test set. (default: 0)')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of trials for statistical significance')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--early_stopping_patience', type=int, default=0, help='Patience for early stopping (0 to disable).')
    args = parser.parse_args()

    if args.eval_only:
        if not args.model_path or not os.path.exists(args.model_path):
            raise ValueError("A valid --model_path is required for --eval_only mode.")
        
        # save evaluation results in a sub-directory of the model's folder
        model_dir = os.path.dirname(args.model_path)
        eval_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        logging_dir = os.path.join(model_dir, f"eval_{eval_timestamp}")
        os.makedirs(logging_dir, exist_ok=True)
        print(f"Evaluation outputs will be saved in: {logging_dir}")
        print(f"Model path: {args.model_path}")
        print(f"Model directory: {model_dir}")
    else:
        model_info = f"{args.model}_h{args.hidden_size}_l{args.layers}_hd{args.gat_heads}"
        if args.aug_type != 'base':
            if args.aug_type == 'PARCON':
                model_info += f"_aug_{args.aug_type}{int(args.aug_fraction * 100)}"
            else:
                model_info += f"_aug_{args.aug_type}"
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_prefix = "multi_trial_" if args.num_trials > 1 else ""
        run_name = f"{run_prefix}{timestamp}_{model_info}"
        logging_dir = os.path.join(args.logdir, run_name)
        os.makedirs(logging_dir, exist_ok=True)

    processed_dir = os.path.join(args.root, 'processed')

    if not os.path.exists(processed_dir):
        print(f"Info: The specified processed directory does not exist: {processed_dir}")
        print("The dataset generation process will create it.")
    if args.num_trials > 1:
        print(f"\n{'='*80}")
        print(f"RUNNING {args.num_trials} TRIALS")
        print(f"{'='*80}")
        all_trial_results = {
            'val_acc_top1': [], 'val_acc_top3': [], 'val_acc_top5': [], 'val_acc_top10': [], 'val_acc_top20': [], 
            'val_acc_top50': [], 'val_acc_top100': [], 'val_balanced_acc': [],
            'val_f1_macro': [], 'val_f1_weighted': [],
            'val_precision_macro': [], 'val_recall_macro': [],
            'val_precision_weighted': [], 'val_recall_weighted': [],
            'test_acc_top1': [], 'test_acc_top3': [], 'test_acc_top5': [], 'test_acc_top10': [], 'test_acc_top20': [], 
            'test_acc_top50': [], 'test_acc_top100': [], 'test_balanced_acc': [],
            'test_f1_macro': [], 'test_f1_weighted': [],
            'test_precision_macro': [], 'test_recall_macro': [],
            'test_precision_weighted': [], 'test_recall_weighted': [],
            'final_train_loss': [], 'final_train_acc': [], 'final_val_loss': [], 'test_loss': []
        }
        
        for trial in range(args.num_trials):
            trial_logdir = os.path.join(logging_dir, f"trial_{trial + 1}")
            os.makedirs(trial_logdir, exist_ok=True)
            log_path = os.path.join(trial_logdir, 'trial.log')
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler(sys.stdout)
                ],
                force=True)
            logging.info("========================================")
            logging.info(f"TRIAL {trial + 1}/{args.num_trials}")
            logging.info("Log file will be saved to: %s", log_path)
            logging.info("RUN ARGUMENTS:")
            for arg, value in sorted(vars(args).items()):
                logging.info("  %s: %s", arg, value)
            logging.info("========================================")

            # seed for reproducibility if specified
            if args.seed is not None:
                trial_seed = args.seed + trial
                torch.manual_seed(trial_seed)
                np.random.seed(trial_seed)
                random.seed(trial_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(trial_seed)
                logging.info(f"Set random seed to: {trial_seed}")
            
            trial_visualize = args.visualize_topk and (trial == 0)
            if trial == 0 and args.visualize_topk:
                logging.info("Visualization enabled for first trial only")
            
            # run single trial
            trial_results = run_single_trial(args, trial_logdir, trial_visualize)            
            for key, value in trial_results.items():
                if key in all_trial_results:
                    all_trial_results[key].append(value)
            print(f"Trial {trial + 1} completed:")
            print(f"  Val Balanced Acc: {trial_results['val_balanced_acc']:.4f}")
            print(f"  Test Balanced Acc: {trial_results['test_balanced_acc']:.4f}")
            print(f"  Test Top-1 Acc: {trial_results['test_acc_top1']:.4f}")
        
        print(f"\n{'='*80}")
        print(f"AGGREGATED RESULTS ACROSS {args.num_trials} TRIALS")
        print(f"{'='*80}")
        
        stats_file = os.path.join(logging_dir, "aggregated_statistics.json")
        aggregated_stats = {}
        for metric, values in all_trial_results.items():
            if values:  # Only process if we have values
                mean_val = np.mean(values)
                std_val = np.std(values)
                aggregated_stats[metric] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'values': [float(v) for v in values]
                }
                print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        with open(stats_file, 'w') as f:
            json.dump(aggregated_stats, f, indent=4)
        print(f"\nAggregated statistics saved to: {stats_file}")
        if not args.eval_only:
            create_trial_summary_plots(all_trial_results, logging_dir)
        
    else:
        log_path = os.path.join(logging_dir, 'train.log')
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Log file will be saved to: %s", log_path)
        logging.info("========================================")
        logging.info("RUN ARGUMENTS:")
        for arg, value in sorted(vars(args).items()):
            logging.info("  %s: %s", arg, value)
        logging.info("========================================")

        print("Running single trial...")
        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
            print(f"Set random seed to: {args.seed}")
        
        run_single_trial(args, logging_dir, args.visualize_topk)

def run_single_trial(args, logging_dir, visualize_topk):
    """Run a single training/evaluation trial."""
    processed_dir = os.path.join(args.root, 'processed')
    embedding_suffix = os.path.basename(args.embeddings_path).replace('.parquet', '')
    command_args = ' '.join(sys.argv)
    log_loss(os.path.join(logging_dir, 'loss_log.json'), 
             epoch=None, 
             command=command_args,
             script_start_time=datetime.now().isoformat())
    wandb_run = getattr(args, 'wandb_run', None)
    print(f"[WANDB DEBUG] wandb_run object in run_single_trial: {wandb_run is not None}")
    config_path = os.path.join(logging_dir, 'config.json')

    contacts_path = None
    contact_embeddings_path = None
    edge_feature_mode = 'none'

    if 'NoEdgeAttr' not in args.model:
        if args.edge_feature_type == 'scalar':
            contacts_path = os.path.join(args.root, 'contacts/contacts.parquet')
            edge_feature_mode = 'scalar'
        elif args.edge_feature_type == 'embedding':
            contact_embeddings_path = args.contact_embeddings_path
            edge_feature_mode = 'embedding'
            if contact_embeddings_path is None:
                raise ValueError('--contact_embeddings_path is required when edge_feature_type="embedding"')
    
    dataset = AssemblyGraphDataset(
        root=args.root,
        model_type=args.model,
        valid_assemblies_path=os.path.join(args.root, 'valid_assemblies.parquet'),
        embeddings_path=args.embeddings_path,
        contacts_path=contacts_path,
        contact_embeddings_path=contact_embeddings_path,
        edge_feature_mode=edge_feature_mode,
        num_clusters=args.num_clusters,
        aug_type=args.aug_type,
        aug_fraction=args.aug_fraction
    )

    edge_attr_size = dataset.num_edge_features
    args.edge_attr_size = edge_attr_size
    node_attr_size = dataset.num_node_features
    args.node_attr_size = node_attr_size
    print(f"Total graphs in dataset: {len(dataset)}")

    train_loader, val_loader, test_loader = get_dataloader(dataset, batch_size=args.batchsize, shuffle=True)

    print("Calculating class weights...")
    all_train_labels = []
    if len(train_loader.dataset) > 0:
        for data in tqdm(train_loader, desc="Extracting labels for class weights"):
            all_train_labels.extend(np.atleast_1d(data.y_cls.squeeze(-1).cpu().numpy()))
    
    class_weights_tensor = torch.ones(args.num_clusters, dtype=torch.float)

    if all_train_labels:
        unique_labels = np.unique(all_train_labels)
        balanced_weights = compute_class_weight('balanced', classes=unique_labels, y=all_train_labels)
        class_weights_tensor[unique_labels] = torch.tensor(balanced_weights, dtype=torch.float)
        print(f"Class weights computed for {len(unique_labels)} classes out of {args.num_clusters} total classes.")
    else:
        print("Warning: No labels found in the training set. Using default class weight of 1.0 for all classes.")

    class_weights_tensor = class_weights_tensor.to(args.device)
    devices = torch.device(args.device)
    
    print(f"Creating model with parameters:")
    print(f"  node_attr_size: {args.node_attr_size}")
    print(f"  edge_attr_size: {args.edge_attr_size}")
    print(f"  hidden_size: {args.hidden_size}")
    print(f"  num_clusters: {args.num_clusters}")
    print(f"  layers: {args.layers}")
    print(f"  gat_heads: {args.gat_heads}")
    print(f"  dropout: {args.dropout}")
    print(f"  attn_drop: {args.attn_drop}")
    print(f"  activation: {args.activation}")
    print(f"  residual: {args.residual}")
    
    model = None

    # load configuration and init model
    if args.eval_only:
        if not args.model_path or not os.path.exists(args.model_path):
            raise FileNotFoundError("--eval_only requires a valid --model_path.")
        
        # Load Configuration from the model's directory
        model_dir = os.path.dirname(args.model_path)
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            config_path_fallback = os.path.join(os.path.dirname(model_dir), 'config.json')
            if os.path.exists(config_path_fallback):
                config_path = config_path_fallback
            else:
                raise FileNotFoundError(f"config.json not found in {model_dir} or parent directory.")

        print(f"Loading model configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Loaded config: {config}")
        model_config = config.get('model_config', {})
        print(f"Model config section: {model_config}")
        param_map = {
            'num_model_layers': 'layers',
            'gat_heads': 'gat_heads', 
            'hidden_size': 'hidden_size',
            'dropout': 'dropout',
            'attn_drop': 'attn_drop',
            'activation': 'activation',
            'residual': 'residual',
            'model': 'model'
        }
        print("Mapping configuration parameters:")
        for config_key, arg_key in param_map.items():
            if config_key in model_config:
                old_value = getattr(args, arg_key, 'NOT_SET')
                new_value = model_config[config_key]
                setattr(args, arg_key, new_value)
                print(f"  {arg_key}: {old_value} -> {new_value}")
            else:
                print(f"  {config_key} not found in model_config")

        if not hasattr(args, 'dropout'):
            args.dropout = 0.0
        
        print("Re-initializing model with loaded configuration for evaluation...")
        print(f"Final model parameters: model={args.model}, hidden_size={args.hidden_size}, layers={args.layers}, gat_heads={args.gat_heads}")

    if args.model == 'GATv2Classification':
        model = GATv2Classification(
            node_attr_size=args.node_attr_size, edge_attr_size=edge_attr_size,
            hidden_size=args.hidden_size, num_clusters=args.num_clusters,
            num_gat_layers=args.layers, dropout=args.dropout,
            gat_heads=args.gat_heads, attn_drop=args.attn_drop,
            activation=args.activation, residual=args.residual
        ).to(devices)
    elif args.model == 'GATClassification':
        model = GATClassification(
            node_attr_size=args.node_attr_size, edge_attr_size=edge_attr_size,
            hidden_size=args.hidden_size, num_clusters=args.num_clusters,
            num_gat_layers=args.layers, dropout=args.dropout,
            gat_heads=args.gat_heads, attn_drop=args.attn_drop,
            activation=args.activation, residual=args.residual
        ).to(devices)
    elif args.model == 'GATv2ClassificationNoEdgeAttr':
        model = GATv2ClassificationNoEdgeAttr(
            node_attr_size=args.node_attr_size, hidden_size=args.hidden_size,
            num_clusters=args.num_clusters, num_gat_layers=args.layers,
            dropout=args.dropout, gat_heads=args.gat_heads,
            attn_drop=args.attn_drop, activation=args.activation,
            residual=args.residual
        ).to(devices)
    
    print("Model initialized successfully:")
    print(f"  Model: {args.model}, Hidden Size: {args.hidden_size}, Layers: {args.layers}, Heads: {args.gat_heads}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=args.label_smoothing)
    
    # Learning rate scheduler
    if args.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.schedule == "cosine_w10":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    elif args.schedule == "step":
        step_size = max(1, args.epochs // 2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    else:
        scheduler = None
    
    if not args.eval_only:
        config = {
            'model_config': {
                'node_attr_size': args.node_attr_size,
                'edge_attr_size': args.edge_attr_size,
                'hidden_size': args.hidden_size,
                'num_clusters': args.num_clusters,
                'num_model_layers': args.layers,
                'gat_heads': args.gat_heads,
                'dropout': args.dropout,
                'attn_drop': args.attn_drop,
                'activation': args.activation,
                'residual': args.residual,
                'model': args.model,
                'criterion': criterion.__class__.__name__
            },
            'training_config': {
                'epochs': args.epochs,
                'batch_size': args.batchsize,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'schedule': args.schedule,
                'clip_grad': args.clip_grad,
                'mask_prob': args.mask_prob,
                'edge_dropout_p': args.edge_dropout_p,
                'feature_noise': args.feature_noise,
                'device': args.device,
                'masknode': args.masknode,
                'label_smoothing': args.label_smoothing
            },
            'logging_config': {
                'logdir': args.logdir,
                'savefreq': args.savefreq
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to: {config_path}")
    else:
        print("Skipping config save during evaluation mode")

    k = [1, 3, 5, 10, 20, 50, 100]
    
    embeddings_df = pd.read_parquet(args.embeddings_path)
    if 'filename' in embeddings_df.columns and 'uuid' not in embeddings_df.columns:
        embeddings_df.rename(columns={'filename': 'uuid'}, inplace=True)
    if 'assembly_ID' in embeddings_df.columns and 'assembly_id' not in embeddings_df.columns:
        embeddings_df.rename(columns={'assembly_ID': 'assembly_id'}, inplace=True)
    embeddings_df = AssemblyGraphDataset.convert_embedding(embeddings_df)

    uuid_to_embedding = {}
    for uuid, emb in zip(embeddings_df['uuid'], embeddings_df['embedding']):
        if emb is not None and isinstance(emb, np.ndarray):
            uuid_to_embedding[uuid] = emb.astype(np.float32)

    if len(uuid_to_embedding) == 0:
        print("WARNING: No valid embeddings found after conversion. Disabling visualize_topk.")
        visualize_topk = False

    part_uuid_to_assembly_id = build_part_uuid_to_assembly_id_map(args.root, embeddings_df)

    # build cluster_to_parts mapping for cluster-based visualization
    cluster_to_parts = None
    if visualize_topk or args.visualize_best_predictions > 0 or args.visualize_worst_predictions > 0:
        print("Building cluster_to_parts mapping for cluster-based visualization...")
        
        precomputed_clusters_path = os.path.join(processed_dir, f'precomputed_clusters_{args.num_clusters}_{embedding_suffix}.parquet')
        
        if os.path.exists(precomputed_clusters_path):
            clusters_df = pd.read_parquet(precomputed_clusters_path)
            cluster_to_parts = {}
            
            for uuid, cluster_label in zip(clusters_df['uuid'], clusters_df['cluster_label']):
                cluster_id = int(cluster_label)
                if cluster_id not in cluster_to_parts:
                    cluster_to_parts[cluster_id] = []
                cluster_to_parts[cluster_id].append(uuid)
            
            print(f"Loaded cluster mappings for {len(cluster_to_parts)} clusters with {len(clusters_df)} total parts")
            
            cluster_sizes = [len(parts) for parts in cluster_to_parts.values()]
            print(f"Cluster size statistics: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
        else:
            print(f"WARNING: Precomputed clusters file not found at '{precomputed_clusters_path}'.")
            print("This file should have been generated automatically during dataset processing.")
            print("Disabling cluster-based visualizations.")
            visualize_topk = False
            args.visualize_best_predictions = 0
            args.visualize_worst_predictions = 0

    kmeans_model_path = os.path.join(processed_dir, f'kmeans_model_{args.num_clusters}_{embedding_suffix}.pkl')
    if not os.path.exists(kmeans_model_path):
        raise FileNotFoundError(
            f"KMeans model file not found at: {kmeans_model_path}\n"
            f"This file is generated during the initial data processing. "
            f"Please ensure it exists or regenerate your dataset by deleting the corresponding .pt file in the 'processed' directory and re-running."
        )
    kmeans = joblib.load(kmeans_model_path)
    cluster_centers = kmeans.cluster_centers_
    print(f"[DEBUG] Loaded cluster centers. Shape: {cluster_centers.shape}, Type: {type(cluster_centers)}")
    distance_matrix = compute_distance_matrix(cluster_centers)
    distance_matrix = torch.tensor(distance_matrix, device=devices)

    if args.eval_only:
        print("Loading pre-trained model for evaluation...")
        checkpoint = torch.load(args.model_path, map_location=devices, weights_only=True)
        max_layer_idx = -1
        for key in checkpoint.keys():
            if key.startswith('gat_layers.'):
                layer_idx = int(key.split('.')[1])
                max_layer_idx = max(max_layer_idx, layer_idx)
        
        inferred_layers = max_layer_idx + 1 if max_layer_idx >= 0 else args.layers
        if inferred_layers != args.layers:
            print(f"WARNING: Command line says {args.layers} layers, but checkpoint has {inferred_layers} layers.")
            print(f"Using {inferred_layers} layers from checkpoint structure.")
            args.layers = inferred_layers
            
            if args.model == 'GATv2Classification':
                model = GATv2Classification(
                    node_attr_size=args.node_attr_size, edge_attr_size=edge_attr_size,
                    hidden_size=args.hidden_size, num_clusters=args.num_clusters,
                    num_gat_layers=args.layers, dropout=args.dropout,
                    gat_heads=args.gat_heads, attn_drop=args.attn_drop,
                    activation=args.activation, residual=args.residual
                ).to(devices)
            elif args.model == 'GATClassification':
                model = GATClassification(
                    node_attr_size=args.node_attr_size, edge_attr_size=edge_attr_size,
                    hidden_size=args.hidden_size, num_clusters=args.num_clusters,
                    num_gat_layers=args.layers, dropout=args.dropout,
                    gat_heads=args.gat_heads, attn_drop=args.attn_drop,
                    activation=args.activation, residual=args.residual
                ).to(devices)
            elif args.model == 'GATv2ClassificationNoEdgeAttr':
                model = GATv2ClassificationNoEdgeAttr(
                    node_attr_size=args.node_attr_size, hidden_size=args.hidden_size,
                    num_clusters=args.num_clusters, num_gat_layers=args.layers,
                    dropout=args.dropout, gat_heads=args.gat_heads,
                    attn_drop=args.attn_drop, activation=args.activation,
                    residual=args.residual
                ).to(devices)
            
            print(f"Model recreated with {args.layers} layers")
        
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {args.model_path}")
    
    train_losses, val_losses, train_accuracies, val_accuracies_top1 = [], [], [], []
    val_accuracies_top3, val_accuracies_top5, val_accuracies_top10, val_accuracies_top20 = [], [], [], []
    val_accuracies_top50, val_accuracies_top100, val_balanced_accuracies = [], [], []
    val_f1s_macro, val_f1s_weighted = [], []
    val_precisions_macro, val_recalls_macro = [], []
    val_precisions_weighted, val_recalls_weighted = [], []
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(logging_dir, 'best_model.pth')
    
    if not args.eval_only:
        print("Starting training...")
        print(f"[WANDB DEBUG] wandb_run available in training loop: {wandb_run is not None}")
        for epoch in tqdm(range(args.epochs), desc=f"Training for {args.epochs} epochs"):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, devices, 
                                        mask_node=args.masknode, mask_prob=args.mask_prob, 
                                        edge_dropout_p=args.edge_dropout_p, feature_noise=args.feature_noise, 
                                        clip_grad=args.clip_grad)
            val_loss, val_acc_top1, val_acc_top3, val_acc_top5, val_acc_top10, val_acc_top20, val_acc_top50, val_acc_top100, val_balanced_acc, val_precision_macro, val_recall_macro, val_f1_macro, val_precision_weighted, val_recall_weighted, val_f1_weighted = validate(model, val_loader, criterion, devices, num_clusters=args.num_clusters)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies_top1.append(val_acc_top1)
            val_accuracies_top3.append(val_acc_top3)
            val_accuracies_top5.append(val_acc_top5)
            val_accuracies_top10.append(val_acc_top10)
            val_accuracies_top20.append(val_acc_top20)
            val_accuracies_top50.append(val_acc_top50)
            val_accuracies_top100.append(val_acc_top100)
            val_balanced_accuracies.append(val_balanced_acc)
            val_f1s_macro.append(val_f1_macro)
            val_f1s_weighted.append(val_f1_weighted)
            val_precisions_macro.append(val_precision_macro)
            val_recalls_macro.append(val_recall_macro)
            val_precisions_weighted.append(val_precision_weighted)
            val_recalls_weighted.append(val_recall_weighted)
            log_loss(os.path.join(logging_dir, 'loss_log.json'), epoch + 1, 
                     train_loss=train_loss, 
                     val_loss=val_loss, 
                     train_acc=train_acc, 
                     val_acc_top1=val_acc_top1, 
                     val_acc_top3=val_acc_top3,
                     val_acc_top5=val_acc_top5, 
                     val_acc_top10=val_acc_top10, 
                     val_acc_top20=val_acc_top20, 
                     val_acc_top50=val_acc_top50, 
                     val_acc_top100=val_acc_top100, 
                     val_balanced_acc=val_balanced_acc,
                     val_precision_macro=val_precision_macro,
                     val_recall_macro=val_recall_macro,
                     val_f1_macro=val_f1_macro,
                     val_precision_weighted=val_precision_weighted,
                     val_recall_weighted=val_recall_weighted,
                     val_f1_weighted=val_f1_weighted)

            if epoch == 0:
                print(f"[WANDB DEBUG] First epoch - wandb_run is: {wandb_run}")
                print(f"[WANDB DEBUG] wandb_run is not None: {wandb_run is not None}")
            if wandb_run:
                metrics_to_log = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc_top1": val_acc_top1,
                    "val_acc_top5": val_acc_top5,
                    "val_acc_top10": val_acc_top10,
                    "val_balanced_acc": val_balanced_acc,
                    "val_f1_macro": val_f1_macro,
                    "val_f1_weighted": val_f1_weighted,
                    "lr": scheduler.get_last_lr()[0] if scheduler else args.lr
                }
                print(f"[WANDB DEBUG] Logging metrics for epoch {epoch + 1}: {metrics_to_log}")
                wandb_run.log(metrics_to_log)
            else:
                if epoch == 0:
                    print(f"[WANDB DEBUG] wandb_run is None, skipping epoch logging")

            if scheduler is not None:
                scheduler.step()

            # early stopping
            if args.early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), best_model_path)
                    logging.info(f"New best validation loss: {best_val_loss:.4f}. Saved model to {best_model_path}")
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= args.early_stopping_patience:
                    logging.info(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
                    break
            if (epoch + 1) % args.savefreq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Top-1 Acc: {val_acc_top1:.4f}, Val Top-3 Acc: {val_acc_top3:.4f}, Val Top-5 Acc: {val_acc_top5:.4f}, Val Top-10 Acc: {val_acc_top10:.4f}, Val Top-20 Acc: {val_acc_top20:.4f}, Val Top-50 Acc: {val_acc_top50:.4f}, Val Top-100 Acc: {val_acc_top100:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}, Val F1 (M): {val_f1_macro:.4f}, Val F1 (W): {val_f1_weighted:.4f}, LR: {current_lr:.6f}')
                checkpoint_path = os.path.join(logging_dir, f'model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Model checkpoint saved to {checkpoint_path}')

    if not args.eval_only and args.early_stopping_patience > 0 and os.path.exists(best_model_path):
        logging.info(f"Loading best model from {best_model_path} for final testing.")
        model.load_state_dict(torch.load(best_model_path, map_location=devices))

    print("Starting testing...")
    test_loss, test_acc_top1, test_acc_top3, test_acc_top5, test_acc_top10, test_acc_top20, test_acc_top50, test_acc_top100, test_balanced_acc, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_weighted, test_recall_weighted, test_f1_weighted = test(
        model, test_loader, criterion, devices, cluster_centers, k, 
        args.root, part_uuid_to_assembly_id, logging_dir, uuid_to_embedding, 
        cluster_to_parts=cluster_to_parts, 
        visualize_topk=visualize_topk,
        visualize_best_predictions=args.visualize_best_predictions,
        visualize_worst_predictions=args.visualize_worst_predictions
    )
    print(f"Test Loss: {test_loss:.4f}, Test Top-1 Accuracy: {test_acc_top1:.4f}, Test Top-3 Accuracy: {test_acc_top3:.4f}, Test Top-5 Accuracy: {test_acc_top5:.4f}, Test Top-10 Accuracy: {test_acc_top10:.4f}, Test Top-20 Accuracy: {test_acc_top20:.4f}, Test Top-50 Accuracy: {test_acc_top50:.4f}, Test Top-100 Accuracy: {test_acc_top100:.4f}, Test Balanced Accuracy: {test_balanced_acc:.4f}, Test F1 (Macro): {test_f1_macro:.4f}, Test F1 (Weighted): {test_f1_weighted:.4f}")
    log_loss(os.path.join(logging_dir, 'loss_log.json'), epoch=None, 
             test_loss=test_loss, 
             test_acc_top1=test_acc_top1, 
             test_acc_top3=test_acc_top3,
             test_acc_top5=test_acc_top5, 
             test_acc_top10=test_acc_top10, 
             test_acc_top20=test_acc_top20, 
             test_acc_top50=test_acc_top50, 
             test_acc_top100=test_acc_top100, 
             test_balanced_acc=test_balanced_acc,
             test_precision_macro=test_precision_macro,
             test_recall_macro=test_recall_macro,
             test_f1_macro=test_f1_macro,
             test_precision_weighted=test_precision_weighted,
             test_recall_weighted=test_recall_weighted,
             test_f1_weighted=test_f1_weighted)

    if args.eval_only:
        print("Evaluation complete.")
        return {
            'test_acc_top1': test_acc_top1, 'test_acc_top3': test_acc_top3, 'test_acc_top5': test_acc_top5, 'test_acc_top10': test_acc_top10,
            'test_acc_top20': test_acc_top20, 'test_acc_top50': test_acc_top50, 'test_acc_top100': test_acc_top100,
            'test_balanced_acc': test_balanced_acc, 'test_f1_macro': test_f1_macro, 'test_f1_weighted': test_f1_weighted, 'test_loss': test_loss
        }
    else:
        print("Training complete.")

        plot_train_val_loss(train_losses, val_losses, save_path=os.path.join(logging_dir, 'loss_plot.png'))
        print(f"Loss plot saved to {os.path.join(logging_dir, 'loss_plot.png')}")

        plot_train_val_accuracy(train_accuracies, val_accuracies_top1, val_accuracies_top5, val_accuracies_top10, val_balanced_accuracies, save_path=os.path.join(logging_dir, 'accuracy_plot.png'))
        print(f"Accuracy plot saved to {os.path.join(logging_dir, 'accuracy_plot.png')}")

        final_model_path = os.path.join(logging_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        print(f"Final Train Loss: {train_losses[-1]:.4f}, Final Train Accuracy: {train_accuracies[-1]:.4f}")
        print(f"Final Validation Loss: {val_losses[-1]:.4f}, Final Validation Top-1 Accuracy: {val_accuracies_top1[-1]:.4f}, Final Validation Top-3 Accuracy: {val_accuracies_top3[-1]:.4f}, Final Validation Top-5 Accuracy: {val_accuracies_top5[-1]:.4f}, Final Validation Top-10 Accuracy: {val_accuracies_top10[-1]:.4f}, Final Validation Top-20 Accuracy: {val_accuracies_top20[-1]:.4f}, Final Validation Top-50 Accuracy: {val_accuracies_top50[-1]:.4f}, Final Validation Top-100 Accuracy: {val_accuracies_top100[-1]:.4f}, Final Validation Balanced Accuracy: {val_balanced_accuracies[-1]:.4f}, Final Validation F1 (Macro): {val_f1s_macro[-1]:.4f}, Final Validation F1 (Weighted): {val_f1s_weighted[-1]:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}, Test Top-1 Accuracy: {test_acc_top1:.4f}, Test Top-5 Accuracy: {test_acc_top5:.4f}, Test Top-10 Accuracy: {test_acc_top10:.4f}, Test Top-20 Accuracy: {test_acc_top20:.4f}, Test Top-50 Accuracy: {test_acc_top50:.4f}, Test Top-100 Accuracy: {test_acc_top100:.4f}, Test Balanced Accuracy: {test_balanced_acc:.4f}, Test F1 (Macro): {test_f1_macro:.4f}, Test F1 (Weighted): {test_f1_weighted:.4f}")

        return {
            'val_acc_top1': val_accuracies_top1[-1], 'val_acc_top3': val_accuracies_top3[-1], 'val_acc_top5': val_accuracies_top5[-1], 'val_acc_top10': val_accuracies_top10[-1],
            'val_acc_top20': val_accuracies_top20[-1], 'val_acc_top50': val_accuracies_top50[-1], 'val_acc_top100': val_accuracies_top100[-1],
            'val_balanced_acc': val_balanced_accuracies[-1], 
            'val_f1_macro': val_f1s_macro[-1], 'val_f1_weighted': val_f1s_weighted[-1],
            'val_precision_macro': val_precisions_macro[-1], 'val_recall_macro': val_recalls_macro[-1],
            'val_precision_weighted': val_precisions_weighted[-1], 'val_recall_weighted': val_recalls_weighted[-1],
            'test_acc_top1': test_acc_top1, 'test_acc_top3': test_acc_top3, 'test_acc_top5': test_acc_top5, 'test_acc_top10': test_acc_top10,
            'test_acc_top20': test_acc_top20, 'test_acc_top50': test_acc_top50, 'test_acc_top100': test_acc_top100,
            'test_balanced_acc': test_balanced_acc, 
            'test_f1_macro': test_f1_macro, 'test_f1_weighted': test_f1_weighted, 
            'test_precision_macro': test_precision_macro, 'test_recall_macro': test_recall_macro,
            'test_precision_weighted': test_precision_weighted, 'test_recall_weighted': test_recall_weighted,
            'test_loss': test_loss,
            'final_train_loss': train_losses[-1], 'final_train_acc': train_accuracies[-1], 'final_val_loss': val_losses[-1]
        }

def create_trial_summary_plots(all_trial_results, logging_dir):
    """Create summary plots showing distribution of metrics across trials."""
    import matplotlib.pyplot as plt
    
    # Metrics to plot
    key_metrics = ['val_balanced_acc', 'test_balanced_acc', 'test_acc_top1', 'test_acc_top5', 'test_acc_top10']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(key_metrics):
        if metric in all_trial_results and all_trial_results[metric]:
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
    
    if len(key_metrics) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(logging_dir, 'trial_summary_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Trial summary plots saved to: {os.path.join(logging_dir, 'trial_summary_statistics.png')}")

if __name__ == '__main__':
    main()