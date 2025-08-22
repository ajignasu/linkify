# import matplotlib.pyplot as plt
# import re

# def read_loss_log(filepath):
#     epochs, train_loss, val_loss = [], [], []
#     with open(filepath, 'r') as f:
#         for line in f:
#             match = re.match(r"Epoch (\d+): Train Loss: ([\d\.e-]+), Validation Loss: ([\d\.e-]+)", line)
#             if match:
#                 epochs.append(int(match.group(1)))
#                 train_loss.append(float(match.group(2)))
#                 val_loss.append(float(match.group(3)))
#     return epochs, train_loss, val_loss

# # Example usage: update these paths to your actual log files
# log1 = r'C:/Users/jignasa/Repo/assembly_graph_interface/logs/node_regression/20250611-101423_ContactLabels_100Epochs/loss_log.txt'
# log2 = r'C:/Users/jignasa/Repo/assembly_graph_interface/logs/node_regression/20250611-101539/loss_log.txt'

# epochs1, train1, val1 = read_loss_log(log1)
# epochs2, train2, val2 = read_loss_log(log2)

# plt.figure(figsize=(10,5))
# plt.plot(epochs1, train1, label='Contacts: Train Loss')
# plt.plot(epochs2, train2, label='No Contacts: Train Loss', linestyle='--')
# plt.xlabel('Epoch')
# plt.ylabel('Train Loss')
# plt.title('Train Loss for Two Runs')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,5))
# plt.plot(epochs1, val1, label='Contacts: Val Loss')
# plt.plot(epochs2, val2, label='No Contacts: Val Loss', linestyle='--')
# plt.xlabel('Epoch')
# plt.ylabel('Validation Loss')
# plt.title('Validation Loss for Two Runs')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



import matplotlib.pyplot as plt
import json
import os

def plot_loss_curves_from_json(log_file):
    """
    Plots training and validation loss curves from a JSON log file.

    Args:
        log_file (str): Path to the JSON log file.
    """
    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                epoch = data.get("epoch")
                if epoch is not None:
                    epochs.append(epoch)
                    train_loss_value = data.get("train_loss")
                    if train_loss_value is not None:
                        train_loss.append(train_loss_value)
                    val_loss_value = data.get("val_loss")
                    if val_loss_value is not None:
                        val_loss.append(val_loss_value)
                    train_acc_value = data.get("train_acc")
                    if train_acc_value is not None:
                        train_acc.append(train_acc_value)
                    val_acc_value = data.get("val_acc")
                    if val_acc_value is not None:
                        val_acc.append(val_acc_value)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue

    # Plotting Loss
    if train_loss and val_loss:
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(log_file), 'loss_plot.png'))  # Save the plot
        plt.show()
    else:
        print("Train Loss or Validation Loss data not found in log file.")

    # Plotting Accuracy
    if train_acc and val_acc:
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, train_acc, label='Train Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(log_file), 'accuracy_plot.png'))  # Save the plot
        plt.show()
    else:
        print("Train Accuracy or Validation Accuracy data not found in log file.")

# Example Usage:
log_file_path = 'C:/Users/jignasa/Repo/assembly_graph_interface/logs/node_classification/20250613-140638_Contacts_100Epochs_PrecomputedClusters_TopK_joblib/loss_log.json'
plot_loss_curves_from_json(log_file_path)