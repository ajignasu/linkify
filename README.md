# Learning from Interface-Augmented Assembly Graphs

## Dependencies
- Python 3.12+
- Pytorch 2.7.0+cu118
- Pytorch Geometric 2.6.1
- Networkx 3.3

## Conda Environment
Build the environment using the provided `env.yml` or `env_nobuild.yml` file:
```bash
conda env create -f env.yml

OR

conda env create -f env_nobuild.yml
```

Activate the environment:
```bash
conda activate linkify
```

## Debug Mode for data generation
```python
python data_filtering.py --destination "PATH TO A SINGLE ASSEMBLY" --debug
```

# Classification

For an overview of all variables,
```python
python train_classifcation.py -h
```

## Using PointMAE params

```python
python train_classification.py --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --num_clusters 500 --lr 0.000607 --hidden_size 256 --gat_heads 2 --residual none --activation gelu --weight_decay 0.000001 --schedule constant --dropout 0.467117 --attn_drop 0.119235 --edge_dropout_p 0.093230 --feature_noise 0.116391 --clip_grad 1.089107 --visualize_topk --edge_feature_type embedding --contact_embeddings_path PATH TO YOUR INTERFACE EMBEDDINGS --visualize_best_predictions 3 --visualize_worst_predictions 3 --epochs 100 --batchsize 64
```

## multi trial with final full scale Optuna params for PointMAE; stores data on EFS file
```python 
python train_classification.py --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --visualize_topk --edge_feature_type embedding --contact_embeddings_path PATH TO YOUR INTERFACE EMBEDDINGS --visualize_best_predictions 3 --visualize_worst_predictions 3 --num_clusters 500 --lr 0.000068 --weight_decay 0.000009 --schedule cosine_w10 --hidden_size 512 --layers 4 --gat_heads 4 --dropout 0.154811 --attn_drop 0.392510 --clip_grad 1.104802 --batchsize 64 --epochs 100 --activation leaky_relu --residual add --edge_dropout_p 0.1 --feature_noise 0.15 --label_smoothing 0.2 --num_trials 10 --savefreq 25 --logdir PATH TO YOUR LOGGING DIRECTORY
```

## run in eval mode
```python
python train_classification.py --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --edge_feature_type embedding --contact_embeddings_path PATH TO YOUR INTERFACE EMBEDDINGS --num_clusters 500 --batchsize 64 --visualize_topk  --visualize_best_predictions 2 --visualize_worst_predictions 2 --eval_only --model_path PATH TO YOUR final_model.pth
```

```python
python train_classification.py --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --edge_feature_type embedding --contact_embeddings_path PATH TO YOUR INTERFACE EMBEDDINGS --num_clusters 500 --hidden_size 512 --layers 4 --gat_heads 4 --dropout 0.154811 --attn_drop 0.392510 --batchsize 64 --activation leaky_relu --residual add --visualize_topk --visualize_best_predictions 2 --visualize_worst_predictions 2 --eval_only --model_path PATH TO YOUR final_model.pth
```

## run with GATClassification
```python
python train_classification.py --model GATClassification --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --visualize_topk --edge_feature_type embedding --contact_embeddings_path PATH TO YOUR INTERFACE EMBEDDINGS --visualize_best_predictions 3 --visualize_worst_predictions 3 --num_clusters 500 --lr 0.000068 --weight_decay 0.000009 --schedule cosine_w10 --hidden_size 512 --layers 4 --gat_heads 4 --dropout 0.154811 --attn_drop 0.392510 --clip_grad 1.104802 --batchsize 64 --epochs 100 --activation leaky_relu --residual add --edge_dropout_p 0.1 --feature_noise 0.15 --label_smoothing 0.2 --num_trials 10 --savefreq 20 --logdir PATH TO YOUR LOGGING DIRECTORY
```

## run with augmentation and without edge attr
```python
python train_classification.py --model GATv2ClassificationNoEdgeAttr --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --visualize_topk --visualize_best_predictions 3 --visualize_worst_predictions 3 --num_clusters 500 --lr 0.000068 --weight_decay 0.000009 --schedule cosine_w10 --hidden_size 512 --layers 4 --gat_heads 4 --dropout 0.154811 --attn_drop 0.392510 --clip_grad 1.104802 --batchsize 64 --epochs 100 --activation leaky_relu --residual add --edge_dropout_p 0.1 --feature_noise 0.15 --label_smoothing 0.2 --aug_type RE --num_trials 10 --savefreq 20 --logdir PATH TO YOUR LOGGING DIRECTORY
```

```python
python train_classification.py --model GATv2ClassificationNoEdgeAttr --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --visualize_topk --visualize_best_predictions 3 --visualize_worst_predictions 3 --num_clusters 500 --lr 0.000068 --weight_decay 0.000009 --schedule cosine_w10 --hidden_size 512 --layers 4 --gat_heads 4 --dropout 0.154811 --attn_drop 0.392510 --clip_grad 1.104802 --batchsize 64 --epochs 100 --activation leaky_relu --residual add --edge_dropout_p 0.1 --feature_noise 0.15 --label_smoothing 0.2 --aug_type PARCON --num_trials 10 --savefreq 20 --logdir PATH TO YOUR LOGGING DIRECTORY
```

## run without edge attr
```python
python train_classification.py --model GATv2ClassificationNoEdgeAttr --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --visualize_topk --visualize_best_predictions 3 --visualize_worst_predictions 3 --num_clusters 500 --lr 0.000068 --weight_decay 0.000009 --schedule cosine_w10 --hidden_size 512 --layers 4 --gat_heads 4 --dropout 0.154811 --attn_drop 0.392510 --clip_grad 1.104802 --batchsize 64 --epochs 100 --activation leaky_relu --residual add --edge_dropout_p 0.1 --feature_noise 0.15 --label_smoothing 0.2 --savefreq 20 --num_trials 10 --logdir PATH TO YOUR LOGGING DIRECTORY
```

## with Optuna Full Scale tuning hyperparameters (PointMAE)
```python
python train_classification.py --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --visualize_topk --edge_feature_type embedding --contact_embeddings_path PATH TO YOUR INTERFACE EMBEDDINGS --visualize_best_predictions 3 --visualize_worst_predictions 3 --num_clusters 500 --lr 0.000068 --weight_decay 0.000009 --schedule cosine_w10 --hidden_size 512 --layers 4 --gat_heads 4 --dropout 0.154811 --attn_drop 0.392510 --clip_grad 1.104802 --batchsize 64 --epochs 100 --activation leaky_relu --residual add --edge_dropout_p 0.1 --feature_noise 0.15 --label_smoothing 0.2 --num_trials 10 --savefreq 20 --logdir PATH TO YOUR LOGGING DIRECTORY
```

## with Random Contact Embeddings + Optuna Full Scale tuning hyperparameter (PointMAE)
```python
python train_classification.py --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --visualize_topk --edge_feature_type embedding --contact_embeddings_path PATH TO YOUR RANDOMLY GENERATED NODE EMBEDDINGS --visualize_best_predictions 3 --visualize_worst_predictions 3 --num_clusters 500 --lr 0.000068 --weight_decay 0.000009 --schedule cosine_w10 --hidden_size 512 --layers 4 --gat_heads 4 --dropout 0.154811 --attn_drop 0.392510 --clip_grad 1.104802 --batchsize 64 --epochs 100 --activation leaky_relu --residual add --edge_dropout_p 0.1 --feature_noise 0.15 --label_smoothing 0.2 --num_trials 10 --savefreq 20 --logdir PATH TO YOUR LOGGING DIRECTORY
```

## Example command for Minimum Spanning Tree augmentation
```python
    python train_classification.py --model GATv2ClassificationNoEdgeAttr --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --visualize_topk --visualize_best_predictions 3 --visualize_worst_predictions 3 --num_clusters 500 --lr 0.000068 --weight_decay 0.000009 --schedule cosine_w10 --hidden_size 512 --layers 4 --gat_heads 4 --dropout 0.154811 --attn_drop 0.392510 --clip_grad 1.104802 --batchsize 64 --epochs 100 --activation leaky_relu --residual add --edge_dropout_p 0.1 --feature_noise 0.15 --label_smoothing 0.2 --aug_type MST
```


# Optuna

## Using optuna_tuning_classification_fast.py
```python
python optuna_tuning_classification_fast.py --root PATH TO YOUR DATASET --embeddings_path PATH TO YOUR NODE EMBEDDINGS --seed 42 --use_wandb --wandb_project NAME OF YOUR WANDB PROJECT --edge_feature_type embedding --contact_embeddings_path PATH TO YOUR CONTACT EMBEDDINGS --data_type dgcnn --debug --logdir PATH TO YOUR LOGGING DIRECTORY
```

## Use optuna_check_results.py
```python
python optuna_check_results.py --task classification_pointmae --top_n 10 --no_plots
```


# Data Filtering

```python
python -m data_generation.data_filtering --destination PATH TO YOUR DATASET --getcontacts
```


# Generate Contacts
```python
python -m scripts.data_generation.contact_generation.generate_contacts_test --source PATH TO DATASET TO BE AUGMENTED --st-source PATH TO SOURCE ASSEMBLIES --augment --workers 4 --num NUMBER OF ASSEMBLIES TO AUGMENT --logs YOUR LOGGING DIR
```