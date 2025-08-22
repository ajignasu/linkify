import os
import io
import sys
import json
import torch
import random
import optuna
import logging
import datetime
import argparse
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import plotly.graph_objects as go
from torch_geometric.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_classification import run_single_trial

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. .env file will not be loaded. Install with: pip install python-dotenv")

# Set WANDB_DIR from WANDB_DIR_CLASSIFICATION if present
if "WANDB_DIR_CLASSIFICATION" in os.environ:
    os.environ["WANDB_DIR"] = os.environ["WANDB_DIR_CLASSIFICATION"]

wandb_env_map = {
    "dgcnn": "WC_WANDB_DIR_CLASSIFICATION_DGCNN",
    "pointmae": "WC_WANDB_DIR_CLASSIFICATION_POINTMAE",
    "pointmae_full_scale": "WC_WANDB_DIR_CLASSIFICATION_POINTMAE_FULL_SCALE",
    "default": "WC_WANDB_DIR_CLASSIFICATION",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for GNN Classification")
    parser.add_argument("--n_trials", type=int, default=500, help="Number of Optuna trials")
    parser.add_argument("--study_name", type=str, default=None, help="Study name for Optuna and W&B grouping")
    parser.add_argument("--logdir", type=str, default=None, help="Directory to store Optuna logs and results")
    parser.add_argument('--debug', action='store_true', help='Run a single trial for debugging')
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="gat-classification-tuning-full", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team name")

    # model/data args
    parser.add_argument("--root", type=str, required=True, help="Dataset root path")
    parser.add_argument('--embeddings_path', type=str, required=True, help='Path to the node embeddings parquet file')
    parser.add_argument('--model', type=str, default='GATv2Classification', choices=['GATv2Classification', 'GATv2ClassificationNoEdgeAttr', 'GATClassification'], help='Model architecture to use')
    parser.add_argument('--edge_feature_type', type=str, default='scalar', choices=['scalar','embedding'], help='Type of edge features to use')
    parser.add_argument('--contact_embeddings_path', type=str, default=None, help='Path to contact embeddings parquet')
    parser.add_argument('--num_clusters', type=int, default=500, help='Number of clusters for classification')
    parser.add_argument("--data_type", type=str, default="default", choices=["default", "dgcnn", "pointmae", "pointmae_full_scale"], help="Data type for W&B directory mapping")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # These will be set by Optuna in each trial, but having them here avoids errors.
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--schedule', type=str, default='constant')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--mask_prob', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--residual', type=str, default='none')
    parser.add_argument('--edge_dropout_p', type=float, default=0.0)
    parser.add_argument('--feature_noise', type=float, default=0.0)
    
    # Arguments needed by run_single_trial but not tuned
    parser.add_argument('--wandb_run', default=None)
    parser.add_argument('--node_attr_size', type=int, default=256)
    parser.add_argument('--masknode', action='store_true', default=True)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--savefreq', type=int, default=1000)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--visualize_topk', action='store_true')
    parser.add_argument('--visualize_best_predictions', type=int, default=0)
    parser.add_argument('--visualize_worst_predictions', type=int, default=0)
    parser.add_argument('--num_trials', type=int, default=1) 

    return parser.parse_args()

def objective(trial, args):
    """
    Optuna objective function for a comprehensive hyperparameter search.
    This function wraps the main training script.
    """
    trial_args = argparse.Namespace(**vars(args))

    # hyperparameter search space
    trial_args.lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    trial_args.weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    trial_args.schedule = trial.suggest_categorical('schedule', ['constant', 'cosine', 'cosine_w10', 'step'])
    trial_args.hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
    trial_args.layers = trial.suggest_int('layers', 1, 4)
    trial_args.gat_heads = trial.suggest_categorical('gat_heads', [2, 4, 8])
    trial_args.dropout = trial.suggest_float('dropout', 0.0, 0.7)
    trial_args.attn_drop = trial.suggest_float('attn_drop', 0.0, 0.6)
    trial_args.clip_grad = trial.suggest_float('clip_grad', 0.0, 2.0)
    trial_args.batchsize = trial.suggest_categorical('batchsize', [16, 32, 64])
    trial_args.epochs = trial.suggest_int("epochs", 25, 100, step=25)

    # augmentations
    trial_args.mask_prob = trial.suggest_float("mask_prob", 0.0, 0.4, step=0.1)
    trial_args.activation = trial.suggest_categorical("activation", ["relu", "elu", "gelu", "leaky_relu"])
    trial_args.residual = trial.suggest_categorical("residual", ["none", "add", "concat"])
    trial_args.edge_dropout_p = trial.suggest_float("edge_dropout_p", 0.0, 0.4, step=0.1)
    trial_args.feature_noise = trial.suggest_float("feature_noise", 0.0, 0.2, step=0.05)
    trial_args.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05)

    wandb_run = None
    if WANDB_AVAILABLE and args.use_wandb:
        current_params = {**trial.params}
        current_params['trial_number'] = trial.number
        current_params['data_type'] = args.data_type
        current_params['edge_feature_type'] = args.edge_feature_type
        current_params['model'] = args.model
        
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.study_name,
            job_type="hyperparameter_search",
            config=current_params,
            reinit=True,
            name=f"trial_{trial.number}",
            tags=["optuna", "hyperparameter_search", "full_search"],
        )
        
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("train_loss", step_metric="epoch")
        wandb_run.define_metric("val_loss", step_metric="epoch")
        wandb_run.define_metric("train_acc", step_metric="epoch")
        wandb_run.define_metric("val_acc_top1", step_metric="epoch")
        wandb_run.define_metric("val_acc_top5", step_metric="epoch")
        wandb_run.define_metric("val_acc_top10", step_metric="epoch")
        wandb_run.define_metric("val_balanced_acc", step_metric="epoch")
        
        trial_args.wandb_run = wandb_run

    trial_logdir = os.path.join(args.logdir, f"trial_{trial.number}")
    os.makedirs(trial_logdir, exist_ok=True)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"STARTING OPTUNA TRIAL {trial.number}")
    logging.info(f"PARAMETERS: {trial.params}")
    logging.info(f"{'='*60}")

    val_balanced_acc = 0.0
    try:
        results = run_single_trial(trial_args, trial_logdir, visualize_topk=False)
        val_balanced_acc = results.get('val_balanced_acc', 0.0)
        if wandb_run:
            results["trial_state"] = "completed"
            wandb_run.log(results)
        trial.report(val_balanced_acc, step=trial_args.epochs)
        if trial.should_prune():
            if wandb_run:
                wandb_run.log({"trial_state": "pruned"})
            raise optuna.exceptions.TrialPruned()
            
    except optuna.exceptions.TrialPruned:
        if wandb_run:
            wandb_run.log({"trial_state": "pruned"})
        logging.warning(f"Trial {trial.number} pruned.")
        raise
    except Exception as e:
        logging.error(f"Trial {trial.number} crashed with error: {e}", exc_info=True)
        if wandb_run:
            wandb_run.log({"trial_state": "failed", "error": str(e)})
        raise
    finally:
        if wandb_run:
            wandb_run.finish()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return val_balanced_acc

def main():
    args = get_args()
        
    if args.debug:
        logging.info("[DEBUG] Debug mode enabled. Overriding n_trials to 1 and epochs to 2.")
        args.n_trials = 1
        args.epochs = 2
    
    set_seed(args.seed)
    env_map = wandb_env_map    
    env_var_key = args.data_type.lower()
    env_var = env_map.get(env_var_key)

    if env_var and env_var in os.environ:
        os.environ["WANDB_DIR"] = os.environ[env_var]
        logging.info(f"WANDB_DIR set from {env_var} -> {os.environ['WANDB_DIR']}")
    elif "WANDB_DIR" not in os.environ and env_map.get("default") in os.environ:
        default_key = env_map.get("default")
        os.environ["WANDB_DIR"] = os.environ[default_key]
        logging.info(f"WANDB_DIR set from default ({default_key}) -> {os.environ['WANDB_DIR']}")

    if args.logdir is None:
        log_dir_map = {
            'default': '/mnt/drive/linkify-data/Experiments/optuna_classification_full',
            'dgcnn': '/mnt/drive/linkify-data/Experiments/optuna_classification_full_dgcnn',
            'pointmae': '/mnt/drive/linkify-data/Experiments/optuna_classification_full_pointmae',
            'pointmae_full_scale': '/mnt/drive/linkify-data/Experiments/optuna_classification_full_pointmae_full_scale'
        }
        optuna_logs_dir = log_dir_map.get(args.data_type, log_dir_map['default'])
    else:
        optuna_logs_dir = os.path.abspath(args.logdir)

    if args.study_name is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        study_name = f"OptunaFull-{args.model}-{timestamp}"
    else:
        study_name = args.study_name
    
    study_dir = os.path.join(optuna_logs_dir, study_name)
    os.makedirs(study_dir, exist_ok=True)
    args.logdir = study_dir
    args.study_name = study_name

    # set optuna study
    storage_path = f"sqlite:///{os.path.join(study_dir, 'optuna_study.db')}"
    logging.info(f"Optuna study database will be saved to: {storage_path}")

    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction='maximize',
        pruner=pruner,
        load_if_exists=True
    )

    # run optimization
    logging.info(f"Starting FULL study '{study_name}' with {args.n_trials} trials.")
    logging.info(f"Logging to study directory: {study_dir}")
    if args.use_wandb:
        logging.info(f"Logging to W&B project: {args.wandb_project}")

    try:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, catch=(Exception,))
    except KeyboardInterrupt:
        logging.info("\nStudy interrupted by user. Results so far will be saved.")

    logging.info("\n" + "="*80)
    logging.info("FULL STUDY COMPLETED")
    logging.info("="*80)

    results_df = study.trials_dataframe()
    results_csv_path = os.path.join(study_dir, 'results.csv')
    results_df.to_csv(results_csv_path, index=False)
    logging.info(f"Full trial results saved to: {results_csv_path}")

    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    if not complete_trials:
        logging.warning("No trials completed successfully. Cannot determine best trial.")
        return

    best_trial = study.best_trial
    logging.info(f"\nBest trial number: {best_trial.number}")
    logging.info(f"  Value (Best Val Balanced Acc): {best_trial.value:.4f}")
    
    logging.info("\n  Best Hyperparameters:")
    all_best_params = {**best_trial.params}
    for key, value in all_best_params.items():
        logging.info(f"    --{key.replace('_', '-')} {value}")

    best_params_path = os.path.join(study_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(all_best_params, f, indent=4)
    logging.info(f"\nBest parameters saved to: {best_params_path}")

    # for re-running best trial
    print("\n" + "="*80)
    print("To re-run the best trial with visualization, use this command:")
    print("="*80)
    
    best_args_str = " ".join([f"--{key.replace('_', '-')} {value}" for key, value in all_best_params.items()])
    contact_path_arg = f"--contact-embeddings-path {args.contact_embeddings_path}" if args.edge_feature_type == 'embedding' and args.contact_embeddings_path else ""
    embeddings_path_arg = f"--embeddings-path {args.embeddings_path}" if args.embeddings_path else ""
    
    command = (f"python scripts/train_classification.py --root {args.root} \\\n"
               f"    --model {args.model} \\\n"
               f"    --edge-feature-type {args.edge_feature_type} \\\n"
               f"    {embeddings_path_arg} \\\n"
               f"    {contact_path_arg} \\\n"
               f"    --num-clusters {args.num_clusters} \\\n"
               f"    {best_args_str} \\\n"
               f"    --visualize-best-predictions 5 --visualize-worst-predictions 5")
    
    print(command)
    print("="*80)

    # plotting
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(study_dir, "optimization_history.html"))
        if len(complete_trials) > 1:
            try:
                fig = optuna.visualization.plot_param_importances(study)
                fig.write_html(os.path.join(study_dir, "param_importance.html"))
            except (ValueError, ZeroDivisionError) as e:
                logging.warning(f"Could not generate param importances plot: {e}")
        logging.info(f"Optuna plots saved in {study_dir}/ directory")
    except ImportError:
        logging.warning("\nInstall plotly for visualization: pip install plotly")
    except Exception as e:
        logging.error(f"\nCould not generate plotly plots due to an error: {e}", exc_info=True)
if __name__ == "__main__":
    main()