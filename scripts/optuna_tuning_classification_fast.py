import optuna
import os
import sys
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import logging
import random
from datetime import datetime
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_classification import run_single_trial

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed, so .env file will not be loaded.")

# W&B integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

_wandb_env_map = {
    "pointmae": "WC_WANDB_DIR_CLASSIFICATION_POINTMAE",
    "default": "WC_WANDB_DIR_CLASSIFICATION",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Define a set of fixed hyperparameters for a faster, more focused search
FIXED_PARAMS = {
    "layers": 2,
    "hidden_size": 256,
    "gat_heads": 2,
    "residual": "none",
    "activation": "gelu"
}

def objective(trial, args):
    """
    Optuna objective function for a fast, focused hyperparameter search.
    """
    trial_args = argparse.Namespace(**vars(args))

    for key, value in FIXED_PARAMS.items():
        setattr(trial_args, key, value)
    
    trial_args.lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    trial_args.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    trial_args.schedule = trial.suggest_categorical('schedule', ['constant', 'cosine'])
    trial_args.dropout = trial.suggest_float('dropout', 0.1, 0.7)
    trial_args.attn_drop = trial.suggest_float('attn_drop', 0.1, 0.7)
    trial_args.edge_dropout_p = trial.suggest_float('edge_dropout_p', 0.0, 0.5)
    trial_args.feature_noise = trial.suggest_float('feature_noise', 0.0, 0.2)
    trial_args.clip_grad = trial.suggest_float('clip_grad', 0.0, 1.5)

    wandb_run = None
    if WANDB_AVAILABLE and args.use_wandb:
        current_params = {**FIXED_PARAMS, **trial.params}
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.study_name,
            job_type="hyperparameter_search_fast",
            config=current_params,
            reinit=True,
            name=f"trial_{trial.number}",
            tags=["optuna", "hyperparameter_search", "fast_search"],
            notes=f"Optuna trial {trial.number} with fixed params: {FIXED_PARAMS}"
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

    trial_logdir = os.path.join(trial_args.logdir, f"trial_{trial.number}")
    os.makedirs(trial_logdir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"STARTING OPTUNA TRIAL {trial.number}")
    print(f"PARAMETERS: { {**FIXED_PARAMS, **trial.params} }")
    print(f"{'='*60}")

    val_balanced_acc = 0.0
    try:
        results = run_single_trial(trial_args, trial_logdir, visualize_topk=False)
        val_balanced_acc = results.get('val_balanced_acc', 0.0)        
        actual_node_attr_size = results.get('node_attr_size')
        actual_edge_attr_size = results.get('edge_attr_size')
        if actual_node_attr_size is not None:
            trial.set_user_attr("actual_node_attr_size", actual_node_attr_size)
        if actual_edge_attr_size is not None:
            trial.set_user_attr("actual_edge_attr_size", actual_edge_attr_size)
        if wandb_run:
            final_metrics = {
                "trial_number": trial.number,
                "final_val_balanced_acc": val_balanced_acc,
                "final_val_acc_top1": results.get('val_acc_top1', 0.0),
                "final_val_acc_top5": results.get('val_acc_top5', 0.0),
                "final_val_acc_top10": results.get('val_acc_top10', 0.0),
                "final_test_balanced_acc": results.get('test_balanced_acc', 0.0),
                "final_test_acc_top1": results.get('test_acc_top1', 0.0),
                "final_test_acc_top5": results.get('test_acc_top5', 0.0),
                "final_test_acc_top10": results.get('test_acc_top10', 0.0),
                "final_test_loss": results.get('test_loss', 0.0),
                "final_train_loss": results.get('final_train_loss', 0.0),
                "final_train_acc": results.get('final_train_acc', 0.0),
                "final_val_loss": results.get('final_val_loss', 0.0),
                "trial_state": "completed"
            }
            
            trial_hyperparams = {f"hp_{key}": value for key, value in trial.params.items()}
            final_metrics.update(trial_hyperparams)
            
            final_metrics.update({
                "actual_node_attr_size": actual_node_attr_size,
                "actual_edge_attr_size": actual_edge_attr_size,
            })
            
            print(f"[WANDB DEBUG] Logging final trial metrics: {final_metrics}")
            wandb_run.log(final_metrics)
        
        trial.report(val_balanced_acc, step=trial_args.epochs)
        if trial.should_prune():
            if wandb_run: 
                wandb_run.log({"trial_state": "pruned", "pruned_at_epoch": trial_args.epochs})
            raise optuna.exceptions.TrialPruned()
            
    except optuna.exceptions.TrialPruned:
        if wandb_run: wandb_run.log({"trial_state": "pruned"})
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        if wandb_run: wandb_run.log({"trial_state": "failed", "error": str(e)})
        raise
    
    finally:
        if wandb_run: wandb_run.finish()

    return val_balanced_acc


def main():
    parser = argparse.ArgumentParser(description='Run a fast hyperparameter search for GNN classification.')
    
    parser.add_argument('--study_name', type=str, default=None, help='Name for the Optuna study.')
    parser.add_argument('--n_trials', type=int, default=150, help='Number of Optuna trials to run.')
    
    parser.add_argument('--root', type=str, required=True, help='Root directory for the dataset')
    parser.add_argument('--embeddings_path', type=str, required=True, help='Path to the node embeddings parquet file')
    parser.add_argument('--model', type=str, default='GATv2Classification', choices=['GATv2Classification', 'GATv2ClassificationNoEdgeAttr', 'GATClassification'], help='Model architecture to use')
    parser.add_argument('--edge_feature_type', type=str, default='scalar', choices=['scalar','embedding'], help='Type of edge features to use')
    parser.add_argument('--contact_embeddings_path', type=str, default=None, help='Path to contact embeddings parquet')
    parser.add_argument('--num_clusters', type=int, default=500, help='Number of clusters for classification')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs per trial')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for training')
    parser.add_argument('--logdir', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'logs', 'optuna_tuning_fast'), help='Parent directory for logs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    
    parser.add_argument('--use_wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb_project', type=str, default='gat-classification-tuning-fast', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help="W&B entity/team name")
    parser.add_argument("--data_type", type=str, default="default", choices=["default", "dgcnn", "pointmae"], help="Data type to set WANDB_DIR")
    parser.add_argument('--debug', action='store_true', help='Run a single trial for debugging')
    
    parser.add_argument('--masknode', action='store_true', default=True, help='Whether to mask a node during training')
    parser.add_argument('--node_attr_size', type=int, default=256)
    parser.add_argument('--mask_prob', type=float, default=0.0)
    parser.add_argument('--early_stopping_patience', type=int, default=0)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--visualize_topk', action='store_true', help='Include --visualize_topk in the final suggested command.')
    parser.add_argument('--visualize_best_predictions', type=int, default=0)
    parser.add_argument('--visualize_worst_predictions', type=int, default=0)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--savefreq', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--wandb_run', default=None)

    args = parser.parse_args()
    
    if args.debug:
        logging.info("--- DEBUG MODE ENABLED ---")
        args.n_trials = 1
        args.epochs = 2
        print(f"Overriding n_trials to {args.n_trials} and epochs to {args.epochs}")
    
    env_map = _wandb_env_map
    env_var_key = args.data_type.lower()
    env_var = env_map.get(env_var_key)

    if env_var and env_var in os.environ:
        os.environ["WANDB_DIR"] = os.environ[env_var]
        logging.info(f"WANDB_DIR set from {env_var} -> {os.environ['WANDB_DIR']}")
    elif "WANDB_DIR" not in os.environ and env_map.get("default") in os.environ:
        default_key = env_map.get("default")
        os.environ["WANDB_DIR"] = os.environ[default_key]
        logging.info(f"WANDB_DIR set from default ({default_key}) -> {os.environ['WANDB_DIR']}")

    # setup study
    if args.study_name is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        args.study_name = f"OptunaFast-{args.model}-{timestamp}"
        
    study_logdir = os.path.join(args.logdir, args.study_name)
    os.makedirs(study_logdir, exist_ok=True)
    args.logdir = study_logdir 

    storage_path = f"sqlite:///{os.path.join(study_logdir, 'optuna_study_fast.db')}"
    print(f"Optuna study database will be saved to: {storage_path}")
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_path,
        direction='maximize',
        pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=args.epochs, reduction_factor=3)
    )

    # run optimization
    print(f"Starting FAST study '{args.study_name}' with {args.n_trials} trials.")
    try:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, catch=(Exception,))
    except KeyboardInterrupt:
        print("Study interrupted by user. Results so far will be saved.")

    print("\n" + "="*80)
    print("FAST STUDY COMPLETED")
    print("="*80)
    
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    
    if not complete_trials:
        print("No trials completed successfully. Cannot determine best trial.")
        return

    best_trial = study.best_trial
    print(f"  Value (Best Val Balanced Acc): {best_trial.value:.4f}")
    
    print("  Best Hyperparameters (tuned):")
    for key, value in best_trial.params.items():
        print(f"    --{key.replace('_', '-')} {value}")
        
    print("  Fixed Hyperparameters:")
    for key, value in FIXED_PARAMS.items():
        print(f"    --{key.replace('_', '-')} {value}")
        
    all_best_params = {**FIXED_PARAMS, **best_trial.params}
    
    best_params_path = os.path.join(study_logdir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(all_best_params, f, indent=4)
    print(f"\nBest parameters saved to: {best_params_path}")

    df = study.trials_dataframe()
    results_csv_path = os.path.join(study_logdir, 'results.csv')
    df.to_csv(results_csv_path, index=False)
    print(f"Full trial results saved to: {results_csv_path}")
    
    print("\n" + "="*80)
    print("To re-run the best trial for 100 epochs, use this command:")
    print("="*80)
    best_args_str = " ".join([f"--{key.replace('_', '-')} {value}" for key, value in all_best_params.items()])
    contact_path_arg = f"--contact_embeddings_path {args.contact_embeddings_path}" if args.edge_feature_type == 'embedding' and args.contact_embeddings_path else ""
    embeddings_path_arg = f"--embeddings_path {args.embeddings_path}" if args.embeddings_path else ""
    
    viz_args = []
    if args.visualize_topk:
        viz_args.append("--visualize_topk")
    if args.visualize_best_predictions > 0:
        viz_args.append(f"--visualize_best_predictions {args.visualize_best_predictions}")
    if args.visualize_worst_predictions > 0:
        viz_args.append(f"--visualize_worst_predictions {args.visualize_worst_predictions}")
    viz_args_str = " ".join(viz_args)

    command = (f"python scripts/train_classification.py --root {args.root} "
               f"--model {args.model} "
               f"--edge_feature_type {args.edge_feature_type} "
               f"{embeddings_path_arg} "
               f"{contact_path_arg} "
               f"--num_clusters {args.num_clusters} "
               f"--epochs 100 --batchsize {args.batchsize} "
               f"{best_args_str} {viz_args_str}")
    print(command)
    print("="*80)

if __name__ == '__main__':
    main()