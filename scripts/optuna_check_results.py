import optuna
import pandas as pd
import json
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Analyze Optuna results for regression or classification.")
parser.add_argument("--task", type=str, choices=["regression", "regression_dgcnn", "regression_pointmae", "classification", "classification_dgcnn", "classification_pointmae", "classification_fast", "classification_dgcnn_fast", "classification_pointmae_fast", "classification_fullscale"], required=True, help="Task type: regression or classification")
parser.add_argument("--study_dir", type=str, default=None, help="Path to the study directory (optional, will use default if not set)")
parser.add_argument("--study_name", type=str, default=None, help="Study name (optional, will use default if not set)")
parser.add_argument("--top_n", type=int, default=10, help="Number of top trials to show")
parser.add_argument("--no_plots", action="store_true", help="Skip interactive plots")
args = parser.parse_args()

if args.task == "classification_pointmae":
    default_dir = "PATH TO YOUR LOGGING DIRECTORY"
    default_prefix = "gat_classification"
    metric_name = "validation_accuracy"
    better = "higher"
elif args.task == "classification_fullscale":
    default_dir = "PATH TO YOUR LOGGING DIRECTORY"
    default_prefix = "optuna_study"
    fast_dir_prefix = "OptunaFull"
    metric_name = "validation_accuracy"
    better = "higher"
elif args.task == "classification_fast":
    default_dir = "PATH TO YOUR LOGGING DIRECTORY"
    default_prefix = "gat_classification_fast"
    metric_name = "validation_accuracy"
    better = "higher"


# Find latest study if not specified
def find_latest_study(study_dir, prefix):
    candidates = [d for d in os.listdir(study_dir) if d.startswith(prefix) and os.path.isdir(os.path.join(study_dir, d))]
    if not candidates:
        raise FileNotFoundError(f"No studies found in {study_dir} with prefix {prefix}")
    return sorted(candidates)[-1]

is_fast_task = "fast" in args.task or args.task == "classification_fullscale"

if is_fast_task:
    study_dir = args.study_dir or default_dir
    if not os.path.isdir(study_dir):
        raise FileNotFoundError(f"Study directory {study_dir} does not exist.")
    study_run_dir = find_latest_study(study_dir, fast_dir_prefix)
    study_path = os.path.join(study_dir, study_run_dir)
    study_name = args.study_name or study_run_dir
    
    db_filename = f"{default_prefix}.db"
    storage = f"sqlite:///{os.path.join(study_path, db_filename)}"
    if not os.path.exists(os.path.join(study_path, db_filename)):
        alt_db_files = [f for f in os.listdir(study_path) if f.endswith(".db")]
        if alt_db_files:
            raise FileNotFoundError(f"Expected database file '{db_filename}' not found in {study_path}. Found these instead: {alt_db_files}")
        else:
            raise FileNotFoundError(f"No .db file found in the study directory: {study_path}")
else:
    study_dir = args.study_dir or default_dir
    if not os.path.isdir(study_dir):
        raise FileNotFoundError(f"Study directory {study_dir} does not exist.")

    if args.study_name:
        study_name = args.study_name
    else:
        study_name = find_latest_study(study_dir, default_prefix)

    study_path = os.path.join(study_dir, study_name)
    storage = f"sqlite:///{os.path.join(study_path, 'optuna.db')}"


print(f"Loading study: {study_name}")
print(f"From directory: {study_path}")
print(f"Task: {args.task} ({metric_name}, {better} is better)")

study = optuna.load_study(study_name=study_name, storage=storage)
print(f"\n=== Study Statistics ===")
print(f"Total trials: {len(study.trials)}")
print(f"Complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
print(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

# best trial
print(f"\n=== Best Trial ===")
print(f"Number: {study.best_trial.number}")
print(f"Value: {study.best_trial.value:.6f}")
print(f"Params:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")

# top n trials
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if better == "lower":
    top_trials = sorted(complete_trials, key=lambda t: t.value)[:args.top_n]
else:
    top_trials = sorted(complete_trials, key=lambda t: t.value, reverse=True)[:args.top_n]

print(f"\n=== Top {args.top_n} Trials ===")
for i, trial in enumerate(top_trials, 1):
    print(f"{i}. Trial {trial.number}: {trial.value:.6f}")
    print(f"   Key params: lr={trial.params.get('lr', 'N/A'):.2e}, "
          f"hidden_size={trial.params.get('hidden_size', 'N/A')}, "
          f"layers={trial.params.get('layers', 'N/A')}, "
          f"dropout={trial.params.get('dropout', 'N/A')}")

# stats
print(f"\n=== Parameter Statistics ===")
if complete_trials:
    param_stats = {}
    for param_name in study.best_trial.params.keys():
        values = [t.params.get(param_name) for t in complete_trials if param_name in t.params]
        if values:
            if isinstance(values[0], (int, float)):
                param_stats[param_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'best': study.best_trial.params[param_name]
                }
            else:
                from collections import Counter
                counter = Counter(values)
                param_stats[param_name] = {
                    'most_common': counter.most_common(3),
                    'best': study.best_trial.params[param_name]
                }
    
    for param, stats in param_stats.items():
        print(f"{param}:")
        if 'mean' in stats:
            print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Best: {stats['best']:.4f}")
        else:
            print(f"  Most common: {stats['most_common']}")
            print(f"  Best: {stats['best']}")

csv_path = os.path.join(study_path, "results.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"\n=== DataFrame Info ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if 'value' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['value'].abs().sort_values(ascending=False)
        print(f"\n=== Parameter Correlations with Objective ===")
        for param, corr in correlations.items():
            if param != 'value':
                print(f"{param}: {corr:.3f}")
else:
    print("\nresults.csv not found.")
json_path = os.path.join(study_path, "best_params.json")

if os.path.exists(json_path):
    with open(json_path) as f:
        best_params = json.load(f)
    print(f"\n=== Best Parameters (JSON) ===")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
else:
    print("\nbest_params.json not found.")

# plot
if not args.no_plots:
    try:
        import optuna.visualization as vis
        import plotly.io as pio

        print(f"\n=== Generating Plots ===")
        fig = vis.plot_optimization_history(study)
        fig.show()
        
        if len(complete_trials) > 1:
            fig = vis.plot_param_importances(study)
            fig.show()
        else:
            print("Skipping parameter importance plot (need >1 complete trial)")
        
        if len(complete_trials) > 1:
            fig = vis.plot_parallel_coordinate(study)
            fig.show()
        
        if len(complete_trials) > 10:
            try:
                fig = vis.plot_contour(study)
                fig.show()
            except Exception as e:
                print(f"Contour plot failed: {e}")
        
    except ImportError:
        print("\nInstall optuna[visualization] and plotly for interactive plots:")
        print("pip install optuna[visualization] plotly")

print(f"\n=== Summary ===")
print(f"Best {metric_name}: {study.best_trial.value:.6f}")
print(f"Best trial number: {study.best_trial.number}")
print(f"Study completed with {len(complete_trials)} successful trials")