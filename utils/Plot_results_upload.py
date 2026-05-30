import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_all_metrics(log_path, save_path=None):
    df = pd.read_csv(log_path)
    if save_path is None:
        save_dir = os.path.dirname(log_path)
        save_path = os.path.join(save_dir, "all_metrics.png")

    metrics = [
        ("Loss", "train_loss", "val_loss"),
        ("Accuracy", "train_acc", "val_acc"),
        ("AUC", "train_auc", "val_auc"),
        ("F1-score", "train_f1", "val_f1"),
        ("Precision", "train_precision", "val_precision"),
    ]

    # 创建 5 行 1 列子图
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 20))

    for idx, (metric_name, train_col, val_col) in enumerate(metrics):
        ax = axes[idx]
        ax.plot(df["epoch"], df[train_col], label=f"Train {metric_name}")
        ax.plot(df["epoch"], df[val_col], label=f"Val {metric_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Curve")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

