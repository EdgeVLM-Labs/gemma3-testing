#!/usr/bin/env python3
"""
Training Statistics Plotter for Mobile-VideoGPT / Google Gemma-3n-E2B

This script parses training logs and generates plots and a PDF report
for loss, gradient norm, learning rate, and validation metrics.

Usage:
    python utils/plot_training_stats.py --log_file <path_to_log> --output_dir plots/google_gemma_3n_E2B
    python utils/plot_training_stats.py --model_name google_gemma_3n_E2B
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# Default dataset split ratios
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# Matplotlib configuration
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (8, 6),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ----------------- Parsing Functions ----------------- #

def parse_training_log(log_file: str):
    """Parse training log file to extract metrics."""
    metrics = {'epoch': [], 'loss': [], 'grad_norm': [], 'learning_rate': [],
               'eval_loss': [], 'eval_epoch': []}

    train_pattern = r"\{'loss': ([\d.]+), 'grad_norm': ([\d.]+), 'learning_rate': ([\de.\-+]+), 'epoch': ([\d.]+)\}"
    eval_pattern = r"\{'eval_loss': ([\d.]+).*?'epoch': ([\d.]+)\}"

    with open(log_file, 'r') as f:
        for line in f:
            m_train = re.search(train_pattern, line)
            if m_train:
                loss, grad_norm, lr, epoch = m_train.groups()
                metrics['loss'].append(float(loss))
                metrics['grad_norm'].append(float(grad_norm))
                metrics['learning_rate'].append(float(lr))
                metrics['epoch'].append(float(epoch))
                continue
            m_eval = re.search(eval_pattern, line)
            if m_eval:
                eval_loss, epoch = m_eval.groups()
                metrics['eval_loss'].append(float(eval_loss))
                metrics['eval_epoch'].append(float(epoch))
    return metrics

def parse_training_summary(log_file: str):
    """Parse final training summary from log file."""
    summary = {}
    pattern = r"\{'train_runtime': ([\d.]+), 'train_samples_per_second': ([\d.]+), 'train_steps_per_second': ([\d.]+), 'train_loss': ([\d.]+), 'epoch': ([\d.]+)\}"
    with open(log_file, 'r') as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                summary['train_runtime'] = float(m.group(1))
                summary['train_samples_per_second'] = float(m.group(2))
                summary['train_steps_per_second'] = float(m.group(3))
                summary['train_loss'] = float(m.group(4))
                summary['final_epoch'] = float(m.group(5))
                break
    return summary

# ----------------- Plotting Functions ----------------- #

def plot_loss(epochs, loss, output_path=None, eval_epochs=None, eval_loss=None, pdf=None):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(epochs, loss, linewidth=2, color='#2E86AB', marker='o', markersize=3, alpha=0.8, label='Training Loss')
    if eval_epochs and eval_loss:
        ax.plot(eval_epochs, eval_loss, linewidth=2, color='#E63946', marker='s', markersize=4, alpha=0.8, label='Validation Loss')
    ax.set_xlabel(r'\textbf{Epoch}')
    ax.set_ylabel(r'\textbf{Loss}')
    ax.set_title(r'\textbf{Training and Validation Loss}')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    plt.tight_layout()
    if output_path: plt.savefig(output_path); print(f"✓ Saved loss plot: {output_path}")
    if pdf: pdf.savefig(fig)
    plt.close()

def plot_gradient_norm(epochs, grad_norm, output_path=None, pdf=None):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(epochs, grad_norm, linewidth=2, color='#F18F01', marker='o', markersize=3, alpha=0.8)
    ax.set_xlabel(r'\textbf{Epoch}'); ax.set_ylabel(r'\textbf{Gradient Norm}')
    ax.set_title(r'\textbf{Gradient Norm}')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if output_path: plt.savefig(output_path); print(f"✓ Saved gradient norm plot: {output_path}")
    if pdf: pdf.savefig(fig)
    plt.close()

def plot_learning_rate(epochs, lr, output_path=None, pdf=None):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(epochs, lr, linewidth=2, color='#06A77D', marker='o', markersize=3, alpha=0.8)
    ax.set_xlabel(r'\textbf{Epoch}'); ax.set_ylabel(r'\textbf{Learning Rate}')
    ax.set_title(r'\textbf{Learning Rate Schedule}')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    plt.tight_layout()
    if output_path: plt.savefig(output_path); print(f"✓ Saved learning rate plot: {output_path}")
    if pdf: pdf.savefig(fig)
    plt.close()

def plot_combined(epochs, metrics, output_path=None, eval_epochs=None, eval_loss=None, pdf=None):
    fig, axes = plt.subplots(3,1, figsize=(10,12))
    axes[0].plot(epochs, metrics['loss'], linewidth=2, color='#2E86AB', marker='o', markersize=2, alpha=0.8, label='Training Loss')
    if eval_epochs and eval_loss:
        axes[0].plot(eval_epochs, eval_loss, linewidth=2, color='#E63946', marker='s', markersize=4, alpha=0.8, label='Validation Loss')
    axes[0].set_xlabel(r'\textbf{Epoch}'); axes[0].set_ylabel(r'\textbf{Loss}')
    axes[0].set_title(r'\textbf{Training and Validation Loss}'); axes[0].grid(True, alpha=0.3, linestyle='--'); axes[0].legend()
    axes[1].plot(epochs, metrics['grad_norm'], linewidth=2, color='#F18F01', marker='o', markersize=2, alpha=0.8)
    axes[1].set_xlabel(r'\textbf{Epoch}'); axes[1].set_ylabel(r'\textbf{Gradient Norm}'); axes[1].set_title(r'\textbf{Gradient Norm}'); axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[2].plot(epochs, metrics['learning_rate'], linewidth=2, color='#06A77D', marker='o', markersize=2, alpha=0.8)
    axes[2].set_xlabel(r'\textbf{Epoch}'); axes[2].set_ylabel(r'\textbf{Learning Rate}'); axes[2].set_title(r'\textbf{Learning Rate Schedule}'); axes[2].grid(True, alpha=0.3, linestyle='--'); axes[2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    plt.tight_layout()
    if output_path: plt.savefig(output_path); print(f"✓ Saved combined plot: {output_path}")
    if pdf: pdf.savefig(fig)
    plt.close()

# ----------------- Summary Functions ----------------- #

def get_summary_text(metrics, summary):
    lines = ["="*60, "Training Statistics Summary", "="*60+"\n"]
    if summary:
        lines.append(f"Total Runtime: {summary.get('train_runtime',0):.2f}s")
        lines.append(f"Train Loss: {summary.get('train_loss',0):.4f}, Final Epoch: {summary.get('final_epoch',0):.2f}")
    lines.append(f"Total Training Steps: {len(metrics['loss'])}")
    lines.append(f"Epoch Range: {min(metrics['epoch']):.2f}-{max(metrics['epoch']):.2f}")
    return "\n".join(lines)

def save_summary_stats(metrics, summary, output_path):
    text = get_summary_text(metrics, summary)
    with open(output_path, 'w') as f: f.write(text)
    print(f"✓ Saved summary stats: {output_path}")

def create_summary_page(pdf, metrics, summary):
    fig = plt.figure(figsize=(8.5,11))
    plt.axis('off')
    plt.text(0.05,0.95,get_summary_text(metrics, summary), transform=fig.transFigure, fontsize=10, verticalalignment='top', fontfamily='monospace')
    pdf.savefig(fig)
    plt.close()

# ----------------- Main Function ----------------- #

def main():
    parser = argparse.ArgumentParser(description="Plot training stats for Google Gemma-3n-E2B / Mobile-VideoGPT")
    parser.add_argument("--log_file", type=str, help="Path to training log file")
    parser.add_argument("--model_name", type=str, default="google_gemma_3n_E2B", help="Model name")
    parser.add_argument("--output_dir", type=str, help="Custom output directory")
    args = parser.parse_args()

    # Auto-detect log file if missing
    if args.log_file is None:
        results_dir = Path(f"results/{args.model_name}")
        if results_dir.exists():
            log_files = list(results_dir.glob("*.log"))
            if log_files: args.log_file = str(max(log_files, key=os.path.getctime))
        if args.log_file is None:
            print("❌ No log file provided or found.")
            return

    if not os.path.exists(args.log_file):
        print(f"❌ Log file not found: {args.log_file}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else Path("plots") / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing training log: {args.log_file}")
    metrics = parse_training_log(args.log_file)
    summary = parse_training_summary(args.log_file)

    if not metrics['epoch']:
        print("❌ No metrics found in log")
        return

    epochs = metrics['epoch']
    eval_epochs = metrics.get('eval_epoch', [])
    eval_loss = metrics.get('eval_loss', [])

    report_path = output_dir / "training_report.pdf"
    with PdfPages(report_path) as pdf:
        create_summary_page(pdf, metrics, summary)
        plot_loss(epochs, metrics['loss'], str(output_dir / "loss.png"), eval_epochs, eval_loss, pdf)
        plot_gradient_norm(epochs, metrics['grad_norm'], str(output_dir / "grad_norm.png"), pdf)
        plot_learning_rate(epochs, metrics['learning_rate'], str(output_dir / "lr.png"), pdf)
        plot_combined(epochs, metrics, str(output_dir / "combined.png"), eval_epochs, eval_loss, pdf)

    save_summary_stats(metrics, summary, str(output_dir / "training_summary.txt"))
    print(f"\n✓ Training report and plots saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
