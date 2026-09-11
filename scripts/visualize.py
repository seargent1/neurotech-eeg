#!/usr/bin/env python3
"""Generate presentation-quality figures from training results.

Each figure is saved separately for easy inclusion in presentations.
"""

from pathlib import Path
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Style settings for clean figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
COLORS = {
    'csp': '#3498db',         # blue
    'riemann': '#e74c3c',     # red
    'within': '#2ecc71',      # green
    'cross': '#9b59b6',       # purple
    'train': '#95a5a6',       # gray
    'chance': '#7f8c8d',      # dark gray
}


def load_results(results_path: str = "results/comparison.json") -> dict:
    with open(results_path) as f:
        return json.load(f)


def plot_accuracy_comparison(results: dict, output_path: Path):
    """Bar chart: within vs cross-subject accuracy for each model."""
    models = results["models"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    within_accs = [m["within_subject"]["accuracy"] for m in models]
    cross_accs = [m["cross_subject"]["accuracy"] for m in models]
    names = [m["name"] for m in models]
    
    bars1 = ax.bar(x - width/2, within_accs, width, label='Within-subject', 
                   color=COLORS['within'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, cross_accs, width, label='Cross-subject', 
                   color=COLORS['cross'], edgecolor='white', linewidth=1.5)
    
    # Chance line
    ax.axhline(y=0.5, color=COLORS['chance'], linestyle='--', linewidth=2, label='Chance (50%)')
    
    # Value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance: Within-Subject vs Cross-Subject')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc='upper right')
    ax.set_ylim(0.4, 0.75)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_subject_csp(results: dict, output_path: Path):
    """Per-subject accuracy for CSP+LDA model."""
    csp_model = next(m for m in results["models"] if "CSP" in m["name"])
    
    within_per_subj = csp_model["within_subject"]["per_subject"]
    cross_per_subj = csp_model["cross_subject"]["per_subject"]
    
    subjects = sorted([int(s) for s in within_per_subj.keys()])
    within_vals = [within_per_subj[str(s)] for s in subjects]
    cross_vals = [cross_per_subj[str(s)] for s in subjects]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(subjects))
    width = 0.35
    
    ax.bar(x - width/2, within_vals, width, label='Within-subject', 
           color=COLORS['within'], edgecolor='white', linewidth=1.5)
    ax.bar(x + width/2, cross_vals, width, label='Cross-subject (held out)', 
           color=COLORS['cross'], edgecolor='white', linewidth=1.5)
    
    ax.axhline(y=0.5, color=COLORS['chance'], linestyle='--', linewidth=2, label='Chance')
    
    ax.set_xlabel('Subject')
    ax.set_ylabel('Accuracy')
    ax.set_title('CSP + LDA: Per-Subject Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s:02d}' for s in subjects])
    ax.legend(loc='upper right')
    ax.set_ylim(0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_subject_riemann(results: dict, output_path: Path):
    """Per-subject accuracy for Riemannian model."""
    riemann_model = next((m for m in results["models"] if "Riemann" in m["name"]), None)
    if riemann_model is None:
        print(f"Skipped: {output_path} (no Riemannian model)")
        return
    
    within_per_subj = riemann_model["within_subject"]["per_subject"]
    cross_per_subj = riemann_model["cross_subject"]["per_subject"]
    
    subjects = sorted([int(s) for s in within_per_subj.keys()])
    within_vals = [within_per_subj[str(s)] for s in subjects]
    cross_vals = [cross_per_subj[str(s)] for s in subjects]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(subjects))
    width = 0.35
    
    ax.bar(x - width/2, within_vals, width, label='Within-subject', 
           color=COLORS['within'], edgecolor='white', linewidth=1.5)
    ax.bar(x + width/2, cross_vals, width, label='Cross-subject (held out)', 
           color=COLORS['cross'], edgecolor='white', linewidth=1.5)
    
    ax.axhline(y=0.5, color=COLORS['chance'], linestyle='--', linewidth=2, label='Chance')
    
    ax.set_xlabel('Subject')
    ax.set_ylabel('Accuracy')
    ax.set_title('Riemannian: Per-Subject Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s:02d}' for s in subjects])
    ax.legend(loc='upper right')
    ax.set_ylim(0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def _annotate_confusion(ax, cm, cmap_name: str):
    """Draw confusion cells; black text with white halo for contrast on any cell."""
    import matplotlib.colors as mcolors
    from matplotlib import patheffects

    ax.grid(False)
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=cm.max())
    ax.imshow(cm, cmap=cmap, norm=norm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="black", fontsize=22, fontweight="bold",
            )
            txt.set_path_effects([
                patheffects.withStroke(linewidth=2.5, foreground="white")
            ])


def plot_confusion_csp(results: dict, output_path: Path):
    """Confusion matrix for CSP+LDA (cross-subject)."""
    csp_model = next(m for m in results["models"] if "CSP" in m["name"])
    cm = np.array(csp_model["cross_subject"]["confusion"])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    _annotate_confusion(ax, cm, "Blues")
    
    ax.set_title(f'CSP + LDA: Confusion Matrix\n(Cross-Subject, Acc: {csp_model["cross_subject"]["accuracy"]:.1%})')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Left', 'Right'])
    ax.set_yticklabels(['Left', 'Right'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_riemann(results: dict, output_path: Path):
    """Confusion matrix for Riemannian (cross-subject)."""
    riemann_model = next((m for m in results["models"] if "Riemann" in m["name"]), None)
    if riemann_model is None:
        print(f"Skipped: {output_path} (no Riemannian model)")
        return
    
    cm = np.array(riemann_model["cross_subject"]["confusion"])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    _annotate_confusion(ax, cm, "Reds")
    
    ax.set_title(f'Riemannian: Confusion Matrix\n(Cross-Subject, Acc: {riemann_model["cross_subject"]["accuracy"]:.1%})')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Left', 'Right'])
    ax.set_yticklabels(['Left', 'Right'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_overfitting(results: dict, output_path: Path):
    """Train vs test accuracy to show overfitting."""
    models = results["models"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    train_accs = [m["within_subject"]["train_accuracy"] for m in models]
    test_accs = [m["within_subject"]["accuracy"] for m in models]
    names = [m["name"] for m in models]
    gaps = [m["within_subject"]["overfitting_gap"] for m in models]
    
    bars1 = ax.bar(x - width/2, train_accs, width, label='Train', 
                   color=COLORS['train'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test', 
                   color=COLORS['within'], edgecolor='white', linewidth=1.5)
    
    ax.axhline(y=0.5, color=COLORS['chance'], linestyle='--', linewidth=2, label='Chance')
    
    # Value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
    
    # Gap annotations
    for i, gap in enumerate(gaps):
        ax.annotate(f'Gap: {gap:.1%}', xy=(x[i], 0.45), ha='center', fontsize=10, 
                   color=COLORS['chance'], style='italic')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Overfitting Check: Train vs Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc='upper right')
    ax.set_ylim(0.4, 0.75)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_generalization_gap(results: dict, output_path: Path):
    """Within vs cross-subject gap visualization."""
    models = results["models"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    names = [m["name"] for m in models]
    within_accs = [m["within_subject"]["accuracy"] for m in models]
    cross_accs = [m["cross_subject"]["accuracy"] for m in models]
    gaps = [m["generalization_gap"] for m in models]
    
    colors = [COLORS['csp'], COLORS['riemann']]
    
    # Scatter plot
    for i, (name, w, c) in enumerate(zip(names, within_accs, cross_accs)):
        ax.scatter(w, c, s=200, c=colors[i], label=name, edgecolors='white', linewidths=2, zorder=3)
        ax.annotate(f'  {name}', (w, c), fontsize=11, va='center')
    
    # Diagonal line (perfect generalization)
    ax.plot([0.4, 0.7], [0.4, 0.7], 'k--', linewidth=1, alpha=0.5, label='Perfect generalization')
    
    # Chance lines
    ax.axhline(y=0.5, color=COLORS['chance'], linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0.5, color=COLORS['chance'], linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Within-Subject Accuracy')
    ax.set_ylabel('Cross-Subject Accuracy')
    ax.set_title('Generalization: Within vs Cross-Subject')
    ax.set_xlim(0.45, 0.7)
    ax.set_ylim(0.45, 0.7)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_subject_variability(results: dict, output_path: Path):
    """Box plot showing subject variability across models."""
    models = results["models"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = []
    labels = []
    colors_list = []
    
    for i, model in enumerate(models):
        cross_per_subj = model["cross_subject"]["per_subject"]
        accs = list(cross_per_subj.values())
        data.append(accs)
        labels.append(model["name"])
        colors_list.append(COLORS['csp'] if 'CSP' in model["name"] else COLORS['riemann'])
    
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0.5, color=COLORS['chance'], linestyle='--', linewidth=2, label='Chance')
    
    # Individual points
    for i, d in enumerate(data):
        x = np.random.normal(i+1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, s=50, c='white', edgecolors='black', zorder=3)
    
    ax.set_ylabel('Cross-Subject Accuracy')
    ax.set_title('Subject Variability: Cross-Subject Performance')
    ax.set_ylim(0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_key_metrics(results: dict, output_path: Path):
    """Summary of key metrics as a clean table-like figure."""
    models = results["models"]
    data_info = results["data_shape"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Motor Imagery Classification: Key Results', 
            fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Dataset info
    dataset_text = (
        f"Dataset: {data_info['n_subjects']} subjects, {data_info['n_epochs']} epochs, "
        f"{data_info['n_channels']} channels"
    )
    ax.text(0.5, 0.85, dataset_text, fontsize=12, ha='center', transform=ax.transAxes,
            color=COLORS['chance'])
    
    # Results table
    y_start = 0.7
    row_height = 0.12
    
    # Header
    ax.text(0.15, y_start, 'Model', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.4, y_start, 'Within-Subject', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.6, y_start, 'Cross-Subject', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.8, y_start, 'Gap', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Separator line
    ax.plot([0.1, 0.9], [y_start - 0.03, y_start - 0.03], color='black', linewidth=1, transform=ax.transAxes)
    
    # Data rows
    for i, model in enumerate(models):
        y = y_start - (i + 1) * row_height
        color = COLORS['csp'] if 'CSP' in model["name"] else COLORS['riemann']
        
        ax.text(0.15, y, model["name"], fontsize=12, transform=ax.transAxes, color=color, fontweight='bold')
        ax.text(0.4, y, f'{model["within_subject"]["accuracy"]:.1%}', fontsize=12, ha='center', transform=ax.transAxes)
        ax.text(0.6, y, f'{model["cross_subject"]["accuracy"]:.1%}', fontsize=12, ha='center', transform=ax.transAxes)
        ax.text(0.8, y, f'{model["generalization_gap"]:.1%}', fontsize=12, ha='center', transform=ax.transAxes)
    
    # Chance baseline
    y = y_start - (len(models) + 1) * row_height
    ax.text(0.15, y, 'Chance baseline', fontsize=11, transform=ax.transAxes, color=COLORS['chance'], style='italic')
    ax.text(0.4, y, '50.0%', fontsize=11, ha='center', transform=ax.transAxes, color=COLORS['chance'])
    ax.text(0.6, y, '50.0%', fontsize=11, ha='center', transform=ax.transAxes, color=COLORS['chance'])
    
    # Key insight
    best_cross = max(models, key=lambda m: m["cross_subject"]["accuracy"])
    insight = f"Best cross-subject generalization: {best_cross['name']} ({best_cross['cross_subject']['accuracy']:.1%})"
    ax.text(0.5, 0.15, insight, fontsize=12, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    results_path = Path("results/comparison.json")
    
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        print("Run 'python scripts/train_compare.py' first.")
        sys.exit(1)
    
    results = load_results(results_path)
    
    # Create figures directory
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    print("Generating figures...\n")
    
    # Individual charts
    plot_accuracy_comparison(results, figures_dir / "01_accuracy_comparison.png")
    plot_per_subject_csp(results, figures_dir / "02_per_subject_csp.png")
    plot_per_subject_riemann(results, figures_dir / "03_per_subject_riemann.png")
    plot_confusion_csp(results, figures_dir / "04_confusion_csp.png")
    plot_confusion_riemann(results, figures_dir / "05_confusion_riemann.png")
    plot_overfitting(results, figures_dir / "06_overfitting.png")
    plot_generalization_gap(results, figures_dir / "07_generalization_gap.png")
    plot_subject_variability(results, figures_dir / "08_subject_variability.png")
    plot_key_metrics(results, figures_dir / "09_key_metrics.png")
    
    print(f"\nAll figures saved to: {figures_dir}/")


if __name__ == "__main__":
    main()
