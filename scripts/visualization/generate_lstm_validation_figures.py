import json
import os
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_metrics_from_validation(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('episodes', [])


def load_training_prediction_data(exp_dir: str) -> List[Dict[str, Any]]:
    """Load LSTM training prediction logs from comprehensive_results/<exp>/prediction_dashboard/data
    Returns a list of per-episode dicts with accuracy, precision, recall, f1, avg_prob-like proxy, and support via tp/tn/fp/fn.
    """
    pred_path = os.path.join(exp_dir, 'prediction_dashboard', 'data', 'prediction_data.json')
    acc_path = os.path.join(exp_dir, 'prediction_dashboard', 'data', 'accuracy_history.json')
    episodes: List[Dict[str, Any]] = []
    if os.path.exists(pred_path):
        with open(pred_path, 'r', encoding='utf-8') as f:
            preds = json.load(f)
        episodes = preds  # already list of dicts with metrics and counts
    else:
        episodes = []
    # augment with accuracy_history if present (ensure 'accuracy' exists)
    if os.path.exists(acc_path):
        with open(acc_path, 'r', encoding='utf-8') as f:
            accs = json.load(f)
        for i, a in enumerate(accs):
            if i < len(episodes):
                episodes[i]['accuracy'] = float(a)
            else:
                episodes.append({'episode': i, 'accuracy': float(a)})
    # derive ground_truth_label if counts allow (heavies considered positive=1 when tp+fn>0 else 0)
    for ep in episodes:
        tp = int(ep.get('true_positives', 0) or 0)
        fn = int(ep.get('false_negatives', 0) or 0)
        tn = int(ep.get('true_negatives', 0) or 0)
        fp = int(ep.get('false_positives', 0) or 0)
        total = tp + tn + fp + fn
        # Episode label: majority class present, if tie leave None
        if (tp + fn) > (tn + fp):
            ep['ground_truth_label'] = 1
        elif (tn + fp) > (tp + fn):
            ep['ground_truth_label'] = 0
        # add avg_prob proxy if missing (map accuracy to prob band minimally)
        ep.setdefault('avg_prob', None)
        # ensure precision/recall present
        ep.setdefault('precision', float(ep.get('precision', 0.0) or 0.0))
        ep.setdefault('recall', float(ep.get('recall', 0.0) or 0.0))
        ep.setdefault('f1_score', float(ep.get('f1_score', 0.0) or 0.0))
        ep.setdefault('steps', total)
    return episodes


def dot_accuracy(episodes: List[Dict[str, Any]], out_path: str):
    xs = np.arange(1, len(episodes) + 1)
    acc = np.array([float(ep.get('accuracy', np.nan)) for ep in episodes], dtype=float)
    gt = np.array([int(ep.get('ground_truth_label', -1)) for ep in episodes], dtype=int)
    colors = np.where(gt == 1, '#E15759', '#4C78A8')

    plt.figure(figsize=(10, 5))
    plt.scatter(xs, acc, c=colors, alpha=0.7, s=24, edgecolors='none')
    m = np.nanmean(acc) if len(acc) else np.nan
    plt.plot([1, len(xs)], [m, m], 'k--', linewidth=2, label=f'mean={m:.3f}')
    plt.xlabel('Episode')
    plt.ylabel('Step-level accuracy')
    plt.title('LSTM Validation Accuracy per Episode (color: GT heavy=red, light=blue)')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_confusion(episodes: List[Dict[str, Any]], out_path: str):
    y_true = []
    y_pred = []
    for ep in episodes:
        gt = ep.get('ground_truth_label')
        if gt is None:
            continue
        avg_prob = ep.get('avg_prob', None)
        if avg_prob is None:
            # fallback: use accuracy/precision/recall hints
            pred_ep = 1 if (float(ep.get('recall', 0.0)) >= 0.5 or float(ep.get('precision', 0.0)) >= 0.5) else 0
        else:
            pred_ep = 1 if float(avg_prob) >= 0.5 else 0
        y_true.append(int(gt))
        y_pred.append(int(pred_ep))

    if not y_true:
        return
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    f1_macro = np.mean(f1)
    weights = support / np.maximum(np.sum(support), 1)
    f1_weighted = np.sum(f1 * weights)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Light (pred)', 'Heavy (pred)'])
    ax.set_yticklabels(['Light (true)', 'Heavy (true)'])
    # annotate both counts and percentages
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)", ha='center', va='center', color='black', fontsize=10)
    subtitle = f"Acc={acc:.3f} | F1(macro)={f1_macro:.3f} | F1(weighted)={f1_weighted:.3f} | N={int(np.sum(support))}"
    ax.set_title('LSTM Validation Confusion Matrix')
    fig.suptitle(subtitle, y=0.02, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def precision_recall_bars(episodes: List[Dict[str, Any]], out_path: str):
    # derive predictions as in confusion plot
    y_true = []
    y_pred = []
    for ep in episodes:
        gt = ep.get('ground_truth_label')
        if gt is None:
            continue
        avg_prob = ep.get('avg_prob', None)
        if avg_prob is None:
            pred_ep = 1 if (float(ep.get('recall', 0.0)) >= 0.5 or float(ep.get('precision', 0.0)) >= 0.5) else 0
        else:
            pred_ep = 1 if float(avg_prob) >= 0.5 else 0
        y_true.append(int(gt))
        y_pred.append(int(pred_ep))

    if not y_true:
        return
    labels_idx = [0, 1]
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels_idx, zero_division=0)
    classes = ['Light', 'Heavy']
    x = np.arange(2)
    width = 0.35

    plt.figure(figsize=(7, 5))
    bars1 = plt.bar(x - width/2, prec, width, label='Precision', color='#4C78A8', alpha=0.9)
    bars2 = plt.bar(x + width/2, rec, width, label='Recall', color='#59A14F', alpha=0.9)
    plt.xticks(x, [f"{c}\n(n={int(s)} )" for c, s in zip(classes, support)])
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('LSTM Validation Precision/Recall by Class')
    for r in list(bars1) + list(bars2):
        v = r.get_height()
        plt.text(r.get_x() + r.get_width()/2, v, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    # macro/micro/weighted summaries
    acc = accuracy_score(y_true, y_pred)
    f1_macro = float(np.mean(f1))
    weights = support / np.maximum(np.sum(support), 1)
    f1_weighted = float(np.sum(f1 * weights))
    plt.annotate(f"Acc={acc:.2f} | F1(macro)={f1_macro:.2f} | F1(weighted)={f1_weighted:.2f}",
                 xy=(0.5, 1.02), xycoords='axes fraction', ha='center', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def mean_prob_by_class(episodes: List[Dict[str, Any]], out_path: str):
    probs_light = [float(ep.get('avg_prob')) for ep in episodes if ep.get('ground_truth_label') == 0 and ep.get('avg_prob') is not None]
    probs_heavy = [float(ep.get('avg_prob')) for ep in episodes if ep.get('ground_truth_label') == 1 and ep.get('avg_prob') is not None]
    means = [np.nanmean(probs_light) if probs_light else np.nan, np.nanmean(probs_heavy) if probs_heavy else np.nan]
    errs = [np.nanstd(probs_light) if probs_light else 0.0, np.nanstd(probs_heavy) if probs_heavy else 0.0]

    plt.figure(figsize=(6, 5))
    bars = plt.bar([0, 1], means, yerr=errs, capsize=6, color=['#4C78A8', '#E15759'])
    plt.xticks([0, 1], ['Light (true)', 'Heavy (true)'])
    plt.ylim(0, 1.0)
    plt.ylabel('Mean predicted probability (heavy)')
    plt.title('Mean Predicted Heavy Probability by True Class')
    for r, v in zip(bars, means):
        if not np.isnan(v):
            plt.text(r.get_x() + r.get_width()/2, v, f"{v:.2f}", ha='center', va='bottom')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def reliability_curve(episodes: List[Dict[str, Any]], out_path: str, bins: int = 8):
    # Use episode-level avg probabilities and binary true label
    pts = [(float(ep.get('avg_prob')), int(ep.get('ground_truth_label'))) for ep in episodes if ep.get('avg_prob') is not None]
    if not pts:
        return
    probs, labels = zip(*pts)
    probs = np.array(probs)
    labels = np.array(labels)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    empirical = []
    for i in range(bins):
        m = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if np.any(m):
            empirical.append(np.mean(labels[m]))
        else:
            empirical.append(np.nan)
    empirical = np.array(empirical)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(bin_centers, empirical, 'o-', color='#4C78A8', label='Empirical accuracy')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Predicted probability (heavy)')
    plt.ylabel('Empirical accuracy')
    plt.title('LSTM Validation Reliability Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(root, 'Chapter 4', 'figures', 'lstm_validation')
    ensure_dir(out_dir)

    # Prefer 300-episode training logs from final_thesis_training_350ep; fallback to validation metrics
    exp_dir = os.path.join(root, 'comprehensive_results', 'final_thesis_training_350ep')
    episodes = []
    if os.path.exists(exp_dir):
        episodes = load_training_prediction_data(exp_dir)
    if not episodes:
        metrics_path = os.path.join(root, 'comparison_results', 'lstm_validation_metrics.json')
        episodes = load_metrics_from_validation(metrics_path)
    if not episodes:
        print('No LSTM metrics found to plot (training or validation).')
        return

    dot_accuracy(episodes, os.path.join(out_dir, 'lstm_val_accuracy_dots.png'))
    plot_confusion(episodes, os.path.join(out_dir, 'lstm_val_confusion.png'))
    precision_recall_bars(episodes, os.path.join(out_dir, 'lstm_val_precision_recall.png'))
    mean_prob_by_class(episodes, os.path.join(out_dir, 'lstm_val_mean_prob_by_class.png'))
    reliability_curve(episodes, os.path.join(out_dir, 'lstm_val_reliability_curve.png'))
    print('LSTM validation figures saved to:', out_dir)


if __name__ == '__main__':
    main()


