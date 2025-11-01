import json
import os
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread

# Global plotting style for clarity
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


def load_validation(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_episode_level_frames(data: Dict[str, Any]):
    episodes = data.get('episodes', [])

    ep_idx: List[int] = []
    ctrl: List[str] = []
    throughput: List[float] = []
    wait: List[float] = []
    queue: List[float] = []
    speed: List[float] = []

    # per intersection metrics across episodes
    intersections = set()
    per_int_records = []

    # vehicle type totals per episode & controller
    veh_type_records = []

    for ep in episodes:
        ep_num = ep.get('episode')
        for controller in ('fixed_time', 'd3qn'):
            metrics = ep.get(controller, {})
            if not metrics:
                continue

            ep_idx.append(ep_num)
            ctrl.append('Fixed-Time' if controller == 'fixed_time' else 'D3QN')
            throughput.append(float(metrics.get('passenger_throughput', np.nan)))
            wait.append(float(metrics.get('avg_waiting_time', np.nan)))
            queue.append(float(metrics.get('avg_queue_length', np.nan)))
            speed.append(float(metrics.get('avg_speed', np.nan)))

            int_metrics = metrics.get('intersection_metrics', {})
            for int_name, int_vals in int_metrics.items():
                intersections.add(int_name)
                per_int_records.append({
                    'episode': ep_num,
                    'controller': 'Fixed-Time' if controller == 'fixed_time' else 'D3QN',
                    'intersection': int_name,
                    'passenger_throughput': float(int_vals.get('passenger_throughput', np.nan)),
                    'avg_waiting': float(int_vals.get('avg_waiting', np.nan)),
                    'total_queue': float(int_vals.get('total_queue', np.nan)),
                })

            # top-level vehicle type counts (processed)
            veh_types = {
                'cars': metrics.get('cars_processed', 0),
                'buses': metrics.get('buses_processed', 0),
                'jeepneys': metrics.get('jeepneys_processed', 0),
                'motorcycles': metrics.get('motorcycles_processed', 0),
                'trucks': metrics.get('trucks_processed', 0),
                'tricycles': metrics.get('tricycles_processed', 0),
            }
            veh_type_records.append({
                'episode': ep_num,
                'controller': 'Fixed-Time' if controller == 'fixed_time' else 'D3QN',
                **veh_types
            })

    # Episode-level combined arrays
    ep_frame = {
        'episode': np.array(ep_idx),
        'controller': np.array(ctrl),
        'passenger_throughput': np.array(throughput, dtype=float),
        'avg_waiting_time': np.array(wait, dtype=float),
        'avg_queue_length': np.array(queue, dtype=float),
        'avg_speed': np.array(speed, dtype=float),
    }

    return ep_frame, per_int_records, veh_type_records, sorted(list(intersections))


def boxplot_two_groups(values: Dict[str, np.ndarray], title: str, ylabel: str, out_path: str):
    labels = list(values.keys())
    data = [np.asarray(values[k], dtype=float) for k in labels]

    # Compute summary stats
    means = [np.nanmean(d) for d in data]
    ns = [np.sum(~np.isnan(d)) for d in data]
    diff_pct = None
    if all(m is not None and not np.isnan(m) for m in means) and means[0] != 0:
        diff_pct = (means[1] - means[0]) / means[0] * 100.0

    plt.figure(figsize=(8, 5))
    bp = plt.boxplot(data, labels=[f"{labels[i]} (n={ns[i]})" for i in range(len(labels))],
                     showmeans=True, notch=True, patch_artist=True)
    colors = ['#4C78A8', '#59A14F']  # colorblind-friendly
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    plt.title(title)
    plt.ylabel(ylabel)
    if diff_pct is not None:
        plt.annotate(f"Δ mean: {means[1]-means[0]:.2f} ({diff_pct:+.2f}%)",
                     xy=(0.5, 1.02), xycoords='axes fraction', ha='center', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def lineplot_episodes(ep: np.ndarray, y_fixed: np.ndarray, y_agent: np.ndarray, title: str, ylabel: str, out_path: str):
    order = np.argsort(ep)
    ep_sorted = ep[order]
    # extract corresponding for each controller by filtering
    plt.figure(figsize=(10, 5))
    y_f = y_fixed[order]
    y_a = y_agent[order]
    plt.plot(ep_sorted, y_f, label='Fixed-Time (raw)', color='#4C78A8', linewidth=1.5, alpha=0.5)
    plt.plot(ep_sorted, y_a, label='D3QN (raw)', color='#59A14F', linewidth=1.5, alpha=0.5)

    # rolling mean for smoothing (window=5)
    def rolling_mean(x, w=5):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w)/w, mode='same')

    y_f_rm = rolling_mean(y_f)
    y_a_rm = rolling_mean(y_a)
    plt.plot(ep_sorted, y_f_rm, linestyle='-', color='#4C78A8', linewidth=2.5, alpha=0.9, label='Fixed-Time (rolling mean)')
    plt.plot(ep_sorted, y_a_rm, linestyle='-', color='#59A14F', linewidth=2.5, alpha=0.9, label='D3QN (rolling mean)')

    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def barplot_intersection(records: List[Dict[str, Any]], intersection: str, metric: str, title: str, ylabel: str, out_path: str):
    # Aggregate mean over episodes per controller
    values = defaultdict(list)
    for r in records:
        if r['intersection'] == intersection:
            values[r['controller']].append(r.get(metric, np.nan))
    labels = ['Fixed-Time', 'D3QN']
    means = [np.nanmean(values.get(lbl, [np.nan])) for lbl in labels]
    errs = [np.nanstd(values.get(lbl, [np.nan])) for lbl in labels]

    x = np.arange(len(labels))
    width = 0.5
    plt.figure(figsize=(6, 5))
    colors = ['#4C78A8', '#59A14F']
    bars = plt.bar(x, means, yerr=errs, color=colors, alpha=0.85, capsize=6)
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for rect, val in zip(bars, means):
        plt.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{val:.1f}",
                 ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def stacked_bars_vehicle_types(records: List[Dict[str, Any]], out_path: str, title: str):
    # Sum across episodes by controller
    totals = {
        'Fixed-Time': defaultdict(float),
        'D3QN': defaultdict(float)
    }
    veh_keys = ['cars', 'buses', 'jeepneys', 'motorcycles', 'trucks', 'tricycles']
    for r in records:
        c = r['controller']
        for k in veh_keys:
            totals[c][k] += float(r.get(k, 0))

    labels = ['Fixed-Time', 'D3QN']
    x = np.arange(len(labels))
    width = 0.55

    bottoms = np.zeros(len(labels))
    colors = {
        'cars': '#7F7F7F',          # gray
        'buses': '#F28E2B',         # orange (colorblind-safe)
        'jeepneys': '#E15759',      # red
        'motorcycles': '#9467BD',   # purple
        'trucks': '#8C564B',        # brown
        'tricycles': '#FF9DA7'      # pink
    }

    plt.figure(figsize=(10.5, 6.2))
    for k in veh_keys:
        vals = np.array([totals[l][k] for l in labels], dtype=float)
        plt.bar(x, vals, width, bottom=bottoms, label=k.capitalize(), color=colors.get(k, None), alpha=0.9, linewidth=0)
        bottoms += vals

    # Axis and labels
    plt.xticks(x, labels)
    plt.ylabel('Total processed vehicles (summed over 66 episodes)')
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.25)

    # Legend: move below plot to avoid overlap, multi-column
    plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False)

    # Add only overall totals above each stacked bar, with some headroom
    totals_summed = bottoms
    y_max = np.max(totals_summed) if len(totals_summed) else 0
    plt.ylim(0, y_max * 1.12 if y_max > 0 else 1)
    for xi, tot in zip(x, totals_summed):
        plt.text(xi, tot * 1.005, f"{int(round(tot)):,}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def pt_totals_stacked(records: List[Dict[str, Any]], out_path: str, title: str):
    # Sum buses and jeepneys only across episodes by controller
    totals = {
        'Fixed-Time': {'buses': 0.0, 'jeepneys': 0.0},
        'D3QN': {'buses': 0.0, 'jeepneys': 0.0}
    }
    for r in records:
        c = r['controller']
        totals[c]['buses'] += float(r.get('buses', r.get('buses_processed', 0)))
        totals[c]['jeepneys'] += float(r.get('jeepneys', r.get('jeepneys_processed', 0)))

    labels = ['Fixed-Time', 'D3QN']
    x = np.arange(len(labels))
    width = 0.6
    colors = {'buses': '#F28E2B', 'jeepneys': '#E15759'}
    bottoms = np.zeros(len(labels))

    plt.figure(figsize=(7.5, 5))
    for k in ['buses', 'jeepneys']:
        vals = np.array([totals[l][k] for l in labels], dtype=float)
        plt.bar(x, vals, width, bottom=bottoms, label=k.capitalize(), color=colors[k], alpha=0.9)
        bottoms += vals

    plt.xticks(x, labels)
    plt.ylabel('Total PT vehicles (sum of episodes)')
    plt.title(title)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    for xi, tot in zip(x, bottoms):
        plt.text(xi, tot, f"{int(tot):,}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def pt_per_intersection_stacked(data: Dict[str, Any], out_path: str, title: str):
    # Compute per-intersection mean buses/jeepneys per episode for each controller
    intersections = []
    summaries = {}
    for ep in data.get('episodes', []):
        for ctrl_key, ctrl_lab in [('fixed_time', 'Fixed-Time'), ('d3qn', 'D3QN')]:
            cm = (ep.get(ctrl_key, {}) or {}).get('intersection_metrics', {}) or {}
            for ix, vals in cm.items():
                intersections.append(ix)
                vtypes = (vals.get('vehicle_types', {}) or {})
                bus = float(vtypes.get('bus', 0) or 0)
                jeep = float(vtypes.get('jeepney', 0) or 0)
                summaries.setdefault(ix, {'Fixed-Time': {'buses': [] ,'jeepneys': []}, 'D3QN': {'buses': [], 'jeepneys': []}})
                summaries[ix][ctrl_lab]['buses'].append(bus)
                summaries[ix][ctrl_lab]['jeepneys'].append(jeep)
    intersections = sorted(list(set(intersections)))

    # Plot stacked bars per intersection (two bars per intersection: FT, D3QN)
    n = len(intersections)
    fig, ax = plt.subplots(figsize=(max(7.5, 4 + n*1.5), 5))
    x = np.arange(n)
    width = 0.35

    colors = {'buses': '#F28E2B', 'jeepneys': '#E15759'}

    ft_b = np.array([np.nanmean(summaries[ix]['Fixed-Time']['buses']) if summaries.get(ix) else 0 for ix in intersections])
    ft_j = np.array([np.nanmean(summaries[ix]['Fixed-Time']['jeepneys']) if summaries.get(ix) else 0 for ix in intersections])
    ag_b = np.array([np.nanmean(summaries[ix]['D3QN']['buses']) if summaries.get(ix) else 0 for ix in intersections])
    ag_j = np.array([np.nanmean(summaries[ix]['D3QN']['jeepneys']) if summaries.get(ix) else 0 for ix in intersections])

    # Fixed-Time stacks
    ax.bar(x - width/2, ft_b, width, label='FT Buses', color=colors['buses'], alpha=0.85)
    ax.bar(x - width/2, ft_j, width, bottom=ft_b, label='FT Jeepneys', color=colors['jeepneys'], alpha=0.85)
    # D3QN stacks
    ax.bar(x + width/2, ag_b, width, label='D3QN Buses', color=colors['buses'], alpha=0.6)
    ax.bar(x + width/2, ag_j, width, bottom=ag_b, label='D3QN Jeepneys', color=colors['jeepneys'], alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(intersections, rotation=0)
    ax.set_ylabel('Mean PT vehicles per episode')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def compose_lstm_dashboard(fig_dir: str, out_path: str):
    # Compose existing LSTM figures into a compact dashboard (2x2)
    files = [
        os.path.join(fig_dir, 'lstm_val_accuracy_dots.png'),
        os.path.join(fig_dir, 'lstm_val_confusion.png'),
        os.path.join(fig_dir, 'lstm_val_precision_recall.png'),
        os.path.join(fig_dir, 'lstm_val_reliability_curve.png'),
    ]
    imgs = []
    for p in files:
        if os.path.exists(p):
            imgs.append(imread(p))
        else:
            imgs.append(None)

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    titles = ['Accuracy per Episode', 'Confusion Matrix', 'Precision/Recall', 'Reliability Curve']
    for i in range(4):
        ax = fig.add_subplot(gs[i//2, i%2])
        if imgs[i] is not None:
            ax.imshow(imgs[i])
            ax.axis('off')
            ax.set_title(titles[i])
        else:
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center')
            ax.axis('off')
    fig.suptitle('LSTM Validation Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def ecdf_plot(values_a: np.ndarray, values_b: np.ndarray, labels: List[str], title: str, xlabel: str, out_path: str):
    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    xa, ya = ecdf(values_a)
    xb, yb = ecdf(values_b)

    plt.figure(figsize=(8, 5))
    plt.step(xa, ya, where='post', label=labels[0], color='#4C78A8', linewidth=2)
    plt.step(xb, yb, where='post', label=labels[1], color='#59A14F', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Cumulative probability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def mean_ci_bar(values: Dict[str, np.ndarray], title: str, ylabel: str, out_path: str):
    labels = list(values.keys())
    data = [np.asarray(values[k], dtype=float) for k in labels]
    means = np.array([np.nanmean(d) for d in data])
    stds = np.array([np.nanstd(d, ddof=1) for d in data])
    ns = np.array([np.sum(~np.isnan(d)) for d in data])
    ses = stds / np.sqrt(np.maximum(ns, 1))
    ci95 = 1.96 * ses

    x = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    colors = ['#4C78A8', '#59A14F']
    bars = plt.bar(x, means, yerr=ci95, capsize=6, color=colors[:len(labels)], alpha=0.9)
    plt.xticks(x, [f"{labels[i]} (n={ns[i]})" for i in range(len(labels))])
    plt.ylabel(ylabel)
    plt.title(title)
    for rect, m in zip(bars, means):
        plt.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{m:.2f}", ha='center', va='bottom', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def simple_bar_with_improvement(values: Dict[str, np.ndarray], title: str, ylabel: str, out_path: str, higher_is_better: bool = True):
    labels = list(values.keys())
    data = [np.asarray(values[k], dtype=float) for k in labels]
    means = np.array([np.nanmean(d) for d in data])
    x = np.arange(len(labels))
    colors = ['#4C78A8', '#59A14F']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, means, color=colors[:len(labels)], alpha=0.9)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    for rect, m in zip(bars, means):
        plt.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{m:.2f}", ha='center', va='bottom', fontsize=10)

    # annotate percent improvement from first to second
    if len(means) >= 2 and means[0] != 0:
        pct = (means[1] - means[0]) / means[0] * 100.0
        direction = '+' if (pct >= 0 and higher_is_better) or (pct < 0 and not higher_is_better) else ''
        note = f"Improvement: {pct:+.2f}%" if higher_is_better else f"Change: {pct:+.2f}%"
        plt.annotate(note, xy=(0.5, 1.02), xycoords='axes fraction', ha='center', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def dot_jitter_with_mean(values: Dict[str, np.ndarray], title: str, ylabel: str, out_path: str):
    labels = list(values.keys())
    data = [np.asarray(values[k], dtype=float) for k in labels]
    x_positions = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    colors = ['#4C78A8', '#59A14F']
    np.random.seed(42)  # Reproducible jitter
    for i, (x, arr) in enumerate(zip(x_positions, data)):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
        # Better jitter distribution - use wider spread and larger markers
        jitter = (np.random.rand(len(arr)) - 0.5) * 0.4
        plt.scatter(np.full_like(arr, x) + jitter, arr, alpha=0.6, s=40, color=colors[i], 
                   edgecolors='white', linewidth=0.5, label=f"{labels[i]} (n={len(arr)})")
        m = float(np.mean(arr)) if len(arr) else np.nan
        std = float(np.std(arr)) if len(arr) else np.nan
        plt.plot([x-0.4, x+0.4], [m, m], color='black', linewidth=3)
        plt.fill_between([x-0.4, x+0.4], [m-std, m-std], [m+std, m+std], 
                        color='black', alpha=0.15)
        plt.text(x, m + std * 0.5, f"μ={m:.1f}", ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='black',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xticks(x_positions, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def per_episode_delta_line(ep_fixed: np.ndarray, y_fixed: np.ndarray, ep_agent: np.ndarray, y_agent: np.ndarray, title: str, ylabel: str, out_path: str):
    # Align by episode id intersection
    common_eps = np.intersect1d(ep_fixed, ep_agent)
    f_map = {e: v for e, v in zip(ep_fixed, y_fixed)}
    a_map = {e: v for e, v in zip(ep_agent, y_agent)}
    deltas = np.array([a_map[e] - f_map[e] for e in common_eps], dtype=float)

    order = np.argsort(common_eps)
    eps_sorted = common_eps[order]
    deltas_sorted = deltas[order]

    plt.figure(figsize=(10, 5))
    plt.axhline(0.0, color='k', linewidth=1)
    plt.plot(eps_sorted, deltas_sorted, color='#F28E2B', linewidth=2, label='Delta (D3QN - Fixed-Time)')
    # Shade positive/negative regions for quick interpretation
    plt.fill_between(eps_sorted, 0, deltas_sorted, where=deltas_sorted>=0, color='#59A14F', alpha=0.2)
    plt.fill_between(eps_sorted, 0, deltas_sorted, where=deltas_sorted<0, color='#E15759', alpha=0.15)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def scatter_improvement(ep_fixed: np.ndarray, veh_fixed: np.ndarray, ep_agent: np.ndarray, veh_agent: np.ndarray,
                        pass_fixed: np.ndarray, pass_agent: np.ndarray, out_path: str):
    # percentage improvements per episode
    common_eps = np.intersect1d(ep_fixed, ep_agent)
    vf_map = {e: v for e, v in zip(ep_fixed, veh_fixed)}
    va_map = {e: v for e, v in zip(ep_agent, veh_agent)}
    pf_map = {e: v for e, v in zip(ep_fixed, pass_fixed)}
    pa_map = {e: v for e, v in zip(ep_agent, pass_agent)}

    veh_imp = []
    pass_imp = []
    for e in common_eps:
        v0 = vf_map[e]
        v1 = va_map[e]
        p0 = pf_map[e]
        p1 = pa_map[e]
        if v0 != 0 and p0 != 0:
            veh_imp.append((v1 - v0) / v0 * 100.0)
            pass_imp.append((p1 - p0) / p0 * 100.0)

    veh_imp = np.array(veh_imp, dtype=float)
    pass_imp = np.array(pass_imp, dtype=float)

    plt.figure(figsize=(6, 6))
    plt.scatter(veh_imp, pass_imp, alpha=0.8, color='#B07AA1', edgecolor='white', linewidth=0.5)
    lim_min = min(np.min(veh_imp), np.min(pass_imp))
    lim_max = max(np.max(veh_imp), np.max(pass_imp))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5)
    plt.xlabel('Vehicle Throughput Improvement (%)')
    plt.ylabel('Passenger Throughput Improvement (%)')
    plt.title('Passenger vs Vehicle Throughput Improvement per Episode')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def per_intersection_pt_stacked(records: List[Dict[str, Any]], out_dir: str):
    # For each intersection, sum buses+jeepneys counts across episodes and controllers
    # Intersection vehicle types are inside per-intersection metrics? Here we only have top-level counts.
    # Use per-intersection averages for passenger_throughput already plotted; for PT uplift use top-level as proxy.
    # Therefore, we provide overall PT uplift bars per controller (buses+jeepneys totals).
    totals = {'Fixed-Time': 0.0, 'D3QN': 0.0}
    # This uses episode-level totals; intersection split not present at top-level for PT sum, so we show overall.
    plt.figure(figsize=(5, 5))
    # This function intentionally left minimal due to data granularity; handled in stacked_bars_vehicle_types
    plt.close()


def main():
    print('Generating Chapter 4 comparison figures...')
    root = os.path.dirname(os.path.dirname(__file__))
    json_path = os.path.join(root, 'comparison_results', 'validation_dashboard_complete.json')
    out_dir = os.path.join(root, 'Chapter 4', 'figures')
    ensure_dir(out_dir)

    data = load_validation(json_path)
    ep_frame, per_int_records, veh_type_records, intersections = build_episode_level_frames(data)

    # Prepare arrays split by controller in consistent episode order
    ep = ep_frame['episode']
    mask_fixed = ep_frame['controller'] == 'Fixed-Time'
    mask_agent = ep_frame['controller'] == 'D3QN'

    # Sort both masks by episode number to align lines
    order_fixed = np.argsort(ep[mask_fixed])
    order_agent = np.argsort(ep[mask_agent])

    # Overall boxplots
    dot_jitter_with_mean(
        {
            'Fixed-Time': ep_frame['passenger_throughput'][mask_fixed],
            'D3QN': ep_frame['passenger_throughput'][mask_agent]
        },
        title='Passenger Throughput per Episode (with group means)',
        ylabel='Passenger Throughput',
        out_path=os.path.join(out_dir, 'overall_passenger_throughput_dots.png')
    )

    dot_jitter_with_mean(
        {
            'Fixed-Time': ep_frame['avg_waiting_time'][mask_fixed],
            'D3QN': ep_frame['avg_waiting_time'][mask_agent]
        },
        title='Average Waiting Time per Episode (with group means)',
        ylabel='Average Waiting Time (s)',
        out_path=os.path.join(out_dir, 'overall_avg_waiting_time_dots.png')
    )

    dot_jitter_with_mean(
        {
            'Fixed-Time': ep_frame['avg_queue_length'][mask_fixed],
            'D3QN': ep_frame['avg_queue_length'][mask_agent]
        },
        title='Average Queue Length per Episode (with group means)',
        ylabel='Average Queue Length (veh)',
        out_path=os.path.join(out_dir, 'overall_avg_queue_length_dots.png')
    )

    # Episode progression lines (throughput)
    lineplot_episodes(
        ep[mask_fixed][order_fixed],
        ep_frame['passenger_throughput'][mask_fixed][order_fixed],
        ep_frame['passenger_throughput'][mask_agent][order_agent],
        title='Passenger Throughput Across Episodes',
        ylabel='Passenger Throughput',
        out_path=os.path.join(out_dir, 'episodes_passenger_throughput_line.png')
    )

    # Note: Delta plots removed for clarity per feedback

    # Average speed (optional but useful)
    boxplot_two_groups(
        {
            'Fixed-Time': ep_frame['avg_speed'][mask_fixed],
            'D3QN': ep_frame['avg_speed'][mask_agent]
        },
        title='Overall Average Speed per Episode',
        ylabel='Average Speed (m/s)',
        out_path=os.path.join(out_dir, 'overall_avg_speed_box.png')
    )

    # ECDF for waiting time (service quality distribution)
    ecdf_plot(
        ep_frame['avg_waiting_time'][mask_fixed],
        ep_frame['avg_waiting_time'][mask_agent],
        labels=['Fixed-Time', 'D3QN'],
        title='ECDF of Average Waiting Time per Episode',
        xlabel='Average Waiting Time (s)',
        out_path=os.path.join(out_dir, 'ecdf_avg_waiting_time.png')
    )

    # Per-intersection bars for throughput and waiting time
    for int_name in intersections:
        safe = int_name.replace(' ', '_')
        barplot_intersection(
            per_int_records,
            intersection=int_name,
            metric='passenger_throughput',
            title=f'Intersection Throughput: {int_name}',
            ylabel='Passenger Throughput',
            out_path=os.path.join(out_dir, f'intersection_{safe}_throughput_bar.png')
        )
        barplot_intersection(
            per_int_records,
            intersection=int_name,
            metric='avg_waiting',
            title=f'Intersection Average Waiting: {int_name}',
            ylabel='Average Waiting Time (s)',
            out_path=os.path.join(out_dir, f'intersection_{safe}_avg_waiting_bar.png')
        )

    # Vehicle-type stacked bars (totals across episodes)
    stacked_bars_vehicle_types(
        veh_type_records,
        out_path=os.path.join(out_dir, 'vehicle_type_totals_stacked.png'),
        title='Total Processed Vehicles by Type (Summed over Episodes)'
    )

    # Scatter: passenger vs vehicle throughput improvement per episode
    # For vehicle throughput, approximate with completed_trips or vehicles if present; use 'completed_trips' if available.
    # Reconstruct arrays for vehicle throughput proxy from original data
    veh_fixed = []
    veh_agent = []
    pass_fixed = []
    pass_agent = []
    ep_f = []
    ep_a = []
    for ep_item in data.get('episodes', []):
        epnum = ep_item.get('episode')
        ft = ep_item.get('fixed_time', {})
        ag = ep_item.get('d3qn', {})
        if ft and ag:
            ep_f.append(epnum)
            ep_a.append(epnum)
            veh_fixed.append(float(ft.get('completed_trips', ft.get('vehicles', np.nan))))
            veh_agent.append(float(ag.get('completed_trips', ag.get('vehicles', np.nan))))
            pass_fixed.append(float(ft.get('passenger_throughput', np.nan)))
            pass_agent.append(float(ag.get('passenger_throughput', np.nan)))

    scatter_improvement(
        np.array(ep_f), np.array(veh_fixed, dtype=float),
        np.array(ep_a), np.array(veh_agent, dtype=float),
        np.array(pass_fixed, dtype=float), np.array(pass_agent, dtype=float),
        out_path=os.path.join(out_dir, 'scatter_passenger_vs_vehicle_improvement.png')
    )

    # Public Transport comparison visuals
    # 1) Per-episode dots for buses, jeepneys, and combined PT
    buses_fixed = []
    buses_agent = []
    jeep_fixed = []
    jeep_agent = []
    for ep_item in data.get('episodes', []):
        ft = ep_item.get('fixed_time', {})
        ag = ep_item.get('d3qn', {})
        if ft and ag:
            buses_fixed.append(float(ft.get('buses_processed', np.nan)))
            buses_agent.append(float(ag.get('buses_processed', np.nan)))
            jeep_fixed.append(float(ft.get('jeepneys_processed', np.nan)))
            jeep_agent.append(float(ag.get('jeepneys_processed', np.nan)))

    # Dots: buses
    dot_jitter_with_mean(
        {'Fixed-Time': np.array(buses_fixed, dtype=float), 'D3QN': np.array(buses_agent, dtype=float)},
        title='Public Transport: Buses per Episode (with group means)',
        ylabel='Buses processed',
        out_path=os.path.join(out_dir, 'pt_buses_per_episode_dots.png')
    )
    # Dots: jeepneys
    dot_jitter_with_mean(
        {'Fixed-Time': np.array(jeep_fixed, dtype=float), 'D3QN': np.array(jeep_agent, dtype=float)},
        title='Public Transport: Jeepneys per Episode (with group means)',
        ylabel='Jeepneys processed',
        out_path=os.path.join(out_dir, 'pt_jeepneys_per_episode_dots.png')
    )
    # Dots: combined PT
    pt_fixed = (np.array(buses_fixed, dtype=float) + np.array(jeep_fixed, dtype=float)) if buses_fixed else np.array([])
    pt_agent = (np.array(buses_agent, dtype=float) + np.array(jeep_agent, dtype=float)) if buses_agent else np.array([])
    dot_jitter_with_mean(
        {'Fixed-Time': pt_fixed, 'D3QN': pt_agent},
        title='Public Transport: Buses+Jeepneys per Episode (with group means)',
        ylabel='PT vehicles processed',
        out_path=os.path.join(out_dir, 'pt_combined_per_episode_dots.png')
    )

    # 2) Per-intersection PT averages (buses+jeepneys)
    pt_per_int = {}
    for ep_item in data.get('episodes', []):
        for ctrl_key, ctrl_lab in [('fixed_time', 'Fixed-Time'), ('d3qn', 'D3QN')]:
            cm = (ep_item.get(ctrl_key, {}) or {}).get('intersection_metrics', {}) or {}
            for ix, vals in cm.items():
                vtypes = (vals.get('vehicle_types', {}) or {})
                pt = float(vtypes.get('bus', 0) or 0) + float(vtypes.get('jeepney', 0) or 0)
                pt_per_int.setdefault(ix, {'Fixed-Time': [], 'D3QN': []})
                pt_per_int[ix][ctrl_lab].append(pt)

    for ix, ctrl_map in pt_per_int.items():
        safe = ix.replace(' ', '_')
        mean_ci_bar(
            {'Fixed-Time': np.array(ctrl_map['Fixed-Time'], dtype=float), 'D3QN': np.array(ctrl_map['D3QN'], dtype=float)},
            title=f'PT (Buses+Jeepneys) per Episode: {ix}',
            ylabel='PT vehicles processed',
            out_path=os.path.join(out_dir, f'pt_per_episode_mean_ci_{safe}.png')
        )

    # 3) New: Overall PT totals stacked (buses vs jeepneys) per controller
    pt_totals_stacked(
        veh_type_records,
        out_path=os.path.join(out_dir, 'pt_totals_stacked.png'),
        title='Public Transport Totals (Buses vs Jeepneys, summed over episodes)'
    )

    # 4) New: Per-intersection mean stacked PT (buses vs jeepneys)
    pt_per_intersection_stacked(
        data,
        out_path=os.path.join(out_dir, 'pt_per_intersection_stacked.png'),
        title='PT Breakdown per Intersection (mean per episode)'
    )

    # 5) New: LSTM compact dashboard (compose existing validation figures)
    lstm_dir = os.path.join(out_dir, 'lstm_validation')
    ensure_dir(lstm_dir)
    compose_lstm_dashboard(
        fig_dir=lstm_dir,
        out_path=os.path.join(lstm_dir, 'lstm_dashboard.png')
    )

    # 6) New: Secondary metrics summary (Waiting Time, Vehicle Throughput, Queue Length)
    # Compute controller means from episode-level arrays
    # Save in main figures directory for portability

    wt_fixed = ep_frame['avg_waiting_time'][mask_fixed]
    wt_agent = ep_frame['avg_waiting_time'][mask_agent]
    vt_fixed = []
    vt_agent = []
    ql_fixed = ep_frame['avg_queue_length'][mask_fixed]
    ql_agent = ep_frame['avg_queue_length'][mask_agent]

    # Vehicle throughput proxy: completed_trips if available, else vehicles
    vt_f_map = []
    vt_a_map = []
    for ep_item in data.get('episodes', []):
        ft = ep_item.get('fixed_time', {}) or {}
        ag = ep_item.get('d3qn', {}) or {}
        if ft and ag:
            vt_f_map.append(float(ft.get('completed_trips', ft.get('vehicles', np.nan))))
            vt_a_map.append(float(ag.get('completed_trips', ag.get('vehicles', np.nan))))
    vt_fixed = np.array(vt_f_map, dtype=float)
    vt_agent = np.array(vt_a_map, dtype=float)

    # Build bar chart
    metrics = ['Mean Waiting Time (s)', 'Mean Vehicle Throughput (veh/h)', 'Mean Queue Length (veh)']
    fixed_means = [np.nanmean(wt_fixed), np.nanmean(vt_fixed), np.nanmean(ql_fixed)]
    agent_means = [np.nanmean(wt_agent), np.nanmean(vt_agent), np.nanmean(ql_agent)]

    x = np.arange(len(metrics))
    width = 0.35
    colors = ['#4C78A8', '#59A14F']

    plt.figure(figsize=(10, 5.5))
    b1 = plt.bar(x - width/2, fixed_means, width, label='Fixed-Time', color=colors[0], alpha=0.9)
    b2 = plt.bar(x + width/2, agent_means, width, label='D3QN', color=colors[1], alpha=0.9)
    plt.xticks(x, metrics, rotation=12)
    plt.ylabel('Value')
    plt.title('Secondary Metrics: Fixed-Time vs D3QN (66 Scenarios)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()

    # Annotate bars and percent improvements
    for rect, val in zip(b1, fixed_means):
        plt.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{val:.2f}", ha='center', va='bottom', fontsize=9)
    for rect, val in zip(b2, agent_means):
        plt.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{val:.2f}", ha='center', va='bottom', fontsize=9)

    for i, (f, a) in enumerate(zip(fixed_means, agent_means)):
        if f == 0 or np.isnan(f) or np.isnan(a):
            continue
        pct = (a - f) / f * 100.0
        sign = '+' if pct >= 0 else ''
        plt.annotate(f"{sign}{pct:.2f}%", xy=(i, max(f, a) * 1.02), ha='center', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'secondary_metrics_summary.png'), bbox_inches='tight')
    plt.close()

    # Also create a dumbbell chart to emphasize relative changes without duplicating the table
    fixed_arr = np.array(fixed_means, dtype=float)
    agent_arr = np.array(agent_means, dtype=float)
    y = np.arange(len(metrics))
    plt.figure(figsize=(10, 5.5))
    # connectors
    for i in range(len(metrics)):
        x0, x1 = fixed_arr[i], agent_arr[i]
        plt.plot([min(x0, x1), max(x0, x1)], [y[i], y[i]], color='#9E9E9E', linewidth=3, zorder=1)
    # endpoints
    plt.scatter(fixed_arr, y, color='#4C78A8', s=80, zorder=2, label='Fixed-Time', edgecolors='white', linewidths=0.5)
    plt.scatter(agent_arr, y, color='#59A14F', s=80, zorder=2, label='D3QN', edgecolors='white', linewidths=0.5)
    # percent labels at midpoints
    for i, (f, a) in enumerate(zip(fixed_arr, agent_arr)):
        if np.isnan(f) or np.isnan(a) or f == 0:
            continue
        pct = (a - f) / f * 100.0
        mid = (f + a) / 2.0
        sign = '+' if pct >= 0 else ''
        plt.text(mid, y[i] + 0.15, f"{sign}{pct:.2f}%", ha='center', va='bottom', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.yticks(y, ['Waiting Time (s)', 'Vehicle Throughput (veh/h)', 'Queue Length (veh)'])
    plt.xlabel('Value')
    plt.title('Secondary Metrics (66 Scenarios): Fixed-Time vs D3QN')
    plt.grid(True, axis='x', alpha=0.3)
    plt.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'secondary_metrics_dumbbell.png'), bbox_inches='tight')
    plt.close()

    print('Chapter 4 figures created in:', out_dir)


if __name__ == '__main__':
    main()


