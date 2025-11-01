import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def load_training_data(filepath):
    """Load training results and extract passenger throughput data"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    episodes = []
    passenger_throughput = []
    
    for result in data['training_results']:
        episodes.append(result['episode'])
        passenger_throughput.append(result['passenger_throughput'])
    
    return episodes, passenger_throughput

def calculate_stability_metrics(data, name):
    """Calculate various stability metrics"""
    data = np.array(data)
    
    metrics = {
        'name': name,
        'mean': np.mean(data),
        'std': np.std(data),
        'cv': np.std(data) / np.mean(data),  # Coefficient of Variation
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }
    
    return metrics

def analyze_trend_stability(data, window=50):
    """Analyze trend stability using rolling statistics"""
    data = np.array(data)
    
    # Rolling mean and std
    rolling_mean = pd.Series(data).rolling(window=window, min_periods=1).mean()
    rolling_std = pd.Series(data).rolling(window=window, min_periods=1).std()
    
    # Trend consistency (lower is more stable)
    trend_volatility = np.std(np.diff(rolling_mean.dropna()))
    
    # Stability score (lower is more stable)
    stability_score = np.mean(rolling_std) / np.mean(rolling_mean)
    
    return {
        'trend_volatility': trend_volatility,
        'stability_score': stability_score,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std
    }

def main():
    # Load data from both training runs
    thesis_episodes, thesis_pt = load_training_data('comprehensive_results/final_thesis_training_350ep/complete_results.json')
    defense_episodes, defense_pt = load_training_data('comprehensive_results/final_defense_training_350ep/complete_results.json')
    
    # Cap both to 300 episodes for fair comparison
    thesis_pt = thesis_pt[:300]
    defense_pt = defense_pt[:300]
    
    print("PASSENGER THROUGHPUT STABILITY ANALYSIS")
    print("=" * 50)
    
    # Calculate stability metrics
    thesis_metrics = calculate_stability_metrics(thesis_pt, "Final Thesis Training")
    defense_metrics = calculate_stability_metrics(defense_pt, "Final Defense Training")
    
    # Print comparison table
    print(f"\n{'Metric':<20} {'Thesis':<15} {'Defense':<15} {'More Stable':<15}")
    print("-" * 65)
    
    metrics_to_compare = ['mean', 'std', 'cv', 'range', 'iqr', 'skewness', 'kurtosis']
    for metric in metrics_to_compare:
        thesis_val = thesis_metrics[metric]
        defense_val = defense_metrics[metric]
        
        if metric in ['cv', 'std', 'range', 'iqr', 'skewness', 'kurtosis']:
            # Lower is better for these metrics
            more_stable = "Defense" if defense_val < thesis_val else "Thesis"
        else:
            # Higher mean is better
            more_stable = "Defense" if defense_val > thesis_val else "Thesis"
        
        print(f"{metric:<20} {thesis_val:<15.2f} {defense_val:<15.2f} {more_stable:<15}")
    
    # Trend stability analysis
    thesis_trend = analyze_trend_stability(thesis_pt)
    defense_trend = analyze_trend_stability(defense_pt)
    
    print(f"\n{'Trend Analysis':<20} {'Thesis':<15} {'Defense':<15} {'More Stable':<15}")
    print("-" * 65)
    print(f"{'Trend Volatility':<20} {thesis_trend['trend_volatility']:<15.2f} {defense_trend['trend_volatility']:<15.2f} {'Defense' if defense_trend['trend_volatility'] < thesis_trend['trend_volatility'] else 'Thesis':<15}")
    print(f"{'Stability Score':<20} {thesis_trend['stability_score']:<15.4f} {defense_trend['stability_score']:<15.4f} {'Defense' if defense_trend['stability_score'] < thesis_trend['stability_score'] else 'Thesis':<15}")
    
    # Statistical tests
    print(f"\nSTATISTICAL TESTS")
    print("-" * 30)
    
    # F-test for variance comparison
    f_stat, f_pvalue = stats.f_oneway(thesis_pt, defense_pt)
    levene_stat, levene_pvalue = stats.levene(thesis_pt, defense_pt)
    
    print(f"F-test p-value: {f_pvalue:.6f}")
    print(f"Levene test p-value: {levene_pvalue:.6f}")
    
    if levene_pvalue < 0.05:
        print("Significant difference in variance (p < 0.05)")
    else:
        print("No significant difference in variance (p >= 0.05)")
    
    # Coefficient of Variation comparison
    thesis_cv = thesis_metrics['cv']
    defense_cv = defense_metrics['cv']
    
    print(f"\nCOEFFICIENT OF VARIATION COMPARISON")
    print(f"Thesis CV: {thesis_cv:.4f}")
    print(f"Defense CV: {defense_cv:.4f}")
    print(f"Relative stability: {'Defense' if defense_cv < thesis_cv else 'Thesis'} is {(abs(thesis_cv - defense_cv) / max(thesis_cv, defense_cv)) * 100:.1f}% more stable")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Passenger Throughput Stability Comparison', fontsize=16, fontweight='bold')
    
    # Raw data comparison
    axes[0,0].plot(thesis_pt, 'b-', alpha=0.7, label='Thesis Training')
    axes[0,0].plot(defense_pt, 'r-', alpha=0.7, label='Defense Training')
    axes[0,0].set_title('Raw Passenger Throughput (300 Episodes)')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Passenger Throughput')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Rolling mean comparison
    thesis_rolling = pd.Series(thesis_pt).rolling(window=50, min_periods=1).mean()
    defense_rolling = pd.Series(defense_pt).rolling(window=50, min_periods=1).mean()
    
    axes[0,1].plot(thesis_rolling, 'b-', linewidth=2, label='Thesis (50-ep rolling mean)')
    axes[0,1].plot(defense_rolling, 'r-', linewidth=2, label='Defense (50-ep rolling mean)')
    axes[0,1].set_title('Rolling Mean Comparison (50-Episode Window)')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Rolling Mean Passenger Throughput')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Distribution comparison
    axes[1,0].hist(thesis_pt, bins=30, alpha=0.7, label='Thesis', color='blue', density=True)
    axes[1,0].hist(defense_pt, bins=30, alpha=0.7, label='Defense', color='red', density=True)
    axes[1,0].set_title('Distribution Comparison')
    axes[1,0].set_xlabel('Passenger Throughput')
    axes[1,0].set_ylabel('Density')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Box plot comparison
    data_for_box = [thesis_pt, defense_pt]
    axes[1,1].boxplot(data_for_box, labels=['Thesis', 'Defense'])
    axes[1,1].set_title('Box Plot Comparison')
    axes[1,1].set_ylabel('Passenger Throughput')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results/training_stability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved: comparison_results/training_stability_comparison.png")
    
    # Summary conclusion
    print(f"\nSUMMARY")
    print("=" * 20)
    
    stability_indicators = {
        'CV': (defense_cv < thesis_cv, 'Defense' if defense_cv < thesis_cv else 'Thesis'),
        'Range': (defense_metrics['range'] < thesis_metrics['range'], 'Defense' if defense_metrics['range'] < thesis_metrics['range'] else 'Thesis'),
        'IQR': (defense_metrics['iqr'] < thesis_metrics['iqr'], 'Defense' if defense_metrics['iqr'] < thesis_metrics['iqr'] else 'Thesis'),
        'Trend Volatility': (defense_trend['trend_volatility'] < thesis_trend['trend_volatility'], 'Defense' if defense_trend['trend_volatility'] < thesis_trend['trend_volatility'] else 'Thesis'),
        'Stability Score': (defense_trend['stability_score'] < thesis_trend['stability_score'], 'Defense' if defense_trend['stability_score'] < thesis_trend['stability_score'] else 'Thesis')
    }
    
    defense_wins = sum(1 for is_better, _ in stability_indicators.values() if is_better)
    thesis_wins = len(stability_indicators) - defense_wins
    
    print(f"Defense Training wins: {defense_wins}/5 stability indicators")
    print(f"Thesis Training wins: {thesis_wins}/5 stability indicators")
    
    if defense_wins > thesis_wins:
        print("CONCLUSION: Final Defense Training is MORE STABLE")
    elif thesis_wins > defense_wins:
        print("CONCLUSION: Final Thesis Training is MORE STABLE")
    else:
        print("CONCLUSION: Both training runs show similar stability")

if __name__ == "__main__":
    main()


