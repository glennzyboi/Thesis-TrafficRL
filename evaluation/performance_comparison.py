"""
Comprehensive Performance Comparison between D3QN and Fixed-Time Control
Implements visualization and statistical analysis based on SUMO+RL research standards
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from datetime import datetime

# Set SUMO_HOME first
if 'SUMO_HOME' not in os.environ:
    possible_paths = [
        r'C:\Program Files (x86)\Eclipse\Sumo',
        r'C:\Program Files\Eclipse\Sumo', 
        r'C:\sumo',
        r'C:\Users\%USERNAME%\sumo'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['SUMO_HOME'] = path
            break

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

import traci

# Import our controllers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.fixed_time_baseline import run_fixed_time_baseline
from experiments.train_d3qn import load_scenarios_index, select_random_bundle
from core.traffic_env import TrafficEnvironment
from algorithms.d3qn_agent import D3QNAgent


class PerformanceComparator:
    """
    Comprehensive performance comparison system for traffic signal control
    Based on established SUMO+RL research methodologies
    """
    
    def __init__(self, output_dir="comparison_results"):
        """Initialize performance comparator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Standard performance metrics from SUMO+RL literature
        self.metrics = [
            'avg_waiting_time',    # Primary metric - user experience
            'avg_throughput',      # Network efficiency
            'avg_speed',           # Traffic flow quality  
            'avg_queue_length',    # Congestion measure
            'completed_trips',     # Service level
            'travel_time_index',   # Mobility efficiency
            'max_queue_length'     # Worst-case performance
        ]
        
        # Results storage
        self.results = {
            'fixed_time': [],
            'd3qn': [],
            'scenarios': []
        }
        
        print(f"📊 Performance Comparator Initialized")
        print(f"   Output Directory: {output_dir}")
        print(f"   Metrics: {len(self.metrics)} standard SUMO+RL metrics")
    
    def run_comprehensive_comparison(self, num_episodes=6):
        """
        Run comprehensive comparison across multiple scenarios
        
        Args:
            num_episodes: Number of episodes to test (should match available bundles)
        """
        print(f"\n🚀 COMPREHENSIVE D3QN vs FIXED-TIME COMPARISON")
        print("=" * 80)
        print(f"📋 Testing {num_episodes} episodes with both control methods")
        print(f"📊 Metrics: {', '.join(self.metrics)}")
        
        # Load available scenarios
        bundles = load_scenarios_index()
        if not bundles:
            print("❌ No traffic bundles available!")
            return
        
        available_bundles = min(len(bundles), num_episodes)
        print(f"✅ Found {len(bundles)} traffic bundles, testing {available_bundles}")
        
        # Test each scenario with both methods
        for episode in range(available_bundles):
            print(f"\n" + "="*60)
            print(f"📺 EPISODE {episode + 1}/{available_bundles}")
            print("="*60)
            
            # Select bundle for this episode
            if episode < len(bundles):
                bundle = bundles[episode]
                route_file = bundle['consolidated_file']
                scenario_name = bundle['name']
            else:
                # Random selection if we run out of bundles
                bundle, route_file = select_random_bundle(bundles)
                scenario_name = bundle['name']
            
            print(f"🎯 Scenario: {scenario_name}")
            print(f"📂 Route File: {os.path.basename(route_file)}")
            
            # Store scenario info
            self.results['scenarios'].append({
                'episode': episode + 1,
                'scenario': scenario_name,
                'route_file': route_file
            })
            
            # Run Fixed-Time Baseline
            print(f"\n🔧 Running Fixed-Time Control...")
            try:
                # Ensure no existing SUMO connection
                if traci.isLoaded():
                    traci.close()
                    
                fixed_metrics = run_fixed_time_baseline(route_file, episodes=1)[0]
                fixed_metrics['episode'] = episode + 1
                fixed_metrics['scenario'] = scenario_name
                self.results['fixed_time'].append(fixed_metrics)
                print(f"✅ Fixed-Time completed: {fixed_metrics['avg_throughput']:.1f} veh/h")
            except Exception as e:
                print(f"❌ Fixed-Time failed: {e}")
                continue
            
            # Run D3QN Agent
            print(f"\n🧠 Running D3QN Agent...")
            try:
                # Ensure no existing SUMO connection
                if traci.isLoaded():
                    traci.close()
                    
                d3qn_metrics = self._run_d3qn_episode(route_file, episode + 1)
                d3qn_metrics['episode'] = episode + 1
                d3qn_metrics['scenario'] = scenario_name
                self.results['d3qn'].append(d3qn_metrics)
                print(f"✅ D3QN completed: {d3qn_metrics['avg_throughput']:.1f} veh/h")
            except Exception as e:
                print(f"❌ D3QN failed: {e}")
                continue
            
            # Quick comparison for this episode
            if len(self.results['fixed_time']) > 0 and len(self.results['d3qn']) > 0:
                self._print_episode_comparison(episode)
        
        # Generate comprehensive analysis
        self._generate_comprehensive_analysis()
        
        print(f"\n🎉 COMPARISON COMPLETED!")
        print(f"📊 Results saved to: {self.output_dir}")
    
    def _run_d3qn_episode(self, route_file, episode_num):
        """Run single D3QN episode and extract metrics"""
        # Initialize environment with realistic constraints
        env = TrafficEnvironment(
            net_file='network/ThesisNetowrk.net.xml',
            rou_file=route_file,
            use_gui=False,  # Disable GUI for batch comparison
            num_seconds=180,
            warmup_time=30,
            step_length=1.0,
            min_phase_time=8,   # Research-based timing constraints
            max_phase_time=90   # Optimized for urban arterials
        )
        
        # Load pre-trained D3QN model if available
        agent = None
        model_path = "models/best_d3qn_model.keras"
        
        if os.path.exists(model_path):
            # Use trained model
            initial_state = env.reset()
            state_size = len(initial_state)
            action_size = env.action_size
            
            agent = D3QNAgent(
                state_size=state_size, 
                action_size=action_size, 
                sequence_length=10  # Match training configuration
            )
            try:
                agent.load(model_path)
                agent.epsilon = 0.0  # No exploration for evaluation
                print(f"   📁 Loaded trained model: {model_path}")
            except Exception as e:
                print(f"   ⚠️ Failed to load model: {e}")
                print(f"   🆕 Using new agent instead")
                agent.epsilon = 0.1  # Small exploration for new agent
        else:
            # Train a quick agent (simplified for comparison)
            initial_state = env.reset()
            state_size = len(initial_state)
            action_size = env.action_size
            
            agent = D3QNAgent(
                state_size=state_size, 
                action_size=action_size, 
                sequence_length=10  # Match training configuration
            )
            print(f"   🆕 Using new agent (no pre-trained model found)")
        
        # Run episode
        state = env.reset()
        step_data = []
        total_reward = 0
        
        for step in range(150):  # 150 steps = 150s after warmup
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            # Collect step metrics
            step_data.append({
                'step': step,
                'vehicles': info.get('vehicles', 0),
                'waiting_time': info.get('waiting_time', 0),
                'avg_speed': info.get('avg_speed', 0),
                'queue_length': info.get('queue_length', 0),
                'completed_trips': info.get('completed_trips', 0),
                'throughput': info.get('throughput', 0)
            })
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Calculate episode metrics
        metrics = self._calculate_episode_metrics(step_data, total_reward)
        
        # Extract public transport metrics from environment reward components
        if hasattr(env, 'reward_components') and env.reward_components:
            last_reward_data = env.reward_components[-1]
            metrics.update({
                'buses_processed': last_reward_data.get('buses_processed', 0),
                'jeepneys_processed': last_reward_data.get('jeepneys_processed', 0),
                'pt_passenger_throughput': last_reward_data.get('pt_passenger_throughput', 0.0),
                'pt_avg_waiting': last_reward_data.get('pt_avg_waiting', 0.0),
                'pt_service_efficiency': last_reward_data.get('pt_service_efficiency', 1.0)
            })
            print(f"   🚌 PT Metrics: {metrics['buses_processed']} buses, {metrics['jeepneys_processed']} jeepneys, "
                  f"{metrics['pt_passenger_throughput']:.0f} PT passengers")
        
        env.close()
        return metrics
    
    def _calculate_episode_metrics(self, step_data, total_reward):
        """Calculate metrics from step data"""
        if not step_data:
            return {}
        
        # Average metrics over episode  
        metrics = {
            'avg_waiting_time': np.mean([d['waiting_time'] for d in step_data]),
            'avg_speed': np.mean([d['avg_speed'] for d in step_data]),
            'avg_queue_length': np.mean([d['queue_length'] for d in step_data]),
            'max_queue_length': max([d['queue_length'] for d in step_data]),
            'completed_trips': step_data[-1]['completed_trips'],  # Cumulative
            'avg_throughput': np.mean([d['throughput'] for d in step_data if d['throughput'] > 0]),
            'total_reward': total_reward,
            'travel_time_index': 40.0 / max(np.mean([d['avg_speed'] for d in step_data]), 1.0),
            # Public Transport Specific Metrics (Research-Based Enhancement)
            'buses_processed': 0,  # Will be updated in D3QN test
            'jeepneys_processed': 0,  # Will be updated in D3QN test  
            'pt_passenger_throughput': 0.0,  # Will be updated in D3QN test
            'pt_avg_waiting': 0.0,  # Will be updated in D3QN test
            'pt_service_efficiency': 1.0  # Will be updated in D3QN test
        }
        
        return metrics
    
    def _print_episode_comparison(self, episode):
        """Print quick comparison for current episode"""
        fixed = self.results['fixed_time'][-1]
        d3qn = self.results['d3qn'][-1]
        
        print(f"\n📊 Episode {episode + 1} Comparison:")
        print(f"   Throughput:    Fixed-Time: {fixed['avg_throughput']:6.1f} veh/h | "
              f"D3QN: {d3qn['avg_throughput']:6.1f} veh/h | "
              f"Improvement: {((d3qn['avg_throughput'] - fixed['avg_throughput']) / fixed['avg_throughput'] * 100):+5.1f}%")
        print(f"   Waiting Time:  Fixed-Time: {fixed['avg_waiting_time']:6.2f}s    | "
              f"D3QN: {d3qn['avg_waiting_time']:6.2f}s    | "
              f"Improvement: {((fixed['avg_waiting_time'] - d3qn['avg_waiting_time']) / fixed['avg_waiting_time'] * 100):+5.1f}%")
        print(f"   Avg Speed:     Fixed-Time: {fixed['avg_speed']:6.1f} km/h | "
              f"D3QN: {d3qn['avg_speed']:6.1f} km/h | "
              f"Improvement: {((d3qn['avg_speed'] - fixed['avg_speed']) / fixed['avg_speed'] * 100):+5.1f}%")
    
    def _generate_comprehensive_analysis(self):
        """Generate comprehensive analysis and visualizations"""
        print(f"\n📈 Generating Comprehensive Analysis...")
        
        # Convert results to DataFrames
        fixed_df = pd.DataFrame(self.results['fixed_time'])
        d3qn_df = pd.DataFrame(self.results['d3qn'])
        
        if fixed_df.empty or d3qn_df.empty:
            print("❌ Insufficient data for analysis")
            return
        
        # Save raw data
        fixed_df.to_csv(f"{self.output_dir}/fixed_time_results.csv", index=False)
        d3qn_df.to_csv(f"{self.output_dir}/d3qn_results.csv", index=False)
        
        # Generate comparison report
        self._generate_comparison_report(fixed_df, d3qn_df)
        
        # Generate visualizations
        self._generate_visualizations(fixed_df, d3qn_df)
        
        # Statistical analysis
        self._generate_statistical_analysis(fixed_df, d3qn_df)
        
        print(f"✅ Analysis complete - files saved to {self.output_dir}")
    
    def _generate_comparison_report(self, fixed_df, d3qn_df):
        """Generate detailed comparison report"""
        report_file = f"{self.output_dir}/performance_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE PERFORMANCE COMPARISON REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Episodes: {len(fixed_df)}\n\n")
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            
            for metric in self.metrics:
                if metric in fixed_df.columns and metric in d3qn_df.columns:
                    fixed_mean = fixed_df[metric].mean()
                    d3qn_mean = d3qn_df[metric].mean()
                    improvement = ((d3qn_mean - fixed_mean) / fixed_mean) * 100
                    
                    if metric == 'avg_waiting_time':  # Lower is better
                        improvement = -improvement
                    
                    f.write(f"{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  Fixed-Time: {fixed_mean:.2f}\n")
                    f.write(f"  D3QN:       {d3qn_mean:.2f}\n")
                    f.write(f"  Improvement: {improvement:+.1f}%\n\n")
            
            f.write("EPISODE-BY-EPISODE RESULTS\n")
            f.write("-" * 30 + "\n")
            
            for i in range(len(fixed_df)):
                f.write(f"Episode {i+1}:\n")
                f.write(f"  Scenario: {fixed_df.iloc[i]['scenario']}\n")
                f.write(f"  Throughput: {fixed_df.iloc[i]['avg_throughput']:.1f} -> {d3qn_df.iloc[i]['avg_throughput']:.1f} veh/h\n")
                f.write(f"  Waiting: {fixed_df.iloc[i]['avg_waiting_time']:.2f} -> {d3qn_df.iloc[i]['avg_waiting_time']:.2f}s\n\n")
        
        print(f"📄 Performance report saved: {report_file}")
    
    def _generate_visualizations(self, fixed_df, d3qn_df):
        """Generate comprehensive visualizations"""
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create comparison plots
        self._plot_metric_comparison(fixed_df, d3qn_df)
        self._plot_episode_trends(fixed_df, d3qn_df)
        self._plot_performance_radar(fixed_df, d3qn_df)
        self._plot_improvement_analysis(fixed_df, d3qn_df)
        
        print(f"📊 Visualizations saved to {self.output_dir}")
    
    def _plot_metric_comparison(self, fixed_df, d3qn_df):
        """Plot side-by-side metric comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Performance Metric Comparison: D3QN vs Fixed-Time', fontsize=16, fontweight='bold')
        
        plot_metrics = ['avg_throughput', 'avg_waiting_time', 'avg_speed', 
                       'avg_queue_length', 'completed_trips', 'max_queue_length']
        
        for i, metric in enumerate(plot_metrics):
            if i >= 6:
                break
            
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            if metric in fixed_df.columns and metric in d3qn_df.columns:
                # Box plot comparison
                data_to_plot = [fixed_df[metric], d3qn_df[metric]]
                box_plot = ax.boxplot(data_to_plot, labels=['Fixed-Time', 'D3QN'], patch_artist=True)
                
                # Color the boxes
                box_plot['boxes'][0].set_facecolor('lightblue')
                box_plot['boxes'][1].set_facecolor('lightgreen')
                
                ax.set_title(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                
                # Add improvement percentage
                fixed_mean = fixed_df[metric].mean()
                d3qn_mean = d3qn_df[metric].mean()
                if metric == 'avg_waiting_time':
                    improvement = ((fixed_mean - d3qn_mean) / fixed_mean) * 100
                else:
                    improvement = ((d3qn_mean - fixed_mean) / fixed_mean) * 100
                
                ax.text(0.5, 0.95, f'Improvement: {improvement:+.1f}%', 
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/metric_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_episode_trends(self, fixed_df, d3qn_df):
        """Plot episode-by-episode trends"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Episode Trends: Performance Over Different Scenarios', fontsize=14, fontweight='bold')
        
        key_metrics = ['avg_throughput', 'avg_waiting_time', 'avg_speed', 'avg_queue_length']
        
        for i, metric in enumerate(key_metrics):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if metric in fixed_df.columns and metric in d3qn_df.columns:
                episodes = range(1, len(fixed_df) + 1)
                ax.plot(episodes, fixed_df[metric], 'o-', label='Fixed-Time', linewidth=2, markersize=6)
                ax.plot(episodes, d3qn_df[metric], 's-', label='D3QN', linewidth=2, markersize=6)
                
                ax.set_xlabel('Episode')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xticks(episodes)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/episode_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_radar(self, fixed_df, d3qn_df):
        """Plot radar chart for overall performance comparison"""
        # Normalize metrics for radar chart
        metrics_for_radar = ['avg_throughput', 'avg_speed', 'completed_trips']
        reverse_metrics = ['avg_waiting_time', 'avg_queue_length']  # Lower is better
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate normalized values
        angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar + reverse_metrics), endpoint=False)
        
        fixed_values = []
        d3qn_values = []
        labels = []
        
        for metric in metrics_for_radar:
            if metric in fixed_df.columns and metric in d3qn_df.columns:
                max_val = max(fixed_df[metric].max(), d3qn_df[metric].max())
                fixed_values.append(fixed_df[metric].mean() / max_val)
                d3qn_values.append(d3qn_df[metric].mean() / max_val)
                labels.append(metric.replace('_', ' ').title())
        
        for metric in reverse_metrics:
            if metric in fixed_df.columns and metric in d3qn_df.columns:
                max_val = max(fixed_df[metric].max(), d3qn_df[metric].max())
                # Invert for "lower is better" metrics
                fixed_values.append(1 - (fixed_df[metric].mean() / max_val))
                d3qn_values.append(1 - (d3qn_df[metric].mean() / max_val))
                labels.append(metric.replace('_', ' ').title() + ' (inv)')
        
        # Close the plot
        fixed_values += fixed_values[:1]
        d3qn_values += d3qn_values[:1]
        angles = np.concatenate([angles, [angles[0]]])
        
        # Plot
        ax.plot(angles, fixed_values, 'o-', linewidth=2, label='Fixed-Time', color='blue')
        ax.fill(angles, fixed_values, alpha=0.25, color='blue')
        ax.plot(angles, d3qn_values, 's-', linewidth=2, label='D3QN', color='green')
        ax.fill(angles, d3qn_values, alpha=0.25, color='green')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.savefig(f"{self.output_dir}/performance_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_analysis(self, fixed_df, d3qn_df):
        """Plot improvement analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('D3QN Improvement Analysis', fontsize=16, fontweight='bold')
        
        # Calculate improvements for each episode
        improvements = {}
        for metric in ['avg_throughput', 'avg_waiting_time', 'avg_speed', 'avg_queue_length']:
            if metric in fixed_df.columns and metric in d3qn_df.columns:
                if metric == 'avg_waiting_time' or metric == 'avg_queue_length':
                    # Lower is better
                    improvements[metric] = ((fixed_df[metric] - d3qn_df[metric]) / fixed_df[metric] * 100).tolist()
                else:
                    # Higher is better
                    improvements[metric] = ((d3qn_df[metric] - fixed_df[metric]) / fixed_df[metric] * 100).tolist()
        
        # Plot 1: Improvement by metric
        metric_names = list(improvements.keys())
        avg_improvements = [np.mean(improvements[metric]) for metric in metric_names]
        
        bars = ax1.bar(range(len(metric_names)), avg_improvements, 
                      color=['green' if x > 0 else 'red' for x in avg_improvements])
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Average Improvement (%)')
        ax1.set_title('Average Improvement by Metric')
        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_improvements):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 2: Improvement distribution
        all_improvements = []
        for metric in improvements:
            all_improvements.extend(improvements[metric])
        
        ax2.hist(all_improvements, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Improvement (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of All Improvements')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
        ax2.axvline(x=np.mean(all_improvements), color='green', linestyle='--', alpha=0.7, 
                   label=f'Mean: {np.mean(all_improvements):.1f}%')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/improvement_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_statistical_analysis(self, fixed_df, d3qn_df):
        """Generate comprehensive statistical significance analysis with academic rigor"""
        from scipy import stats
        from scipy.stats import shapiro, levene, wilcoxon
        from statsmodels.stats.multitest import multipletests
        
        analysis_file = f"{self.output_dir}/statistical_analysis.json"
        
        # Check minimum sample size requirement
        sample_size = len(fixed_df)
        min_required = 20  # Academic standard
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': sample_size,
            'sample_size_adequate': sample_size >= min_required,
            'power_analysis': self._calculate_power_analysis(sample_size),
            'metrics_analysis': {}
        }
        
        # Collect p-values for multiple comparison correction
        p_values = []
        
        for metric in self.metrics:
            if metric in fixed_df.columns and metric in d3qn_df.columns:
                fixed_data = fixed_df[metric].values
                d3qn_data = d3qn_df[metric].values
                
                # Test statistical assumptions
                assumptions = self._test_statistical_assumptions(fixed_data, d3qn_data)
                
                # Perform appropriate statistical test
                if assumptions['normality'] and assumptions['equal_variance']:
                    # Parametric: Paired t-test
                    t_stat, p_value = stats.ttest_rel(fixed_data, d3qn_data)
                    test_used = "paired_t_test"
                    statistic = t_stat
                else:
                    # Non-parametric: Wilcoxon signed-rank test
                    try:
                        statistic, p_value = wilcoxon(fixed_data, d3qn_data)
                        test_used = "wilcoxon_signed_rank"
                    except ValueError:
                        # Fallback to t-test if Wilcoxon fails
                        statistic, p_value = stats.ttest_rel(fixed_data, d3qn_data)
                        test_used = "paired_t_test_fallback"
                
                # Calculate effect size (Cohen's d)
                effect_size = self._calculate_cohens_d(fixed_data, d3qn_data)
                
                # Calculate confidence interval
                confidence_interval = self._calculate_confidence_interval(fixed_data, d3qn_data)
                
                analysis['metrics_analysis'][metric] = {
                    'test_used': test_used,
                    'fixed_time_mean': float(np.mean(fixed_data)),
                    'fixed_time_std': float(np.std(fixed_data)),
                    'd3qn_mean': float(np.mean(d3qn_data)),
                    'd3qn_std': float(np.std(d3qn_data)),
                    'test_statistic': float(statistic),
                    'p_value': float(p_value),
                    'effect_size_cohens_d': float(effect_size),
                    'effect_magnitude': self._interpret_effect_size(effect_size),
                    'confidence_interval_95': [float(ci) for ci in confidence_interval],
                    'significant': bool(p_value < 0.05),
                    'assumptions': assumptions
                }
                
                p_values.append(p_value)
        
        # Multiple comparison correction
        if len(p_values) > 1:
            rejected, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
            
            metric_names = list(analysis['metrics_analysis'].keys())
            for i, metric in enumerate(metric_names):
                analysis['metrics_analysis'][metric]['corrected_p_value'] = float(corrected_p[i])
                analysis['metrics_analysis'][metric]['significant_corrected'] = bool(rejected[i])
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Print academic-grade summary
        self._print_statistical_summary(analysis)
        print(f"📈 Enhanced statistical analysis saved: {analysis_file}")
    
    def _calculate_power_analysis(self, sample_size):
        """Calculate statistical power for given sample size"""
        # Simplified power calculation for paired t-test
        # For effect size = 0.5 (medium effect), alpha = 0.05
        if sample_size < 17:
            power = "< 0.8 (inadequate)"
        elif sample_size < 25:
            power = "0.8-0.9 (adequate)"
        else:
            power = "> 0.9 (excellent)"
        
        return {
            'sample_size': sample_size,
            'power_estimate': power,
            'minimum_required': 17,
            'recommended': 25
        }
    
    def _test_statistical_assumptions(self, group1, group2):
        """Test statistical assumptions for parametric tests"""
        # Normality test (Shapiro-Wilk)
        _, p_norm1 = shapiro(group1)
        _, p_norm2 = shapiro(group2)
        normality = (p_norm1 > 0.05) and (p_norm2 > 0.05)
        
        # Equal variance test (Levene's test)
        _, p_levene = levene(group1, group2)
        equal_variance = p_levene > 0.05
        
        return {
            'normality': normality,
            'equal_variance': equal_variance,
            'shapiro_p_group1': float(p_norm1),
            'shapiro_p_group2': float(p_norm2),
            'levene_p': float(p_levene)
        }
    
    def _calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        mean_diff = np.mean(group2) - np.mean(group1)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        return mean_diff / pooled_std if pooled_std != 0 else 0
    
    def _calculate_confidence_interval(self, group1, group2, confidence=0.95):
        """Calculate confidence interval for mean difference"""
        diff = np.array(group2) - np.array(group1)
        mean_diff = np.mean(diff)
        sem_diff = stats.sem(diff)
        ci = stats.t.interval(confidence, len(diff)-1, mean_diff, sem_diff)
        return ci
    
    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size magnitude"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _print_statistical_summary(self, analysis):
        """Print academic-grade statistical summary"""
        print(f"\n📊 STATISTICAL ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Sample Size: {analysis['sample_size']} ({'✅ Adequate' if analysis['sample_size_adequate'] else '❌ Inadequate'})")
        print(f"Power: {analysis['power_analysis']['power_estimate']}")
        
        for metric, stats_data in analysis['metrics_analysis'].items():
            print(f"\n{metric.upper()}:")
            print(f"  Test: {stats_data['test_used']}")
            print(f"  p-value: {stats_data['p_value']:.6f}")
            if 'corrected_p_value' in stats_data:
                print(f"  Corrected p-value: {stats_data['corrected_p_value']:.6f}")
            print(f"  Effect size (Cohen's d): {stats_data['effect_size_cohens_d']:.3f} ({stats_data['effect_magnitude']})")
            print(f"  95% CI: [{stats_data['confidence_interval_95'][0]:.3f}, {stats_data['confidence_interval_95'][1]:.3f}]")
            significant = stats_data.get('significant_corrected', stats_data['significant'])
            print(f"  Significant: {'✅ Yes' if significant else '❌ No'}")
        print(f"{'='*50}")
    
    def run_enhanced_comparison(self, num_episodes=25):  # Renamed to avoid conflict
        """Run comprehensive comparison with adequate sample size"""
        if num_episodes < 20:
            print(f"⚠️ WARNING: {num_episodes} episodes is below academic minimum (20)")
            print(f"   Consider increasing to 25+ for robust statistical analysis")
        
        return self.run_comprehensive_comparison(num_episodes)


if __name__ == "__main__":
    # Run comprehensive comparison
    comparator = PerformanceComparator()
    comparator.run_comprehensive_comparison(num_episodes=3)  # Test with 3 episodes first
