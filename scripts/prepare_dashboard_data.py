"""
Prepare Dashboard Data from Training Results
Extracts data from existing training logs and prepares for dashboard consumption
Author: D3QN Thesis Project
Date: October 13, 2025

NOTE: If Supabase is enabled, prefer streaming metrics directly to Postgres
      instead of writing local JSON packages.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dashboard_data_logger import DashboardDataLogger, create_vehicle_breakdown_estimate


def load_training_log(log_file: str):
    """Load training log from JSONL file."""
    
    episodes = []
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                episode_data = json.loads(line.strip())
                episodes.append(episode_data)
            except json.JSONDecodeError:
                continue
    
    return episodes


def load_evaluation_results(eval_dir: str):
    """Load evaluation results from directory."""
    
    d3qn_file = os.path.join(eval_dir, 'd3qn_results.json')
    fixed_file = os.path.join(eval_dir, 'fixed_time_results.json')
    
    d3qn_episodes = []
    fixed_episodes = []
    
    if os.path.exists(d3qn_file):
        with open(d3qn_file, 'r') as f:
            d3qn_data = json.load(f)
            d3qn_episodes = d3qn_data.get('episodes', [])
    
    if os.path.exists(fixed_file):
        with open(fixed_file, 'r') as f:
            fixed_data = json.load(f)
            fixed_episodes = fixed_data.get('episodes', [])
    
    return d3qn_episodes, fixed_episodes


def convert_training_to_dashboard_format(training_episodes, logger):
    """Convert training log format to dashboard format."""
    
    dashboard_episodes = []
    
    for i, ep in enumerate(training_episodes):
        episode_num = ep.get('episode', i + 1)
        
        # Extract metrics
        metrics = {
            'completed_trips': ep.get('completed_trips', 0),
            'waiting_time': ep.get('avg_waiting_time', 0),
            'avg_speed': ep.get('avg_speed', 0),
            'queue_length': ep.get('avg_queue_length', 0),
            'max_queue_length': ep.get('max_queue_length', 0),
            'total_reward': ep.get('total_reward', 0),
            'loss': ep.get('avg_loss', 0),
            'epsilon': ep.get('epsilon', 0),
            'scenario': ep.get('scenario', f'Episode {episode_num}'),
            'steps': ep.get('steps', 60)
        }
        
        # Estimate vehicle breakdown
        vehicle_breakdown = create_vehicle_breakdown_estimate(
            int(metrics['completed_trips'])
        )
        
        # Determine phase
        phase = 'offline' if episode_num <= 70 else 'online'
        
        # Log episode
        dashboard_ep = logger.log_episode(
            episode_number=episode_num,
            metrics=metrics,
            vehicle_breakdown=vehicle_breakdown,
            phase=phase
        )
        
        dashboard_episodes.append(dashboard_ep)
    
    return dashboard_episodes


def convert_evaluation_to_dashboard_format(eval_episodes, logger, agent_type='d3qn'):
    """Convert evaluation format to dashboard format."""
    
    dashboard_episodes = []
    
    for i, ep in enumerate(eval_episodes):
        episode_num = i + 1
        
        # Extract metrics
        metrics = {
            'completed_trips': ep.get('completed_trips', 0),
            'waiting_time': ep.get('avg_waiting_time', 0),
            'avg_speed': ep.get('avg_speed', 0),
            'queue_length': ep.get('avg_queue_length', 0),
            'max_queue_length': ep.get('max_queue_length', 0),
            'total_reward': ep.get('total_reward', 0) if agent_type == 'd3qn' else 0,
            'loss': 0,
            'epsilon': 0,
            'scenario': ep.get('scenario', f'Eval {episode_num}'),
            'steps': 60
        }
        
        # Estimate vehicle breakdown
        vehicle_breakdown = create_vehicle_breakdown_estimate(
            int(metrics['completed_trips'])
        )
        
        # Create episode data (don't log to file, just format)
        dashboard_ep = {
            'episode_number': episode_num,
            'vehicles_completed': int(metrics['completed_trips']),
            'passengers_completed': int(logger._calculate_passengers(vehicle_breakdown)),
            'vehicle_breakdown': vehicle_breakdown,
            'passenger_breakdown': logger._calculate_passenger_breakdown(vehicle_breakdown),
            'avg_waiting_time': float(metrics['waiting_time']),
            'avg_speed': float(metrics['avg_speed']),
            'avg_queue_length': float(metrics['queue_length']),
            'max_queue_length': int(metrics['max_queue_length']),
            'total_reward': float(metrics['total_reward']),
            'phase': 'evaluation'
        }
        
        dashboard_episodes.append(dashboard_ep)
    
    return dashboard_episodes


def main():
    parser = argparse.ArgumentParser(description='Prepare dashboard data from training results')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Name of experiment (e.g., final_defense_training_350ep)')
    parser.add_argument('--training_log', type=str, default=None,
                       help='Path to training_log.jsonl (optional)')
    parser.add_argument('--eval_dir', type=str, default=None,
                       help='Path to evaluation results directory (optional)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("📊 PREPARING DASHBOARD DATA")
    print("="*80)
    print(f"Experiment: {args.experiment_name}")
    print("="*80 + "\n")
    
    # Create logger
    output_dir = f'dashboard_data/{args.experiment_name}'
    logger = DashboardDataLogger(output_dir=output_dir)
    
    # Process training log if available
    if args.training_log and os.path.exists(args.training_log):
        print(f"📖 Loading training log: {args.training_log}")
        training_episodes = load_training_log(args.training_log)
        print(f"✅ Loaded {len(training_episodes)} training episodes")
        
        print("🔄 Converting to dashboard format...")
        dashboard_episodes = convert_training_to_dashboard_format(training_episodes, logger)
        print(f"✅ Converted {len(dashboard_episodes)} episodes\n")
    
    # Process evaluation results if available
    d3qn_eval = []
    fixed_eval = []
    
    if args.eval_dir and os.path.exists(args.eval_dir):
        print(f"📖 Loading evaluation results: {args.eval_dir}")
        d3qn_episodes, fixed_episodes = load_evaluation_results(args.eval_dir)
        
        if d3qn_episodes:
            print(f"✅ Loaded {len(d3qn_episodes)} D3QN evaluation episodes")
            d3qn_eval = convert_evaluation_to_dashboard_format(d3qn_episodes, logger, 'd3qn')
        
        if fixed_episodes:
            print(f"✅ Loaded {len(fixed_episodes)} Fixed-Time evaluation episodes")
            fixed_eval = convert_evaluation_to_dashboard_format(fixed_episodes, logger, 'fixed_time')
        
        # Generate summary comparison
        if d3qn_eval and fixed_eval:
            print("\n🔄 Generating summary comparison...")
            summary = logger.generate_summary(d3qn_eval, fixed_eval)
            print("✅ Summary generated\n")
            
            # Print improvements
            improvements = summary['improvements']
            print("📊 PERFORMANCE IMPROVEMENTS:")
            print(f"  • Throughput: {improvements['throughput']:+.1f}%")
            print(f"    └─ {improvements['explanations']['throughput']}")
            print(f"  • Passengers: {improvements['passengers']:+.1f}%")
            print(f"    └─ {improvements['explanations']['passengers']}")
            print(f"  • Waiting Time: {improvements['waiting_time']:+.1f}%")
            print(f"    └─ {improvements['explanations']['waiting_time']}")
            print(f"  • Speed: {improvements['speed']:+.1f}%")
            print(f"    └─ {improvements['explanations']['speed']}")
            print(f"  • Queue Length: {improvements['queue_length']:+.1f}%")
            print(f"    └─ {improvements['explanations']['queue_length']}")
            print()
    
    # Prepare final dashboard package
    print("📦 Preparing dashboard package...")
    package_file = logger.prepare_dashboard_package(args.experiment_name)
    
    print("\n✅ DASHBOARD DATA PREPARATION COMPLETE!")
    print(f"📁 Dashboard package: {package_file}")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("1. Copy dashboard_package.json to your dashboard frontend")
    print("2. Update dashboard to read from the JSON file")
    print("3. Test all dashboard components with real data")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

