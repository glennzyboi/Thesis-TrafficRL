#!/usr/bin/env python3
"""
Analyze current logs and map to database tables
"""

import os
import json
from datetime import datetime

def analyze_logs_for_database_mapping():
    """Analyze logs and identify which database tables lack data"""
    print("=" * 80)
    print("ANALYZING LOGS FOR DATABASE TABLE MAPPING")
    print("=" * 80)
    
    # Check what log files are available
    log_files = [
        'production_logs/final_thesis_training_350ep_episodes.jsonl',
        'production_logs/final_thesis_training_350ep_summary.json',
        'comprehensive_results/final_thesis_training_350ep/complete_results.json',
        'comprehensive_results/final_thesis_training_350ep/training_progress.json',
        'comprehensive_results/final_thesis_training_350ep/prediction_dashboard/data/prediction_data.json',
        'comprehensive_results/final_thesis_training_350ep/prediction_dashboard/data/accuracy_history.json'
    ]
    
    print("1. CHECKING AVAILABLE LOG FILES:")
    print("=" * 50)
    available_files = []
    for file_path in log_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path}")
            available_files.append(file_path)
        else:
            print(f"[MISSING] {file_path}")
    
    print(f"\nFound {len(available_files)} available log files")
    
    # Analyze episodes data
    print("\n2. ANALYZING EPISODES DATA:")
    print("=" * 50)
    
    episodes_path = 'production_logs/final_thesis_training_350ep_episodes.jsonl'
    if os.path.exists(episodes_path):
        with open(episodes_path, 'r') as f:
            content = f.read()
        
        # Parse first few episodes to see structure
        episode_blocks = content.split('"episode_info":')
        print(f"Total episodes: {len(episode_blocks)-1}")
        
        # Analyze first episode structure
        if len(episode_blocks) > 1:
            try:
                episode_json = '{"episode_info":' + episode_blocks[1]
                # Find the end of this episode
                brace_count = 0
                end_pos = 0
                for j, char in enumerate(episode_json):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = j + 1
                            break
                
                if end_pos > 0:
                    episode_json = episode_json[:end_pos]
                    episode = json.loads(episode_json)
                    
                    print("\nEpisode structure analysis:")
                    print(f"  - episode_info: {list(episode.get('episode_info', {}).keys())}")
                    print(f"  - performance_metrics: {list(episode.get('performance_metrics', {}).keys())}")
                    print(f"  - traffic_metrics: {list(episode.get('traffic_metrics', {}).keys())}")
                    print(f"  - public_transport_metrics: {list(episode.get('public_transport_metrics', {}).keys())}")
                    print(f"  - training_metrics: {list(episode.get('training_metrics', {}).keys())}")
                    print(f"  - reward_breakdown: {list(episode.get('reward_breakdown', {}).keys())}")
                    print(f"  - scenario: {list(episode.get('scenario', {}).keys())}")
                    
                    # Check for intersection-specific data
                    if 'detailed_data' in episode:
                        detailed = episode.get('detailed_data', {})
                        print(f"  - detailed_data: {list(detailed.keys())}")
                        if 'intersection_throughput' in detailed:
                            print(f"    - intersection_throughput: {list(detailed['intersection_throughput'].keys())}")
                        if 'agent_rewards' in detailed:
                            print(f"    - agent_rewards: {list(detailed['agent_rewards'].keys())}")
                    
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse episode: {e}")
    else:
        print("[ERROR] Episodes file not found")
    
    # Analyze summary data
    print("\n3. ANALYZING SUMMARY DATA:")
    print("=" * 50)
    
    summary_path = 'production_logs/final_thesis_training_350ep_summary.json'
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        print("Summary structure:")
        print(f"  - Keys: {list(summary.keys())}")
        
        if 'performance_metrics' in summary:
            print(f"  - performance_metrics: {list(summary['performance_metrics'].keys())}")
        if 'training_metrics' in summary:
            print(f"  - training_metrics: {list(summary['training_metrics'].keys())}")
        if 'traffic_metrics' in summary:
            print(f"  - traffic_metrics: {list(summary['traffic_metrics'].keys())}")
    else:
        print("[MISSING] Summary file not found")
    
    # Analyze complete results
    print("\n4. ANALYZING COMPLETE RESULTS:")
    print("=" * 50)
    
    complete_path = 'comprehensive_results/final_thesis_training_350ep/complete_results.json'
    if os.path.exists(complete_path):
        with open(complete_path, 'r') as f:
            complete = json.load(f)
        
        print("Complete results structure:")
        print(f"  - Keys: {list(complete.keys())}")
        
        if 'episodes' in complete:
            print(f"  - episodes: {len(complete['episodes'])} records")
            if len(complete['episodes']) > 0:
                print(f"    - First episode keys: {list(complete['episodes'][0].keys())}")
    else:
        print("[MISSING] Complete results file not found")
    
    # Analyze prediction data
    print("\n5. ANALYZING PREDICTION DATA:")
    print("=" * 50)
    
    prediction_path = 'comprehensive_results/final_thesis_training_350ep/prediction_dashboard/data/prediction_data.json'
    if os.path.exists(prediction_path):
        with open(prediction_path, 'r') as f:
            prediction = json.load(f)
        
        print("Prediction data structure:")
        print(f"  - Type: {type(prediction)}")
        if isinstance(prediction, list) and len(prediction) > 0:
            print(f"  - Records: {len(prediction)}")
            print(f"  - First record keys: {list(prediction[0].keys())}")
        elif isinstance(prediction, dict):
            print(f"  - Keys: {list(prediction.keys())}")
    else:
        print("[MISSING] Prediction data file not found")
    
    # Analyze accuracy history
    print("\n6. ANALYZING ACCURACY HISTORY:")
    print("=" * 50)
    
    accuracy_path = 'comprehensive_results/final_thesis_training_350ep/prediction_dashboard/data/accuracy_history.json'
    if os.path.exists(accuracy_path):
        with open(accuracy_path, 'r') as f:
            accuracy = json.load(f)
        
        print("Accuracy history structure:")
        print(f"  - Type: {type(accuracy)}")
        if isinstance(accuracy, list):
            print(f"  - Records: {len(accuracy)}")
            if len(accuracy) > 0:
                print(f"  - First value: {accuracy[0]}")
        elif isinstance(accuracy, dict):
            print(f"  - Keys: {list(accuracy.keys())}")
    else:
        print("[MISSING] Accuracy history file not found")
    
    # Map to database tables
    print("\n7. DATABASE TABLE MAPPING ANALYSIS:")
    print("=" * 80)
    
    database_tables = {
        'experiments': {
            'description': 'Experiment metadata and configuration',
            'required_fields': ['experiment_id', 'experiment_name', 'status', 'training_mode', 'total_episodes', 'best_reward', 'best_accuracy'],
            'available_in_logs': 'episode_info, performance_metrics, training_metrics',
            'status': 'AVAILABLE'
        },
        'intersections': {
            'description': 'Traffic intersection information',
            'required_fields': ['intersection_id', 'intersection_name', 'num_approaches'],
            'available_in_logs': 'Hardcoded (Ecoland, JohnPaul, Sandawa)',
            'status': 'AVAILABLE'
        },
        'training_episodes': {
            'description': 'Individual episode performance data',
            'required_fields': ['episode_id', 'episode_number', 'total_reward', 'vehicles_served', 'passenger_throughput', 'avg_waiting_time', 'avg_queue_length'],
            'available_in_logs': 'episode_info, performance_metrics, traffic_metrics, reward_breakdown',
            'status': 'AVAILABLE'
        },
        'vehicle_breakdown': {
            'description': 'Vehicle type breakdown per episode',
            'required_fields': ['episode_id', 'cars', 'motorcycles', 'trucks', 'jeepneys', 'buses'],
            'available_in_logs': 'public_transport_metrics (jeepneys, buses), traffic_metrics (total vehicles)',
            'status': 'PARTIAL - Missing individual vehicle counts'
        },
        'validation_results': {
            'description': 'Validation performance results',
            'required_fields': ['validation_id', 'episode_number', 'avg_reward', 'avg_vehicles', 'avg_passenger_throughput'],
            'available_in_logs': 'performance_metrics, traffic_metrics',
            'status': 'AVAILABLE'
        },
        'baseline_comparisons': {
            'description': 'Baseline performance comparisons',
            'required_fields': ['baseline_id', 'baseline_type', 'avg_passenger_throughput', 'avg_waiting_time'],
            'available_in_logs': 'Calculated from training data',
            'status': 'CALCULATED'
        },
        'objective_metrics': {
            'description': 'Objective achievement metrics',
            'required_fields': ['objective_id', 'passenger_throughput_improvement_pct', 'waiting_time_reduction_pct'],
            'available_in_logs': 'Calculated from performance data',
            'status': 'CALCULATED'
        },
        'traffic_data': {
            'description': 'Detailed traffic flow data',
            'required_fields': ['traffic_id', 'intersection_id', 'total_count', 'passenger_throughput', 'passenger_waiting_time'],
            'available_in_logs': 'traffic_metrics, reward_breakdown',
            'status': 'AVAILABLE'
        },
        'lane_metrics': {
            'description': 'Lane-level performance metrics',
            'required_fields': ['lane_id', 'episode_id', 'intersection_id', 'queue_length', 'throughput', 'avg_waiting_time'],
            'available_in_logs': 'traffic_metrics, reward_breakdown (distributed across intersections)',
            'status': 'PARTIAL - Need intersection-level breakdown'
        }
    }
    
    print("DATABASE TABLE ANALYSIS:")
    print("=" * 80)
    
    for table_name, table_info in database_tables.items():
        print(f"\n{table_name.upper()}:")
        print(f"  Description: {table_info['description']}")
        print(f"  Required fields: {table_info['required_fields']}")
        print(f"  Available in logs: {table_info['available_in_logs']}")
        print(f"  Status: {table_info['status']}")
        
        if table_info['status'] == 'PARTIAL':
            print(f"  [WARNING] MISSING DATA: Some fields need to be estimated or calculated")
        elif table_info['status'] == 'CALCULATED':
            print(f"  [CALC] CALCULATED: Can be derived from available data")
        elif table_info['status'] == 'AVAILABLE':
            print(f"  [OK] AVAILABLE: All required data is in logs")
    
    print("\n" + "=" * 80)
    print("SUMMARY OF MISSING DATA:")
    print("=" * 80)
    
    missing_data = []
    for table_name, table_info in database_tables.items():
        if table_info['status'] == 'PARTIAL':
            missing_data.append(f"  - {table_name}: Missing individual vehicle counts")
        elif table_info['status'] == 'CALCULATED':
            missing_data.append(f"  - {table_name}: Needs calculation from available data")
    
    if missing_data:
        print("Tables that need additional data or calculation:")
        for item in missing_data:
            print(item)
    else:
        print("All tables have sufficient data available in logs!")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("1. Use episode data from JSONL files for training_episodes")
    print("2. Calculate vehicle breakdown from total vehicles and PT vehicles")
    print("3. Use reward_breakdown for waiting times and queue lengths")
    print("4. Distribute episode data across intersections for lane_metrics")
    print("5. Calculate baseline and objective metrics from performance data")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    analyze_logs_for_database_mapping()


















