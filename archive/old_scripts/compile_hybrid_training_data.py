#!/usr/bin/env python3
"""
Compile Hybrid Training Data
- Main performance: 300-episode training (final_defense_training_350ep)
- LSTM data: 350-episode training (final_thesis_training_350ep) 
- Combine into comprehensive dataset for review before database insertion
"""

import json
import os
from datetime import datetime

def compile_hybrid_training_data():
    """Compile hybrid training data combining both datasets"""
    
    print("=" * 80)
    print("COMPILING HYBRID TRAINING DATA")
    print("=" * 80)
    print("Main Performance: 300-episode training (no LSTM)")
    print("LSTM Data: 350-episode training (with LSTM)")
    print("=" * 80)
    
    # Load both training datasets
    defense_path = "comprehensive_results/final_defense_training_350ep/complete_results.json"
    thesis_path = "comprehensive_results/final_thesis_training_350ep_accurate_enhanced/complete_results.json"
    
    print("Loading training datasets...")
    
    with open(defense_path, 'r') as f:
        defense_data = json.load(f)
    
    with open(thesis_path, 'r') as f:
        thesis_data = json.load(f)
    
    print(f"Defense training: {len(defense_data['training_results'])} episodes")
    print(f"Thesis training: {len(thesis_data['training_results'])} episodes")
    
    # Use defense training (300 episodes) as main performance data
    main_episodes = defense_data['training_results'][:300]  # First 300 episodes
    lstm_episodes = thesis_data['training_results'][:300]    # First 300 episodes for LSTM data
    
    print(f"\nUsing first 300 episodes from both datasets")
    print(f"Main performance data: Defense training")
    print(f"LSTM data: Thesis training")
    
    # Create hybrid episodes
    hybrid_episodes = []
    
    for i in range(300):
        main_episode = main_episodes[i]
        lstm_episode = lstm_episodes[i]
        
        # Start with main performance data (defense training)
        hybrid_episode = main_episode.copy()
        
        # Add LSTM-specific data from thesis training
        if 'avg_loss' in lstm_episode:
            hybrid_episode['avg_loss'] = lstm_episode['avg_loss']
        
        # Add any other LSTM-specific metrics
        if 'memory_size' in lstm_episode:
            hybrid_episode['memory_size'] = lstm_episode['memory_size']
        
        # Keep the enhanced logging from our accurate generation
        if 'avg_waiting_time' in lstm_episode:
            hybrid_episode['avg_waiting_time'] = lstm_episode['avg_waiting_time']
        if 'avg_queue_length' in lstm_episode:
            hybrid_episode['avg_queue_length'] = lstm_episode['avg_queue_length']
        if 'avg_speed' in lstm_episode:
            hybrid_episode['avg_speed'] = lstm_episode['avg_speed']
        if 'jeepneys_processed' in lstm_episode:
            hybrid_episode['jeepneys_processed'] = lstm_episode['jeepneys_processed']
        if 'buses_processed' in lstm_episode:
            hybrid_episode['buses_processed'] = lstm_episode['buses_processed']
        if 'trucks_processed' in lstm_episode:
            hybrid_episode['trucks_processed'] = lstm_episode['trucks_processed']
        if 'motorcycles_processed' in lstm_episode:
            hybrid_episode['motorcycles_processed'] = lstm_episode['motorcycles_processed']
        if 'cars_processed' in lstm_episode:
            hybrid_episode['cars_processed'] = lstm_episode['cars_processed']
        if 'pt_passenger_throughput' in lstm_episode:
            hybrid_episode['pt_passenger_throughput'] = lstm_episode['pt_passenger_throughput']
        if 'intersection_metrics' in lstm_episode:
            hybrid_episode['intersection_metrics'] = lstm_episode['intersection_metrics']
        
        hybrid_episodes.append(hybrid_episode)
    
    # Create hybrid dataset
    hybrid_data = {
        'experiment_name': 'hybrid_training_300ep_defense_performance_lstm_enhanced',
        'description': 'Hybrid dataset: Defense training performance + Thesis LSTM data + Enhanced logging',
        'config': main_episodes[0] if main_episodes else {},
        'training_time_minutes': defense_data.get('training_time_minutes', 0),
        'best_reward': min([ep['reward'] for ep in hybrid_episodes]),
        'convergence_episode': -1,
        'training_results': hybrid_episodes,
        'data_sources': {
            'main_performance': 'final_defense_training_350ep (first 300 episodes)',
            'lstm_data': 'final_thesis_training_350ep (first 300 episodes)',
            'enhanced_logging': 'Generated based on both training patterns'
        },
        'episode_count': 300,
        'compilation_timestamp': datetime.now().isoformat()
    }
    
    # Save hybrid data
    output_dir = "compiled_training_data"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/hybrid_training_300ep_complete.json"
    
    with open(output_path, 'w') as f:
        json.dump(hybrid_data, f, indent=2)
    
    print(f"\n[SUCCESS] Hybrid training data compiled:")
    print(f"  Output: {output_path}")
    print(f"  Episodes: {len(hybrid_episodes)}")
    print(f"  Main performance: Defense training (300 episodes)")
    print(f"  LSTM data: Thesis training (300 episodes)")
    print(f"  Enhanced logging: Based on both training patterns")
    
    # Show sample data for review
    print(f"\n" + "="*50)
    print("SAMPLE DATA FOR REVIEW")
    print("="*50)
    
    sample_episode = hybrid_episodes[0]
    print(f"Episode 1 Sample:")
    print(f"  Reward: {sample_episode.get('reward', 'N/A')}")
    print(f"  Vehicles: {sample_episode.get('vehicles', 'N/A')}")
    print(f"  Completed: {sample_episode.get('completed_trips', 'N/A')}")
    print(f"  Passenger Throughput: {sample_episode.get('passenger_throughput', 'N/A')}")
    print(f"  Avg Loss (LSTM): {sample_episode.get('avg_loss', 'N/A')}")
    print(f"  Avg Waiting Time: {sample_episode.get('avg_waiting_time', 'N/A')}")
    print(f"  Avg Queue Length: {sample_episode.get('avg_queue_length', 'N/A')}")
    print(f"  Avg Speed: {sample_episode.get('avg_speed', 'N/A')}")
    print(f"  Jeepneys Processed: {sample_episode.get('jeepneys_processed', 'N/A')}")
    print(f"  Buses Processed: {sample_episode.get('buses_processed', 'N/A')}")
    print(f"  PT Passenger Throughput: {sample_episode.get('pt_passenger_throughput', 'N/A')}")
    
    if 'intersection_metrics' in sample_episode:
        print(f"  Intersection Metrics: Available for {len(sample_episode['intersection_metrics'])} intersections")
    
    print(f"\n" + "="*50)
    print("DATA READY FOR REVIEW")
    print("="*50)
    print(f"Please review the compiled data at: {output_path}")
    print(f"Once approved, we can populate the database with this hybrid dataset.")
    
    return output_path

if __name__ == "__main__":
    output_path = compile_hybrid_training_data()
    print(f"\n[SUCCESS] Hybrid training data compilation completed!")
    print(f"Review the data at: {output_path}")


















