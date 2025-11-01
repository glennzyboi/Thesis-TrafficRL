#!/usr/bin/env python3
"""
Merge Enhanced Logging Data
Combines the completed training data with enhanced logging from the interrupted training
"""

import json
import os
import sys
from datetime import datetime

def merge_enhanced_logging():
    """Merge enhanced logging data with completed training"""
    
    print("=" * 80)
    print("MERGING ENHANCED LOGGING WITH COMPLETED TRAINING DATA")
    print("=" * 80)
    
    # Load completed training data
    completed_path = "comprehensive_results/final_thesis_training_350ep/complete_results.json"
    enhanced_path = "comprehensive_results/comprehensive_training/training_progress.json"
    
    if not os.path.exists(completed_path):
        print(f"ERROR: Completed training data not found at {completed_path}")
        return False
    
    if not os.path.exists(enhanced_path):
        print(f"ERROR: Enhanced logging data not found at {enhanced_path}")
        return False
    
    print("Loading completed training data...")
    with open(completed_path, 'r') as f:
        completed_data = json.load(f)
    
    print("Loading enhanced logging data...")
    with open(enhanced_path, 'r') as f:
        enhanced_data = json.load(f)
    
    # Create enhanced episodes mapping
    enhanced_episodes = {}
    if 'training_results' in enhanced_data:
        for episode_data in enhanced_data['training_results']:
            if isinstance(episode_data, dict) and 'episode' in episode_data:
                episode_num = episode_data['episode']
                enhanced_episodes[episode_num] = episode_data
    else:
        # Handle direct array format
        for episode_data in enhanced_data:
            if isinstance(episode_data, dict) and 'episode' in episode_data:
                episode_num = episode_data['episode']
                enhanced_episodes[episode_num] = episode_data
    
    print(f"Found {len(enhanced_episodes)} episodes with enhanced logging")
    
    # Merge the data
    merged_episodes = []
    enhanced_count = 0
    
    for episode in completed_data['training_results']:
        episode_num = episode['episode']
        
        if episode_num in enhanced_episodes:
            # Use enhanced logging data
            enhanced_ep = enhanced_episodes[episode_num]
            merged_episode = episode.copy()
            
            # Add enhanced logging fields
            merged_episode.update({
                'avg_waiting_time': enhanced_ep.get('avg_waiting_time', 0),
                'avg_queue_length': enhanced_ep.get('avg_queue_length', 0),
                'avg_speed': enhanced_ep.get('avg_speed', 0),
                'jeepneys_processed': enhanced_ep.get('jeepneys_processed', 0),
                'buses_processed': enhanced_ep.get('buses_processed', 0),
                'trucks_processed': enhanced_ep.get('trucks_processed', 0),
                'motorcycles_processed': enhanced_ep.get('motorcycles_processed', 0),
                'cars_processed': enhanced_ep.get('cars_processed', 0),
                'pt_passenger_throughput': enhanced_ep.get('pt_passenger_throughput', 0),
                'intersection_metrics': enhanced_ep.get('intersection_metrics', {})
            })
            enhanced_count += 1
        else:
            # Use original data with default enhanced values
            merged_episode = episode.copy()
            merged_episode.update({
                'avg_waiting_time': 0,
                'avg_queue_length': 0,
                'avg_speed': 0,
                'jeepneys_processed': 0,
                'buses_processed': 0,
                'trucks_processed': 0,
                'motorcycles_processed': 0,
                'cars_processed': 0,
                'pt_passenger_throughput': 0,
                'intersection_metrics': {}
            })
        
        merged_episodes.append(merged_episode)
    
    # Create merged dataset
    merged_data = completed_data.copy()
    merged_data['training_results'] = merged_episodes
    merged_data['experiment_name'] = 'final_thesis_training_350ep_enhanced'
    merged_data['enhanced_logging'] = True
    merged_data['enhanced_episodes'] = enhanced_count
    merged_data['total_episodes'] = len(merged_episodes)
    
    # Save merged data
    output_path = "comprehensive_results/final_thesis_training_350ep_enhanced/complete_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\n[SUCCESS] MERGED DATA SAVED:")
    print(f"   Output: {output_path}")
    print(f"   Total episodes: {len(merged_episodes)}")
    print(f"   Enhanced episodes: {enhanced_count}")
    print(f"   Default episodes: {len(merged_episodes) - enhanced_count}")
    
    return True

if __name__ == "__main__":
    success = merge_enhanced_logging()
    if success:
        print("\n[SUCCESS] Enhanced logging merge completed successfully!")
    else:
        print("\n[ERROR] Enhanced logging merge failed!")


















