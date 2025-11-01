#!/usr/bin/env python3
"""
Fix passenger throughput calculation in training data
Corrects unrealistic values based on actual intersection vehicle counts
"""

import json
import sys
from pathlib import Path

def calculate_realistic_passenger_throughput(intersection_metrics):
    """Calculate realistic passenger throughput based on intersection vehicle counts"""
    
    # Realistic passenger capacities (Davao City specific)
    passenger_capacities = {
        'car': 1.3,
        'bus': 35.0,
        'jeepney': 14.0,
        'motorcycle': 1.4,
        'truck': 1.5,
        'tricycle': 2.5  # Added for completeness
    }
    
    total_passengers = 0
    
    for intersection_id, metrics in intersection_metrics.items():
        if 'vehicle_types' in metrics:
            intersection_passengers = 0
            for vehicle_type, count in metrics['vehicle_types'].items():
                if vehicle_type in passenger_capacities:
                    intersection_passengers += count * passenger_capacities[vehicle_type]
            
            total_passengers += intersection_passengers
            print(f"  {intersection_id}: {intersection_passengers:.1f} passengers")
    
    return total_passengers

def fix_training_data():
    """Fix passenger throughput in training data"""
    
    input_file = Path("compiled_training_data/hybrid_training_300ep_complete.json")
    output_file = Path("compiled_training_data/hybrid_training_300ep_corrected.json")
    
    print("Loading training data...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data['training_results'])} episodes...")
    
    corrections_made = 0
    
    for i, episode in enumerate(data['training_results']):
        episode_num = episode['episode']
        
        # Calculate realistic passenger throughput
        if 'intersection_metrics' in episode:
            realistic_pt = calculate_realistic_passenger_throughput(episode['intersection_metrics'])
            original_pt = episode.get('passenger_throughput', 0)
            
            if abs(realistic_pt - original_pt) > 100:  # Significant difference
                print(f"\nEpisode {episode_num}:")
                print(f"  Original: {original_pt:.1f} passengers")
                print(f"  Corrected: {realistic_pt:.1f} passengers")
                print(f"  Difference: {realistic_pt - original_pt:.1f} passengers")
                
                # Update the episode data
                episode['passenger_throughput'] = realistic_pt
                episode['pt_passenger_throughput'] = realistic_pt  # Also update PT throughput
                corrections_made += 1
        
        # Also fix the config section for episode 1
        if episode_num == 1 and 'config' in data:
            if 'intersection_metrics' in episode:
                realistic_pt = calculate_realistic_passenger_throughput(episode['intersection_metrics'])
                data['config']['passenger_throughput'] = realistic_pt
                print(f"  Config updated: {realistic_pt:.1f} passengers")
    
    print(f"\nCorrections made: {corrections_made} episodes")
    
    # Save corrected data
    print(f"Saving corrected data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Training data corrected successfully!")
    
    # Show sample of corrected data
    print("\nSample of corrected data (first 3 episodes):")
    for i in range(min(3, len(data['training_results']))):
        episode = data['training_results'][i]
        print(f"Episode {episode['episode']}: {episode['passenger_throughput']:.1f} passengers")

if __name__ == "__main__":
    fix_training_data()








