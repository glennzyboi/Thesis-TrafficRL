#!/usr/bin/env python3
"""
Test Enhanced Logging - 5 Episode Test
Verify that all required metrics are being captured before running full training
"""

import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.comprehensive_training import ComprehensiveTrainer

def test_enhanced_logging():
    """Run 5-episode test to verify logging"""
    print("=" * 80)
    print("TESTING ENHANCED LOGGING - 5 EPISODE TEST")
    print("=" * 80)
    
    # Create trainer with 5 episodes
    trainer = ComprehensiveTrainer(experiment_name="logging_test_5ep")
    trainer.config['episodes'] = 5
    trainer.config['save_freq'] = 5
    trainer.config['validation_freq'] = 100  # Disable validation for quick test
    
    # Run training
    print("\nRunning 5-episode test...")
    results = trainer.run_comprehensive_training()
    
    # Check results
    print("\n" + "=" * 80)
    print("LOGGING TEST RESULTS")
    print("=" * 80)
    
    if len(results['training_results']) < 5:
        print(f"[ERROR] Only {len(results['training_results'])} episodes logged!")
        return False
    
    # Check first episode for required fields
    episode_1 = results['training_results'][0]
    
    required_fields = [
        'episode', 'scenario', 'reward', 'steps', 'time_minutes', 'avg_loss',
        'epsilon', 'vehicles', 'completed_trips', 'passenger_throughput',
        'memory_size', 'avg_waiting_time', 'avg_queue_length', 'avg_speed',
        'jeepneys_processed', 'buses_processed', 'trucks_processed',
        'motorcycles_processed', 'cars_processed', 'pt_passenger_throughput',
        'intersection_metrics'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in episode_1:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"[ERROR] Missing fields in episode data:")
        for field in missing_fields:
            print(f"  - {field}")
        return False
    
    # Print sample episode data
    print("\n[OK] All required fields present!")
    print("\nSample Episode 1 Data:")
    print(f"  Episode: {episode_1['episode']}")
    print(f"  Scenario: {episode_1['scenario']}")
    print(f"  Reward: {episode_1['reward']:.2f}")
    print(f"  Vehicles: {episode_1['vehicles']}")
    print(f"  Avg Waiting Time: {episode_1['avg_waiting_time']:.2f}s")
    print(f"  Avg Queue Length: {episode_1['avg_queue_length']:.2f}")
    print(f"  Avg Speed: {episode_1['avg_speed']:.2f} km/h")
    print(f"  Jeepneys: {episode_1['jeepneys_processed']}")
    print(f"  Buses: {episode_1['buses_processed']}")
    print(f"  PT Throughput: {episode_1['pt_passenger_throughput']:.0f}")
    print(f"  Intersection Metrics: {len(episode_1.get('intersection_metrics', {}))} intersections")
    
    # Check intersection metrics
    if 'intersection_metrics' in episode_1 and episode_1['intersection_metrics']:
        print("\n  Per-Intersection Data:")
        for intersection_id, metrics in episode_1['intersection_metrics'].items():
            print(f"    {intersection_id}:")
            print(f"      Vehicles: {metrics.get('total_vehicles', 0)}")
            print(f"      Queue: {metrics.get('total_queue', 0)}")
            print(f"      Avg Waiting: {metrics.get('avg_waiting', 0):.2f}s")
    
    print("\n" + "=" * 80)
    print("LOGGING TEST: PASSED [OK]")
    print("=" * 80)
    print("\nAll required metrics are being captured correctly!")
    print("You can now proceed with full 350-episode training.")
    print("\nTo run full training:")
    print("  cd experiments")
    print("  python comprehensive_training.py --episodes 350")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_logging()
    if success:
        print("\n[SUCCESS] Enhanced logging verified!")
    else:
        print("\n[FAILED] Logging test failed!")
        sys.exit(1)
