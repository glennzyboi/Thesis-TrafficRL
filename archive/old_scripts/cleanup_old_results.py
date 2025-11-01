"""
Cleanup Old Comprehensive Results
Removes old/test training runs to reduce clutter
Keeps the important ones: final_defense_training_350ep and comprehensive_training
"""

import os
import shutil

# Directories to KEEP
KEEP = [
    'final_defense_training_350ep',  # Your best +14% throughput result
    'comprehensive_training',         # Current corrected LSTM test
]

# Directories to REMOVE (all the test/old runs)
REMOVE = [
    'comprehensive_training_20251005_031757',
    'comprehensive_training_20251006_142714',
    'enhanced_benchmarking_60',
    'enhanced_training_fixed_rewards',
    'headless_sanity_stability',
    'headless_sanity_tweaked',
    'long_sanity_tweaked_60',
    'lstm_50_episodes',
    'lstm_50_episodes_repeat',
    'lstm_agent_test',
    'lstm_comparison_test',
    'lstm_extended_training',
    'lstm_progressive_test_50ep',
    'lstm_rebalanced_rewards',
    'lstm_retune_thr045_spill_100',
    'lstm_stabilized_moderate_200ep',
    'lstm_vs_no_lstm_comparison',
    'lstm_vs_no_lstm_simple',
    'main_training_run',
    'non_lstm_50_episodes',
    'non_lstm_comparison_test',
    'non_lstm_extended_training',
    'non_lstm_rebalanced_rewards',
    'non_lstm_retune_thr045_spill_100',
    'sanity_aggressive_reward',
]

def cleanup():
    base_dir = "comprehensive_results"
    
    print("Cleaning up old comprehensive results...")
    print(f"\nKEEPING:")
    for keep_dir in KEEP:
        path = os.path.join(base_dir, keep_dir)
        if os.path.exists(path):
            print(f"  [KEEP] {keep_dir}")
        else:
            print(f"  [MISS] {keep_dir} (not found)")
    
    print(f"\nREMOVING:")
    removed_count = 0
    for remove_dir in REMOVE:
        path = os.path.join(base_dir, remove_dir)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"  [DONE] Removed {remove_dir}")
                removed_count += 1
            except Exception as e:
                print(f"  [FAIL] Failed to remove {remove_dir}: {e}")
        else:
            print(f"  [SKIP] {remove_dir} (not found)")
    
    print(f"\nCleanup complete! Removed {removed_count} directories.")

if __name__ == "__main__":
    cleanup()

