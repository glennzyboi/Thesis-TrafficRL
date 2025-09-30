#!/usr/bin/env python3
"""
Training Progress Monitor for 200-Episode D3QN Training
Provides real-time updates on training progress, performance metrics, and dashboard generation status.
"""

import os
import json
import time
import glob
from datetime import datetime

def monitor_training_progress(experiment_name="thesis_200ep_dashboard_training"):
    """Monitor the 200-episode training progress"""
    
    base_path = f"comprehensive_results/{experiment_name}"
    
    print("🔍 D3QN Training Progress Monitor")
    print("=" * 50)
    print(f"Experiment: {experiment_name}")
    print(f"Target Episodes: 200")
    print(f"Expected Duration: 3-4 hours")
    print("=" * 50)
    
    last_episode = 0
    start_time = time.time()
    
    while True:
        try:
            # Check if results file exists
            results_file = f"{base_path}/complete_results.json"
            progress_file = f"{base_path}/training_progress.json"
            
            current_episode = 0
            current_reward = 0
            current_loss = 0
            best_reward = 0
            
            # Try to read progress from multiple sources
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    if 'training_results' in progress:
                        current_episode = len(progress['training_results'])
                        if progress['training_results']:
                            latest = progress['training_results'][-1]
                            current_reward = latest.get('reward', 0)
                            current_loss = latest.get('avg_loss', 0)
                        best_reward = progress.get('best_reward', 0)
            
            elif os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    if 'training_results' in results:
                        current_episode = len(results['training_results'])
                        if results['training_results']:
                            latest = results['training_results'][-1]
                            current_reward = latest.get('reward', 0)
                            current_loss = latest.get('avg_loss', 0)
                        best_reward = results.get('best_reward', 0)
            
            # Calculate progress
            progress_percent = (current_episode / 200) * 100
            elapsed_time = time.time() - start_time
            
            # Estimate remaining time
            if current_episode > 0:
                time_per_episode = elapsed_time / current_episode
                remaining_episodes = 200 - current_episode
                estimated_remaining = remaining_episodes * time_per_episode
            else:
                estimated_remaining = 0
            
            # Training phase
            if current_episode <= 140:
                phase = f"Offline Phase ({current_episode}/140)"
                phase_progress = (current_episode / 140) * 100
            else:
                phase = f"Online Phase ({current_episode-140}/60)"
                phase_progress = ((current_episode - 140) / 60) * 100
            
            # Clear screen and display status
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("🔍 D3QN Training Progress Monitor")
            print("=" * 60)
            print(f"📊 Experiment: {experiment_name}")
            print(f"⏰ Started: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
            print(f"🕐 Elapsed: {int(elapsed_time//3600):02d}:{int((elapsed_time%3600)//60):02d}:{int(elapsed_time%60):02d}")
            
            if estimated_remaining > 0:
                print(f"⏳ Remaining: {int(estimated_remaining//3600):02d}:{int((estimated_remaining%3600)//60):02d}:{int(estimated_remaining%60):02d}")
            
            print("=" * 60)
            
            # Progress bars
            progress_bar = "█" * int(progress_percent // 5) + "░" * (20 - int(progress_percent // 5))
            phase_bar = "█" * int(phase_progress // 5) + "░" * (20 - int(phase_progress // 5))
            
            print(f"📈 Overall Progress: [{progress_bar}] {progress_percent:.1f}%")
            print(f"   Episode {current_episode}/200")
            
            print(f"🔄 Current Phase: [{phase_bar}] {phase_progress:.1f}%")
            print(f"   {phase}")
            
            print("=" * 60)
            
            # Performance metrics
            print("🎯 Performance Metrics:")
            print(f"   Current Reward: {current_reward:.2f}")
            print(f"   Best Reward: {best_reward:.2f}")
            print(f"   Current Loss: {current_loss:.6f}")
            
            # Check for dashboard files
            dashboard_dir = f"{base_path}/plots/dashboard"
            dashboard_files = []
            if os.path.exists(dashboard_dir):
                dashboard_files = [f for f in os.listdir(dashboard_dir) if f.endswith('.png')]
            
            print(f"📊 Dashboard Files: {len(dashboard_files)} generated")
            if dashboard_files:
                for file in dashboard_files:
                    print(f"   ✅ {file}")
            
            # Check for model files
            model_dir = f"{base_path}/models"
            model_files = []
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
            
            print(f"💾 Model Files: {len(model_files)} saved")
            
            # Training status
            if current_episode >= 200:
                print("\n🎉 TRAINING COMPLETED!")
                print("📊 Dashboard generation in progress...")
                break
            elif current_episode == last_episode and elapsed_time > 300:  # No progress for 5 minutes
                print("\n⚠️ WARNING: No progress detected for 5 minutes")
                print("   Training may have stopped or encountered an error")
            elif current_episode > last_episode:
                print(f"\n✅ Progress detected: Episode {current_episode}")
            
            last_episode = current_episode
            
            print("\n" + "=" * 60)
            print("Press Ctrl+C to exit monitor")
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error monitoring training: {e}")
            time.sleep(60)  # Wait longer on error

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor D3QN Training Progress')
    parser.add_argument('--experiment', type=str, default="thesis_200ep_dashboard_training",
                       help='Experiment name to monitor')
    
    args = parser.parse_args()
    
    monitor_training_progress(args.experiment)

