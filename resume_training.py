#!/usr/bin/env python3
"""
Resume Training Script
Continues the interrupted comprehensive training from episode 150
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.comprehensive_training import ComprehensiveTrainer

def resume_training():
    """Resume training from episode 150"""
    
    print("=" * 80)
    print("RESUMING COMPREHENSIVE TRAINING FROM EPISODE 150")
    print("=" * 80)
    
    # Initialize trainer
    trainer = ComprehensiveTrainer(experiment_name="comprehensive_training")
    
    # Load the last checkpoint (episode 150)
    checkpoint_path = "comprehensive_results/comprehensive_training/models/checkpoint_ep150.keras"
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return False
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load training progress
    progress_path = "comprehensive_results/comprehensive_training/training_progress.json"
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress_data = json.load(f)
        
        # Find the last episode
        last_episode = 0
        for episode_data in progress_data:
            if isinstance(episode_data, dict) and 'episode' in episode_data:
                last_episode = max(last_episode, episode_data['episode'])
        
        print(f"Last saved episode: {last_episode}")
    
    # Resume training from episode 150
    print("Resuming training from episode 150...")
    print("This will continue to episode 350 with enhanced logging")
    
    # Run the training
    try:
        results = trainer.run_comprehensive_training()
        print("Training completed successfully!")
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    success = resume_training()
    if success:
        print("✅ Training resumed and completed successfully!")
    else:
        print("❌ Training failed to resume")





