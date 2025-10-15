"""
CORRECTED Training Script for D3QN with Traffic Prediction as Primary Function
Implements the correct LSTM architecture where traffic prediction is PRIMARY
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import corrected agent
from algorithms.d3qn_agent_corrected import D3QNAgentCorrected
from core.traffic_env import TrafficEnvironment
from utils.production_logger import create_production_logger
from utils.traffic_prediction_dashboard import TrafficPredictionDashboard
from evaluation.performance_comparison import PerformanceComparator

class CorrectedTrainer:
    """
    CORRECTED trainer that implements LSTM with traffic prediction as PRIMARY function
    """
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"corrected_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Training configuration
        self.config = {
            'learning_rate': 0.0005,
            'epsilon_decay': 0.9995,
            'memory_size': 50000,
            'batch_size': 64,
            'gamma': 0.98,
            'sequence_length': 10,
            'episodes': 200,
            'episode_duration': 300,
            'warmup_time': 30,
            'min_phase_time': 5,
            'max_phase_time': 60,
            'validation_freq': 15,
            'target_update_freq': 20
        }
        
        # Create output directory
        self.output_dir = f"comprehensive_results/{self.experiment_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        
        # Initialize components
        self.training_results = []
        self.validation_results = []
        self.prediction_dashboard = TrafficPredictionDashboard(f"{self.output_dir}/prediction_dashboard")
        
        # Setup logger
        self.logger = create_production_logger(
            f"{self.output_dir}/logs/training.log",
            f"{self.output_dir}/logs/training.jsonl"
        )
        
        print(f"CORRECTED TRAINING INITIALIZED")
        print(f"Experiment: {self.experiment_name}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Architecture: LSTM → Traffic Prediction → Q-Value Estimation")
    
    def load_training_data(self):
        """Load training bundles"""
        bundles_file = "data/training_bundles.json"
        
        if not os.path.exists(bundles_file):
            raise FileNotFoundError(f"Training bundles file not found: {bundles_file}")
        
        with open(bundles_file, 'r') as f:
            bundles = json.load(f)
        
        # Split into train/validation
        train_bundles = bundles[:int(len(bundles) * 0.8)]
        val_bundles = bundles[int(len(bundles) * 0.8):]
        
        print(f"Loaded {len(train_bundles)} training bundles and {len(val_bundles)} validation bundles")
        
        return train_bundles, val_bundles
    
    def select_random_bundle(self, bundles):
        """Select random training bundle"""
        return random.choice(bundles)
    
    def run_single_episode(self, env, agent, episode, scenario_info):
        """Run single episode with traffic prediction monitoring"""
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_predictions = []
        episode_actual_labels = []
        episode_traffic_metrics = []
        
        # Episode tracking
        start_time = time.time()
        
        while steps < self.config['episode_duration']:
            # Get action from agent (includes traffic prediction)
            action = agent.act(state)
            
            # Get traffic prediction for monitoring
            if len(agent.state_history) >= agent.sequence_length:
                state_sequence = np.array(list(agent.state_history))
                traffic_prediction = agent.predict_traffic(state_sequence)
                episode_predictions.append(traffic_prediction)
                
                # Determine actual traffic label
                traffic_metrics = {
                    'queue_length': env.metrics.get('queue_length', 0),
                    'waiting_time': env.metrics.get('waiting_time', 0),
                    'vehicle_density': env.metrics.get('vehicle_density', 0),
                    'congestion_level': env.metrics.get('congestion_level', 0)
                }
                episode_traffic_metrics.append(traffic_metrics)
                
                is_heavy_traffic = agent.is_heavy_traffic(traffic_metrics)
                episode_actual_labels.append(1 if is_heavy_traffic else 0)
            
            # Execute action
            next_state, reward, done = env.step(action)
            
            # Store experience with traffic metrics
            traffic_metrics = {
                'queue_length': env.metrics.get('queue_length', 0),
                'waiting_time': env.metrics.get('waiting_time', 0),
                'vehicle_density': env.metrics.get('vehicle_density', 0),
                'congestion_level': env.metrics.get('congestion_level', 0)
            }
            
            agent.remember(state, action, reward, next_state, done, traffic_metrics)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Calculate episode metrics
        episode_time = time.time() - start_time
        avg_loss = 0.0
        
        # Train agent if enough experiences
        if len(agent.memory) > agent.batch_size:
            training_metrics = agent.train(agent.memory)
            avg_loss = training_metrics['q_loss']
            
            # Log prediction metrics
            if episode_predictions and episode_actual_labels:
                prediction_data = self.prediction_dashboard.log_prediction(
                    episode, episode_predictions, episode_actual_labels, episode_traffic_metrics
                )
        
        # Get final metrics
        final_metrics = env.get_metrics()
        
        episode_result = {
            'episode': episode,
            'scenario': scenario_info['bundle_name'],
            'reward': total_reward,
            'steps': steps,
            'time_minutes': episode_time / 60,
            'avg_loss': avg_loss,
            'epsilon': agent.epsilon,
            'vehicles': final_metrics.get('vehicles', 0),
            'completed_trips': final_metrics.get('completed_trips', 0),
            'passenger_throughput': final_metrics.get('passenger_throughput', 0),
            'waiting_time': final_metrics.get('waiting_time', 0),
            'avg_speed': final_metrics.get('avg_speed', 0),
            'queue_length': final_metrics.get('queue_length', 0),
            'memory_size': len(agent.memory)
        }
        
        # Add prediction metrics if available
        if episode_predictions and episode_actual_labels:
            binary_predictions = (np.array(episode_predictions) > 0.5).astype(int)
            accuracy = np.mean(binary_predictions == episode_actual_labels)
            episode_result['prediction_accuracy'] = accuracy
            episode_result['heavy_traffic_predictions'] = np.sum(binary_predictions)
            episode_result['actual_heavy_traffic'] = np.sum(episode_actual_labels)
        
        return episode_result
    
    def run_validation(self, agent, val_bundles, episode):
        """Run validation with traffic prediction monitoring"""
        val_rewards = []
        val_predictions = []
        val_actual_labels = []
        
        for i, bundle in enumerate(val_bundles[:5]):  # Validate on 5 scenarios
            route_file = f"data/routes/consolidated/bundle_{bundle['Day']}_cycle_{bundle['CycleNum']}.rou.xml"
            
            if not os.path.exists(route_file):
                continue
            
            # Create validation environment
            val_env = TrafficEnvironment(
                net_file='network/ThesisNetowrk.net.xml',
                rou_file=route_file,
                use_gui=False,
                num_seconds=self.config['episode_duration'],
                warmup_time=self.config['warmup_time'],
                step_length=1.0,
                min_phase_time=self.config['min_phase_time'],
                max_phase_time=self.config['max_phase_time']
            )
            
            # Run validation episode
            state = val_env.reset()
            total_reward = 0
            steps = 0
            episode_predictions = []
            episode_actual_labels = []
            
            while steps < self.config['episode_duration']:
                action = agent.act(state)
                
                # Get traffic prediction
                if len(agent.state_history) >= agent.sequence_length:
                    state_sequence = np.array(list(agent.state_history))
                    traffic_prediction = agent.predict_traffic(state_sequence)
                    episode_predictions.append(traffic_prediction)
                    
                    # Determine actual traffic label
                    traffic_metrics = {
                        'queue_length': val_env.metrics.get('queue_length', 0),
                        'waiting_time': val_env.metrics.get('waiting_time', 0),
                        'vehicle_density': val_env.metrics.get('vehicle_density', 0),
                        'congestion_level': val_env.metrics.get('congestion_level', 0)
                    }
                    is_heavy_traffic = agent.is_heavy_traffic(traffic_metrics)
                    episode_actual_labels.append(1 if is_heavy_traffic else 0)
                
                next_state, reward, done = val_env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            val_rewards.append(total_reward)
            val_predictions.extend(episode_predictions)
            val_actual_labels.extend(episode_actual_labels)
            
            val_env.close()
        
        # Calculate validation metrics
        avg_reward = np.mean(val_rewards) if val_rewards else 0
        
        val_result = {
            'episode': episode,
            'avg_reward': avg_reward,
            'val_episodes': len(val_rewards)
        }
        
        # Add prediction metrics if available
        if val_predictions and val_actual_labels:
            binary_predictions = (np.array(val_predictions) > 0.5).astype(int)
            accuracy = np.mean(binary_predictions == val_actual_labels)
            val_result['prediction_accuracy'] = accuracy
        
        return val_result
    
    def train(self):
        """Run corrected training with traffic prediction as primary function"""
        print(f"\n{'='*80}")
        print(f"CORRECTED TRAINING STARTED")
        print(f"Architecture: LSTM → Traffic Prediction → Q-Value Estimation")
        print(f"Episodes: {self.config['episodes']}")
        print(f"{'='*80}")
        
        # Load training data
        train_bundles, val_bundles = self.load_training_data()
        
        # Create initial environment
        initial_bundle = self.select_random_bundle(train_bundles)
        route_file = f"data/routes/consolidated/bundle_{initial_bundle['Day']}_cycle_{initial_bundle['CycleNum']}.rou.xml"
        
        env = TrafficEnvironment(
            net_file='network/ThesisNetowrk.net.xml',
            rou_file=route_file,
            use_gui=False,
            num_seconds=self.config['episode_duration'],
            warmup_time=self.config['warmup_time'],
            step_length=1.0,
            min_phase_time=self.config['min_phase_time'],
            max_phase_time=self.config['max_phase_time']
        )
        
        # Get initial state
        initial_state = env.reset()
        
        # Create corrected agent
        agent = D3QNAgentCorrected(
            state_size=len(initial_state),
            action_size=env.action_size,
            learning_rate=self.config['learning_rate'],
            epsilon_decay=self.config['epsilon_decay'],
            memory_size=self.config['memory_size'],
            batch_size=self.config['batch_size'],
            sequence_length=self.config['sequence_length']
        )
        
        # Training loop
        start_time = time.time()
        best_reward = float('-inf')
        
        for episode in range(self.config['episodes']):
            # Select scenario
            bundle = self.select_random_bundle(train_bundles)
            route_file = f"data/routes/consolidated/bundle_{bundle['Day']}_cycle_{bundle['CycleNum']}.rou.xml"
            
            # Reinitialize environment with new scenario
            if episode > 0:
                env.close()
                env = TrafficEnvironment(
                    net_file='network/ThesisNetowrk.net.xml',
                    rou_file=route_file,
                    use_gui=False,
                    num_seconds=self.config['episode_duration'],
                    warmup_time=self.config['warmup_time'],
                    step_length=1.0,
                    min_phase_time=self.config['min_phase_time'],
                    max_phase_time=self.config['max_phase_time']
                )
            
            # Run episode
            scenario_info = {
                'bundle_name': f"Day {bundle['Day']}, Cycle {bundle['CycleNum']}",
                'route_file': route_file,
                'day': bundle['Day'],
                'cycle': bundle['CycleNum']
            }
            
            episode_result = self.run_single_episode(env, agent, episode, scenario_info)
            self.training_results.append(episode_result)
            
            # Log episode
            self.logger.info(f"Episode {episode + 1:03d}: Reward={episode_result['reward']:.2f}, "
                           f"Throughput={episode_result['passenger_throughput']:.1f}, "
                           f"Prediction Accuracy={episode_result.get('prediction_accuracy', 0):.3f}")
            
            # Update best model
            if episode_result['reward'] > best_reward:
                best_reward = episode_result['reward']
                agent.save(f"{self.output_dir}/models/best_model")
                print(f"   New best model saved! Reward: {best_reward:.2f}")
            
            # Validation
            if (episode + 1) % self.config['validation_freq'] == 0:
                val_result = self.run_validation(agent, val_bundles, episode + 1)
                self.validation_results.append(val_result)
                
                print(f"   Validation Reward: {val_result['avg_reward']:.2f}, "
                      f"Prediction Accuracy: {val_result.get('prediction_accuracy', 0):.3f}")
            
            # Update target network
            if (episode + 1) % self.config['target_update_freq'] == 0:
                agent.update_target_model()
            
            # Print progress
            if (episode + 1) % 10 == 0:
                elapsed_time = (time.time() - start_time) / 60
                avg_reward = np.mean([r['reward'] for r in self.training_results[-10:]])
                avg_throughput = np.mean([r['passenger_throughput'] for r in self.training_results[-10:]])
                avg_prediction_accuracy = np.mean([r.get('prediction_accuracy', 0) for r in self.training_results[-10:]])
                
                print(f"Episode {episode + 1:03d}: Avg Reward={avg_reward:.2f}, "
                      f"Avg Throughput={avg_throughput:.1f}, "
                      f"Avg Prediction Accuracy={avg_prediction_accuracy:.3f}, "
                      f"Time={elapsed_time:.1f}min")
        
        # Training complete
        total_time = (time.time() - start_time) / 60
        print(f"\n{'='*80}")
        print(f"CORRECTED TRAINING COMPLETED")
        print(f"Total Time: {total_time:.1f} minutes")
        print(f"Best Reward: {best_reward:.2f}")
        print(f"{'='*80}")
        
        # Save results
        self.save_results()
        
        # Create prediction dashboard
        self.prediction_dashboard.create_dashboard()
        
        # Close environment
        env.close()
        
        return self.training_results, self.validation_results
    
    def save_results(self):
        """Save training results"""
        # Save training results
        with open(f"{self.output_dir}/training_results.json", 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        # Save validation results
        with open(f"{self.output_dir}/validation_results.json", 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Save configuration
        with open(f"{self.output_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Results saved to: {self.output_dir}")

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CORRECTED D3QN Training with Traffic Prediction')
    parser.add_argument('--episodes', type=int, default=200, help='Number of training episodes')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CorrectedTrainer(args.experiment_name)
    
    # Update episodes if specified
    if args.episodes:
        trainer.config['episodes'] = args.episodes
    
    # Run training
    training_results, validation_results = trainer.train()
    
    print(f"\nTraining completed successfully!")
    print(f"Results saved to: {trainer.output_dir}")
    print(f"Prediction dashboard created in: {trainer.output_dir}/prediction_dashboard")

if __name__ == "__main__":
    main()
