"""
Production-Grade Logging System for D3QN Traffic Signal Control
Designed for Supabase integration and real-world deployment
Based on research standards from SUMO+RL literature for performance evaluation
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from collections import defaultdict, deque

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    else:
        return obj

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

class ProductionLogger:
    """
    High-performance logging system for traffic signal control deployment
    
    Features:
    - Interval-based logging to prevent performance impact
    - Supabase-ready JSON format
    - Research-grade metrics tracking
    - Automatic episode summarization
    - Memory-efficient buffering
    """
    
    def __init__(self, 
                 log_interval: int = 30,  # Log every 30 seconds (research standard)
                 buffer_size: int = 100,  # Buffer 100 entries before write
                 output_dir: str = "production_logs",
                 experiment_name: str = None):
        """
        Initialize production logger
        
        Args:
            log_interval: Seconds between detailed logs (30s standard from Li et al. 2022)
            buffer_size: Number of entries to buffer before writing
            output_dir: Directory for log files
            experiment_name: Unique experiment identifier
        """
        self.log_interval = log_interval
        self.buffer_size = buffer_size
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize buffers (memory efficient)
        self.step_buffer = deque(maxlen=buffer_size)
        self.episode_buffer = deque(maxlen=50)  # Keep last 50 episodes in memory
        
        # Tracking variables
        self.last_log_time = 0
        self.current_episode = 0
        self.session_id = str(uuid.uuid4())
        
        # Episode accumulators
        self.episode_data = self._init_episode_data()
        
        # File paths
        self.step_log_file = os.path.join(output_dir, f"{self.experiment_name}_steps.jsonl")
        self.episode_log_file = os.path.join(output_dir, f"{self.experiment_name}_episodes.jsonl")
        self.summary_file = os.path.join(output_dir, f"{self.experiment_name}_summary.json")
        
        print(f"🗂️ Production Logger Initialized:")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Log Interval: {log_interval}s")
        print(f"   Output: {output_dir}")
    
    def _init_episode_data(self) -> Dict:
        """Initialize episode data structure"""
        return {
            'episode_id': str(uuid.uuid4()),
            'episode_number': self.current_episode,
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'scenario_info': {},
            'step_count': 0,
            'total_reward': 0.0,
            
            # Core Traffic Metrics (Supabase ready)
            'vehicles_served': 0,
            'passenger_throughput': 0.0,
            'avg_queue_length': 0.0,
            'avg_delay_per_vehicle': 0.0,
            'avg_travel_time': 0.0,
            'phase_utilization_ratio': 0.0,
            
            # Public Transport Metrics (Novel)
            'buses_processed': 0,
            'jeepneys_processed': 0,
            'pt_passenger_throughput': 0.0,
            'pt_avg_waiting': 0.0,
            'pt_service_efficiency': 0.0,
            
            # RL-Specific Metrics
            'reward_components': {},
            'action_distribution': defaultdict(int),
            'cumulative_reward': 0.0,
            
            # MARL Coordination (if applicable)
            'agent_rewards': {},
            'coordination_score': 0.0,
            'intersection_throughput': {},
            
            # Temporal tracking
            'step_metrics': [],
            'reward_history': []
        }
    
    def log_step(self, 
                 step: int,
                 reward: float,
                 info: Dict[str, Any],
                 actions: Dict[str, int] = None,
                 reward_components: Dict[str, float] = None):
        """
        Log step data with interval-based writing for performance
        
        Args:
            step: Current simulation step
            reward: Step reward (single-agent) or average reward (MARL)
            info: Environment info dictionary
            actions: Agent actions (optional for MARL)
            reward_components: Breakdown of reward components
        """
        current_time = time.time()
        
        # Update episode accumulators
        self.episode_data['step_count'] = step
        self.episode_data['total_reward'] += reward
        self.episode_data['cumulative_reward'] += reward
        
        # Update traffic metrics
        self._update_traffic_metrics(info)
        
        # Update action distribution
        if actions:
            for agent_id, action in actions.items():
                self.episode_data['action_distribution'][action] += 1
        
        # Store reward components
        if reward_components:
            self.episode_data['reward_components'] = reward_components
        
        # Interval-based detailed logging (performance optimization)
        if current_time - self.last_log_time >= self.log_interval:
            step_entry = self._create_step_entry(step, reward, info, actions, reward_components)
            self.step_buffer.append(step_entry)
            self.last_log_time = current_time
            
            # Write buffer if full
            if len(self.step_buffer) >= self.buffer_size:
                self._flush_step_buffer()
        
        # Always track for episode summary (lightweight)
        self.episode_data['reward_history'].append(reward)
    
    def _update_traffic_metrics(self, info: Dict[str, Any]):
        """Update cumulative traffic metrics"""
        # Standard metrics
        if 'vehicles' in info:
            self.episode_data['vehicles_served'] = info['vehicles']
        if 'completed_trips' in info:
            self.episode_data['vehicles_served'] = info['completed_trips']
        if 'passenger_throughput' in info:
            self.episode_data['passenger_throughput'] = info['passenger_throughput']
        
        # Public transport metrics (if available)
        if 'buses_processed' in info:
            self.episode_data['buses_processed'] = info['buses_processed']
        if 'jeepneys_processed' in info:
            self.episode_data['jeepneys_processed'] = info['jeepneys_processed']
        if 'pt_passenger_throughput' in info:
            self.episode_data['pt_passenger_throughput'] = info['pt_passenger_throughput']
    
    def _create_step_entry(self, 
                          step: int,
                          reward: float,
                          info: Dict[str, Any],
                          actions: Dict[str, int] = None,
                          reward_components: Dict[str, float] = None) -> Dict:
        """Create detailed step entry for interval logging"""
        return {
            'step_id': str(uuid.uuid4()),
            'episode_id': self.episode_data['episode_id'],
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'simulation_time': step,  # Assuming 1 second per step
            
            # Core metrics
            'reward': reward,
            'vehicles': info.get('vehicles', 0),
            'avg_speed': info.get('avg_speed', 0),
            'queue_length': info.get('queue_length', 0),
            'waiting_time': info.get('waiting_time', 0),
            'completed_trips': info.get('completed_trips', 0),
            'throughput': info.get('throughput', 0),
            'passenger_throughput': info.get('passenger_throughput', 0),
            
            # Public transport (if available)
            'buses_active': info.get('buses_active', 0),
            'jeepneys_active': info.get('jeepneys_active', 0),
            'pt_waiting': info.get('pt_avg_waiting', 0),
            'pt_efficiency': info.get('pt_service_efficiency', 0),
            
            # Actions
            'actions': actions or {},
            'reward_components': reward_components or {},
            
            # Intersection-specific (MARL)
            'intersection_metrics': info.get('intersection_metrics', {})
        }
    
    def complete_episode(self, 
                        scenario_info: Dict[str, Any] = None,
                        final_metrics: Dict[str, Any] = None):
        """
        Complete episode logging with comprehensive summary
        
        Args:
            scenario_info: Information about the traffic scenario used
            final_metrics: Final episode metrics from environment
        """
        # Finalize episode data
        self.episode_data['end_time'] = datetime.now().isoformat()
        self.episode_data['scenario_info'] = scenario_info or {}
        
        # Calculate episode statistics
        if self.episode_data['reward_history']:
            rewards = self.episode_data['reward_history']
            self.episode_data['avg_reward'] = np.mean(rewards)
            self.episode_data['reward_std'] = np.std(rewards)
            self.episode_data['min_reward'] = np.min(rewards)
            self.episode_data['max_reward'] = np.max(rewards)
        
        # Add final metrics
        if final_metrics:
            self.episode_data.update(final_metrics)
        
        # Calculate derived metrics
        self._calculate_derived_metrics()
        
        # Store episode
        self.episode_buffer.append(self.episode_data.copy())
        
        # Write episode to file
        self._write_episode_log()
        
        # Prepare for next episode
        self.current_episode += 1
        self.episode_data = self._init_episode_data()
        
        # Get the actual completed episode data from buffer (last completed episode)
        completed_episode = self.episode_buffer[-1] if self.episode_buffer else self.episode_data
        
        print(f"📝 Episode {self.current_episode} logged: "
              f"Reward={completed_episode.get('total_reward', 0):.2f}, "
              f"Vehicles={completed_episode.get('vehicles_served', 0)}, "
              f"Completed={completed_episode.get('final_completed_trips', 0)}, "
              f"Time={completed_episode.get('episode_time', 0):.1f}s")
    
    def _calculate_derived_metrics(self):
        """Calculate research-standard derived metrics"""
        steps = self.episode_data['step_count']
        
        if steps > 0:
            # Phase utilization (effectiveness metric)
            if self.episode_data['passenger_throughput'] > 0:
                self.episode_data['phase_utilization_ratio'] = (
                    self.episode_data['passenger_throughput'] / 
                    (steps * 0.5)  # Theoretical maximum throughput
                )
            
            # Travel time index (mobility metric)
            if self.episode_data.get('avg_travel_time', 0) > 0:
                free_flow_time = 60  # Assumed 1-minute free flow time
                self.episode_data['travel_time_index'] = (
                    self.episode_data['avg_travel_time'] / free_flow_time
                )
            
            # PT service quality
            if self.episode_data['pt_passenger_throughput'] > 0:
                total_pt = (self.episode_data['buses_processed'] + 
                           self.episode_data['jeepneys_processed'])
                if total_pt > 0:
                    self.episode_data['pt_service_quality'] = (
                        self.episode_data['pt_passenger_throughput'] / total_pt
                    )
    
    def _flush_step_buffer(self):
        """Write step buffer to file"""
        with open(self.step_log_file, 'a', encoding='utf-8') as f:
            for entry in self.step_buffer:
                clean_entry = convert_numpy_types(entry)
                f.write(json.dumps(clean_entry) + '\n')
        self.step_buffer.clear()
    
    def _write_episode_log(self):
        """Write episode to file"""
        with open(self.episode_log_file, 'a', encoding='utf-8') as f:
            clean_data = convert_numpy_types(self.episode_data)
            f.write(json.dumps(clean_data) + '\n')
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive experiment summary for Supabase upload
        """
        if not self.episode_buffer:
            return {}
        
        episodes = list(self.episode_buffer)
        
        summary = {
            'experiment_id': self.experiment_name,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'total_episodes': len(episodes),
            
            # Performance Statistics
            'performance_metrics': {
                'avg_reward': np.mean([ep['avg_reward'] for ep in episodes if 'avg_reward' in ep]),
                'best_reward': np.max([ep['avg_reward'] for ep in episodes if 'avg_reward' in ep]),
                'reward_improvement': self._calculate_improvement_trend(episodes),
                'avg_passenger_throughput': np.mean([ep['passenger_throughput'] for ep in episodes]),
                'avg_vehicles_served': np.mean([ep['vehicles_served'] for ep in episodes]),
                'avg_pt_throughput': np.mean([ep['pt_passenger_throughput'] for ep in episodes])
            },
            
            # Training Progress
            'learning_metrics': {
                'convergence_episode': self._find_convergence_point(episodes),
                'stability_score': self._calculate_stability_score(episodes),
                'exploration_rate': self._calculate_exploration_rate(episodes)
            },
            
            # System Configuration
            'config': {
                'log_interval': self.log_interval,
                'buffer_size': self.buffer_size,
                'total_steps': sum([ep['step_count'] for ep in episodes])
            }
        }
        
        # Write summary
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            clean_summary = convert_numpy_types(summary)
            json.dump(clean_summary, f, indent=2)
        
        return summary
    
    def _calculate_improvement_trend(self, episodes: List[Dict]) -> float:
        """Calculate learning improvement trend"""
        if len(episodes) < 10:
            return 0.0
        
        early_rewards = [ep['avg_reward'] for ep in episodes[:5] if 'avg_reward' in ep]
        late_rewards = [ep['avg_reward'] for ep in episodes[-5:] if 'avg_reward' in ep]
        
        if early_rewards and late_rewards:
            early_avg = np.mean(early_rewards)
            late_avg = np.mean(late_rewards)
            return (late_avg - early_avg) / abs(early_avg) * 100 if early_avg != 0 else 0.0
        return 0.0
    
    def _find_convergence_point(self, episodes: List[Dict]) -> int:
        """Find episode where training converged"""
        if len(episodes) < 20:
            return -1
        
        rewards = [ep['avg_reward'] for ep in episodes if 'avg_reward' in ep]
        window_size = 10
        
        for i in range(window_size, len(rewards) - window_size):
            window = rewards[i:i+window_size]
            if np.std(window) < np.mean(window) * 0.1:  # 10% coefficient of variation
                return i
        return -1
    
    def _calculate_stability_score(self, episodes: List[Dict]) -> float:
        """Calculate training stability score"""
        if len(episodes) < 10:
            return 0.0
        
        rewards = [ep['avg_reward'] for ep in episodes if 'avg_reward' in ep]
        if not rewards:
            return 0.0
        
        # Coefficient of variation (lower is more stable)
        cv = np.std(rewards) / abs(np.mean(rewards)) if np.mean(rewards) != 0 else float('inf')
        return max(0.0, 1.0 - cv)  # Convert to stability score (0-1)
    
    def _calculate_exploration_rate(self, episodes: List[Dict]) -> float:
        """Calculate average exploration rate across episodes"""
        action_distributions = [ep['action_distribution'] for ep in episodes if ep['action_distribution']]
        if not action_distributions:
            return 0.0
        
        total_entropy = 0.0
        for action_dist in action_distributions:
            if action_dist:
                total_actions = sum(action_dist.values())
                if total_actions > 0:
                    probs = [count/total_actions for count in action_dist.values()]
                    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                    total_entropy += entropy
        
        return total_entropy / len(action_distributions) if action_distributions else 0.0
    
    def export_for_supabase(self, table_name: str = "traffic_experiments") -> Dict[str, Any]:
        """
        Export data in Supabase-ready format
        
        Returns:
            Dictionary ready for Supabase insertion
        """
        summary = self.generate_summary()
        
        # Flatten for Supabase table structure
        supabase_data = {
            'experiment_id': self.experiment_name,
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'total_episodes': summary.get('total_episodes', 0),
            
            # Performance metrics (flattened)
            'avg_reward': summary.get('performance_metrics', {}).get('avg_reward', 0),
            'best_reward': summary.get('performance_metrics', {}).get('best_reward', 0),
            'reward_improvement_pct': summary.get('performance_metrics', {}).get('reward_improvement', 0),
            'avg_passenger_throughput': summary.get('performance_metrics', {}).get('avg_passenger_throughput', 0),
            'avg_vehicles_served': summary.get('performance_metrics', {}).get('avg_vehicles_served', 0),
            'avg_pt_throughput': summary.get('performance_metrics', {}).get('avg_pt_throughput', 0),
            
            # Learning metrics
            'convergence_episode': summary.get('learning_metrics', {}).get('convergence_episode', -1),
            'stability_score': summary.get('learning_metrics', {}).get('stability_score', 0),
            'exploration_rate': summary.get('learning_metrics', {}).get('exploration_rate', 0),
            
            # Configuration
            'total_steps': summary.get('config', {}).get('total_steps', 0),
            'log_interval': self.log_interval,
            
            # Raw data paths (for detailed analysis)
            'step_log_file': self.step_log_file,
            'episode_log_file': self.episode_log_file,
            'summary_file': self.summary_file
        }
        
        return supabase_data
    
    def close(self):
        """Clean shutdown of logger"""
        # Flush remaining buffers
        if self.step_buffer:
            self._flush_step_buffer()
        
        # Generate final summary
        summary = self.generate_summary()
        
        print(f"📊 Production Logger Summary:")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Episodes: {summary.get('total_episodes', 0)}")
        print(f"   Avg Reward: {summary.get('performance_metrics', {}).get('avg_reward', 0):.2f}")
        print(f"   Files: {self.step_log_file}, {self.episode_log_file}")
        
        return summary


# Example usage for integration
def create_production_logger(experiment_name: str = None) -> ProductionLogger:
    """
    Factory function for creating production logger
    
    Usage in training:
        logger = create_production_logger("d3qn_marl_v1")
        
        # During training loop:
        logger.log_step(step, reward, info, actions, reward_components)
        
        # At episode end:
        logger.complete_episode(scenario_info, final_metrics)
        
        # At training end:
        supabase_data = logger.export_for_supabase()
        logger.close()
    """
    return ProductionLogger(
        log_interval=30,  # 30-second intervals (research standard)
        buffer_size=100,  # 100 entries before write
        experiment_name=experiment_name
    )
