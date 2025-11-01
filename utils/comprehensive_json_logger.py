"""
Comprehensive JSON Logger for D3QN Traffic Signal Control
Logs all vehicle, signal, and lane data in JSON format for dashboard
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

class ComprehensiveJSONLogger:
    """Comprehensive JSON logger for all simulation data"""
    
    def __init__(self, experiment_name: str, base_dir: str = "comprehensive_results"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(base_dir, f"{experiment_name}_{self.timestamp}")
        
        # Create experiment directory
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize log files
        self.vehicle_log_file = os.path.join(self.experiment_dir, "vehicle_data.jsonl")
        self.signal_log_file = os.path.join(self.experiment_dir, "signal_phases.jsonl")
        self.lane_log_file = os.path.join(self.experiment_dir, "lane_metrics.jsonl")
        self.episode_log_file = os.path.join(self.experiment_dir, "episode_summaries.jsonl")
        
        # Episode tracking
        self.current_episode = 0
        self.episode_data = {
            'vehicles': [],
            'signals': [],
            'lanes': []
        }
    
    def log_vehicle_data(self, step: int, vehicle_data: Dict[str, Any]):
        """Log vehicle data for a single step"""
        log_entry = {
            'episode': self.current_episode,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'data': vehicle_data
        }
        
        with open(self.vehicle_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.episode_data['vehicles'].append(log_entry)
    
    def log_signal_phase(self, step: int, signal_data: Dict[str, Any]):
        """Log signal phase data for a single step"""
        log_entry = {
            'episode': self.current_episode,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'data': signal_data
        }
        
        with open(self.signal_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.episode_data['signals'].append(log_entry)
    
    def log_lane_metrics(self, step: int, lane_data: Dict[str, Any]):
        """Log lane metrics for a single step"""
        log_entry = {
            'episode': self.current_episode,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'data': lane_data
        }
        
        with open(self.lane_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.episode_data['lanes'].append(log_entry)
    
    def finalize_episode(self, episode_num: int, episode_summary: Dict[str, Any]):
        """Finalize episode logging with summary"""
        episode_summary['episode'] = episode_num
        episode_summary['timestamp'] = datetime.now().isoformat()
        
        with open(self.episode_log_file, 'a') as f:
            f.write(json.dumps(episode_summary) + '\n')
        
        # Reset episode data
        self.episode_data = {
            'vehicles': [],
            'signals': [],
            'lanes': []
        }
    
    def set_episode(self, episode: int):
        """Set current episode number"""
        self.current_episode = episode
