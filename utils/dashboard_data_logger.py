"""
Dashboard Data Logger for D3QN Traffic Signal Control
Extracts training/evaluation data in dashboard-friendly format
Author: D3QN Thesis Project
Date: October 13, 2025
"""

import os
import json
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Optional

class DashboardDataLogger:
    """
    Extract and format data from training/evaluation for dashboard consumption.
    Uses RAW EPISODE VALUES (not hourly rates) for better dashboard visualization.
    """
    
    def __init__(self, output_dir: str = 'dashboard_data'):
        """
        Initialize dashboard data logger.
        
        Args:
            output_dir: Directory to save dashboard data files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize JSON files
        self.summary_file = os.path.join(output_dir, 'summary_metrics.json')
        self.episodes_file = os.path.join(output_dir, 'episodes_data.json')
        self.vehicles_file = os.path.join(output_dir, 'vehicle_breakdown.json')
        self.stats_file = os.path.join(output_dir, 'statistical_analysis.json')
        
        # Passenger capacity multipliers (from thesis methodology)
        self.passenger_capacity = {
            'cars': 1.3,
            'jeepneys': 14.0,      # PUBLIC TRANSPORT - HIGH PRIORITY
            'buses': 35.0,         # PUBLIC TRANSPORT - HIGH PRIORITY
            'motorcycles': 1.4,
            'trucks': 1.5,
            'tricycles': 2.5,
            'other': 1.0
        }
    
    def log_episode(self, episode_number: int, metrics: Dict[str, Any], 
                   vehicle_breakdown: Dict[str, int], phase: str = 'online') -> Dict[str, Any]:
        """
        Log single episode data for dashboard.
        
        Args:
            episode_number: Episode number
            metrics: Episode metrics dictionary
            vehicle_breakdown: Vehicle type counts
            phase: Training phase ('offline' or 'online')
        
        Returns:
            Episode data dictionary
        """
        
        # Calculate passenger throughput
        passengers_completed = self._calculate_passengers(vehicle_breakdown)
        
        episode_data = {
            # Episode Info
            'episode_number': episode_number,
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'scenario_name': metrics.get('scenario', f'Episode {episode_number}'),
            'duration_seconds': 300,  # Fixed duration
            'steps': metrics.get('steps', 60),
            
            # RAW VALUES (NOT hourly rates)
            'vehicles_completed': int(metrics.get('completed_trips', 0)),
            'passengers_completed': int(passengers_completed),
            
            # Vehicle Type Breakdown
            'vehicle_breakdown': vehicle_breakdown,
            'passenger_breakdown': self._calculate_passenger_breakdown(vehicle_breakdown),
            
            # Performance Metrics
            'avg_waiting_time': float(metrics.get('waiting_time', 0)),
            'avg_speed': float(metrics.get('avg_speed', 0)),
            'avg_queue_length': float(metrics.get('queue_length', 0)),
            'max_queue_length': int(metrics.get('max_queue_length', 0)),
            
            # Training Metrics
            'total_reward': float(metrics.get('total_reward', 0)),
            'avg_loss': float(metrics.get('loss', 0)),
            'epsilon': float(metrics.get('epsilon', 0)),
            
            # Reward Components (if available)
            'reward_components': {
                'throughput_reward': float(metrics.get('throughput_reward', 0)),
                'waiting_penalty': float(metrics.get('waiting_penalty', 0)),
                'speed_reward': float(metrics.get('speed_reward', 0)),
                'queue_penalty': float(metrics.get('queue_penalty', 0)),
                'emergency_penalty': float(metrics.get('emergency_penalty', 0))
            }
        }
        
        # Append to episodes file
        self._append_to_json(self.episodes_file, episode_data)
        
        return episode_data
    
    def log_vehicle_breakdown_from_traci(self, traci) -> Dict[str, int]:
        """
        Extract vehicle type breakdown from SUMO/TRACI.
        
        Args:
            traci: SUMO TRACI connection
        
        Returns:
            Dictionary of vehicle type counts
        """
        
        breakdown = {
            'cars': 0,
            'jeepneys': 0,
            'buses': 0,
            'motorcycles': 0,
            'trucks': 0,
            'tricycles': 0,
            'other': 0
        }
        
        try:
            # Count arrived vehicles by type
            arrived_vehicles = traci.simulation.getArrivedIDList()
            
            for veh_id in arrived_vehicles:
                veh_id_lower = veh_id.lower()
                
                if 'car' in veh_id_lower:
                    breakdown['cars'] += 1
                elif 'jeepney' in veh_id_lower or 'jeep' in veh_id_lower:
                    breakdown['jeepneys'] += 1
                elif 'bus' in veh_id_lower:
                    breakdown['buses'] += 1
                elif 'motor' in veh_id_lower:
                    breakdown['motorcycles'] += 1
                elif 'truck' in veh_id_lower:
                    breakdown['trucks'] += 1
                elif 'trike' in veh_id_lower or 'tricycle' in veh_id_lower:
                    breakdown['tricycles'] += 1
                else:
                    breakdown['other'] += 1
        except Exception as e:
            print(f"Warning: Could not extract vehicle breakdown: {e}")
        
        return breakdown
    
    def _calculate_passengers(self, vehicle_breakdown: Dict[str, int]) -> float:
        """Calculate total passengers from vehicle breakdown."""
        
        total_passengers = 0
        for veh_type, count in vehicle_breakdown.items():
            capacity = self.passenger_capacity.get(veh_type, 1.0)
            total_passengers += count * capacity
        
        return total_passengers
    
    def _calculate_passenger_breakdown(self, vehicle_breakdown: Dict[str, int]) -> Dict[str, int]:
        """Calculate passenger breakdown by vehicle type."""
        
        passenger_breakdown = {}
        for veh_type, count in vehicle_breakdown.items():
            capacity = self.passenger_capacity.get(veh_type, 1.0)
            passenger_breakdown[veh_type] = int(count * capacity)
        
        return passenger_breakdown
    
    def generate_summary(self, d3qn_episodes: List[Dict], 
                        fixed_time_episodes: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary metrics comparing D3QN vs Fixed-Time.
        
        Args:
            d3qn_episodes: List of D3QN episode data
            fixed_time_episodes: List of Fixed-Time episode data
        
        Returns:
            Summary dictionary with comparisons and improvements
        """
        
        # Calculate aggregates
        d3qn_avg = self._aggregate_episodes(d3qn_episodes)
        fixed_avg = self._aggregate_episodes(fixed_time_episodes)
        
        # Calculate improvements
        improvements = {
            'throughput': self._calculate_improvement(
                d3qn_avg['vehicles'], fixed_avg['vehicles']
            ),
            'passengers': self._calculate_improvement(
                d3qn_avg['passengers'], fixed_avg['passengers']
            ),
            'waiting_time': self._calculate_improvement(
                d3qn_avg['waiting'], fixed_avg['waiting'], lower_is_better=True
            ),
            'speed': self._calculate_improvement(
                d3qn_avg['speed'], fixed_avg['speed']
            ),
            'queue_length': self._calculate_improvement(
                d3qn_avg['queue'], fixed_avg['queue'], lower_is_better=True
            )
        }
        
        # Add layman's terms explanations
        improvements['explanations'] = {
            'throughput': f"{abs(improvements['throughput']):.1f}% improvement = {int(d3qn_avg['vehicles'] - fixed_avg['vehicles'])} more vehicles per episode",
            'passengers': f"{abs(improvements['passengers']):.1f}% improvement = {int(d3qn_avg['passengers'] - fixed_avg['passengers'])} more passengers per episode",
            'waiting_time': f"{abs(improvements['waiting_time']):.1f}% reduction = {abs(d3qn_avg['waiting'] - fixed_avg['waiting']):.2f} seconds less waiting",
            'speed': f"{abs(improvements['speed']):.1f}% improvement = {abs(d3qn_avg['speed'] - fixed_avg['speed']):.2f} km/h faster",
            'queue_length': f"{abs(improvements['queue_length']):.1f}% reduction = {abs(d3qn_avg['queue'] - fixed_avg['queue']):.0f} fewer vehicles in queue"
        }
        
        summary = {
            'd3qn_performance': d3qn_avg,
            'fixed_time_performance': fixed_avg,
            'improvements': improvements,
            'training_info': {
                'total_episodes': len(d3qn_episodes),
                'convergence_episode': self._find_convergence(d3qn_episodes),
                'best_episode': self._find_best_episode(d3qn_episodes),
                'best_episode_vehicles': max(ep.get('vehicles_completed', 0) for ep in d3qn_episodes),
            },
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        # Save to file
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _aggregate_episodes(self, episodes: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate statistics for episodes."""
        
        if not episodes:
            return {
                'vehicles': 0, 'passengers': 0, 'waiting': 0,
                'speed': 0, 'queue': 0, 'reward': 0
            }
        
        return {
            'vehicles': np.mean([ep.get('vehicles_completed', 0) for ep in episodes]),
            'passengers': np.mean([ep.get('passengers_completed', 0) for ep in episodes]),
            'waiting': np.mean([ep.get('avg_waiting_time', 0) for ep in episodes]),
            'speed': np.mean([ep.get('avg_speed', 0) for ep in episodes]),
            'queue': np.mean([ep.get('avg_queue_length', 0) for ep in episodes]),
            'reward': np.mean([ep.get('total_reward', 0) for ep in episodes])
        }
    
    def _calculate_improvement(self, d3qn_value: float, fixed_value: float, 
                              lower_is_better: bool = False) -> float:
        """Calculate percentage improvement."""
        
        if fixed_value == 0:
            return 0.0
        
        improvement = ((d3qn_value - fixed_value) / fixed_value) * 100
        
        # Invert sign if lower is better
        if lower_is_better:
            improvement = -improvement
        
        return improvement
    
    def _find_convergence(self, episodes: List[Dict]) -> int:
        """Find episode where agent converged."""
        
        if len(episodes) < 10:
            return len(episodes)
        
        # Calculate rolling average of vehicles completed
        vehicles = [ep.get('vehicles_completed', 0) for ep in episodes]
        window = 10
        
        for i in range(window, len(vehicles)):
            rolling_avg = np.mean(vehicles[i-window:i])
            rolling_std = np.std(vehicles[i-window:i])
            
            # Convergence: std < 5% of mean for sustained period
            if rolling_std < 0.05 * rolling_avg:
                return i - window
        
        return len(episodes)
    
    def _find_best_episode(self, episodes: List[Dict]) -> int:
        """Find episode with best vehicle throughput."""
        
        if not episodes:
            return 0
        
        best_idx = np.argmax([ep.get('vehicles_completed', 0) for ep in episodes])
        return episodes[best_idx].get('episode_number', best_idx + 1)
    
    def _append_to_json(self, filepath: str, data: Dict[str, Any]):
        """Append data to JSON array file."""
        
        # Read existing data
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []
        
        # Append new data
        existing_data.append(data)
        
        # Write back
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def prepare_dashboard_package(self, experiment_name: str) -> str:
        """
        Prepare complete dashboard data package.
        
        Args:
            experiment_name: Name of the experiment
        
        Returns:
            Path to dashboard package JSON file
        """
        
        # Load episodes data
        episodes_data = []
        if os.path.exists(self.episodes_file):
            with open(self.episodes_file, 'r') as f:
                episodes_data = json.load(f)
        
        # Calculate training progress
        training_progress = {
            'episodes': [ep['episode_number'] for ep in episodes_data],
            'vehicles': [ep['vehicles_completed'] for ep in episodes_data],
            'passengers': [ep['passengers_completed'] for ep in episodes_data],
            'rewards': [ep['total_reward'] for ep in episodes_data],
            'losses': [ep['avg_loss'] for ep in episodes_data],
            'waiting_times': [ep['avg_waiting_time'] for ep in episodes_data],
            'speeds': [ep['avg_speed'] for ep in episodes_data],
            'queue_lengths': [ep['avg_queue_length'] for ep in episodes_data]
        }
        
        # Aggregate vehicle breakdown
        total_vehicles = {veh_type: 0 for veh_type in self.passenger_capacity.keys()}
        total_passengers = {veh_type: 0 for veh_type in self.passenger_capacity.keys()}
        
        for ep in episodes_data:
            veh_breakdown = ep.get('vehicle_breakdown', {})
            pass_breakdown = ep.get('passenger_breakdown', {})
            
            for veh_type in total_vehicles.keys():
                total_vehicles[veh_type] += veh_breakdown.get(veh_type, 0)
                total_passengers[veh_type] += pass_breakdown.get(veh_type, 0)
        
        # Create dashboard package
        dashboard_package = {
            'metadata': {
                'experiment_name': experiment_name,
                'generated_at': datetime.now().isoformat(),
                'total_episodes': len(episodes_data),
                'version': '1.0.0'
            },
            'episodes': episodes_data,
            'training_progress': training_progress,
            'vehicle_breakdown_total': total_vehicles,
            'passenger_breakdown_total': total_passengers,
            'public_transport_stats': {
                'total_jeepneys': total_vehicles['jeepneys'],
                'total_buses': total_vehicles['buses'],
                'total_public_transport_passengers': total_passengers['jeepneys'] + total_passengers['buses'],
                'percentage_passengers_public_transport': (
                    (total_passengers['jeepneys'] + total_passengers['buses']) / 
                    sum(total_passengers.values()) * 100
                ) if sum(total_passengers.values()) > 0 else 0
            }
        }
        
        # Save dashboard package
        output_file = os.path.join(self.output_dir, 'dashboard_package.json')
        with open(output_file, 'w') as f:
            json.dump(dashboard_package, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("✅ DASHBOARD DATA PACKAGE PREPARED")
        print("="*80)
        print(f"📁 Output File: {output_file}")
        print(f"📊 Total Episodes: {len(episodes_data)}")
        print(f"🚗 Total Vehicles: {sum(total_vehicles.values())}")
        print(f"👥 Total Passengers: {sum(total_passengers.values())}")
        print(f"🚌 Public Transport Passengers: {dashboard_package['public_transport_stats']['total_public_transport_passengers']} ({dashboard_package['public_transport_stats']['percentage_passengers_public_transport']:.1f}%)")
        print("="*80)
        
        return output_file


# Utility functions for external use
def create_vehicle_breakdown_estimate(completed_trips: int) -> Dict[str, int]:
    """
    Create estimated vehicle breakdown when TRACI is not available.
    Uses typical Manila traffic distribution.
    
    Args:
        completed_trips: Total completed trips
    
    Returns:
        Estimated vehicle breakdown
    """
    
    # Typical Manila traffic distribution (estimated from literature)
    distribution = {
        'cars': 0.35,          # 35% cars
        'jeepneys': 0.25,      # 25% jeepneys (PUBLIC TRANSPORT)
        'buses': 0.08,         # 8% buses (PUBLIC TRANSPORT)
        'motorcycles': 0.20,   # 20% motorcycles
        'trucks': 0.07,        # 7% trucks
        'tricycles': 0.05,     # 5% tricycles
    }
    
    breakdown = {}
    for veh_type, percentage in distribution.items():
        breakdown[veh_type] = int(completed_trips * percentage)
    
    # Adjust for rounding errors
    total = sum(breakdown.values())
    if total < completed_trips:
        breakdown['cars'] += (completed_trips - total)
    
    return breakdown



