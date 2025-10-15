"""
Real-Time Dashboard Logger for D3QN Traffic Signal Control
Logs training data to SQLite database during training for live dashboard updates
Author: D3QN Thesis Project
Date: October 13, 2025
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading


class RealtimeDashboardLogger:
    """
    Real-time logger that writes training data to SQLite database during training.
    Frontend fetches from this database for live updates.
    """
    
    def __init__(self, experiment_name: str, db_path: str = None):
        """
        Initialize real-time dashboard logger.
        
        Args:
            experiment_name: Name of the experiment
            db_path: Path to SQLite database (default: dashboard_data/training.db)
        """
        self.experiment_name = experiment_name
        self.db_path = db_path or 'dashboard_data/training.db'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Thread lock for concurrent writes
        self.lock = threading.Lock()
        
        # Passenger capacity multipliers
        self.passenger_capacity = {
            'cars': 1.3,
            'jeepneys': 14.0,
            'buses': 35.0,
            'motorcycles': 1.4,
            'trucks': 1.5,
            'tricycles': 2.5,
            'other': 1.0
        }
        
        # Initialize database
        self._init_database()
        
        # Create experiment record
        self._create_experiment_record()
    
    def _init_database(self):
        """Initialize database schema."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    total_episodes INTEGER DEFAULT 0,
                    completed_episodes INTEGER DEFAULT 0,
                    config TEXT
                )
            ''')
            
            # Episodes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    episode_number INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    phase TEXT,
                    scenario_name TEXT,
                    duration_seconds REAL,
                    steps INTEGER,
                    
                    -- Raw metrics
                    vehicles_completed INTEGER,
                    passengers_completed INTEGER,
                    
                    -- Performance metrics
                    avg_waiting_time REAL,
                    avg_speed REAL,
                    avg_queue_length REAL,
                    max_queue_length INTEGER,
                    
                    -- Training metrics
                    total_reward REAL,
                    avg_loss REAL,
                    epsilon REAL,
                    
                    -- Reward components
                    throughput_reward REAL,
                    waiting_penalty REAL,
                    speed_reward REAL,
                    queue_penalty REAL,
                    emergency_penalty REAL,
                    
                    FOREIGN KEY (experiment_name) REFERENCES experiments(name),
                    UNIQUE(experiment_name, episode_number)
                )
            ''')
            
            # Vehicle breakdown table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicle_breakdown (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    episode_number INTEGER NOT NULL,
                    
                    -- Vehicle counts
                    cars INTEGER DEFAULT 0,
                    jeepneys INTEGER DEFAULT 0,
                    buses INTEGER DEFAULT 0,
                    motorcycles INTEGER DEFAULT 0,
                    trucks INTEGER DEFAULT 0,
                    tricycles INTEGER DEFAULT 0,
                    other INTEGER DEFAULT 0,
                    
                    -- Passenger counts
                    cars_passengers INTEGER DEFAULT 0,
                    jeepneys_passengers INTEGER DEFAULT 0,
                    buses_passengers INTEGER DEFAULT 0,
                    motorcycles_passengers INTEGER DEFAULT 0,
                    trucks_passengers INTEGER DEFAULT 0,
                    tricycles_passengers INTEGER DEFAULT 0,
                    other_passengers INTEGER DEFAULT 0,
                    
                    FOREIGN KEY (experiment_name) REFERENCES experiments(name),
                    UNIQUE(experiment_name, episode_number)
                )
            ''')
            
            # Evaluation results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    episode_number INTEGER NOT NULL,
                    
                    vehicles_completed INTEGER,
                    passengers_completed INTEGER,
                    avg_waiting_time REAL,
                    avg_speed REAL,
                    avg_queue_length REAL,
                    max_queue_length INTEGER,
                    total_reward REAL,
                    
                    FOREIGN KEY (experiment_name) REFERENCES experiments(name),
                    UNIQUE(experiment_name, agent_type, episode_number)
                )
            ''')
            
            # Summary statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS summary_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT UNIQUE NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- D3QN performance
                    d3qn_avg_vehicles REAL,
                    d3qn_avg_passengers REAL,
                    d3qn_avg_waiting REAL,
                    d3qn_avg_speed REAL,
                    d3qn_avg_queue REAL,
                    
                    -- Fixed-Time performance
                    fixed_avg_vehicles REAL,
                    fixed_avg_passengers REAL,
                    fixed_avg_waiting REAL,
                    fixed_avg_speed REAL,
                    fixed_avg_queue REAL,
                    
                    -- Improvements
                    throughput_improvement REAL,
                    passengers_improvement REAL,
                    waiting_improvement REAL,
                    speed_improvement REAL,
                    queue_improvement REAL,
                    
                    -- Statistical significance
                    p_value REAL,
                    cohens_d REAL,
                    
                    FOREIGN KEY (experiment_name) REFERENCES experiments(name)
                )
            ''')
            
            # Create indices for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_episodes_experiment 
                ON episodes(experiment_name, episode_number)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_vehicle_breakdown_experiment 
                ON vehicle_breakdown(experiment_name, episode_number)
            ''')
            
            conn.commit()
    
    def _create_experiment_record(self):
        """Create or update experiment record."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR IGNORE INTO experiments (name, status)
                VALUES (?, 'running')
            ''', (self.experiment_name,))
            
            conn.commit()
    
    def log_episode(self, episode_number: int, metrics: Dict[str, Any], 
                   vehicle_breakdown: Dict[str, int] = None, phase: str = 'online'):
        """
        Log episode data to database in real-time.
        Called during training after each episode completes.
        
        Args:
            episode_number: Episode number
            metrics: Episode metrics dictionary
            vehicle_breakdown: Vehicle type counts (optional, estimated if not provided)
            phase: Training phase ('offline' or 'online')
        """
        
        with self.lock:
            # Estimate vehicle breakdown if not provided
            if vehicle_breakdown is None:
                vehicle_breakdown = self._estimate_vehicle_breakdown(
                    int(metrics.get('completed_trips', 0))
                )
            
            # Calculate passengers
            passengers_completed = self._calculate_passengers(vehicle_breakdown)
            passenger_breakdown = self._calculate_passenger_breakdown(vehicle_breakdown)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert episode data
                cursor.execute('''
                    INSERT OR REPLACE INTO episodes (
                        experiment_name, episode_number, phase, scenario_name,
                        duration_seconds, steps,
                        vehicles_completed, passengers_completed,
                        avg_waiting_time, avg_speed, avg_queue_length, max_queue_length,
                        total_reward, avg_loss, epsilon,
                        throughput_reward, waiting_penalty, speed_reward, 
                        queue_penalty, emergency_penalty
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.experiment_name,
                    episode_number,
                    phase,
                    metrics.get('scenario', f'Episode {episode_number}'),
                    metrics.get('duration_seconds', 300),
                    metrics.get('steps', 60),
                    int(metrics.get('completed_trips', 0)),
                    int(passengers_completed),
                    float(metrics.get('waiting_time', 0)),
                    float(metrics.get('avg_speed', 0)),
                    float(metrics.get('queue_length', 0)),
                    int(metrics.get('max_queue_length', 0)),
                    float(metrics.get('total_reward', 0)),
                    float(metrics.get('loss', 0)),
                    float(metrics.get('epsilon', 0)),
                    float(metrics.get('throughput_reward', 0)),
                    float(metrics.get('waiting_penalty', 0)),
                    float(metrics.get('speed_reward', 0)),
                    float(metrics.get('queue_penalty', 0)),
                    float(metrics.get('emergency_penalty', 0))
                ))
                
                # Insert vehicle breakdown
                cursor.execute('''
                    INSERT OR REPLACE INTO vehicle_breakdown (
                        experiment_name, episode_number,
                        cars, jeepneys, buses, motorcycles, trucks, tricycles, other,
                        cars_passengers, jeepneys_passengers, buses_passengers,
                        motorcycles_passengers, trucks_passengers, tricycles_passengers,
                        other_passengers
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.experiment_name,
                    episode_number,
                    vehicle_breakdown.get('cars', 0),
                    vehicle_breakdown.get('jeepneys', 0),
                    vehicle_breakdown.get('buses', 0),
                    vehicle_breakdown.get('motorcycles', 0),
                    vehicle_breakdown.get('trucks', 0),
                    vehicle_breakdown.get('tricycles', 0),
                    vehicle_breakdown.get('other', 0),
                    passenger_breakdown.get('cars', 0),
                    passenger_breakdown.get('jeepneys', 0),
                    passenger_breakdown.get('buses', 0),
                    passenger_breakdown.get('motorcycles', 0),
                    passenger_breakdown.get('trucks', 0),
                    passenger_breakdown.get('tricycles', 0),
                    passenger_breakdown.get('other', 0)
                ))
                
                # Update experiment progress
                cursor.execute('''
                    UPDATE experiments 
                    SET completed_episodes = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE name = ?
                ''', (episode_number, self.experiment_name))
                
                conn.commit()
    
    def log_evaluation_results(self, agent_type: str, episodes: List[Dict[str, Any]]):
        """
        Log evaluation results to database.
        Called after evaluation completes.
        
        Args:
            agent_type: 'd3qn' or 'fixed_time'
            episodes: List of episode result dictionaries
        """
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for i, ep in enumerate(episodes):
                    # Estimate vehicle breakdown
                    vehicle_breakdown = self._estimate_vehicle_breakdown(
                        int(ep.get('completed_trips', 0))
                    )
                    passengers = self._calculate_passengers(vehicle_breakdown)
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO evaluation_results (
                            experiment_name, agent_type, episode_number,
                            vehicles_completed, passengers_completed,
                            avg_waiting_time, avg_speed, avg_queue_length,
                            max_queue_length, total_reward
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        self.experiment_name,
                        agent_type,
                        i + 1,
                        int(ep.get('completed_trips', 0)),
                        int(passengers),
                        float(ep.get('avg_waiting_time', 0)),
                        float(ep.get('avg_speed', 0)),
                        float(ep.get('avg_queue_length', 0)),
                        int(ep.get('max_queue_length', 0)),
                        float(ep.get('total_reward', 0))
                    ))
                
                conn.commit()
    
    def update_summary_statistics(self, d3qn_avg: Dict, fixed_avg: Dict, 
                                  improvements: Dict, stats: Dict = None):
        """
        Update summary statistics in database.
        Called after evaluation completes.
        
        Args:
            d3qn_avg: D3QN average performance
            fixed_avg: Fixed-Time average performance
            improvements: Improvement percentages
            stats: Statistical test results (optional)
        """
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO summary_statistics (
                        experiment_name,
                        d3qn_avg_vehicles, d3qn_avg_passengers, d3qn_avg_waiting,
                        d3qn_avg_speed, d3qn_avg_queue,
                        fixed_avg_vehicles, fixed_avg_passengers, fixed_avg_waiting,
                        fixed_avg_speed, fixed_avg_queue,
                        throughput_improvement, passengers_improvement, waiting_improvement,
                        speed_improvement, queue_improvement,
                        p_value, cohens_d
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.experiment_name,
                    d3qn_avg.get('vehicles', 0),
                    d3qn_avg.get('passengers', 0),
                    d3qn_avg.get('waiting', 0),
                    d3qn_avg.get('speed', 0),
                    d3qn_avg.get('queue', 0),
                    fixed_avg.get('vehicles', 0),
                    fixed_avg.get('passengers', 0),
                    fixed_avg.get('waiting', 0),
                    fixed_avg.get('speed', 0),
                    fixed_avg.get('queue', 0),
                    improvements.get('throughput', 0),
                    improvements.get('passengers', 0),
                    improvements.get('waiting_time', 0),
                    improvements.get('speed', 0),
                    improvements.get('queue_length', 0),
                    stats.get('p_value', 0) if stats else 0,
                    stats.get('cohens_d', 0) if stats else 0
                ))
                
                conn.commit()
    
    def mark_experiment_complete(self, total_episodes: int):
        """Mark experiment as completed."""
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE experiments
                    SET status = 'completed',
                        total_episodes = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE name = ?
                ''', (total_episodes, self.experiment_name))
                
                conn.commit()
    
    def _estimate_vehicle_breakdown(self, completed_trips: int) -> Dict[str, int]:
        """Estimate vehicle breakdown from total trips using Manila traffic distribution."""
        
        distribution = {
            'cars': 0.35,
            'jeepneys': 0.25,
            'buses': 0.08,
            'motorcycles': 0.20,
            'trucks': 0.07,
            'tricycles': 0.05,
        }
        
        breakdown = {}
        for veh_type, percentage in distribution.items():
            breakdown[veh_type] = int(completed_trips * percentage)
        
        # Adjust for rounding errors
        total = sum(breakdown.values())
        if total < completed_trips:
            breakdown['cars'] += (completed_trips - total)
        
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


# API functions for frontend to fetch data
def get_experiment_status(db_path: str, experiment_name: str) -> Dict:
    """Get current experiment status."""
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, created_at, updated_at, status, total_episodes, completed_episodes
            FROM experiments WHERE name = ?
        ''', (experiment_name,))
        
        row = cursor.fetchone()
        if row:
            return {
                'name': row[0],
                'created_at': row[1],
                'updated_at': row[2],
                'status': row[3],
                'total_episodes': row[4],
                'completed_episodes': row[5]
            }
        return None


def get_training_progress(db_path: str, experiment_name: str) -> List[Dict]:
    """Get all training episodes for progress charts."""
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM episodes 
            WHERE experiment_name = ?
            ORDER BY episode_number
        ''', (experiment_name,))
        
        return [dict(row) for row in cursor.fetchall()]


def get_vehicle_breakdown_aggregated(db_path: str, experiment_name: str) -> Dict:
    """Get aggregated vehicle breakdown across all episodes."""
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                SUM(cars) as total_cars,
                SUM(jeepneys) as total_jeepneys,
                SUM(buses) as total_buses,
                SUM(motorcycles) as total_motorcycles,
                SUM(trucks) as total_trucks,
                SUM(tricycles) as total_tricycles,
                SUM(cars_passengers) as total_cars_passengers,
                SUM(jeepneys_passengers) as total_jeepneys_passengers,
                SUM(buses_passengers) as total_buses_passengers,
                SUM(motorcycles_passengers) as total_motorcycles_passengers,
                SUM(trucks_passengers) as total_trucks_passengers,
                SUM(tricycles_passengers) as total_tricycles_passengers
            FROM vehicle_breakdown
            WHERE experiment_name = ?
        ''', (experiment_name,))
        
        row = cursor.fetchone()
        if row:
            return {
                'vehicles': {
                    'cars': row[0],
                    'jeepneys': row[1],
                    'buses': row[2],
                    'motorcycles': row[3],
                    'trucks': row[4],
                    'tricycles': row[5]
                },
                'passengers': {
                    'cars': row[6],
                    'jeepneys': row[7],
                    'buses': row[8],
                    'motorcycles': row[9],
                    'trucks': row[10],
                    'tricycles': row[11]
                }
            }
        return None


def get_summary_statistics(db_path: str, experiment_name: str) -> Dict:
    """Get summary statistics for D3QN vs Fixed-Time comparison."""
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM summary_statistics
            WHERE experiment_name = ?
        ''', (experiment_name,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None



