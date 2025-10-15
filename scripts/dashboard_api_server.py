"""
Dashboard API Server for D3QN Training Data
Provides REST API endpoints for frontend to fetch real-time training data
Author: D3QN Thesis Project
Date: October 13, 2025
"""

import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.realtime_dashboard_logger import (
    get_experiment_status,
    get_training_progress,
    get_vehicle_breakdown_aggregated,
    get_summary_statistics
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

DB_PATH = 'dashboard_data/training.db'


@app.route('/api/experiments', methods=['GET'])
def list_experiments():
    """List all experiments in database."""
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT name, created_at, updated_at, status, 
                       total_episodes, completed_episodes
                FROM experiments
                ORDER BY created_at DESC
            ''')
            
            experiments = [dict(row) for row in cursor.fetchall()]
            
            return jsonify({
                'success': True,
                'experiments': experiments
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/experiments/<experiment_name>/status', methods=['GET'])
def experiment_status(experiment_name):
    """Get experiment status."""
    
    try:
        status = get_experiment_status(DB_PATH, experiment_name)
        
        if status:
            return jsonify({
                'success': True,
                'status': status
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Experiment not found'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/experiments/<experiment_name>/episodes', methods=['GET'])
def training_episodes(experiment_name):
    """Get all training episodes."""
    
    try:
        episodes = get_training_progress(DB_PATH, experiment_name)
        
        return jsonify({
            'success': True,
            'episodes': episodes,
            'total': len(episodes)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/experiments/<experiment_name>/episodes/<int:episode_number>', methods=['GET'])
def episode_detail(experiment_name, episode_number):
    """Get specific episode details."""
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get episode data
            cursor.execute('''
                SELECT * FROM episodes
                WHERE experiment_name = ? AND episode_number = ?
            ''', (experiment_name, episode_number))
            
            episode = cursor.fetchone()
            
            if not episode:
                return jsonify({
                    'success': False,
                    'error': 'Episode not found'
                }), 404
            
            # Get vehicle breakdown
            cursor.execute('''
                SELECT * FROM vehicle_breakdown
                WHERE experiment_name = ? AND episode_number = ?
            ''', (experiment_name, episode_number))
            
            breakdown = cursor.fetchone()
            
            result = dict(episode)
            if breakdown:
                result['vehicle_breakdown'] = {
                    'cars': breakdown['cars'],
                    'jeepneys': breakdown['jeepneys'],
                    'buses': breakdown['buses'],
                    'motorcycles': breakdown['motorcycles'],
                    'trucks': breakdown['trucks'],
                    'tricycles': breakdown['tricycles']
                }
                result['passenger_breakdown'] = {
                    'cars': breakdown['cars_passengers'],
                    'jeepneys': breakdown['jeepneys_passengers'],
                    'buses': breakdown['buses_passengers'],
                    'motorcycles': breakdown['motorcycles_passengers'],
                    'trucks': breakdown['trucks_passengers'],
                    'tricycles': breakdown['tricycles_passengers']
                }
            
            return jsonify({
                'success': True,
                'episode': result
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/experiments/<experiment_name>/progress', methods=['GET'])
def training_progress(experiment_name):
    """Get training progress data for charts."""
    
    try:
        episodes = get_training_progress(DB_PATH, experiment_name)
        
        # Format for charts
        progress = {
            'episodes': [ep['episode_number'] for ep in episodes],
            'vehicles': [ep['vehicles_completed'] for ep in episodes],
            'passengers': [ep['passengers_completed'] for ep in episodes],
            'rewards': [ep['total_reward'] for ep in episodes],
            'losses': [ep['avg_loss'] for ep in episodes],
            'waiting_times': [ep['avg_waiting_time'] for ep in episodes],
            'speeds': [ep['avg_speed'] for ep in episodes],
            'queue_lengths': [ep['avg_queue_length'] for ep in episodes],
            'phases': [ep['phase'] for ep in episodes]
        }
        
        return jsonify({
            'success': True,
            'progress': progress
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/experiments/<experiment_name>/vehicles', methods=['GET'])
def vehicle_breakdown(experiment_name):
    """Get vehicle and passenger breakdown."""
    
    try:
        breakdown = get_vehicle_breakdown_aggregated(DB_PATH, experiment_name)
        
        if breakdown:
            # Calculate percentages
            total_vehicles = sum(breakdown['vehicles'].values())
            total_passengers = sum(breakdown['passengers'].values())
            
            result = {
                'vehicles': breakdown['vehicles'],
                'passengers': breakdown['passengers'],
                'vehicle_percentages': {
                    k: (v / total_vehicles * 100) if total_vehicles > 0 else 0
                    for k, v in breakdown['vehicles'].items()
                },
                'passenger_percentages': {
                    k: (v / total_passengers * 100) if total_passengers > 0 else 0
                    for k, v in breakdown['passengers'].items()
                },
                'totals': {
                    'vehicles': total_vehicles,
                    'passengers': total_passengers
                },
                'public_transport': {
                    'vehicles': breakdown['vehicles']['jeepneys'] + breakdown['vehicles']['buses'],
                    'passengers': breakdown['passengers']['jeepneys'] + breakdown['passengers']['buses'],
                    'percentage': (
                        (breakdown['passengers']['jeepneys'] + breakdown['passengers']['buses']) / 
                        total_passengers * 100
                    ) if total_passengers > 0 else 0
                }
            }
            
            return jsonify({
                'success': True,
                'breakdown': result
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No data found'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/experiments/<experiment_name>/summary', methods=['GET'])
def summary_statistics(experiment_name):
    """Get summary statistics (D3QN vs Fixed-Time)."""
    
    try:
        summary = get_summary_statistics(DB_PATH, experiment_name)
        
        if summary:
            return jsonify({
                'success': True,
                'summary': summary
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No summary statistics available'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/experiments/<experiment_name>/evaluation', methods=['GET'])
def evaluation_results(experiment_name):
    """Get evaluation results."""
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get D3QN results
            cursor.execute('''
                SELECT * FROM evaluation_results
                WHERE experiment_name = ? AND agent_type = 'd3qn'
                ORDER BY episode_number
            ''', (experiment_name,))
            d3qn_episodes = [dict(row) for row in cursor.fetchall()]
            
            # Get Fixed-Time results
            cursor.execute('''
                SELECT * FROM evaluation_results
                WHERE experiment_name = ? AND agent_type = 'fixed_time'
                ORDER BY episode_number
            ''', (experiment_name,))
            fixed_episodes = [dict(row) for row in cursor.fetchall()]
            
            return jsonify({
                'success': True,
                'd3qn': d3qn_episodes,
                'fixed_time': fixed_episodes
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/experiments/<experiment_name>/latest', methods=['GET'])
def latest_episode(experiment_name):
    """Get latest episode for real-time monitoring."""
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM episodes
                WHERE experiment_name = ?
                ORDER BY episode_number DESC
                LIMIT 1
            ''', (experiment_name,))
            
            episode = cursor.fetchone()
            
            if episode:
                result = dict(episode)
                
                # Get vehicle breakdown
                cursor.execute('''
                    SELECT * FROM vehicle_breakdown
                    WHERE experiment_name = ? AND episode_number = ?
                ''', (experiment_name, result['episode_number']))
                
                breakdown = cursor.fetchone()
                if breakdown:
                    result['vehicle_breakdown'] = dict(breakdown)
                
                return jsonify({
                    'success': True,
                    'latest': result
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No episodes found'
                }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    
    return jsonify({
        'success': True,
        'status': 'healthy',
        'database': os.path.exists(DB_PATH)
    })


if __name__ == '__main__':
    print("="*80)
    print("DASHBOARD API SERVER")
    print("="*80)
    print(f"Database: {DB_PATH}")
    print(f"Server: http://localhost:5000")
    print("="*80)
    print("\nAvailable Endpoints:")
    print("  GET  /api/experiments")
    print("  GET  /api/experiments/<name>/status")
    print("  GET  /api/experiments/<name>/episodes")
    print("  GET  /api/experiments/<name>/episodes/<number>")
    print("  GET  /api/experiments/<name>/progress")
    print("  GET  /api/experiments/<name>/vehicles")
    print("  GET  /api/experiments/<name>/summary")
    print("  GET  /api/experiments/<name>/evaluation")
    print("  GET  /api/experiments/<name>/latest")
    print("  GET  /health")
    print("="*80)
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)



