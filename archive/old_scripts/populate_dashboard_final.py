#!/usr/bin/env python3
"""
Final Dashboard Population Script
Populates Supabase with complete validation data including intersection metrics
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def load_validation_data():
    """Load the complete validation data with intersection metrics"""
    print("Loading validation data...")
    
    with open('validation_with_intersection_data_complete.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {data['metadata']['total_scenarios']} scenarios")
    return data

def load_training_data():
    """Load training data from production logs"""
    print("Loading training data...")
    
    training_episodes = []
    with open('production_logs/final_thesis_training_350ep_episodes.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                episode = json.loads(line)
                training_episodes.append(episode)
    
    print(f"Loaded {len(training_episodes)} training episodes")
    return training_episodes

def load_vehicle_breakdown():
    """Load vehicle breakdown data"""
    print("Loading vehicle breakdown data...")
    
    with open('vehicle_breakdown_from_routes.json', 'r') as f:
        vehicle_breakdown = json.load(f)
    
    print(f"Loaded vehicle breakdown for {len(vehicle_breakdown)} episodes")
    return vehicle_breakdown

def calculate_research_objectives(validation_data):
    """Calculate research objective improvements"""
    print("Calculating research objectives...")
    
    scenarios = validation_data['scenarios']
    
    # Calculate averages across all scenarios
    fixed_metrics = []
    d3qn_metrics = []
    
    for scenario in scenarios:
        fixed_metrics.append(scenario['fixed_time'])
        d3qn_metrics.append(scenario['d3qn'])
    
    # Calculate improvements
    # FIXED: Calculate passenger throughput as rate per hour (like vehicle throughput)
    # Convert cumulative passenger throughput to rate per hour
    simulation_duration_hours = (300 - 30) / 3600  # 270 seconds = 0.075 hours
    
    fixed_avg_passenger = np.mean([m['passenger_throughput'] / simulation_duration_hours for m in fixed_metrics])
    d3qn_avg_passenger = np.mean([m['passenger_throughput'] / simulation_duration_hours for m in d3qn_metrics])
    passenger_improvement = ((d3qn_avg_passenger - fixed_avg_passenger) / fixed_avg_passenger) * 100
    
    fixed_avg_pt = np.mean([m.get('pt_passenger_throughput', 0) for m in fixed_metrics])
    d3qn_avg_pt = np.mean([m.get('pt_passenger_throughput', 0) for m in d3qn_metrics])
    pt_improvement = ((d3qn_avg_pt - fixed_avg_pt) / max(fixed_avg_pt, 1)) * 100 if fixed_avg_pt > 0 else 0
    
    fixed_avg_waiting = np.mean([m['avg_waiting_time'] for m in fixed_metrics])
    d3qn_avg_waiting = np.mean([m['avg_waiting_time'] for m in d3qn_metrics])
    waiting_improvement = ((fixed_avg_waiting - d3qn_avg_waiting) / fixed_avg_waiting) * 100
    
    fixed_avg_throughput = np.mean([m['avg_throughput'] for m in fixed_metrics])
    d3qn_avg_throughput = np.mean([m['avg_throughput'] for m in d3qn_metrics])
    throughput_improvement = ((d3qn_avg_throughput - fixed_avg_throughput) / fixed_avg_throughput) * 100
    
    objectives = {
        'passenger_throughput': {
            'improvement': passenger_improvement,
            'target': 10.0,
            'achieved': passenger_improvement >= 10.0
        },
        'public_vehicle_throughput': {
            'improvement': pt_improvement,
            'target': 15.0,
            'achieved': pt_improvement >= 15.0
        },
        'waiting_time': {
            'improvement': waiting_improvement,
            'target': 10.0,
            'achieved': waiting_improvement >= 10.0
        },
        'overall_vehicle_throughput': {
            'improvement': throughput_improvement,
            'target': 12.0,
            'achieved': throughput_improvement >= 12.0
        }
    }
    
    print(f"Research Objectives Calculated:")
    print(f"   Passenger Throughput: {passenger_improvement:.1f}% (target: 10%)")
    print(f"   Public Vehicle Throughput: {pt_improvement:.1f}% (target: 15%)")
    print(f"   Waiting Time: {waiting_improvement:.1f}% (target: 10%)")
    print(f"   Overall Vehicle Throughput: {throughput_improvement:.1f}% (target: 12%)")
    
    return objectives

def populate_supabase():
    """Populate Supabase with all dashboard data"""
    print("=" * 80)
    print("POPULATING DASHBOARD WITH COMPLETE DATA")
    print("=" * 80)
    
    # Initialize Supabase client
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("DATABASE_URL not found in environment variables")
        return False
    
    supabase: Client = create_client(DATABASE_URL, os.getenv('SUPABASE_KEY', ''))
    
    try:
        # Test connection
        print("Testing Supabase connection...")
        supabase.table('experiments').select('*').limit(1).execute()
        print("Connected to Supabase")
    except Exception as e:
        print(f"Failed to connect to Supabase: {e}")
        return False
    
    # Load all data
    validation_data = load_validation_data()
    training_episodes = load_training_data()
    vehicle_breakdown = load_vehicle_breakdown()
    
    # Calculate research objectives
    objectives = calculate_research_objectives(validation_data)
    
    # 1. Insert Experiment Metadata
    print("\n1. Inserting experiment metadata...")
    experiment_data = {
        'experiment_id': 'final_thesis_training_350ep',
        'experiment_name': 'D3QN Multi-Agent Traffic Signal Control',
        'description': 'Final thesis training with LSTM traffic prediction',
        'total_episodes': len(training_episodes),
        'validation_scenarios': validation_data['metadata']['total_scenarios'],
        'created_at': datetime.now().isoformat(),
        'status': 'completed'
    }
    
    try:
        supabase.table('experiments').upsert(experiment_data).execute()
        print("Experiment metadata inserted")
    except Exception as e:
        print(f"Failed to insert experiment: {e}")
        return False
    
    # 2. Insert Intersections
    print("\n2. Inserting intersections...")
    intersections = [
        {
            'intersection_id': 'Ecoland_TrafficSignal',
            'name': 'Ecoland',
            'region': 'Davao City',
            'num_approaches': 4,
            'created_at': datetime.now().isoformat()
        },
        {
            'intersection_id': 'JohnPaul_TrafficSignal', 
            'name': 'John Paul',
            'region': 'Davao City',
            'num_approaches': 4,
            'created_at': datetime.now().isoformat()
        },
        {
            'intersection_id': 'Sandawa_TrafficSignal',
            'name': 'Sandawa', 
            'region': 'Davao City',
            'num_approaches': 4,
            'created_at': datetime.now().isoformat()
        }
    ]
    
    try:
        supabase.table('intersections').upsert(intersections).execute()
        print("Intersections inserted")
    except Exception as e:
        print(f"Failed to insert intersections: {e}")
        return False
    
    # 3. Insert Training Episodes
    print("\n3. Inserting training episodes...")
    training_data = []
    for i, episode in enumerate(training_episodes):
        episode_data = {
            'episode_id': episode['episode_id'],
            'episode_number': episode['episode_number'],
            'experiment_id': 'final_thesis_training_350ep',
            'total_reward': episode['total_reward'],
            'vehicles_served': episode['vehicles_served'],
            'passenger_throughput': episode['passenger_throughput'],
            'avg_waiting_time': episode.get('avg_waiting_time', 0),
            'avg_speed': episode.get('avg_speed', 0),
            'avg_queue_length': episode.get('avg_queue_length', 0),
            'buses_processed': episode.get('buses_processed', 0),
            'jeepneys_processed': episode.get('jeepneys_processed', 0),
            'pt_passenger_throughput': episode.get('pt_passenger_throughput', 0),
            'created_at': episode['start_time']
        }
        training_data.append(episode_data)
    
    try:
        # Insert in batches
        batch_size = 100
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            supabase.table('training_episodes').upsert(batch).execute()
        print(f"{len(training_data)} training episodes inserted")
    except Exception as e:
        print(f"Failed to insert training episodes: {e}")
        return False
    
    # 4. Insert Validation Results
    print("\n4. Inserting validation results...")
    simulation_duration_hours = (300 - 30) / 3600  # 270 seconds = 0.075 hours
    validation_results = []
    for scenario in validation_data['scenarios']:
        scenario_data = {
            'scenario_id': f"scenario_{scenario['episode_num']}",
            'experiment_id': 'final_thesis_training_350ep',
            'scenario_name': scenario['scenario_info'].get('bundle_name', f"Scenario {scenario['episode_num']}"),
            'day': scenario['scenario_info'].get('day', 0),
            'cycle': scenario['scenario_info'].get('cycle', 0),
            'fixed_time_throughput': scenario['fixed_time']['avg_throughput'],
            'fixed_time_waiting': scenario['fixed_time']['avg_waiting_time'],
            'fixed_time_speed': scenario['fixed_time']['avg_speed'],
             'fixed_time_passenger_throughput': scenario['fixed_time']['passenger_throughput'] / simulation_duration_hours,
             'd3qn_throughput': scenario['d3qn']['avg_throughput'],
             'd3qn_waiting': scenario['d3qn']['avg_waiting_time'],
             'd3qn_speed': scenario['d3qn']['avg_speed'],
             'd3qn_passenger_throughput': scenario['d3qn']['passenger_throughput'] / simulation_duration_hours,
            'throughput_improvement': ((scenario['d3qn']['avg_throughput'] - scenario['fixed_time']['avg_throughput']) / scenario['fixed_time']['avg_throughput']) * 100,
            'waiting_improvement': ((scenario['fixed_time']['avg_waiting_time'] - scenario['d3qn']['avg_waiting_time']) / scenario['fixed_time']['avg_waiting_time']) * 100,
            'created_at': datetime.now().isoformat()
        }
        validation_results.append(scenario_data)
    
    try:
        supabase.table('evaluation_results').upsert(validation_results).execute()
        print(f"{len(validation_results)} validation results inserted")
    except Exception as e:
        print(f"Failed to insert validation results: {e}")
        return False
    
    # 5. Insert Intersection Performance
    print("\n5. Inserting intersection performance data...")
    intersection_performance = []
    for scenario in validation_data['scenarios']:
        for intersection_id in ['Ecoland_TrafficSignal', 'JohnPaul_TrafficSignal', 'Sandawa_TrafficSignal']:
            if intersection_id in scenario['fixed_time']['intersection_throughput']:
                fixed_data = scenario['fixed_time']['intersection_throughput'][intersection_id]
                d3qn_data = scenario['d3qn']['intersection_throughput'][intersection_id]
                
                perf_data = {
                    'performance_id': f"{scenario['episode_num']}_{intersection_id}",
                    'scenario_id': f"scenario_{scenario['episode_num']}",
                    'intersection_id': intersection_id,
                    'fixed_time_vehicles': fixed_data['total_vehicles'],
                    'fixed_time_queue': fixed_data['total_queue'],
                    'fixed_time_waiting': fixed_data['avg_waiting'],
                    'fixed_time_cars': fixed_data['vehicle_types'].get('car', 0),
                    'fixed_time_buses': fixed_data['vehicle_types'].get('bus', 0),
                    'fixed_time_jeepneys': fixed_data['vehicle_types'].get('jeepney', 0),
                    'fixed_time_motorcycles': fixed_data['vehicle_types'].get('motorcycle', 0),
                    'fixed_time_trucks': fixed_data['vehicle_types'].get('truck', 0),
                    'd3qn_vehicles': d3qn_data['total_vehicles'],
                    'd3qn_queue': d3qn_data['total_queue'],
                    'd3qn_waiting': d3qn_data['avg_waiting'],
                    'd3qn_cars': d3qn_data['vehicle_types'].get('car', 0),
                    'd3qn_buses': d3qn_data['vehicle_types'].get('bus', 0),
                    'd3qn_jeepneys': d3qn_data['vehicle_types'].get('jeepney', 0),
                    'd3qn_motorcycles': d3qn_data['vehicle_types'].get('motorcycle', 0),
                    'd3qn_trucks': d3qn_data['vehicle_types'].get('truck', 0),
                    'created_at': datetime.now().isoformat()
                }
                intersection_performance.append(perf_data)
    
    try:
        supabase.table('intersection_performance').upsert(intersection_performance).execute()
        print(f"{len(intersection_performance)} intersection performance records inserted")
    except Exception as e:
        print(f"Failed to insert intersection performance: {e}")
        return False
    
    # 6. Insert Vehicle Breakdown
    print("\n6. Inserting vehicle breakdown data...")
    vehicle_breakdown_data = []
    for i, breakdown in enumerate(vehicle_breakdown):
        breakdown_data = {
            'breakdown_id': f"episode_{i}",
            'episode_id': training_episodes[i]['episode_id'] if i < len(training_episodes) else f"episode_{i}",
            'cars': breakdown.get('cars', 0),
            'motorcycles': breakdown.get('motorcycles', 0),
            'trucks': breakdown.get('trucks', 0),
            'tricycles': breakdown.get('tricycles', 0),
            'jeepneys': breakdown.get('jeepneys', 0),
            'buses': breakdown.get('buses', 0),
            'modern_jeepneys': 0,
            'car_passengers': int(breakdown.get('cars', 0) * 1.3),
            'motorcycle_passengers': int(breakdown.get('motorcycles', 0) * 1.4),
            'truck_passengers': int(breakdown.get('trucks', 0) * 1.5),
            'tricycle_passengers': int(breakdown.get('tricycles', 0) * 2.5),
            'jeepney_passengers': int(breakdown.get('jeepneys', 0) * 14),
            'bus_passengers': int(breakdown.get('buses', 0) * 35),
            'modern_jeepney_passengers': 0,
            'created_at': datetime.now().isoformat()
        }
        vehicle_breakdown_data.append(breakdown_data)
    
    try:
        supabase.table('vehicle_breakdown').upsert(vehicle_breakdown_data).execute()
        print(f"{len(vehicle_breakdown_data)} vehicle breakdown records inserted")
    except Exception as e:
        print(f"Failed to insert vehicle breakdown: {e}")
        return False
    
    # 7. Insert Objective Metrics
    print("\n7. Inserting objective metrics...")
    objective_metrics = {
        'metric_id': 'research_objectives',
        'experiment_id': 'final_thesis_training_350ep',
        'passenger_throughput_improvement': objectives['passenger_throughput']['improvement'],
        'passenger_throughput_target': objectives['passenger_throughput']['target'],
        'passenger_throughput_achieved': objectives['passenger_throughput']['achieved'],
        'public_vehicle_throughput_improvement': objectives['public_vehicle_throughput']['improvement'],
        'public_vehicle_throughput_target': objectives['public_vehicle_throughput']['target'],
        'public_vehicle_throughput_achieved': objectives['public_vehicle_throughput']['achieved'],
        'waiting_time_improvement': objectives['waiting_time']['improvement'],
        'waiting_time_target': objectives['waiting_time']['target'],
        'waiting_time_achieved': objectives['waiting_time']['achieved'],
        'overall_vehicle_throughput_improvement': objectives['overall_vehicle_throughput']['improvement'],
        'overall_vehicle_throughput_target': objectives['overall_vehicle_throughput']['target'],
        'overall_vehicle_throughput_achieved': objectives['overall_vehicle_throughput']['achieved'],
        'created_at': datetime.now().isoformat()
    }
    
    try:
        supabase.table('objective_metrics').upsert(objective_metrics).execute()
        print("Objective metrics inserted")
    except Exception as e:
        print(f"Failed to insert objective metrics: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("DASHBOARD POPULATION COMPLETE!")
    print("=" * 80)
    print("All data has been successfully inserted into Supabase")
    print("Your dashboard at https://traffic-compare-17.vercel.app/ is now ready!")
    print("=" * 80)
    
    return True

def main():
    print("FINAL DASHBOARD POPULATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Dashboard URL: https://traffic-compare-17.vercel.app/")
    print("=" * 80)
    
    success = populate_supabase()
    
    if success:
        print("\nSUCCESS! Dashboard is ready with complete data!")
        print("All intersection-level data is now available")
        print("Visit: https://traffic-compare-17.vercel.app/")
    else:
        print("\nFAILED! Check the errors above")
        sys.exit(1)

if __name__ == "__main__":
    main()





