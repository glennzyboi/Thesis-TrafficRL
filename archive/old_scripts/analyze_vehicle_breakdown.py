#!/usr/bin/env python3
"""
Analyze vehicle type breakdown and verify passenger throughput calculations
"""

import json
import pandas as pd

def analyze_vehicle_breakdown():
    """Analyze vehicle type breakdown from validation data"""
    print("Analyzing vehicle type breakdown and passenger throughput calculations...")
    
    # Load validation data
    with open('comparison_results/validation_dashboard_series.json', 'r') as f:
        data = json.load(f)
    
    # Passenger capacity mapping (from D3QN environment)
    passenger_capacities = {
        'car': 1.3,
        'motorcycle': 1.4,
        'jeepney': 14.0,
        'bus': 35.0,
        'truck': 1.5,
        'tricycle': 2.5
    }
    
    print("\n" + "="*80)
    print("VEHICLE TYPE BREAKDOWN ANALYSIS")
    print("="*80)
    
    for episode in data:
        episode_num = episode['episode']
        scenario = episode['scenario']
        
        print(f"\nEPISODE {episode_num}: {scenario}")
        print("-" * 60)
        
        # Analyze Fixed-Time
        print("\nFIXED-TIME BREAKDOWN:")
        fixed_data = episode['fixed_time']
        fixed_intersections = fixed_data['intersections']
        
        # Aggregate vehicle types across all intersections
        fixed_total_vehicles = 0
        fixed_vehicle_types = {'car': 0, 'bus': 0, 'jeepney': 0, 'motorcycle': 0, 'truck': 0, 'tricycle': 0}
        fixed_passenger_calc = 0
        
        for intersection, data in fixed_intersections.items():
            print(f"  {intersection}:")
            print(f"    Total vehicles: {data['total_vehicles']}")
            print(f"    Vehicle types: {data['vehicle_types']}")
            
            fixed_total_vehicles += data['total_vehicles']
            for vtype, count in data['vehicle_types'].items():
                if vtype in fixed_vehicle_types:
                    fixed_vehicle_types[vtype] += count
                    # Calculate passenger contribution
                    passenger_contribution = count * passenger_capacities.get(vtype, 1.5)
                    fixed_passenger_calc += passenger_contribution
                    print(f"      {vtype}: {count} vehicles × {passenger_capacities.get(vtype, 1.5)} = {passenger_contribution:.1f} passengers")
        
        print(f"\n  FIXED-TIME TOTALS:")
        print(f"    Total vehicles: {fixed_total_vehicles}")
        print(f"    Vehicle type breakdown: {fixed_vehicle_types}")
        print(f"    Calculated passengers: {fixed_passenger_calc:.1f}")
        print(f"    Logged passengers: {fixed_data['passenger_throughput']}")
        print(f"    Match: {'YES' if abs(fixed_passenger_calc - fixed_data['passenger_throughput']) < 1 else 'NO'}")
        
        # Analyze D3QN
        print(f"\nD3QN BREAKDOWN:")
        d3qn_data = episode['d3qn']
        d3qn_intersections = d3qn_data['intersections']
        
        # Aggregate vehicle types across all intersections
        d3qn_total_vehicles = 0
        d3qn_vehicle_types = {'car': 0, 'bus': 0, 'jeepney': 0, 'motorcycle': 0, 'truck': 0, 'tricycle': 0}
        d3qn_passenger_calc = 0
        
        for intersection, data in d3qn_intersections.items():
            print(f"  {intersection}:")
            print(f"    Total vehicles: {data['total_vehicles']}")
            print(f"    Vehicle types: {data['vehicle_types']}")
            
            d3qn_total_vehicles += data['total_vehicles']
            for vtype, count in data['vehicle_types'].items():
                if vtype in d3qn_vehicle_types:
                    d3qn_vehicle_types[vtype] += count
                    # Calculate passenger contribution
                    passenger_contribution = count * passenger_capacities.get(vtype, 1.5)
                    d3qn_passenger_calc += passenger_contribution
                    print(f"      {vtype}: {count} vehicles × {passenger_capacities.get(vtype, 1.5)} = {passenger_contribution:.1f} passengers")
        
        print(f"\n  D3QN TOTALS:")
        print(f"    Total vehicles: {d3qn_total_vehicles}")
        print(f"    Vehicle type breakdown: {d3qn_vehicle_types}")
        print(f"    Calculated passengers: {d3qn_passenger_calc:.1f}")
        print(f"    Logged passengers: {d3qn_data['passenger_throughput']}")
        print(f"    Match: {'YES' if abs(d3qn_passenger_calc - d3qn_data['passenger_throughput']) < 1 else 'NO'}")
        
        # Performance comparison
        print(f"\nPERFORMANCE COMPARISON:")
        print(f"    Vehicle throughput improvement: {((d3qn_total_vehicles - fixed_total_vehicles) / fixed_total_vehicles * 100):+.1f}%")
        print(f"    Passenger throughput improvement: {((d3qn_data['passenger_throughput'] - fixed_data['passenger_throughput']) / fixed_data['passenger_throughput'] * 100):+.1f}%")
        
        # Check if passenger throughput matches vehicle breakdown
        print(f"\nACCURACY CHECK:")
        print(f"    Fixed-Time: Calculated {fixed_passenger_calc:.1f} vs Logged {fixed_data['passenger_throughput']} = {'ACCURATE' if abs(fixed_passenger_calc - fixed_data['passenger_throughput']) < 1 else 'INACCURATE'}")
        print(f"    D3QN: Calculated {d3qn_passenger_calc:.1f} vs Logged {d3qn_data['passenger_throughput']} = {'ACCURATE' if abs(d3qn_passenger_calc - d3qn_data['passenger_throughput']) < 1 else 'INACCURATE'}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("This analysis shows:")
    print("1. Vehicle type breakdown per intersection")
    print("2. Passenger capacity calculations per vehicle type")
    print("3. Verification of logged passenger throughput values")
    print("4. Performance improvements (vehicle vs passenger throughput)")

if __name__ == "__main__":
    analyze_vehicle_breakdown()










