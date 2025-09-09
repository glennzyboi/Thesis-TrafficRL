"""
Fixed compile_bundles.py
Reads per-cycle Excel workbooks from data/raw/ and produces processed data.
Updated to handle the actual data structure with lane-level data.
"""

import os
import sys
import glob
import pandas as pd
import re
import json
from collections import defaultdict

# Configuration - use absolute paths relative to project root

# Get the project root directory (parent of scripts directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")  
SCEN_DIR = os.path.join(PROJECT_ROOT, "out", "scenarios")

# Vehicle type mapping (your data uses these names)
VEHICLE_TYPES = {
    'Car': 'car',
    'Motorcycle': 'motor', 
    'Jeepney': 'jeepney',
    'Bus': 'bus',
    'Truck': 'truck'
}

def normalize_intersection_name(intersection_id):
    """Extract intersection name from lane ID or intersection ID"""
    if '_' in intersection_id:
        return intersection_id.split('_')[0].upper()
    return intersection_id.upper()

def process_excel_file(filepath):
    """Process a single Excel file and return aggregated data per intersection"""
    print(f"[INFO] Processing {filepath}")
    
    try:
        # Try to read the Raw_Annotations sheet first
        xls = pd.ExcelFile(filepath)
        if 'Raw_Annotations' in xls.sheet_names:
            df = pd.read_excel(filepath, sheet_name='Raw_Annotations')
        elif 'Aggregates' in xls.sheet_names:
            df = pd.read_excel(filepath, sheet_name='Aggregates')
        else:
            # Use first sheet
            df = pd.read_excel(filepath, sheet_name=xls.sheet_names[0])
        
        # Extract metadata from filename
        filename = os.path.basename(filepath)
        parts = filename.replace('.xlsx', '').split('_')
        intersection = parts[0].upper()
        day = parts[1] if len(parts) > 1 else "unknown"
        cycle = parts[2].replace('cycle', '') if len(parts) > 2 else "1"
        
        # Group by VehicleType and sum counts
        vehicle_totals = {}
        cycle_time = df['CycleTime_s'].iloc[0] if 'CycleTime_s' in df.columns else 300
        
        # Initialize totals
        for vtype in VEHICLE_TYPES.keys():
            vehicle_totals[VEHICLE_TYPES[vtype]] = {
                'count': 0,
                'passenger_equivalent': 0.0,
                'passenger_throughput': 0.0
            }
        
        # Aggregate by vehicle type across all lanes
        for vtype in VEHICLE_TYPES.keys():
            vtype_data = df[df['VehicleType'] == vtype]
            if not vtype_data.empty:
                total_count = vtype_data['Count'].sum()
                total_passenger_eq = vtype_data['PassengerEquivalent'].sum() if 'PassengerEquivalent' in df.columns else 0
                total_passenger_throughput = vtype_data['Pass throughput per hr'].sum() if 'Pass throughput per hr' in df.columns else 0
                
                vehicle_totals[VEHICLE_TYPES[vtype]] = {
                    'count': int(total_count),
                    'passenger_equivalent': float(total_passenger_eq),
                    'passenger_throughput': float(total_passenger_throughput)
                }
        
        # Calculate totals
        total_vehicles = sum(v['count'] for v in vehicle_totals.values())
        total_passenger_throughput = sum(v['passenger_throughput'] for v in vehicle_totals.values())
        
        result = {
            'IntersectionID': intersection,
            'Day': day,
            'CycleNum': cycle,
            'CycleTime_s': int(cycle_time),
            'TotalVehicles': total_vehicles,
            'TotalPassengerThroughput': total_passenger_throughput
        }
        
        # Add vehicle-specific data
        for vtype, data in vehicle_totals.items():
            result[f'{vtype}_count'] = data['count']
            result[f'{vtype}_passenger_equivalent'] = data['passenger_equivalent']
            result[f'{vtype}_passenger_throughput'] = data['passenger_throughput']
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to process {filepath}: {e}")
        return None

def compile_bundles():
    """Main function to compile all bundles"""
    
    # Create output directories
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SCEN_DIR, exist_ok=True)
    
    # Find all Excel files
    excel_files = glob.glob(os.path.join(RAW_DIR, "*.xlsx"))
    if not excel_files:
        print(f"[ERROR] No Excel files found in {RAW_DIR}")
        return
    
    print(f"[INFO] Found {len(excel_files)} Excel files")
    
    # Process each file
    all_data = []
    intersections_found = set()
    
    for filepath in excel_files:
        result = process_excel_file(filepath)
        if result:
            all_data.append(result)
            intersections_found.add(result['IntersectionID'])
    
    if not all_data:
        print("[ERROR] No data processed successfully")
        return
    
    print(f"[INFO] Found intersections: {sorted(intersections_found)}")
    
    # Create master bundles DataFrame
    master_df = pd.DataFrame(all_data)
    master_path = os.path.join(OUT_DIR, "master_bundles.csv")
    master_df.to_csv(master_path, index=False)
    print(f"[INFO] Saved master bundles to {master_path}")
    
    # Create scenario folders and individual intersection files
    scenarios_index = []
    
    # Group by Day and CycleNum
    for (day, cycle), group in master_df.groupby(['Day', 'CycleNum']):
        # Create scenario folder
        scenario_dir = os.path.join(SCEN_DIR, str(day), f"cycle_{cycle}")
        os.makedirs(scenario_dir, exist_ok=True)
        
        intersections_in_cycle = []
        
        # Save individual intersection files
        for _, row in group.iterrows():
            intersection = row['IntersectionID']
            intersections_in_cycle.append(intersection)
            
            # Create minimal CSV for this intersection
            minimal_data = {
                'IntersectionID': intersection,
                'Day': day,
                'CycleNum': cycle,
                'CycleTime_s': row['CycleTime_s'],
                'TotalVehicles': row['TotalVehicles']
            }
            
            # Add vehicle counts
            for vtype in VEHICLE_TYPES.values():
                minimal_data[f'{vtype}_count'] = row[f'{vtype}_count']
                minimal_data[f'{vtype}_passenger_equivalent'] = row[f'{vtype}_passenger_equivalent']
            
            # Save to CSV
            intersection_file = os.path.join(scenario_dir, f"{intersection}_cycle{cycle}.csv")
            pd.DataFrame([minimal_data]).to_csv(intersection_file, index=False)
        
        # Add to scenarios index
        scenarios_index.append({
            'Day': day,
            'CycleNum': cycle,
            'Intersections': ','.join(intersections_in_cycle),
            'ScenarioPath': os.path.relpath(scenario_dir, OUT_DIR)
        })
        
        # Create bundle metadata
        bundle_meta = pd.DataFrame([{
            'Day': day,
            'CycleNum': cycle,
            'Intersections': ','.join(intersections_in_cycle)
        }])
        bundle_meta_path = os.path.join(os.path.dirname(scenario_dir), 'bundle_meta.csv')
        bundle_meta.to_csv(bundle_meta_path, index=False)
    
    # Save scenarios index
    scenarios_df = pd.DataFrame(scenarios_index)
    scenarios_path = os.path.join(OUT_DIR, "scenarios_index.csv")
    scenarios_df.to_csv(scenarios_path, index=False)
    print(f"[INFO] Saved scenarios index to {scenarios_path}")
    
    print("[SUCCESS] Bundle compilation completed!")
    print(f"[INFO] Check {OUT_DIR} and {SCEN_DIR} for outputs")

if __name__ == "__main__":
    compile_bundles()
