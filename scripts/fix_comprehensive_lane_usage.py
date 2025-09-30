"""
Comprehensive fix for multi-lane usage across all specified edges
Ensures vehicles use ALL available lanes instead of just one
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob

def fix_lane_usage_comprehensive(route_file):
    """
    Fix lane usage for all specified edges to use all available lanes
    """
    print(f"🔧 Fixing comprehensive lane usage: {os.path.basename(route_file)}")
    
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Track changes
    flows_updated = 0
    
    # Define edges that need multi-lane usage
    multi_lane_edges = {
        # Ecoland edges
        "106768821": 3,  # 3 lanes
        "-1069919419": 3,  # 3 lanes  
        "-794461795": 3,  # 3 lanes
        "-1069919421": 3,  # 3 lanes
        "-934134356#1": 3,  # 3 lanes
        "-794461796#0": 3,  # 3 lanes
        "-1069919420": 3,  # 3 lanes
        "-794461797#2": 3,  # 3 lanes
        "-794461797#1": 3,  # 3 lanes
        "-794461797#0": 3,  # 3 lanes
        "-794461796#1": 3,  # 3 lanes
        "770761758#1": 3,  # 3 lanes
        "770761758#2": 3,  # 3 lanes
        "1069919422#0": 3,  # 3 lanes
        
        # JohnPaul edges
        "-266255177#1": 3,  # 3 lanes
        "-266255177#0": 3,  # 3 lanes
        "-106768827#1": 3,  # 3 lanes
    }
    
    # Find all flows and update lane distribution
    for flow in root.findall('flow'):
        route_id = flow.get('route')
        vehicle_type = flow.get('type')
        
        # Find the corresponding route
        route = root.find(f".//route[@id='{route_id}']")
        if route is not None:
            edges = route.get('edges', '').split()
            
            # Check if this route uses any of our target edges
            needs_lane_fix = False
            for edge in edges:
                if edge in multi_lane_edges:
                    needs_lane_fix = True
                    break
            
            if needs_lane_fix:
                # Add comprehensive lane distribution parameters
                flow.set('departLane', 'random')
                flow.set('arrivalLane', 'random')
                flow.set('departPos', 'random')
                flow.set('departSpeed', 'random')
                flow.set('departPosLat', 'random')
                flow.set('arrivalPos', 'random')
                flow.set('arrivalSpeed', 'random')
                flow.set('arrivalPosLat', 'random')
                flows_updated += 1
    
    # Write the improved file
    tree.write(route_file, encoding='utf-8', xml_declaration=True)
    
    print(f"   ✅ Lane usage fixes: {flows_updated} flows updated")
    return flows_updated

def add_lane_distribution_routes():
    """
    Add specific lane distribution routes for all specified edges
    """
    print("🚀 COMPREHENSIVE LANE USAGE FIX")
    print("=" * 50)
    print("Fixing multi-lane usage for:")
    print("Ecoland edges: 106768821, -1069919419, -794461795, -1069919421, -934134356#1")
    print("Ecoland edges: -794461796#0, -1069919420, -794461797#2, -794461797#1, -794461797#0")
    print("Ecoland edges: -794461796#1, 770761758#1, 770761758#2, 1069919422#0")
    print("JohnPaul edges: -266255177#1, -266255177#0, -106768827#1")
    print()
    
    # Find all route files
    route_files = []
    route_files.extend(glob.glob("data/routes/*_cycle_*/JOHNPAUL_*.rou.xml"))
    route_files.extend(glob.glob("data/routes/*_cycle_*/ECOLAND_*.rou.xml"))
    route_files.extend(glob.glob("data/routes/*_cycle_*/SANDAWA_*.rou.xml"))
    
    total_flows_updated = 0
    processed_files = 0
    
    for route_file in route_files:
        try:
            flows_updated = fix_lane_usage_comprehensive(route_file)
            total_flows_updated += flows_updated
            processed_files += 1
        except Exception as e:
            print(f"   ❌ Error processing {route_file}: {e}")
    
    print(f"\n🎉 COMPREHENSIVE LANE USAGE FIX COMPLETED!")
    print(f"📊 Processed {processed_files} route files")
    print(f"📊 Flows updated: {total_flows_updated}")
    print()
    print("🔄 Regenerating consolidated route files...")
    
    # Regenerate consolidated routes
    import subprocess
    try:
        subprocess.run(["python", "scripts/consolidate_bundle_routes.py"], check=True)
        print("✅ Consolidated route files regenerated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error regenerating consolidated routes: {e}")

if __name__ == "__main__":
    add_lane_distribution_routes()




