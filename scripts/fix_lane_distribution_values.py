"""
Fix lane distribution values to use proper SUMO syntax
Replace 'random' with valid SUMO lane distribution values
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob

def fix_lane_distribution_values(route_file):
    """
    Fix lane distribution values to use proper SUMO syntax
    """
    print(f"🔧 Fixing lane distribution values: {os.path.basename(route_file)}")
    
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Track changes
    flows_updated = 0
    
    # Find all flows and fix lane distribution parameters
    for flow in root.findall('flow'):
        # Fix lane distribution parameters with valid SUMO values
        if flow.get('departLane') == 'random':
            flow.set('departLane', 'best')
        if flow.get('arrivalLane') == 'random':
            flow.set('arrivalLane', 'current')
        if flow.get('departPos') == 'random':
            flow.set('departPos', 'free')
        if flow.get('departSpeed') == 'random':
            flow.set('departSpeed', 'max')
        if flow.get('departPosLat') == 'random':
            flow.set('departPosLat', 'free')
        if flow.get('arrivalPos') == 'random':
            flow.set('arrivalPos', 'max')
        if flow.get('arrivalSpeed') == 'random':
            flow.set('arrivalSpeed', 'current')
        if flow.get('arrivalPosLat') == 'random':
            flow.set('arrivalPosLat', 'free')
        
        flows_updated += 1
    
    # Write the fixed file
    tree.write(route_file, encoding='utf-8', xml_declaration=True)
    
    print(f"   ✅ Lane distribution fixes: {flows_updated} flows updated")
    return flows_updated

def fix_all_route_files():
    """
    Fix lane distribution values in all route files
    """
    print("🚀 FIXING LANE DISTRIBUTION VALUES")
    print("=" * 50)
    print("Replacing 'random' with valid SUMO lane distribution values:")
    print("• departLane: 'random' → 'best'")
    print("• arrivalLane: 'random' → 'current'")
    print("• departPos: 'random' → 'free'")
    print("• departSpeed: 'random' → 'max'")
    print("• departPosLat: 'random' → 'free'")
    print("• arrivalPos: 'random' → 'max'")
    print("• arrivalSpeed: 'random' → 'current'")
    print("• arrivalPosLat: 'random' → 'free'")
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
            flows_updated = fix_lane_distribution_values(route_file)
            total_flows_updated += flows_updated
            processed_files += 1
        except Exception as e:
            print(f"   ❌ Error processing {route_file}: {e}")
    
    print(f"\n🎉 LANE DISTRIBUTION VALUES FIXED!")
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
    fix_all_route_files()




