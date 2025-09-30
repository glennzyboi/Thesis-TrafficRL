"""
Effective fix for lane distribution - use SUMO's lane change model and positioning
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob

def fix_lane_distribution_effective(route_file):
    """
    Effective fix for lane distribution using SUMO's proper lane change model
    """
    print(f"🔧 Effective lane distribution fix: {os.path.basename(route_file)}")
    
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Track changes
    flows_updated = 0
    
    # Find all flows and add effective lane distribution parameters
    for flow in root.findall('flow'):
        # Use SUMO's lane change model parameters for better lane distribution
        flow.set('departLane', 'random')  # Random lane selection
        flow.set('departSpeed', 'max')
        flow.set('departPos', 'random')   # Random position within lane
        flow.set('arrivalLane', 'current')
        flow.set('departPosLat', 'random')  # Random lateral position
        
        # Add lane change model parameters
        flow.set('lcStrategic', '1.0')    # Strategic lane changing
        flow.set('lcCooperative', '1.0')  # Cooperative lane changing
        flow.set('lcSpeedGain', '1.0')    # Speed gain lane changing
        flow.set('lcKeepRight', '0.0')    # Disable keep right preference
        flow.set('lcOvertakeRight', '1.0') # Allow overtaking on right
        
        flows_updated += 1
    
    # Write the fixed file
    tree.write(route_file, encoding='utf-8', xml_declaration=True)
    
    print(f"   ✅ Effective lane distribution fixes: {flows_updated} flows updated")
    return flows_updated

def fix_all_route_files_effective():
    """
    Apply effective lane distribution fixes to all route files
    """
    print("🚀 EFFECTIVE LANE DISTRIBUTION FIX")
    print("=" * 50)
    print("Using SUMO's lane change model for better lane distribution:")
    print("• departLane: 'random' - Random lane selection")
    print("• departPos: 'random' - Random position within lane")
    print("• departPosLat: 'random' - Random lateral position")
    print("• lcStrategic: '1.0' - Strategic lane changing")
    print("• lcCooperative: '1.0' - Cooperative lane changing")
    print("• lcSpeedGain: '1.0' - Speed gain lane changing")
    print("• lcKeepRight: '0.0' - Disable keep right preference")
    print("• lcOvertakeRight: '1.0' - Allow overtaking on right")
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
            flows_updated = fix_lane_distribution_effective(route_file)
            total_flows_updated += flows_updated
            processed_files += 1
        except Exception as e:
            print(f"   ❌ Error processing {route_file}: {e}")
    
    print(f"\n🎉 EFFECTIVE LANE DISTRIBUTION FIX COMPLETED!")
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
    fix_all_route_files_effective()




