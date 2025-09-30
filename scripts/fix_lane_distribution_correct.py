"""
Correct fix for lane distribution - add lane change parameters to vType elements
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob

def fix_lane_distribution_correct(route_file):
    """
    Correct fix for lane distribution - add lane change parameters to vType elements
    """
    print(f"🔧 Correct lane distribution fix: {os.path.basename(route_file)}")
    
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Track changes
    vtypes_updated = 0
    flows_updated = 0
    
    # Fix vType elements with lane change parameters
    for vtype in root.findall('vType'):
        # Add lane change model parameters to vType
        vtype.set('lcStrategic', '1.0')    # Strategic lane changing
        vtype.set('lcCooperative', '1.0')  # Cooperative lane changing
        vtype.set('lcSpeedGain', '1.0')    # Speed gain lane changing
        vtype.set('lcKeepRight', '0.0')    # Disable keep right preference
        vtype.set('lcOvertakeRight', '1.0') # Allow overtaking on right
        vtype.set('lcPushy', '0.0')        # Disable pushy behavior
        vtype.set('lcImpatience', '0.0')   # Disable impatience
        vtypes_updated += 1
    
    # Fix flow elements with proper lane distribution
    for flow in root.findall('flow'):
        # Use proper lane distribution parameters for flows
        flow.set('departLane', 'random')  # Random lane selection
        flow.set('departSpeed', 'max')
        flow.set('departPos', 'random')   # Random position within lane
        flow.set('arrivalLane', 'current')
        flow.set('departPosLat', 'random')  # Random lateral position
        
        # Remove any invalid lane change parameters from flows
        for attr in ['lcStrategic', 'lcCooperative', 'lcSpeedGain', 'lcKeepRight', 'lcOvertakeRight', 'lcPushy', 'lcImpatience']:
            if attr in flow.attrib:
                del flow.attrib[attr]
        
        flows_updated += 1
    
    # Write the fixed file
    tree.write(route_file, encoding='utf-8', xml_declaration=True)
    
    print(f"   ✅ vTypes updated: {vtypes_updated}")
    print(f"   ✅ Flows updated: {flows_updated}")
    return vtypes_updated, flows_updated

def fix_all_route_files_correct():
    """
    Apply correct lane distribution fixes to all route files
    """
    print("🚀 CORRECT LANE DISTRIBUTION FIX")
    print("=" * 50)
    print("Adding lane change parameters to vType elements:")
    print("• lcStrategic: '1.0' - Strategic lane changing")
    print("• lcCooperative: '1.0' - Cooperative lane changing")
    print("• lcSpeedGain: '1.0' - Speed gain lane changing")
    print("• lcKeepRight: '0.0' - Disable keep right preference")
    print("• lcOvertakeRight: '1.0' - Allow overtaking on right")
    print("• lcPushy: '0.0' - Disable pushy behavior")
    print("• lcImpatience: '0.0' - Disable impatience")
    print()
    print("Fixing flow elements with proper lane distribution:")
    print("• departLane: 'random' - Random lane selection")
    print("• departPos: 'random' - Random position within lane")
    print("• departPosLat: 'random' - Random lateral position")
    print()
    
    # Find all route files
    route_files = []
    route_files.extend(glob.glob("data/routes/*_cycle_*/JOHNPAUL_*.rou.xml"))
    route_files.extend(glob.glob("data/routes/*_cycle_*/ECOLAND_*.rou.xml"))
    route_files.extend(glob.glob("data/routes/*_cycle_*/SANDAWA_*.rou.xml"))
    
    total_vtypes_updated = 0
    total_flows_updated = 0
    processed_files = 0
    
    for route_file in route_files:
        try:
            vtypes_updated, flows_updated = fix_lane_distribution_correct(route_file)
            total_vtypes_updated += vtypes_updated
            total_flows_updated += flows_updated
            processed_files += 1
        except Exception as e:
            print(f"   ❌ Error processing {route_file}: {e}")
    
    print(f"\n🎉 CORRECT LANE DISTRIBUTION FIX COMPLETED!")
    print(f"📊 Processed {processed_files} route files")
    print(f"📊 vTypes updated: {total_vtypes_updated}")
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
    fix_all_route_files_correct()




