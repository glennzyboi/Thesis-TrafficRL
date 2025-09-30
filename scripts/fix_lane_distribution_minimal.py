"""
Minimal fix for lane distribution - only add essential parameters without breaking existing functionality
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob

def fix_lane_distribution_minimal(route_file):
    """
    Minimal fix for lane distribution - only add essential parameters
    """
    print(f"🔧 Minimal lane distribution fix: {os.path.basename(route_file)}")
    
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Track changes
    flows_updated = 0
    
    # Find all flows and add only essential lane distribution parameters
    for flow in root.findall('flow'):
        # Only add parameters that don't exist and are safe
        if 'departLane' not in flow.attrib:
            flow.set('departLane', 'best')
        if 'departSpeed' not in flow.attrib:
            flow.set('departSpeed', 'max')
        if 'departPos' not in flow.attrib:
            flow.set('departPos', 'free')
        if 'arrivalLane' not in flow.attrib:
            flow.set('arrivalLane', 'current')
        if 'departPosLat' not in flow.attrib:
            flow.set('departPosLat', 'free')
        
        # Remove any problematic parameters that might cause issues
        if 'arrivalPos' in flow.attrib and flow.get('arrivalPos') == 'max':
            del flow.attrib['arrivalPos']
        if 'arrivalSpeed' in flow.attrib and flow.get('arrivalSpeed') == 'current':
            del flow.attrib['arrivalSpeed']
        if 'arrivalPosLat' in flow.attrib and flow.get('arrivalPosLat') == 'free':
            del flow.attrib['arrivalPosLat']
        
        flows_updated += 1
    
    # Write the fixed file
    tree.write(route_file, encoding='utf-8', xml_declaration=True)
    
    print(f"   ✅ Minimal lane distribution fixes: {flows_updated} flows updated")
    return flows_updated

def fix_all_route_files_minimal():
    """
    Apply minimal lane distribution fixes to all route files
    """
    print("🚀 MINIMAL LANE DISTRIBUTION FIX")
    print("=" * 50)
    print("Adding only essential lane distribution parameters:")
    print("• departLane: 'best' (if not present)")
    print("• departSpeed: 'max' (if not present)")
    print("• departPos: 'free' (if not present)")
    print("• arrivalLane: 'current' (if not present)")
    print("• departPosLat: 'free' (if not present)")
    print("• Removing problematic parameters")
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
            flows_updated = fix_lane_distribution_minimal(route_file)
            total_flows_updated += flows_updated
            processed_files += 1
        except Exception as e:
            print(f"   ❌ Error processing {route_file}: {e}")
    
    print(f"\n🎉 MINIMAL LANE DISTRIBUTION FIX COMPLETED!")
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
    fix_all_route_files_minimal()




