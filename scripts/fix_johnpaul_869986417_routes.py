"""
Fix specific JohnPaul routes from edge 869986417#1 to -935563495#7
Reduce traffic on this specific path
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob

def fix_johnpaul_869986417_routes(route_file):
    """
    Fix JohnPaul routes from 869986417#1 to -935563495#7:
    Reduce frequency of vehicles taking this specific path
    """
    print(f"🔧 Fixing JohnPaul 869986417#1 routes: {os.path.basename(route_file)}")
    
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Track changes
    frequencies_reduced = 0
    
    # Find flows that go through the specific path: 869986417#1 -> -935563495#7
    for flow in root.findall('flow'):
        route_id = flow.get('route')
        vehicle_type = flow.get('type')
        
        # Find the corresponding route
        route = root.find(f".//route[@id='{route_id}']")
        if route is not None:
            edges = route.get('edges', '')
            
            # Check if route goes from 869986417#1 to -935563495#7
            if '869986417#1' in edges and '-935563495#7' in edges:
                current_period = float(flow.get('period', 21.56))
                # Increase period by 100% to reduce frequency by half
                new_period = current_period * 2.0
                flow.set('period', str(new_period))
                frequencies_reduced += 1
                print(f"   📉 Reduced 869986417#1->-935563495#7: {vehicle_type} -> {route_id} (period: {current_period:.2f} -> {new_period:.2f})")
    
    # Write the improved file
    tree.write(route_file, encoding='utf-8', xml_declaration=True)
    
    print(f"   ✅ JohnPaul 869986417#1 fixes: {frequencies_reduced} frequencies reduced")
    return frequencies_reduced

def fix_all_johnpaul_869986417_routes():
    """
    Apply fixes to all JohnPaul route files for the specific 869986417#1 path
    """
    print("🚀 FIXING JOHNPAUL 869986417#1 ROUTES")
    print("=" * 50)
    print("Reducing traffic from 869986417#1 to -935563495#7")
    print("This will make the traffic distribution more realistic")
    print()
    
    # Find all JohnPaul route files
    johnpaul_files = glob.glob("data/routes/*_cycle_*/JOHNPAUL_*.rou.xml")
    
    total_frequencies_reduced = 0
    processed_files = 0
    
    for johnpaul_file in johnpaul_files:
        try:
            freq_reduced = fix_johnpaul_869986417_routes(johnpaul_file)
            total_frequencies_reduced += freq_reduced
            processed_files += 1
        except Exception as e:
            print(f"   ❌ Error processing {johnpaul_file}: {e}")
    
    print(f"\n🎉 JOHNPAUL 869986417#1 ROUTE FIXES COMPLETED!")
    print(f"📊 Processed {processed_files} JohnPaul files")
    print(f"📊 Frequencies reduced: {total_frequencies_reduced}")
    print()
    print("🔄 Now regenerating consolidated route files...")
    
    # Regenerate consolidated routes
    import subprocess
    try:
        subprocess.run(["python", "scripts/consolidate_bundle_routes.py"], check=True)
        print("✅ Consolidated route files regenerated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error regenerating consolidated routes: {e}")

if __name__ == "__main__":
    fix_all_johnpaul_869986417_routes()



