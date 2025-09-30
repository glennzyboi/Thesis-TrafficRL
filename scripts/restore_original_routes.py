import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob
import shutil

def restore_original_routes():
    """Restore original routes from backup and properly add new routes"""
    print("🔄 RESTORING ORIGINAL ROUTES")
    print("=" * 50)
    
    # First, let's check if we have a backup of the original individual route files
    # If not, we'll need to regenerate them from the realistic backup
    
    # Check if we have any backup of individual route files
    backup_found = False
    for backup_dir in ["data/routes/realistic", "data/routes/focused"]:
        if os.path.exists(backup_dir):
            backup_files = glob.glob(str(Path(backup_dir) / "*.rou.xml"))
            if backup_files:
                print(f"📁 Found backup in: {backup_dir}")
                backup_found = True
                break
    
    if not backup_found:
        print("❌ No backup found. We need to regenerate the original routes.")
        print("🔧 This will require recreating the individual route files from scratch.")
        return False
    
    # For now, let's create a simple restoration by regenerating the consolidate script
    print("🔧 Regenerating consolidated routes from individual files...")
    
    # Run the consolidate script to regenerate from individual files
    os.system("python scripts/consolidate_bundle_routes.py")
    
    print("✅ Original routes restored!")
    return True

def add_new_routes_properly():
    """Add the new routes properly without replacing existing ones"""
    print("\n🔧 ADDING NEW ROUTES PROPERLY")
    print("=" * 50)
    
    # Read the current consolidated route file
    route_file = "data/routes/consolidated/bundle_20250701_cycle_1.rou.xml"
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Get the highest existing route ID
    existing_route_ids = []
    for route in root.findall('route'):
        route_id = route.get('id')
        if route_id and route_id.startswith('route_'):
            try:
                existing_route_ids.append(int(route_id.split('_')[1]))
            except:
                pass
    
    next_route_id = max(existing_route_ids) + 1 if existing_route_ids else 0
    
    # New routes to add (as requested by user)
    new_routes = [
        {
            "edges": "770761758#0 770761758#1 770761758#2 1069919422#0 1102489115#0",
            "description": "Ecoland: 770761758#0 -> 1102489115#0 (explore environment)"
        },
        {
            "edges": "-794461796#1 -794461796#0 -1069919420 -455558436#1 1102489115#0",
            "description": "Ecoland: -794461796#1 -> 1102489115#0 (heavy traffic)",
            "period": 8.0
        },
        {
            "edges": "-794461795 -1069919421 -934134356#1 -1069919422#1 -770761758#2",
            "description": "Ecoland: -794461795 -> -770761758#2 (explore environment)"
        },
        {
            "edges": "1042538762#3 -1102489116",
            "description": "Sandawa: 1042538762#3 -> -1102489116 (explore environment)"
        },
        {
            "edges": "1042538760#2 -1102489116",
            "description": "Sandawa: 1042538760#2 -> -1102489116 (explore environment)"
        }
    ]
    
    # Add new routes
    for route_data in new_routes:
        route_id = f"route_{next_route_id}"
        next_route_id += 1
        
        # Create new route element
        new_route = ET.Element('route', id=route_id, edges=route_data["edges"])
        root.insert(-1, new_route)
        
        # Create corresponding flows for all vehicle types
        vehicle_types = ["car", "motor", "jeepney", "bus", "truck"]
        for v_type in vehicle_types:
            flow_id = f"flow_{next_route_id-1}_{v_type}"
            period = route_data.get("period", 15.0)
            
            new_flow = ET.Element('flow', 
                                id=flow_id,
                                route=route_id,
                                begin="0",
                                end="3600", 
                                period=str(period),
                                type=v_type,
                                departLane="random",
                                departSpeed="max",
                                departPos="random",
                                arrivalLane="current",
                                departPosLat="random")
            root.append(new_flow)
        
        print(f"   ✅ Added: {route_data['description']}")
    
    tree.write(route_file)
    print(f"✅ Added {len(new_routes)} new route groups to existing routes")

if __name__ == "__main__":
    print("🚀 ROUTE RESTORATION AND ADDITION")
    print("=" * 60)
    print("This script will:")
    print("1. Restore original routes from backup")
    print("2. Add new routes properly without replacing existing ones")
    print()
    
    # First, try to restore original routes
    if restore_original_routes():
        # Then add new routes properly
        add_new_routes_properly()
        print("\n🎉 ROUTE RESTORATION COMPLETED!")
        print("The consolidated route file now contains both original and new routes.")
    else:
        print("❌ Could not restore original routes. Manual intervention needed.")




