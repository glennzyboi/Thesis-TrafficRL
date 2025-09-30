import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob

def fix_ecoland_routes(route_file):
    """Fix Ecoland intersection routes to ensure proper connections to all target edges"""
    print(f"🔧 Fixing Ecoland route connections: {os.path.basename(route_file)}")
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    routes_modified = 0
    
    # Define proper route mappings for Ecoland
    ecoland_routes = {
        # From -794461796#0, -794461796#1, -794461797#0, -794461797#1, -794461797#2
        # Should go to -1069919420, -1069919422#1, 1102489115#0
        "from_794461796_797": [
            "-794461797#2 -794461797#1 -794461797#0 -794461796#1 -794461796#0 -1069919420",
            "-794461797#2 -794461797#1 -794461797#0 -794461796#1 -794461796#0 -1069919422#1",
            "-794461797#2 -794461797#1 -794461797#0 -794461796#1 -794461796#0 1102489115#0"
        ],
        
        # From -934134356#1, -1069919421, -794461795, -1069919419, 106768821
        # Should go to -1069919422#1, 1102489115#0
        "from_934134356_1069919421": [
            "-934134356#1 -1069919422#1",
            "-934134356#1 1102489115#0",
            "-1069919421 -1069919422#1", 
            "-1069919421 1102489115#0",
            "-794461795 -1069919422#1",
            "-794461795 1102489115#0",
            "-1069919419 -1069919422#1",
            "-1069919419 1102489115#0",
            "106768821 -1069919422#1",
            "106768821 1102489115#0"
        ],
        
        # From 1069919422#0, 770761758#2, 770761758#1, 770761758#0
        # Should go to 1102489115#0 (and reduce traffic to 455558436#0)
        "from_1069919422_770761758": [
            "1069919422#0 1102489115#0",
            "770761758#0 770761758#1 770761758#2 1102489115#0",
            "770761758#0 770761758#1 770761758#2 1069919422#0 1102489115#0"
        ]
    }
    
    # Add new routes for Ecoland
    route_id_counter = max([int(route.get('id').split('_')[1]) for route in root.findall('route')]) + 1
    
    for route_group, routes in ecoland_routes.items():
        for route_edges in routes:
            route_id = f"route_{route_id_counter}"
            route_id_counter += 1
            
            # Create new route element
            new_route = ET.Element('route', id=route_id, edges=route_edges)
            root.insert(-1, new_route)  # Insert before flows
            
            # Create corresponding flow
            flow_id = f"flow_{route_id_counter-1}"
            period = 15.0 if "1102489115#0" in route_edges else 12.0  # Higher frequency for main exits
            
            new_flow = ET.Element('flow', 
                                id=flow_id,
                                route=route_id,
                                begin="0",
                                end="3600", 
                                period=str(period),
                                type="car",
                                departLane="random",
                                departSpeed="max",
                                departPos="random",
                                arrivalLane="current",
                                departPosLat="random")
            root.append(new_flow)
            routes_modified += 1
    
    # Reduce traffic to 455558436#0 (should be rare)
    for flow in root.findall('flow'):
        route_id = flow.get('route')
        route = root.find(f".//route[@id='{route_id}']")
        if route is not None and "455558436#0" in route.get('edges'):
            period = float(flow.get('period'))
            flow.set('period', str(period * 3))  # Reduce frequency by 3x
    
    tree.write(route_file)
    print(f"   ✅ Added {routes_modified} new Ecoland routes")

def fix_johnpaul_routes(route_file):
    """Fix JohnPaul intersection routes to ensure proper connections to 106768827#0"""
    print(f"🔧 Fixing JohnPaul route connections: {os.path.basename(route_file)}")
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    routes_modified = 0
    
    # Define proper route mappings for JohnPaul
    johnpaul_routes = [
        # From 1069919425#1, 1069919425#0 should go to 106768827#0
        "1069919425#1 106768827#0",
        "1069919425#0 106768827#0",
        
        # From 1046997838#3, 1046997838#2, 1046997838#1, 1046997838#0 should go to 106768827#0
        "1046997838#3 1046997838#2 1046997838#1 1046997838#0 106768827#0",
        "1046997838#2 1046997838#1 1046997838#0 106768827#0",
        "1046997838#1 1046997838#0 106768827#0",
        "1046997838#0 106768827#0",
        
        # From 1046997839#7, 1046997839#6 should go to 106768827#0
        "1046997839#7 1046997839#6 106768827#0",
        "1046997839#6 106768827#0"
    ]
    
    # Add new routes for JohnPaul
    route_id_counter = max([int(route.get('id').split('_')[1]) for route in root.findall('route')]) + 1
    
    for route_edges in johnpaul_routes:
        route_id = f"route_{route_id_counter}"
        route_id_counter += 1
        
        # Create new route element
        new_route = ET.Element('route', id=route_id, edges=route_edges)
        root.insert(-1, new_route)  # Insert before flows
        
        # Create corresponding flow
        flow_id = f"flow_{route_id_counter-1}"
        period = 10.0  # High frequency for main exit
        
        new_flow = ET.Element('flow', 
                            id=flow_id,
                            route=route_id,
                            begin="0",
                            end="3600", 
                            period=str(period),
                            type="car",
                            departLane="random",
                            departSpeed="max",
                            departPos="random",
                            arrivalLane="current",
                            departPosLat="random")
        root.append(new_flow)
        routes_modified += 1
    
    tree.write(route_file)
    print(f"   ✅ Added {routes_modified} new JohnPaul routes")

def fix_comprehensive_route_connections():
    """Fix all route files to ensure proper connections to target edges"""
    print("🚀 FIXING COMPREHENSIVE ROUTE CONNECTIONS")
    print("=" * 60)
    print("Fixing route connections to ensure vehicles reach all target edges:")
    print("• Ecoland: -1069919420, -1069919422#1, 1102489115#0")
    print("• JohnPaul: 106768827#0")
    print("• Reducing traffic to 455558436#0 (should be rare)")
    print()
    
    # Process all individual route files
    route_dirs = glob.glob(str(Path(__file__).parent.parent / "data" / "routes" / "2025*"))
    
    for route_dir in route_dirs:
        print(f"📁 Processing directory: {os.path.basename(route_dir)}")
        
        # Process Ecoland files
        ecoland_files = glob.glob(str(Path(route_dir) / "ECOLAND_*.rou.xml"))
        for ecoland_file in ecoland_files:
            fix_ecoland_routes(ecoland_file)
        
        # Process JohnPaul files  
        johnpaul_files = glob.glob(str(Path(route_dir) / "JOHNPAUL_*.rou.xml"))
        for johnpaul_file in johnpaul_files:
            fix_johnpaul_routes(johnpaul_file)
    
    print("\n✅ All individual route files updated with comprehensive route connections!")
    print("🔄 Now consolidating updated routes...")

if __name__ == "__main__":
    fix_comprehensive_route_connections()
    # Re-consolidate all route files
    os.system("python scripts/consolidate_bundle_routes.py")




