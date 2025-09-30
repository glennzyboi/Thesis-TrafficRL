import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob

def get_valid_connections():
    """Define valid edge connections based on the actual network topology"""
    return {
        # Ecoland intersection valid connections
        "ecoland": {
            # From -794461796#0, -794461796#1, -794461797#0, -794461797#1, -794461797#2
            "from_794461796_797": {
                "valid_exits": ["-1069919420", "1102489115#0", "934134356#0"],
                "invalid_exits": ["-1069919422#1"]  # This doesn't connect directly
            },
            # From -934134356#1, -1069919421, -794461795, -1069919419, 106768821
            "from_934134356_1069919421": {
                "valid_exits": ["1102489115#0", "934134356#0", "455558436#0"],
                "invalid_exits": ["-1069919422#1"]  # This doesn't connect directly
            },
            # From 1069919422#0, 770761758#2, 770761758#1, 770761758#0
            "from_1069919422_770761758": {
                "valid_exits": ["1102489115#0", "934134356#0"],
                "invalid_exits": ["455558436#0"]  # Should be rare
            }
        },
        # JohnPaul intersection valid connections
        "johnpaul": {
            "from_1069919425": {
                "valid_exits": ["106768827#0", "1046997833#0", "106609720#0"],
                "invalid_exits": []
            },
            "from_1046997838": {
                "valid_exits": ["106768827#0", "1046997833#0"],
                "invalid_exits": []
            },
            "from_1046997839": {
                "valid_exits": ["106768827#0", "1046997833#0"],
                "invalid_exits": []
            }
        },
        # Sandawa intersection valid connections
        "sandawa": {
            "from_1042538762": {
                "valid_exits": ["-1102489116", "1102489115#0"],
                "invalid_exits": []
            },
            "from_1042538760": {
                "valid_exits": ["-1102489116", "1102489115#0"],
                "invalid_exits": []
            }
        }
    }

def create_valid_ecoland_routes(route_file):
    """Create valid Ecoland routes with proper edge connections"""
    print(f"🔧 Creating valid Ecoland routes: {os.path.basename(route_file)}")
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Remove invalid routes that were added previously
    routes_to_remove = []
    for route in root.findall('route'):
        route_id = route.get('id')
        if route_id and route_id.startswith('route_') and int(route_id.split('_')[1]) > 100:  # New routes
            routes_to_remove.append(route_id)
    
    for route_id in routes_to_remove:
        route_elem = root.find(f".//route[@id='{route_id}']")
        if route_elem is not None:
            root.remove(route_elem)
        flow_elem = root.find(f".//flow[@route='{route_id}']")
        if flow_elem is not None:
            root.remove(flow_elem)
    
    # Add valid Ecoland routes
    route_id_counter = max([int(route.get('id').split('_')[1]) for route in root.findall('route')]) + 1
    
    # Valid routes for Ecoland
    valid_ecoland_routes = [
        # From -794461797#2 through the sequence to valid exits
        "-794461797#2 -794461797#1 -794461797#0 -794461796#1 -794461796#0 -1069919420",
        "-794461797#2 -794461797#1 -794461797#0 -794461796#1 -794461796#0 1102489115#0",
        "-794461797#2 -794461797#1 -794461797#0 -794461796#1 -794461796#0 934134356#0",
        
        # From -934134356#1 to valid exits
        "-934134356#1 1102489115#0",
        "-934134356#1 934134356#0",
        "-934134356#1 455558436#0",
        
        # From -1069919421 to valid exits
        "-1069919421 1102489115#0",
        "-1069919421 934134356#0",
        "-1069919421 455558436#0",
        
        # From 106768821 to valid exits
        "106768821 1102489115#0",
        "106768821 934134356#0",
        "106768821 455558436#0",
        
        # From 1069919422#0, 770761758#2, 770761758#1, 770761758#0 to valid exits
        "1069919422#0 1102489115#0",
        "1069919422#0 934134356#0",
        "770761758#0 770761758#1 770761758#2 1102489115#0",
        "770761758#0 770761758#1 770761758#2 934134356#0"
    ]
    
    for route_edges in valid_ecoland_routes:
        route_id = f"route_{route_id_counter}"
        route_id_counter += 1
        
        # Create new route element
        new_route = ET.Element('route', id=route_id, edges=route_edges)
        root.insert(-1, new_route)
        
        # Create corresponding flow
        flow_id = f"flow_{route_id_counter-1}"
        period = 12.0 if "1102489115#0" in route_edges else 15.0
        
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
    
    # Reduce traffic to 455558436#0 (should be rare)
    for flow in root.findall('flow'):
        route_id = flow.get('route')
        route = root.find(f".//route[@id='{route_id}']")
        if route is not None and "455558436#0" in route.get('edges'):
            period = float(flow.get('period'))
            flow.set('period', str(period * 2))  # Reduce frequency by 2x
    
    tree.write(route_file)
    print(f"   ✅ Added {len(valid_ecoland_routes)} valid Ecoland routes")

def create_valid_johnpaul_routes(route_file):
    """Create valid JohnPaul routes with proper edge connections"""
    print(f"🔧 Creating valid JohnPaul routes: {os.path.basename(route_file)}")
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Remove invalid routes that were added previously
    routes_to_remove = []
    for route in root.findall('route'):
        route_id = route.get('id')
        if route_id and route_id.startswith('route_') and int(route_id.split('_')[1]) > 100:  # New routes
            routes_to_remove.append(route_id)
    
    for route_id in routes_to_remove:
        route_elem = root.find(f".//route[@id='{route_id}']")
        if route_elem is not None:
            root.remove(route_elem)
        flow_elem = root.find(f".//flow[@route='{route_id}']")
        if flow_elem is not None:
            root.remove(flow_elem)
    
    # Add valid JohnPaul routes
    route_id_counter = max([int(route.get('id').split('_')[1]) for route in root.findall('route')]) + 1
    
    # Valid routes for JohnPaul
    valid_johnpaul_routes = [
        # From 1069919425#1, 1069919425#0 to 106768827#0
        "1069919425#1 106768827#0",
        "1069919425#0 106768827#0",
        
        # From 1046997838#3, 1046997838#2, 1046997838#1, 1046997838#0 to 106768827#0
        "1046997838#3 1046997838#2 1046997838#1 1046997838#0 106768827#0",
        "1046997838#2 1046997838#1 1046997838#0 106768827#0",
        "1046997838#1 1046997838#0 106768827#0",
        "1046997838#0 106768827#0",
        
        # From 1046997839#7, 1046997839#6 to 106768827#0
        "1046997839#7 1046997839#6 106768827#0",
        "1046997839#6 106768827#0"
    ]
    
    for route_edges in valid_johnpaul_routes:
        route_id = f"route_{route_id_counter}"
        route_id_counter += 1
        
        # Create new route element
        new_route = ET.Element('route', id=route_id, edges=route_edges)
        root.insert(-1, new_route)
        
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
    
    tree.write(route_file)
    print(f"   ✅ Added {len(valid_johnpaul_routes)} valid JohnPaul routes")

def create_valid_sandawa_routes(route_file):
    """Create valid Sandawa routes with proper edge connections"""
    print(f"🔧 Creating valid Sandawa routes: {os.path.basename(route_file)}")
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Add valid Sandawa routes
    route_id_counter = max([int(route.get('id').split('_')[1]) for route in root.findall('route')]) + 1
    
    # Valid routes for Sandawa
    valid_sandawa_routes = [
        # From 1042538762#3, 1042538762#2, 1042538762#1, 1042538762#0 to -1102489116
        "1042538762#3 1042538762#2 1042538762#1 1042538762#0 -1102489116",
        "1042538762#2 1042538762#1 1042538762#0 -1102489116",
        "1042538762#1 1042538762#0 -1102489116",
        "1042538762#0 -1102489116",
        
        # From 1042538760#2, 1042538760#1 to -1102489116
        "1042538760#2 1042538760#1 -1102489116",
        "1042538760#1 -1102489116",
        
        # Also add routes to 1102489115#0 for balance
        "1042538762#3 1042538762#2 1042538762#1 1042538762#0 1102489115#0",
        "1042538760#2 1042538760#1 1102489115#0"
    ]
    
    for route_edges in valid_sandawa_routes:
        route_id = f"route_{route_id_counter}"
        route_id_counter += 1
        
        # Create new route element
        new_route = ET.Element('route', id=route_id, edges=route_edges)
        root.insert(-1, new_route)
        
        # Create corresponding flow
        flow_id = f"flow_{route_id_counter-1}"
        period = 12.0 if "-1102489116" in route_edges else 15.0
        
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
    
    tree.write(route_file)
    print(f"   ✅ Added {len(valid_sandawa_routes)} valid Sandawa routes")

def fix_valid_route_connections():
    """Fix all route files with valid edge connections"""
    print("🚀 FIXING VALID ROUTE CONNECTIONS")
    print("=" * 60)
    print("Creating routes with valid edge connections:")
    print("• Ecoland: Proper connections to -1069919420, 1102489115#0, 934134356#0")
    print("• JohnPaul: Proper connections to 106768827#0")
    print("• Sandawa: Proper connections to -1102489116, 1102489115#0")
    print("• Reducing traffic to 455558436#0 (should be rare)")
    print()
    
    # Process all individual route files
    route_dirs = glob.glob(str(Path(__file__).parent.parent / "data" / "routes" / "2025*"))
    
    for route_dir in route_dirs:
        print(f"📁 Processing directory: {os.path.basename(route_dir)}")
        
        # Process Ecoland files
        ecoland_files = glob.glob(str(Path(route_dir) / "ECOLAND_*.rou.xml"))
        for ecoland_file in ecoland_files:
            create_valid_ecoland_routes(ecoland_file)
        
        # Process JohnPaul files  
        johnpaul_files = glob.glob(str(Path(route_dir) / "JOHNPAUL_*.rou.xml"))
        for johnpaul_file in johnpaul_files:
            create_valid_johnpaul_routes(johnpaul_file)
        
        # Process Sandawa files
        sandawa_files = glob.glob(str(Path(route_dir) / "SANDAWA_*.rou.xml"))
        for sandawa_file in sandawa_files:
            create_valid_sandawa_routes(sandawa_file)
    
    print("\n✅ All individual route files updated with valid route connections!")
    print("🔄 Now consolidating updated routes...")

if __name__ == "__main__":
    fix_valid_route_connections()
    # Re-consolidate all route files
    os.system("python scripts/consolidate_bundle_routes.py")




