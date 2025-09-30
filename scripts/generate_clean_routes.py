"""
Clean Route Generation System
Generates specific routes for Ecoland and Sandawa intersections as requested.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob
from datetime import datetime, timedelta

def create_clean_routes():
    """Create clean route files with specific routes for Ecoland and Sandawa"""
    print("🚀 CLEAN ROUTE GENERATION SYSTEM")
    print("=" * 50)
    print("Generating specific routes:")
    print("• Ecoland: 3 specific routes as requested")
    print("• Sandawa: 2 specific routes as requested")
    print("• Vehicles explore environment (no premature exits)")
    print()

def generate_vehicle_types():
    """Generate vehicle type definitions"""
    return {
        "car": {
            "accel": "2.6", "decel": "4.5", "sigma": "0.5", "length": "5", 
            "minGap": "2.5", "maxSpeed": "11.11", "guiShape": "passenger"
        },
        "bus": {
            "accel": "1.2", "decel": "4.0", "sigma": "0.5", "length": "12", 
            "minGap": "3.0", "maxSpeed": "11.11", "guiShape": "bus"
        },
        "jeepney": {
            "accel": "1.8", "decel": "4.2", "sigma": "0.5", "length": "8", 
            "minGap": "2.8", "maxSpeed": "11.11", "guiShape": "bus"
        },
        "motor": {
            "accel": "3.0", "decel": "5.0", "sigma": "0.3", "length": "2", 
            "minGap": "1.5", "maxSpeed": "11.11", "guiShape": "motorcycle"
        },
        "truck": {
            "accel": "1.5", "decel": "3.5", "sigma": "0.4", "length": "10", 
            "minGap": "3.5", "maxSpeed": "11.11", "guiShape": "truck"
        }
    }

def get_specific_routes():
    """Get the specific routes as requested by the user"""
    return {
        "ECOLAND": [
            {
                "edges": "770761758#0 770761758#1 770761758#2 1069919422#0 1102489115#0",
                "description": "Ecoland: 770761758#0 -> 1102489115#0 (explore environment)",
                "weight": 0.3,
                "period_multiplier": 1.0
            },
            {
                "edges": "-794461796#1 -794461796#0 -1069919420 -455558436#1 1102489115#0",
                "description": "Ecoland: -794461796#1 -> 1102489115#0 (heavy traffic)",
                "weight": 0.5,
                "period_multiplier": 0.5  # Higher frequency for heavy traffic
            },
            {
                "edges": "-794461795 -1069919421 -934134356#1 -1069919422#1 -770761758#2",
                "description": "Ecoland: -794461795 -> -770761758#2 (explore environment)",
                "weight": 0.2,
                "period_multiplier": 1.0
            }
        ],
        "SANDAWA": [
            {
                "edges": "1042538762#3 -1102489116",
                "description": "Sandawa: 1042538762#3 -> -1102489116 (explore environment)",
                "weight": 0.5,
                "period_multiplier": 1.0
            },
            {
                "edges": "1042538760#2 -1102489116",
                "description": "Sandawa: 1042538760#2 -> -1102489116 (explore environment)",
                "weight": 0.5,
                "period_multiplier": 1.0
            }
        ]
    }

def generate_route_file(intersection_name, routes_data, date_str, cycle_num, output_dir):
    """Generate a route file for a specific intersection"""
    # Create root element
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    
    # Add vehicle types
    vtypes = generate_vehicle_types()
    for vtype_id, vtype_attrs in vtypes.items():
        vtype_elem = ET.SubElement(root, "vType", id=vtype_id, **vtype_attrs)
    
    # Add routes and flows
    route_id_counter = 0
    
    for route_data in routes_data:
        route_id = f"route_{route_id_counter}"
        route_id_counter += 1
        
        # Create route element
        route_elem = ET.SubElement(root, "route", id=route_id, edges=route_data["edges"])
        
        # Create flows for all vehicle types
        vehicle_types = ["car", "motor", "jeepney", "bus", "truck"]
        for v_type in vehicle_types:
            # Calculate period based on weight and vehicle type
            base_period = 20.0  # Base period in seconds
            weight = route_data.get("weight", 0.5)
            period_multiplier = route_data.get("period_multiplier", 1.0)
            
            # Adjust period based on weight (higher weight = more frequent)
            period = base_period / weight * period_multiplier
            
            # Adjust period based on vehicle type
            if v_type == "motor":
                period *= 0.8  # Motorcycles more frequent
            elif v_type in ["bus", "jeepney"]:
                period *= 1.5  # PT vehicles less frequent
            elif v_type == "truck":
                period *= 2.0  # Trucks less frequent
            
            flow_id = f"flow_{route_id_counter-1}_{v_type}"
            flow_elem = ET.SubElement(root, "flow",
                                    id=flow_id,
                                    route=route_id,
                                    begin="0",
                                    end="3600",
                                    period=str(period),
                                    type=v_type,
                                    departLane="random",
                                    departSpeed="max",
                                    departPos="random")
    
    # Write to file
    filename = f"{intersection_name}_{date_str}_cycle{cycle_num}.rou.xml"
    filepath = os.path.join(output_dir, filename)
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)  # Pretty print with proper indentation
    tree.write(filepath, encoding='utf-8', xml_declaration=True)
    
    print(f"   ✅ Generated: {filename} ({len(routes_data)} routes)")
    return filepath

def generate_all_route_files():
    """Generate all route files for all dates and cycles"""
    print("🔧 GENERATING ALL ROUTE FILES")
    print("=" * 50)
    
    # Get specific routes
    all_routes = get_specific_routes()
    
    # Define date range and cycles
    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 8, 31)
    cycles = [1, 2, 3]
    
    # Generate routes for each date and cycle
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        
        for cycle in cycles:
            print(f"📅 Generating routes for {date_str} cycle {cycle}")
            
            # Create date-cycle directory
            date_cycle_dir = f"data/routes/{date_str}_cycle_{cycle}"
            os.makedirs(date_cycle_dir, exist_ok=True)
            
            # Generate routes for each intersection
            for intersection_name, routes_data in all_routes.items():
                generate_route_file(intersection_name, routes_data, date_str, cycle, date_cycle_dir)
        
        current_date += timedelta(days=1)
    
    print(f"\n✅ Generated route files for {len(all_routes)} intersections")

def consolidate_routes():
    """Consolidate all generated route files"""
    print("🔧 CONSOLIDATING ROUTE FILES")
    print("=" * 50)
    
    # Find all date-cycle directories
    route_dirs = glob.glob("data/routes/2025*_cycle_*")
    
    consolidated_dir = "data/routes/consolidated"
    os.makedirs(consolidated_dir, exist_ok=True)
    
    for route_dir in route_dirs:
        # Extract date and cycle from directory name
        dir_name = os.path.basename(route_dir)
        if "_cycle_" in dir_name:
            date_part = dir_name.split("_cycle_")[0]
            cycle_part = dir_name.split("_cycle_")[1]
            
            # Find all route files in this directory
            route_files = glob.glob(os.path.join(route_dir, "*.rou.xml"))
            
            if not route_files:
                print(f"⚠️ No route files found in {route_dir}")
                continue
            
            print(f"📁 Processing {len(route_files)} files from {dir_name}")
            
            # Create consolidated file
            output_file = os.path.join(consolidated_dir, f"bundle_{date_part}_cycle_{cycle_part}.rou.xml")
            consolidate_bundle_routes(route_files, output_file)

def consolidate_bundle_routes(route_files, output_file):
    """Consolidate multiple route files into one"""
    if not route_files:
        return False
    
    # Create root element
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    
    # Track what we've added
    added_vtypes = set()
    route_counter = 0
    flow_counter = 0
    
    for i, route_file in enumerate(route_files):
        if not os.path.exists(route_file):
            continue
        
        print(f"   📁 Processing: {os.path.basename(route_file)}")
        
        try:
            tree = ET.parse(route_file)
            file_root = tree.getroot()
            
            # Add vehicle types only from the first file
            if i == 0:
                for vtype in file_root.findall('.//vType'):
                    vtype_id = vtype.get('id')
                    if vtype_id not in added_vtypes:
                        root.append(vtype)
                        added_vtypes.add(vtype_id)
            
            # Add routes with unique IDs
            for route in file_root.findall('.//route'):
                original_id = route.get('id')
                unique_id = f"route_{route_counter}"
                route.set('id', unique_id)
                root.append(route)
                route_counter += 1
                
                # Update flows that reference this route
                for flow in file_root.findall('.//flow'):
                    if flow.get('route') == original_id:
                        flow.set('route', unique_id)
            
            # Add flows with unique IDs
            for flow in file_root.findall('.//flow'):
                unique_id = f"flow_{flow_counter}"
                flow.set('id', unique_id)
                root.append(flow)
                flow_counter += 1
        
        except ET.ParseError as e:
            print(f"❌ Error parsing {route_file}: {e}")
            continue
    
    # Write consolidated file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    print(f"✅ Created: {os.path.basename(output_file)}")
    print(f"   📊 Vehicle types: {len(added_vtypes)}")
    print(f"   📊 Routes: {route_counter}")
    print(f"   📊 Flows: {flow_counter}")
    
    return True

def main():
    """Main function to generate clean routes"""
    create_clean_routes()
    
    # Generate all route files
    generate_all_route_files()
    
    # Consolidate all routes
    consolidate_routes()
    
    print("\n🎉 CLEAN ROUTE GENERATION COMPLETED!")
    print("=" * 60)
    print("✅ Ecoland: 3 specific routes implemented")
    print("✅ Sandawa: 2 specific routes implemented")
    print("✅ Vehicles explore environment properly")
    print("✅ Clean, simple codebase")
    print("\n📁 Route files are ready for training!")

if __name__ == "__main__":
    main()




