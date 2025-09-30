"""
Generate realistic traffic routes based on actual field data and user specifications
Creates proper traffic flow patterns for JohnPaul, Ecoland, and Sandawa intersections
"""

import os
import sys
import random
import argparse
import pandas as pd
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

# Set SUMO_HOME
if 'SUMO_HOME' not in os.environ:
    possible_paths = [
        r'C:\Program Files (x86)\Eclipse\Sumo',
        r'C:\Program Files\Eclipse\Sumo', 
        r'C:\sumo',
        r'C:\Users\%USERNAME%\sumo'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['SUMO_HOME'] = path
            break

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

import sumolib

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Updated realistic edge mappings based on user specifications
REALISTIC_EDGE_MAPPINGS = {
    "JOHNPAUL": {
        "entry_edges": {
            "1069919425#1": {"weight": 0.4, "description": "Main heavy traffic entry"},
            "869986417#1": {"weight": 0.3, "description": "Secondary heavy traffic entry"},
            "935563495#2": {"weight": 0.1, "description": "Light traffic entry (rare vehicles)"},
            "-106768827#1": {"weight": 0.2, "description": "Cross traffic entry"}
        },
        "exit_edges": {
            "1046997833#0": {"weight": 0.45, "description": "Primary exit - heavy traffic destination"},
            "106768827#0": {"weight": 0.45, "description": "Primary exit - heavy traffic destination"},
            "106609720#0": {"weight": 0.08, "description": "Secondary exit - moderate traffic"},
            "-1069919425#1": {"weight": 0.02, "description": "Rare exit - minimal traffic"}
        },
        "restrictions": {
            "935563495#7": {"allowed_vehicles": ["bus", "jeepney"], "description": "PT vehicles only"},
            "106609720#0": {"from_edges": ["-106768827#1"], "weight": 0.8, "description": "Mainly from -106768827#1"}
        }
    },
    "ECOLAND": {
        "entry_edges": {
            "-794461797#2": {"weight": 0.15, "description": "Light traffic entry - needs more vehicles"},
            "1069919421": {"weight": 0.25, "description": "Moderate traffic entry"},
            "-934134356#1": {"weight": 0.25, "description": "Moderate traffic entry"},
            "1069919420": {"weight": 0.05, "description": "PT vehicles only - through traffic"},
            "1069919422#0": {"weight": 0.15, "description": "Heavy traffic entry"},
            "770761758#2": {"weight": 0.15, "description": "Heavy traffic entry"}
        },
        "exit_edges": {
            "-455558436#1": {"weight": 0.25, "description": "Primary exit - needs more traffic"},
            "455558436#0": {"weight": 0.25, "description": "Secondary exit - moderate traffic"},
            "-1069919422#1": {"weight": 0.25, "description": "Secondary exit - should match 455558436#0"},
            "934134356#0": {"weight": 0.15, "description": "Tertiary exit - from 1069919422#0 & 770761758#2"},
            "1102489115#0": {"weight": 0.10, "description": "Rare exit - from 1069919422#0 & 770761758#2"}
        },
        "restrictions": {
            "1069919420": {"through_traffic": True, "pt_only": True, "description": "PT vehicles only, through traffic"},
            "455558436#0": {"from_edges": ["1069919421", "-934134356#1"], "equal_split": True, "description": "Equal split from main entries"},
            "-1069919422#1": {"from_edges": ["1069919421", "-934134356#1"], "equal_split": True, "description": "Equal split from main entries"}
        }
    },
    "SANDAWA": {
        "entry_edges": {
            "1042538762#0": {"weight": 0.6, "description": "Main heavy traffic entry"},
            "934492020#7": {"weight": 0.4, "description": "Secondary heavy traffic entry"}
        },
        "exit_edges": {
            "934492019#8": {"weight": 0.5, "description": "Primary exit"},
            "1102489120": {"weight": 0.5, "description": "Primary exit"}
        },
        "restrictions": {
            "increase_volume": True,
            "description": "Increase overall traffic volume for more realistic scenario"
        }
    }
}

# Vehicle type properties (all vehicles limited to 40 km/hr = 11.11 m/s)
VEHICLE_TYPES = {
    "car": {"accel": "2.6", "decel": "4.5", "sigma": "0.5", "length": "5", "minGap": "2.5", "maxSpeed": "11.11", "guiShape": "passenger"},
    "bus": {"accel": "1.2", "decel": "4.0", "sigma": "0.5", "length": "12", "minGap": "3.0", "maxSpeed": "11.11", "guiShape": "bus"},
    "jeepney": {"accel": "1.8", "decel": "4.2", "sigma": "0.5", "length": "8", "minGap": "2.8", "maxSpeed": "11.11", "guiShape": "bus"},
    "motor": {"accel": "3.0", "decel": "5.0", "sigma": "0.3", "length": "2", "minGap": "1.5", "maxSpeed": "11.11", "guiShape": "motorcycle"},
    "truck": {"accel": "1.5", "decel": "3.5", "sigma": "0.4", "length": "10", "minGap": "3.5", "maxSpeed": "11.11", "guiShape": "truck"}
}

# Mapping from data vehicle types to SUMO vehicle types
DATA_TO_SUMO_VEHICLE_TYPES = {
    "car_count": "car",
    "motor_count": "motor", 
    "jeepney_count": "jeepney",
    "bus_count": "bus",
    "truck_count": "truck"
}

def load_traffic_data(data_file=None):
    """
    Load real traffic data from processed CSV files
    Returns a dictionary with intersection traffic volumes by vehicle type
    """
    if data_file is None:
        data_file = os.path.join(PROJECT_ROOT, "data", "processed", "master_bundles.csv")
    
    print(f"📊 Loading real traffic data from: {data_file}")
    
    try:
        df = pd.read_csv(data_file)
        traffic_data = {}
        
        for _, row in df.iterrows():
            intersection = row['IntersectionID']
            cycle_time = row['CycleTime_s']
            
            # Extract vehicle counts for each type
            vehicles = {}
            for data_col, sumo_type in DATA_TO_SUMO_VEHICLE_TYPES.items():
                if data_col in row and pd.notna(row[data_col]):
                    vehicles[sumo_type] = int(row[data_col])
            
            traffic_data[intersection] = {
                'cycle_time': cycle_time,
                'total_vehicles': row['TotalVehicles'],
                'vehicles': vehicles
            }
            
        print(f"✅ Loaded traffic data for {len(traffic_data)} intersections:")
        for intersection, data in traffic_data.items():
            total = data['total_vehicles']
            time = data['cycle_time']
            print(f"   {intersection}: {total} vehicles in {time}s cycle")
            
        return traffic_data
        
    except Exception as e:
        print(f"❌ Error loading traffic data: {e}")
        return {}

def create_realistic_route_combinations():
    """
    Create realistic route combinations based on user specifications
    """
    combinations = []
    
    print("🛣️ Creating realistic route combinations based on user specifications...")
    
    for intersection, mapping in REALISTIC_EDGE_MAPPINGS.items():
        print(f"\n📍 {intersection} Intersection:")
        
        entry_edges = mapping["entry_edges"]
        exit_edges = mapping["exit_edges"]
        restrictions = mapping.get("restrictions", {})
        
        # Create weighted combinations based on realistic traffic patterns
        for entry_id, entry_data in entry_edges.items():
            entry_weight = entry_data["weight"]
            print(f"   Entry {entry_id}: {entry_data['description']} (weight: {entry_weight})")
            
            for exit_id, exit_data in exit_edges.items():
                exit_weight = exit_data["weight"]
                
                # Check restrictions
                if exit_id in restrictions:
                    restriction = restrictions[exit_id]
                    
                    # Check if this entry is allowed for this exit
                    if "from_edges" in restriction and entry_id not in restriction["from_edges"]:
                        continue
                    
                    # Check vehicle type restrictions
                    if "allowed_vehicles" in restriction:
                        # This exit only allows certain vehicle types
                        for vehicle_type in restriction["allowed_vehicles"]:
                            combinations.append((entry_id, exit_id, vehicle_type, "restricted"))
                        continue
                
                # Regular combinations with realistic weights
                combined_weight = entry_weight * exit_weight
                
                # Apply special rules for equal splits
                if exit_id in restrictions and "equal_split" in restrictions[exit_id]:
                    combined_weight = 0.5  # Equal weight for equal splits
                
                # Create combinations for all vehicle types with realistic distribution
                vehicle_weights = {
                    "car": 0.5,      # 50% cars
                    "motor": 0.3,    # 30% motorcycles
                    "jeepney": 0.1,  # 10% jeepneys
                    "bus": 0.05,     # 5% buses
                    "truck": 0.05    # 5% trucks
                }
                
                for vehicle_type, vtype_weight in vehicle_weights.items():
                    # Skip PT-only restrictions for non-PT vehicles
                    if exit_id in restrictions and "pt_only" in restrictions[exit_id] and vehicle_type not in ["bus", "jeepney"]:
                        continue
                    
                    # Calculate final weight considering vehicle type distribution
                    final_weight = combined_weight * vtype_weight
                    combinations.append((entry_id, exit_id, vehicle_type, "realistic", final_weight))
        
        print(f"   Created {len([c for c in combinations if c[0] in entry_edges])} combinations for {intersection}")
    
    print(f"\n✅ Total realistic combinations created: {len(combinations)}")
    return combinations

def generate_realistic_routes(net_file, combinations, max_routes=100):
    """Generate realistic routes with proper traffic distribution"""
    print(f"🔍 Loading network: {os.path.basename(net_file)}")
    net = sumolib.net.readNet(net_file)
    
    valid_routes = []
    failed_routes = []
    seen_routes = set()
    
    print(f"🛣️ Generating realistic routes with proper traffic distribution...")
    
    # Group combinations by intersection for better organization
    intersection_routes = {}
    
    for entry_edge, exit_edge, vehicle_type, route_type, weight in combinations:
        intersection = None
        for int_name, mapping in REALISTIC_EDGE_MAPPINGS.items():
            if entry_edge in mapping["entry_edges"]:
                intersection = int_name
                break
        
        if intersection not in intersection_routes:
            intersection_routes[intersection] = []
        
        intersection_routes[intersection].append((entry_edge, exit_edge, vehicle_type, route_type, weight))
    
    # Generate routes for each intersection
    for intersection, int_combinations in intersection_routes.items():
        print(f"\n📍 Processing {intersection} intersection...")
        
        # Sort by weight (highest first) to prioritize realistic routes
        int_combinations.sort(key=lambda x: x[4], reverse=True)
        
        intersection_count = 0
        max_per_intersection = max_routes // len(intersection_routes)
        
        # Group by entry-exit pairs to avoid duplicate paths
        entry_exit_pairs = {}
        for entry_edge, exit_edge, vehicle_type, route_type, weight in int_combinations:
            pair_key = (entry_edge, exit_edge)
            if pair_key not in entry_exit_pairs:
                entry_exit_pairs[pair_key] = []
            entry_exit_pairs[pair_key].append((vehicle_type, route_type, weight))
        
        # Generate routes for each unique entry-exit pair
        for (entry_edge, exit_edge), vehicle_types in entry_exit_pairs.items():
            if intersection_count >= max_per_intersection:
                break
                
            try:
                route = net.getShortestPath(net.getEdge(entry_edge), net.getEdge(exit_edge))
                
                if route[0] and len(route[0]) > 1:
                    route_edges = [e.getID() for e in route[0]]
                    route_edges_str = " ".join(route_edges)
                    
                    # Check if the exact same route path already exists
                    if route_edges_str not in [r['edges_str'] for r in valid_routes]:
                        # Create routes for all vehicle types for this path
                        for vehicle_type, route_type, weight in vehicle_types:
                            route_key = (entry_edge, exit_edge, vehicle_type)
                            if route_key not in seen_routes:
                                valid_routes.append({
                                    'entry': entry_edge,
                                    'exit': exit_edge,
                                    'vehicle_type': vehicle_type,
                                    'edges': route_edges,
                                    'edges_str': route_edges_str,
                                    'length': len(route_edges),
                                    'type': route_type,
                                    'weight': weight,
                                    'intersection': intersection
                                })
                                seen_routes.add(route_key)
                                intersection_count += 1
                                print(f"   ✅ {entry_edge} -> {exit_edge} ({vehicle_type})")
            except Exception as e:
                print(f"   ❌ {entry_edge} -> {exit_edge}: {str(e)[:50]}...")
                for vehicle_type, route_type, weight in vehicle_types:
                    failed_routes.append((entry_edge, exit_edge, vehicle_type, str(e)))
        
        print(f"   Generated {intersection_count} routes for {intersection}")
    
    print(f"✅ Generated {len(valid_routes)} realistic routes")
    print(f"❌ Failed to create {len(failed_routes)} routes")
    
    if failed_routes:
        print("Failed routes:")
        for fail in failed_routes[:10]:  # Show first 10 failures
            print(f"   {fail[0]} -> {fail[1]} ({fail[2]}): {fail[3]}")
    
    return valid_routes

def create_route_file(routes, flows, output_file, scenario_name="realistic_traffic"):
    """Create SUMO route file with realistic traffic patterns"""
    print(f"📝 Creating route file: {output_file}")
    
    # Create XML structure
    root = Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    
    # Add vehicle types
    for vtype_id, vtype_props in VEHICLE_TYPES.items():
        vtype_elem = SubElement(root, "vType")
        vtype_elem.set("id", vtype_id)
        for prop, value in vtype_props.items():
            vtype_elem.set(prop, value)
    
    # Add flows for each route
    flow_count = 0
    for route in routes:
        # Create route element
        route_elem = SubElement(root, "route")
        route_elem.set("id", f"route_{flow_count}")
        route_elem.set("edges", route['edges_str'])
        
        # Create flow element
        flow_elem = SubElement(root, "flow")
        flow_elem.set("id", f"flow_{flow_count}")
        flow_elem.set("route", f"route_{flow_count}")
        flow_elem.set("type", route['vehicle_type'])
        flow_elem.set("begin", "0")
        flow_elem.set("end", "300")  # 5 minutes
        flow_elem.set("vehsPerHour", str(flows[route['vehicle_type']]))
        
        flow_count += 1
    
    # Write to file
    tree_str = tostring(root, encoding='unicode')
    dom = minidom.parseString(tree_str)
    pretty_xml = dom.toprettyxml(indent="  ")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    print(f"✅ Route file created with {flow_count} flows")
    return flow_count

def main():
    parser = argparse.ArgumentParser(description="Generate realistic traffic routes")
    parser.add_argument("--net_file", default="network/ThesisNetowrk.net.xml", help="SUMO network file")
    parser.add_argument("--output_dir", default="data/routes/realistic", help="Output directory for route files")
    parser.add_argument("--max_routes", type=int, default=100, help="Maximum routes per intersection")
    parser.add_argument("--scenario", default="realistic_traffic", help="Scenario name")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 REALISTIC ROUTE GENERATION")
    print("=" * 50)
    print(f"Network: {args.net_file}")
    print(f"Output: {args.output_dir}")
    print(f"Max routes per intersection: {args.max_routes}")
    print(f"Scenario: {args.scenario}")
    
    # Load traffic data
    traffic_data = load_traffic_data()
    if not traffic_data:
        print("❌ No traffic data loaded. Using default flows.")
        flows = {"car": 50, "motor": 30, "jeepney": 20, "bus": 10, "truck": 5}
    else:
        # Calculate flows from traffic data
        flows = {}
        for vehicle_type in VEHICLE_TYPES.keys():
            total_count = sum(data['vehicles'].get(vehicle_type, 0) for data in traffic_data.values())
            flows[vehicle_type] = max(total_count // 10, 10)  # Convert to veh/h
    
    print(f"\n📊 Vehicle flows (veh/h):")
    for vtype, flow in flows.items():
        print(f"   {vtype}: {flow}")
    
    # Create realistic route combinations
    combinations = create_realistic_route_combinations()
    
    # Generate routes
    routes = generate_realistic_routes(args.net_file, combinations, args.max_routes)
    
    # Create route file
    output_file = os.path.join(args.output_dir, f"{args.scenario}.rou.xml")
    flow_count = create_route_file(routes, flows, output_file, args.scenario)
    
    # Generate summary report
    print(f"\n📊 ROUTE GENERATION SUMMARY")
    print("=" * 50)
    print(f"Total routes generated: {len(routes)}")
    print(f"Total flows created: {flow_count}")
    print(f"Output file: {output_file}")
    
    # Intersection breakdown
    intersection_counts = {}
    for route in routes:
        intersection = route['intersection']
        intersection_counts[intersection] = intersection_counts.get(intersection, 0) + 1
    
    print(f"\n📍 Routes per intersection:")
    for intersection, count in intersection_counts.items():
        print(f"   {intersection}: {count} routes")
    
    print(f"\n✅ Realistic route generation completed!")
    print(f"📁 Route file saved: {output_file}")

if __name__ == "__main__":
    main()




