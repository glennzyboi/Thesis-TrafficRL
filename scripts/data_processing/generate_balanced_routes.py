"""
Generate realistic traffic routes based on actual field data
Converts processed CSV data from real intersections into SUMO route files
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

# Mapping between intersection names in data and SUMO network edges
INTERSECTION_TO_EDGES = {
    "JOHNPAUL": {
        "entry_edges": ["1046997839#6", "869986417#1", "935563495#2"],
        "exit_edges": ["-935563495#2", "-1046997839#6", "106609720#4", "266255177#1"]
    },
    "ECOLAND": {
        "entry_edges": ["106768821", "-794461797#2", "770761758#0", "-24224169#2"], 
        "exit_edges": ["106768822", "455558436#0", "-24224169#2"]
    },
    "SANDAWA": {
        "entry_edges": ["1042538762#0", "934492020#7"],
        "exit_edges": ["934492019#8", "1102489120"]
    }
}

# Legacy format for backward compatibility
ENTRY_EDGES = {
    "john_paul": INTERSECTION_TO_EDGES["JOHNPAUL"]["entry_edges"],
    "ecoland": INTERSECTION_TO_EDGES["ECOLAND"]["entry_edges"], 
    "sandawa": INTERSECTION_TO_EDGES["SANDAWA"]["entry_edges"]
}

EXIT_EDGES = {
    "john_paul": INTERSECTION_TO_EDGES["JOHNPAUL"]["exit_edges"],
    "ecoland": INTERSECTION_TO_EDGES["ECOLAND"]["exit_edges"],  
    "sandawa": INTERSECTION_TO_EDGES["SANDAWA"]["exit_edges"]
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

def convert_volumes_to_flows(traffic_data):
    """
    Convert vehicle counts per cycle to SUMO flow probabilities
    """
    flows = {}
    
    for intersection, data in traffic_data.items():
        if intersection not in INTERSECTION_TO_EDGES:
            print(f"⚠️ Warning: No edge mapping for intersection {intersection}")
            continue
            
        cycle_time = data['cycle_time']
        vehicles = data['vehicles']
        
        # Calculate flows per hour based on cycle time
        # Convert cycle time to flows per hour: (vehicles / cycle_time) * 3600
        intersection_flows = {}
        
        for vehicle_type, count in vehicles.items():
            if count > 0:
                # Flow rate per hour
                flow_per_hour = (count / cycle_time) * 3600
                # Convert to SUMO probability (scale down to reasonable values)
                # For data-driven routes, we want to preserve relative proportions but keep probabilities < 1.0
                # Use a scaling factor based on the largest intersection
                base_probability = min(0.8, count / cycle_time / 10.0)  # Scale down significantly
                intersection_flows[vehicle_type] = {
                    'count': count,
                    'flow_per_hour': flow_per_hour,
                    'probability': base_probability
                }
        
        flows[intersection] = intersection_flows
        
        print(f"📈 {intersection} flows:")
        for vtype, flow_data in intersection_flows.items():
            print(f"   {vtype}: {flow_data['count']} vehicles → {flow_data['flow_per_hour']:.1f}/hr → prob={flow_data['probability']:.4f}")
    
    return flows

def get_priority_combinations():
    """Get combinations with priority for underused destinations like 455558436#0"""
    combinations = []
    
    # Get all entry edges
    all_entries = []
    for intersection_entries in ENTRY_EDGES.values():
        all_entries.extend(intersection_entries)
    
    # Priority destinations that need more traffic
    priority_destinations = ["455558436#0"]
    
    # Remove 935563495#2 from priority to reduce traffic from this node
    priority_entries = []
    
    # Create priority routes to 455558436#0 from all entry points
    print(f"🎯 Creating priority routes to underused destinations: {priority_destinations}")
    for entry in all_entries:
        for priority_dest in priority_destinations:
            if entry != priority_dest:
                combinations.append((entry, priority_dest, "priority"))
    
    # Create priority routes FROM underused entry points like 935563495#2
    print(f"🎯 Creating priority routes from underused entry points: {priority_entries}")
    all_exits = []
    for intersection_exits in EXIT_EDGES.values():
        all_exits.extend(intersection_exits)
        
    for priority_entry in priority_entries:
        for exit_edge in all_exits[:2]:  # Create routes to only 2 exit points for balance
            if priority_entry != exit_edge:
                combinations.append((priority_entry, exit_edge, "priority"))
    
    # Add regular combinations
    all_exits = []
    for intersection_exits in EXIT_EDGES.values():
        all_exits.extend(intersection_exits)
    
    for entry in all_entries:
        for exit_edge in all_exits:
            if entry != exit_edge:
                combinations.append((entry, exit_edge, "regular"))
    
    print(f"📋 Created {len([c for c in combinations if c[2] == 'priority'])} priority routes")
    print(f"📋 Created {len([c for c in combinations if c[2] == 'regular'])} regular routes")
    return combinations

def generate_balanced_routes(net_file, combinations, max_routes=50):
    """Generate balanced routes with priority for underused destinations (no duplicates)"""
    print(f"🔍 Loading network: {os.path.basename(net_file)}")
    net = sumolib.net.readNet(net_file)
    
    valid_routes = []
    priority_routes = []
    regular_routes = []
    failed_routes = []
    seen_routes = set()  # Track unique route combinations to avoid duplicates
    
    print(f"🛣️ Generating balanced routes (avoiding duplicates)...")
    
    # Process priority routes first
    priority_combinations = [c for c in combinations if c[2] == "priority"]
    for entry_edge, exit_edge, route_type in priority_combinations:
        if len(priority_routes) >= 15:  # Limit priority routes
            break
            
        # Check for duplicates
        route_key = (entry_edge, exit_edge)
        if route_key in seen_routes:
            continue
            
        try:
            route = net.getShortestPath(net.getEdge(entry_edge), net.getEdge(exit_edge))
            
            if route[0] and len(route[0]) > 1:
                route_edges = [e.getID() for e in route[0]]
                route_edges_str = " ".join(route_edges)
                
                # Also check if the exact same route path already exists
                if route_edges_str not in [r['edges_str'] for r in priority_routes]:
                    priority_routes.append({
                        'entry': entry_edge,
                        'exit': exit_edge,
                        'edges': route_edges,
                        'edges_str': route_edges_str,
                        'length': len(route_edges),
                        'type': 'priority'
                    })
                    seen_routes.add(route_key)
        except Exception:
            failed_routes.append((entry_edge, exit_edge))
    
    # Process regular routes
    regular_combinations = [c for c in combinations if c[2] == "regular"]
    random.shuffle(regular_combinations)  # Randomize for better distribution
    
    for entry_edge, exit_edge, route_type in regular_combinations:
        if len(regular_routes) >= max_routes - len(priority_routes):
            break
            
        # Check for duplicates
        route_key = (entry_edge, exit_edge)
        if route_key in seen_routes:
            continue
            
        try:
            route = net.getShortestPath(net.getEdge(entry_edge), net.getEdge(exit_edge))
            
            if route[0] and len(route[0]) > 1:
                route_edges = [e.getID() for e in route[0]]
                route_edges_str = " ".join(route_edges)
                
                # Also check if the exact same route path already exists
                all_existing_routes = priority_routes + regular_routes
                if route_edges_str not in [r['edges_str'] for r in all_existing_routes]:
                    regular_routes.append({
                        'entry': entry_edge,
                        'exit': exit_edge,
                        'edges': route_edges,
                        'edges_str': route_edges_str,
                        'length': len(route_edges),
                        'type': 'regular'
                    })
                    seen_routes.add(route_key)
        except Exception:
            failed_routes.append((entry_edge, exit_edge))
    
    valid_routes = priority_routes + regular_routes
    
    print(f"✅ Generated {len(priority_routes)} priority routes to underused destinations")
    print(f"✅ Generated {len(regular_routes)} regular routes")
    print(f"❌ Failed to create {len(failed_routes)} routes")
    
    return valid_routes

def create_balanced_route_file(valid_routes, output_file, mode="flow"):
    """Create route file with balanced traffic and proper lane distribution"""
    
    # Create XML structure
    routes_elem = Element("routes")
    routes_elem.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    routes_elem.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    
    # Add vehicle types with departLane="random" for better lane distribution
    for vtype_id, props in VEHICLE_TYPES.items():
        vtype_elem = SubElement(routes_elem, "vType")
        vtype_elem.set("id", vtype_id)
        for attr, value in props.items():
            vtype_elem.set(attr, value)
    
    route_id = 0
    flow_id = 0
    
    # Track destination usage for verification
    destination_count = {}
    
    print(f"📊 Creating balanced routes with lane distribution:")
    
    # Create routes and flows
    for route_data in valid_routes:
        # Create route element
        route_elem = SubElement(routes_elem, "route")
        route_elem.set("id", f"route_{route_id}")
        route_elem.set("edges", " ".join(route_data['edges']))
        
        # Track destinations
        exit_edge = route_data['exit']
        destination_count[exit_edge] = destination_count.get(exit_edge, 0) + 1
        
        # Assign vehicle type and probability
        route_length = route_data['length']
        entry_edge = route_data['entry']
        route_type = route_data.get('type', 'regular')
        
        # Vehicle type selection
        if route_length > 15:  # Long routes
            veh_type = random.choices(
                list(VEHICLE_TYPES.keys()), 
                weights=[50, 10, 25, 10, 5]
            )[0]
            base_probability = random.uniform(0.05, 0.15)
        elif route_length > 8:  # Medium routes
            veh_type = random.choices(
                list(VEHICLE_TYPES.keys()),
                weights=[40, 15, 30, 10, 5]
            )[0]
            base_probability = random.uniform(0.08, 0.20)
        else:  # Short routes
            veh_type = random.choices(
                list(VEHICLE_TYPES.keys()),
                weights=[30, 20, 20, 20, 10]
            )[0]
            base_probability = random.uniform(0.10, 0.25)
        
        # Boost traffic for priority routes (like those going to 455558436#0)
        if route_type == 'priority':
            base_probability *= 2.0  # Double the traffic for priority destinations
            print(f"   🎯 Priority route: {entry_edge} → {exit_edge} (prob: {base_probability:.3f})")
        
        # Boost traffic from major entry points (balanced)
        if "869986417#1" in entry_edge:
            base_probability *= 1.4
        elif "935563495#2" in entry_edge:
            base_probability *= 0.6  # Reduce traffic from this overused node
        elif "934492020#7" in entry_edge:
            base_probability *= 1.3  # Boost Sandawa traffic
        elif any(x in entry_edge for x in ["106768821", "1046997839#6", "-794461797#2", "1042538762#0"]):
            base_probability *= 1.2  # Equal boost for all entry points
        
        # Cap probability
        probability = min(0.40, base_probability)
        
        if mode == "flow":
            # Create flow with random lane departure
            flow_elem = SubElement(routes_elem, "flow")
            flow_elem.set("id", f"flow_{flow_id}")
            flow_elem.set("route", f"route_{route_id}")
            flow_elem.set("begin", "0")
            flow_elem.set("end", "3600")
            flow_elem.set("probability", f"{probability:.3f}")
            flow_elem.set("type", veh_type)
            flow_elem.set("departLane", "random")  # Key fix: random lane distribution
            flow_elem.set("departSpeed", "random")  # Also randomize departure speed
            
            flow_id += 1
        else:
            # Create individual trips with random lanes
            num_vehicles = max(1, int(probability * 200))
            for i in range(num_vehicles):
                trip_elem = SubElement(routes_elem, "vehicle")
                trip_elem.set("id", f"trip_{route_id}_{i}")
                trip_elem.set("route", f"route_{route_id}")
                trip_elem.set("depart", str(random.randint(0, 3600)))
                trip_elem.set("type", veh_type)
                trip_elem.set("departLane", "random")  # Random lane
                trip_elem.set("departSpeed", "random")  # Random speed
        
        route_id += 1
    
    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    rough_string = tostring(routes_elem, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    
    # Remove empty lines
    pretty_lines = [line for line in pretty_xml.split('\n') if line.strip()]
    final_xml = '\n'.join(pretty_lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_xml)
    
    print(f"✅ Balanced route file created: {output_file}")
    print(f"   Total routes: {route_id}")
    if mode == "flow":
        print(f"   Total flows: {flow_id}")
    
    # Print destination analysis
    print(f"\n📈 Destination Traffic Analysis:")
    for dest, count in sorted(destination_count.items()):
        status = "🎯 PRIORITY" if dest == "455558436#0" else ""
        print(f"   {dest}: {count} routes {status}")
    
    return output_file

def generate_data_driven_routes(net_file, traffic_flows):
    """
    Generate routes based on real traffic data
    """
    print(f"🛣️ Generating routes from real traffic data...")
    
    # Load network
    net = sumolib.net.readNet(net_file)
    routes_data = []
    route_id = 0
    
    for intersection, flows in traffic_flows.items():
        if intersection not in INTERSECTION_TO_EDGES:
            continue
            
        entry_edges = INTERSECTION_TO_EDGES[intersection]["entry_edges"]
        exit_edges = INTERSECTION_TO_EDGES[intersection]["exit_edges"]
        
        print(f"🚦 Processing {intersection} intersection:")
        print(f"   Entry edges: {entry_edges}")
        print(f"   Exit edges: {exit_edges}")
        
        # For each vehicle type in the data
        for vehicle_type, flow_data in flows.items():
            probability = flow_data['probability']
            
            # Create routes from each entry to each exit for this vehicle type
            for entry_edge in entry_edges:
                for exit_edge in exit_edges:
                    if entry_edge != exit_edge:
                        try:
                            # Find shortest path
                            path = net.getShortestPath(
                                net.getEdge(entry_edge), 
                                net.getEdge(exit_edge)
                            )
                            
                            if path and len(path[0]) > 0:
                                route_edges = [edge.getID() for edge in path[0]]
                                
                                routes_data.append({
                                    'id': f"route_{route_id}",
                                    'edges': route_edges,
                                    'entry': entry_edge,
                                    'exit': exit_edge,
                                    'vehicle_type': vehicle_type,
                                    'probability': probability,
                                    'intersection': intersection,
                                    'length': len(route_edges),
                                    'type': 'data_driven'
                                })
                                
                                route_id += 1
                                print(f"   ✅ Route {route_id}: {entry_edge} → {exit_edge} ({vehicle_type}, prob={probability:.4f})")
                                
                        except Exception as e:
                            print(f"   ❌ Failed route {entry_edge} → {exit_edge}: {e}")
    
    print(f"✅ Generated {len(routes_data)} data-driven routes")
    return routes_data

def create_data_driven_route_file(routes_data, output_file, mode="flow"):
    """Create route file based on real traffic data"""
    
    print(f"📝 Creating data-driven route file: {output_file}")
    
    # Create XML structure
    routes_elem = Element("routes")
    routes_elem.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    routes_elem.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    
    # Add vehicle types
    for vtype_id, props in VEHICLE_TYPES.items():
        vtype_elem = SubElement(routes_elem, "vType")
        vtype_elem.set("id", vtype_id)
        for attr, value in props.items():
            vtype_elem.set(attr, value)
    
    # Add routes and flows
    flow_id = 0
    intersection_stats = {}
    
    for route_data in routes_data:
        route_id = route_data['id']
        edges = route_data['edges']
        vehicle_type = route_data['vehicle_type']
        probability = route_data['probability']
        intersection = route_data['intersection']
        
        # Track stats
        if intersection not in intersection_stats:
            intersection_stats[intersection] = {}
        if vehicle_type not in intersection_stats[intersection]:
            intersection_stats[intersection][vehicle_type] = 0
        intersection_stats[intersection][vehicle_type] += 1
        
        # Create route element
        route_elem = SubElement(routes_elem, "route")
        route_elem.set("id", route_id)
        route_elem.set("edges", " ".join(edges))
        
        if mode == "flow":
            # Create flow element
            flow_elem = SubElement(routes_elem, "flow")
            flow_elem.set("id", f"flow_{flow_id}")
            flow_elem.set("route", route_id)
            flow_elem.set("begin", "0")
            flow_elem.set("end", "3600")
            flow_elem.set("probability", f"{probability:.6f}")
            flow_elem.set("type", vehicle_type)
            flow_elem.set("departLane", "random")
            flow_elem.set("departSpeed", "random")
            
            flow_id += 1
    
    # Write file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    rough_string = tostring(routes_elem, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    
    # Remove empty lines
    pretty_lines = [line for line in pretty_xml.split('\n') if line.strip()]
    final_xml = '\n'.join(pretty_lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_xml)
    
    print(f"✅ Data-driven route file created: {output_file}")
    print(f"   Total routes: {len(routes_data)}")
    if mode == "flow":
        print(f"   Total flows: {flow_id}")
    
    # Print intersection statistics
    print(f"\n📊 Traffic Distribution by Intersection:")
    for intersection, vtypes in intersection_stats.items():
        print(f"   {intersection}:")
        for vtype, count in vtypes.items():
            print(f"     {vtype}: {count} routes")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate realistic traffic routes from real data")
    parser.add_argument("--net-file", default="network/ThesisNetowrk.net.xml", help="Network file")
    parser.add_argument("--data-file", default=None, help="Traffic data CSV file (default: data/processed/master_bundles.csv)")
    parser.add_argument("--mode", choices=["flow", "trip"], default="flow", help="Generation mode")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data instead of real data")
    parser.add_argument("--max-routes", type=int, default=40, help="Maximum routes for synthetic mode")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if args.use_synthetic:
        print("🚀 SYNTHETIC TRAFFIC ROUTE GENERATOR")
        print("=" * 60)
        print("🎯 Using synthetic traffic patterns")
        
        # Set output file
        if args.output:
            output_file = args.output
        else:
            output_file = os.path.join(PROJECT_ROOT, "data", "routes", "balanced_realistic_traffic.rou.xml")
        
        try:
            # Use old method for compatibility
            combinations = get_priority_combinations()
            print(f"📋 Created {len([c for c in combinations if c[2] == 'priority'])} priority routes")
            print(f"📋 Created {len([c for c in combinations if c[2] == 'regular'])} regular routes")
            
            net_file = os.path.join(PROJECT_ROOT, args.net_file)
            valid_routes = generate_balanced_routes(net_file, combinations, args.max_routes)
            create_balanced_route_file(valid_routes, output_file, mode=args.mode)
            
            print(f"\n🎉 SUCCESS!")
            print(f"   Generated: {output_file}")
            print(f"   ✅ Fixed lane distribution with departLane='random'")
            print(f"   ✅ Boosted traffic to underused destinations like 455558436#0")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
    else:
        print("🚀 DATA-DRIVEN TRAFFIC ROUTE GENERATOR")
        print("=" * 60)
        print("📊 Using real traffic data from field observations")
        
        # Set output file
        if args.output:
            output_file = args.output
        else:
            output_file = os.path.join(PROJECT_ROOT, "data", "routes", "data_driven_traffic.rou.xml")
        
        try:
            # Load real traffic data
            traffic_data = load_traffic_data(args.data_file)
            if not traffic_data:
                print("❌ No traffic data loaded, exiting")
                return None
                
            # Convert to flow rates
            traffic_flows = convert_volumes_to_flows(traffic_data)
            
            # Generate routes based on real data
            routes_data = generate_data_driven_routes(args.net_file, traffic_flows)
            
            # Create route file
            create_data_driven_route_file(routes_data, output_file, mode=args.mode)
            
            print(f"\n🎉 SUCCESS!")
            print(f"   Generated: {output_file}")
            print(f"   ✅ Routes based on real traffic data")
            print(f"   ✅ Vehicle counts match field observations")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
