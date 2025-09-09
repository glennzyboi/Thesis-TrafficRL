"""
Generate SUMO route files from scenario CSV files
Follows the correct lifecycle: Bundle → Individual .rou.xml per intersection
"""

import os
import sys
import pandas as pd
import argparse
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

# Enhanced mapping with more realistic traffic distribution and alternative routes
INTERSECTION_TO_EDGES = {
    "ECOLAND": {
        "entry_edges": ["106768821", "-794461797#2"], 
        "exit_edges": ["106768822", "455558436#0", "1102489115#0"],  # Added 1102489115#0
        "alternative_routes": {
            # Force usage of underutilized edges
            "106768821": ["455558436#0", "1102489115#0"],  # More traffic to these exits
            "-794461797#2": ["106768822", "455558436#0"]
        }
    },
    "JOHNPAUL": {
        "entry_edges": ["1046997839#6", "869986417#1", "935563495#2"],
        "exit_edges": ["-935563495#2", "-1046997839#6", "106609720#4", "266255177#1"],
        "alternative_routes": {
            # Reduce 935563495#2 dominance, add variety
            "1046997839#6": ["106609720#4", "-935563495#2", "266255177#1"],
            "869986417#1": ["-1046997839#6", "106609720#4"], 
            "935563495#2": ["-1046997839#6"]  # Limited routes from this edge
        }
    },
    "SANDAWA": {
        "entry_edges": ["1042538762#0", "934492020#7"],
        "exit_edges": ["934492019#8", "1102489120"],
        "alternative_routes": {
            # Ensure 934492020#7 uses multiple lanes/paths
            "934492020#7": ["934492019#8", "1102489120"],
            "1042538762#0": ["934492019#8", "1102489120"]
        }
    }
}

# Vehicle type properties (40 km/hr = 11.11 m/s)
VEHICLE_TYPES = {
    "car": {"accel": "2.6", "decel": "4.5", "sigma": "0.5", "length": "5", "minGap": "2.5", "maxSpeed": "11.11", "guiShape": "passenger"},
    "bus": {"accel": "1.2", "decel": "4.0", "sigma": "0.5", "length": "12", "minGap": "3.0", "maxSpeed": "11.11", "guiShape": "bus"},
    "jeepney": {"accel": "1.8", "decel": "4.2", "sigma": "0.5", "length": "8", "minGap": "2.8", "maxSpeed": "11.11", "guiShape": "bus"},
    "motor": {"accel": "3.0", "decel": "5.0", "sigma": "0.3", "length": "2", "minGap": "1.5", "maxSpeed": "11.11", "guiShape": "motorcycle"},
    "truck": {"accel": "1.5", "decel": "3.5", "sigma": "0.4", "length": "10", "minGap": "3.5", "maxSpeed": "11.11", "guiShape": "truck"}
}

# Mapping from scenario CSV columns to SUMO vehicle types
CSV_TO_SUMO_VEHICLE_TYPES = {
    "car_count": "car",
    "motor_count": "motor", 
    "jeepney_count": "jeepney",
    "bus_count": "bus",
    "truck_count": "truck"
}

def load_scenario_data(scenario_csv_path):
    """Load vehicle counts from a single intersection scenario CSV file"""
    print(f"📊 Loading scenario data from: {scenario_csv_path}")
    
    try:
        df = pd.read_csv(scenario_csv_path)
        if len(df) == 0:
            print(f"❌ Empty scenario file: {scenario_csv_path}")
            return None
            
        row = df.iloc[0]  # Each scenario file has one row
        intersection = row['IntersectionID']
        cycle_time = row['CycleTime_s']
        day = row['Day']
        cycle_num = row['CycleNum']
        
        # Extract vehicle counts
        vehicles = {}
        for csv_col, sumo_type in CSV_TO_SUMO_VEHICLE_TYPES.items():
            if csv_col in row and pd.notna(row[csv_col]) and row[csv_col] > 0:
                vehicles[sumo_type] = int(row[csv_col])
        
        scenario_data = {
            'intersection': intersection,
            'day': day,
            'cycle_num': cycle_num,
            'cycle_time': cycle_time,
            'total_vehicles': row['TotalVehicles'],
            'vehicles': vehicles
        }
        
        print(f"✅ Loaded {intersection} scenario: {len(vehicles)} vehicle types, {scenario_data['total_vehicles']} total vehicles")
        return scenario_data
        
    except Exception as e:
        print(f"❌ Error loading scenario data: {e}")
        return None

def convert_counts_to_flows(scenario_data):
    """Convert vehicle counts to SUMO flow probabilities"""
    vehicles = scenario_data['vehicles']
    cycle_time = scenario_data['cycle_time']
    
    flows = {}
    for vehicle_type, count in vehicles.items():
        if count > 0:
            # Calculate flow probability (scale down for SUMO)
            # Use cycle time to determine realistic probability
            base_probability = min(0.6, count / cycle_time / 8.0)  # Conservative scaling
            flows[vehicle_type] = {
                'count': count,
                'probability': base_probability
            }
    
    print(f"📈 {scenario_data['intersection']} flows:")
    for vtype, flow_data in flows.items():
        print(f"   {vtype}: {flow_data['count']} vehicles → prob={flow_data['probability']:.4f}")
    
    return flows

def generate_intersection_routes(net_file, scenario_data, flows):
    """Generate diverse and realistic routes for a single intersection"""
    intersection = scenario_data['intersection']
    
    if intersection not in INTERSECTION_TO_EDGES:
        print(f"❌ No edge mapping for intersection {intersection}")
        return []
    
    print(f"🛣️ Generating enhanced routes for {intersection}...")
    
    # Load network
    net = sumolib.net.readNet(net_file)
    routes_data = []
    route_id = 0
    
    entry_edges = INTERSECTION_TO_EDGES[intersection]["entry_edges"]
    exit_edges = INTERSECTION_TO_EDGES[intersection]["exit_edges"]
    alternative_routes = INTERSECTION_TO_EDGES[intersection].get("alternative_routes", {})
    
    print(f"   Entry edges: {entry_edges}")
    print(f"   Exit edges: {exit_edges}")
    
    # For each vehicle type in the scenario
    for vehicle_type, flow_data in flows.items():
        base_probability = flow_data['probability']
        
        # Create diverse routes with realistic distribution
        for entry_edge in entry_edges:
            # Use alternative route preferences if defined
            if entry_edge in alternative_routes:
                preferred_exits = alternative_routes[entry_edge]
                # Higher probability for preferred routes, lower for others
                for exit_edge in preferred_exits:
                    enhanced_prob = base_probability * 1.2  # 20% boost for realistic routes
                    routes_data.extend(create_route_variants(
                        net, entry_edge, exit_edge, vehicle_type, enhanced_prob, route_id, "preferred"
                    ))
                    route_id += len(routes_data) - route_id
                
                # Lower probability for non-preferred routes (realistic variety)
                other_exits = [e for e in exit_edges if e not in preferred_exits]
                for exit_edge in other_exits:
                    reduced_prob = base_probability * 0.3  # Reduced for variety
                    routes_data.extend(create_route_variants(
                        net, entry_edge, exit_edge, vehicle_type, reduced_prob, route_id, "alternative"
                    ))
                    route_id += len(routes_data) - route_id
            else:
                # Standard routes for entries without preferences
                for exit_edge in exit_edges:
                    if entry_edge != exit_edge:
                        routes_data.extend(create_route_variants(
                            net, entry_edge, exit_edge, vehicle_type, base_probability, route_id, "standard"
                        ))
                        route_id += len(routes_data) - route_id
    
    print(f"✅ Generated {len(routes_data)} diverse routes for {intersection}")
    return routes_data

def create_route_variants(net, entry_edge, exit_edge, vehicle_type, probability, route_id, route_type):
    """Create multiple route variants for better lane distribution"""
    variants = []
    
    try:
        # Find shortest path
        path = net.getShortestPath(net.getEdge(entry_edge), net.getEdge(exit_edge))
        
        if path and len(path[0]) > 0:
            route_edges = [edge.getID() for edge in path[0]]
            
            # Create main route
            variants.append({
                'id': f"route_{route_id}",
                'edges': route_edges,
                'entry': entry_edge,
                'exit': exit_edge,
                'vehicle_type': vehicle_type,
                'probability': probability,
                'length': len(route_edges),
                'route_type': route_type
            })
            
            print(f"   ✅ Route {route_id}: {entry_edge} → {exit_edge} ({vehicle_type}, {route_type}, prob={probability:.4f})")
            
    except Exception as e:
        print(f"   ❌ Failed route {entry_edge} → {exit_edge}: {e}")
    
    return variants

def create_intersection_route_file(routes_data, scenario_data, output_file):
    """Create SUMO route file for a single intersection"""
    intersection = scenario_data['intersection']
    print(f"📝 Creating route file for {intersection}: {output_file}")
    
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
    vehicle_stats = {}
    
    for route_data in routes_data:
        route_id = route_data['id']
        edges = route_data['edges']
        vehicle_type = route_data['vehicle_type']
        probability = route_data['probability']
        
        # Track stats
        if vehicle_type not in vehicle_stats:
            vehicle_stats[vehicle_type] = 0
        vehicle_stats[vehicle_type] += 1
        
        # Create route element
        route_elem = SubElement(routes_elem, "route")
        route_elem.set("id", route_id)
        route_elem.set("edges", " ".join(edges))
        
        # Create flow element with enhanced lane distribution
        flow_elem = SubElement(routes_elem, "flow")
        flow_elem.set("id", f"flow_{flow_id}")
        flow_elem.set("route", route_id)
        flow_elem.set("begin", "0")
        flow_elem.set("end", "3600")
        flow_elem.set("probability", f"{probability:.6f}")
        flow_elem.set("type", vehicle_type)
        
        # Enhanced traffic behavior settings
        route_type = route_data.get('route_type', 'standard')
        if route_type == "preferred":
            flow_elem.set("departLane", "best")  # Use best available lane for main routes
        else:
            flow_elem.set("departLane", "random")  # More varied for alternatives
            
        flow_elem.set("departSpeed", "random")  # Vary departure speeds
        flow_elem.set("departPos", "random")    # Vary starting positions within lane
        
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
    
    print(f"✅ Route file created: {output_file}")
    print(f"   Total routes: {len(routes_data)} | Total flows: {flow_id}")
    print(f"   Vehicle distribution: {vehicle_stats}")
    
    return output_file

def process_bundle(day, cycle_num, net_file):
    """Process a complete bundle (all intersections for a specific day/cycle)"""
    print(f"\n🎯 PROCESSING BUNDLE: Day {day}, Cycle {cycle_num}")
    print("=" * 60)
    
    scenario_dir = os.path.join(PROJECT_ROOT, "out", "scenarios", str(day), f"cycle_{cycle_num}")
    route_output_dir = os.path.join(PROJECT_ROOT, "data", "routes", f"{day}_cycle_{cycle_num}")
    
    if not os.path.exists(scenario_dir):
        print(f"❌ Scenario directory not found: {scenario_dir}")
        return []
    
    generated_routes = []
    
    # Process each intersection in the bundle
    for intersection in ["ECOLAND", "JOHNPAUL", "SANDAWA"]:
        scenario_file = os.path.join(scenario_dir, f"{intersection}_cycle{cycle_num}.csv")
        
        if not os.path.exists(scenario_file):
            print(f"⚠️ Scenario file not found: {scenario_file}")
            continue
        
        # Load scenario data
        scenario_data = load_scenario_data(scenario_file)
        if not scenario_data:
            continue
        
        # Convert to flows
        flows = convert_counts_to_flows(scenario_data)
        if not flows:
            print(f"⚠️ No vehicle flows for {intersection}")
            continue
        
        # Generate routes
        routes_data = generate_intersection_routes(net_file, scenario_data, flows)
        if not routes_data:
            print(f"⚠️ No routes generated for {intersection}")
            continue
        
        # Create route file
        output_file = os.path.join(route_output_dir, f"{intersection}_{day}_cycle{cycle_num}.rou.xml")
        create_intersection_route_file(routes_data, scenario_data, output_file)
        
        generated_routes.append({
            'intersection': intersection,
            'route_file': output_file,
            'route_count': len(routes_data)
        })
    
    print(f"\n🎉 Bundle processing complete!")
    print(f"   Generated route files for {len(generated_routes)} intersections")
    for route_info in generated_routes:
        print(f"   {route_info['intersection']}: {route_info['route_count']} routes → {route_info['route_file']}")
    
    return generated_routes

def main():
    parser = argparse.ArgumentParser(description="Generate SUMO route files from scenario bundles")
    parser.add_argument("--net-file", default="network/ThesisNetowrk.net.xml", help="SUMO network file")
    parser.add_argument("--day", default="20250828", help="Day to process")
    parser.add_argument("--cycle", type=int, default=1, help="Cycle number to process")
    parser.add_argument("--all-bundles", action="store_true", help="Process all bundles from scenarios_index.csv")
    
    args = parser.parse_args()
    
    print("🚀 SCENARIO-BASED ROUTE GENERATOR")
    print("=" * 60)
    print("📋 Generates individual .rou.xml files per intersection per cycle")
    print("🎯 Follows the correct lifecycle: Bundle → Synchronized route files")
    
    net_file_path = os.path.join(PROJECT_ROOT, args.net_file)
    
    if args.all_bundles:
        # Process all bundles from scenarios_index.csv
        scenarios_index_file = os.path.join(PROJECT_ROOT, "data", "processed", "scenarios_index.csv")
        
        if not os.path.exists(scenarios_index_file):
            print(f"❌ scenarios_index.csv not found: {scenarios_index_file}")
            return
        
        df = pd.read_csv(scenarios_index_file)
        print(f"📊 Found {len(df)} bundles to process")
        
        for _, row in df.iterrows():
            day = row['Day']
            cycle_num = row['CycleNum']
            process_bundle(day, cycle_num, net_file_path)
    else:
        # Process single bundle
        process_bundle(args.day, args.cycle, net_file_path)

if __name__ == "__main__":
    main()
