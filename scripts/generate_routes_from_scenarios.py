
import os
import pandas as pd
import sumolib
import random
import glob
import pickle
import sys
import subprocess
import xml.etree.ElementTree as ET

# Set SUMO_HOME environment variable if not already set
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = '/usr/share/sumo'

# Add SUMO tools to the Python path
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

import traci

# Define vehicle types and their PCU equivalents and lengths
VEHICLE_TYPES = {
    "car": {"pcu": 1.0, "length": 5.0, "vType_params": "accel=\"0.8\" decel=\"4.5\" sigma=\"0.5\" minGap=\"2.5\" maxSpeed=\"100\" guiShape=\"passenger\""},
    "motor": {"pcu": 0.5, "length": 2.0, "vType_params": "accel=\"1.0\" decel=\"3.0\" sigma=\"0.3\" minGap=\"1.0\" maxSpeed=\"80\" guiShape=\"motorcycle\""},
    "jeepney": {"pcu": 1.5, "length": 7.0, "vType_params": "accel=\"0.7\" decel=\"4.0\" sigma=\"0.6\" minGap=\"3.0\" maxSpeed=\"60\" guiShape=\"bus\""},
    "bus": {"pcu": 2.0, "length": 12.0, "vType_params": "accel=\"0.6\" decel=\"4.5\" sigma=\"0.7\" minGap=\"4.0\" maxSpeed=\"60\" guiShape=\"bus\""},
    "truck": {"pcu": 2.5, "length": 15.0, "vType_params": "accel=\"0.5\" decel=\"5.0\" sigma=\"0.8\" minGap=\"5.0\" maxSpeed=\"50\" guiShape=\"truck\""},
    "tricycle": {"pcu": 0.8, "length": 3.0, "vType_params": "accel=\"0.9\" decel=\"3.5\" sigma=\"0.4\" minGap=\"2.0\" maxSpeed=\"40\" guiShape=\"motorcycle\""}
}

# Cache file for valid routes
VALID_ROUTES_CACHE = "../network/valid_routes_cache.pkl"

def test_route_in_sumo(net_file, route_edges_ids):
    """
    Tests if a given route is valid in SUMO by attempting a quick simulation.
    Returns True if valid, False otherwise.
    """
    temp_rou_file = "temp_test_route.rou.xml"
    with open(temp_rou_file, 'w') as f:
        f.write('<routes>\n')
        f.write(f'    <vType id="test_type" length="5.0" accel="0.8" decel="4.5" sigma="0.5" minGap="2.5" maxSpeed="100" guiShape="passenger"/>\n')
        f.write(f'    <vehicle id="test_veh" type="test_type" depart="0" departLane="random" departPos="random" departSpeed="max">\n')
        f.write(f'        <route edges="{" ".join(route_edges_ids)}"/>\n')
        f.write(f'    </vehicle>\n')
        f.write('</routes>\n')

    sumo_binary = sumolib.checkBinary("sumo")
    sumo_cmd = [sumo_binary, "-n", net_file, "-r", temp_rou_file, "--quit-on-end", "--no-warnings", "-e", "1"]
    
    try:
        result = subprocess.run(sumo_cmd, capture_output=True, text=True, check=False)
        os.remove(temp_rou_file)
        
        if "has no valid route" in result.stderr or "Error: " in result.stderr:
            return False
        return True
    except Exception as e:
        print(f"Error during route test simulation: {e}")
        os.remove(temp_rou_file)
        return False

def precompute_valid_routes(net_file):
    print("Precomputing valid routes using TraCI and testing in SUMO... This may take a while.")
    sumo_binary = sumolib.checkBinary("sumo")
    sumo_cmd = [sumo_binary, "-n", net_file, "--quit-on-end", "--no-warnings"]
    
    try:
        traci.start(sumo_cmd, label="precompute_routes")
    except traci.exceptions.TraCIException as e:
        print(f"Error starting TraCI for precomputation: {e}")
        print("Please ensure SUMO is correctly installed and the network file is valid.")
        return []

    net = sumolib.net.readNet(net_file)
    all_edges = net.getEdges()
    external_edges = [edge for edge in all_edges if not edge.getID().startswith(":")]

    valid_paths = []
    
    # Process a limited number of source-destination pairs to avoid excessive precomputation time
    # and to find a reasonable set of working routes.
    # The goal is to find *some* working routes, not necessarily all possible ones.
    max_pairs_to_check = 1000 # Limit the number of (source, dest) pairs to check with findRoute
    checked_pairs = 0

    # Shuffle external edges to get a random distribution of source/dest pairs
    random.shuffle(external_edges)

    for i, source_edge in enumerate(external_edges):
        if checked_pairs >= max_pairs_to_check: break
        for dest_edge in external_edges:
            if checked_pairs >= max_pairs_to_check: break
            if source_edge != dest_edge:
                try:
                    route = traci.simulation.findRoute(source_edge.getID(), dest_edge.getID())
                    if route.edges: # If a route is found by TraCI
                        # Further test this route in a quick SUMO simulation
                        if test_route_in_sumo(net_file, route.edges):
                            valid_paths.append(route.edges)
                except traci.exceptions.TraCIException as e:
                    pass 
                checked_pairs += 1
    
    traci.close()

    with open(VALID_ROUTES_CACHE, 'wb') as f:
        pickle.dump(valid_paths, f)
    print(f"Precomputation complete. Found {len(valid_paths)} valid paths. Saved to {VALID_ROUTES_CACHE}")
    return valid_paths

def get_random_valid_route(net_file):
    if not os.path.exists(VALID_ROUTES_CACHE):
        valid_paths = precompute_valid_routes(net_file)
    else:
        try:
            with open(VALID_ROUTES_CACHE, 'rb') as f:
                valid_paths = pickle.load(f)
        except Exception as e:
            print(f"Error loading valid routes cache: {e}. Recomputing...")
            valid_paths = precompute_valid_routes(net_file)

    if not valid_paths:
        print("Error: No valid connected edge pairs found in the network.")
        return None

    return random.choice(valid_paths)

def generate_route_file_from_scenario(
    scenario_df,
    net_file,
    output_rou_file,
    begin_time=0,
    end_time=3600,
    depart_speed="max"):

    print(f"Loading network from {net_file}...")
    net = sumolib.net.readNet(net_file)
    edges = net.getEdges()

    if not edges:
        print("Error: No edges found in the network. Cannot generate routes.")
        return

    with open(output_rou_file, 'w') as f:
        f.write('<routes>\n')

        for v_type_id, params in VEHICLE_TYPES.items():
            f.write(f'    <vType id="{v_type_id}" length="{params["length"]}" {params["vType_params"]}/>\n')

        vehicle_id_counter = 0
        for vehicle_class, count in scenario_df.iloc[0].items():
            if vehicle_class in VEHICLE_TYPES and count > 0:
                num_vehicles = int(count)
                duration_seconds = end_time - begin_time

                if num_vehicles > 0 and duration_seconds > 0:
                    population = range(begin_time, end_time)
                    # Ensure sample size is not larger than population
                    sample_size = min(num_vehicles, len(population))
                    depart_times = sorted(random.sample(population, sample_size))

                    for depart_time in depart_times:
                        # Attempt to get a valid route multiple times
                        max_route_attempts = 5
                        route_found = False
                        for _ in range(max_route_attempts):
                            route_edges_ids = get_random_valid_route(net_file)
                            if route_edges_ids is not None and len(route_edges_ids) >= 2:
                                route_found = True
                                break
                        
                        if not route_found:
                            print(f"Skipping vehicle for {vehicle_class} due to inability to find a valid route after {max_route_attempts} attempts.")
                            continue
                        
                        source_edge_id = route_edges_ids[0]
                        source_edge = net.getEdge(source_edge_id)

                        try:
                            if source_edge.getLaneNumber() > 0:
                                depart_lane = random.choice([str(i) for i in range(source_edge.getLaneNumber())])
                            else:
                                print(f"Warning: Source edge {source_edge.getID()} has no lanes. Skipping vehicle for {vehicle_class}.")
                                continue
                        except ValueError:
                            print(f"Warning: No lanes found for edge {source_edge.getID()}. Skipping vehicle for {vehicle_class}.")
                            continue
                        depart_pos = 'random'

                        vehicle_id = f"{vehicle_class}_{vehicle_id_counter}"
                        f.write(f'    <vehicle id="{vehicle_id}" type="{vehicle_class}" depart="{depart_time}" departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}">\n')
                        f.write(f'        <route edges="{" ".join(route_edges_ids)}"/>\n')
                        f.write(f'    </vehicle>\n')
                        vehicle_id_counter += 1

        f.write('</routes>\n')
    print(f"Generated route file: {output_rou_file}")


if __name__ == "__main__":
    NET_FILE = "../network/ThesisNetowrk_validated.net.xml" # Use the validated network file
    SCENARIOS_BASE_PATH = "../out/scenarios/"
    OUTPUT_ROU_DIR = "../traffic_files/generated_routes/"

    os.makedirs(OUTPUT_ROU_DIR, exist_ok=True)

    scenario_csv_files = [f for f in glob.glob(os.path.join(SCENARIOS_BASE_PATH, "**", "*.csv"), recursive=True)
                          if os.path.basename(f) not in ["bundle_meta.csv", "scenarios_index.csv"]]

    if not scenario_csv_files:
        print(f"No valid scenario CSVs found in {SCENARIOS_BASE_PATH}. Generating dummy data...")
        dummy_scenario_data = {
            'car': [10],
            'motor': [5],
            'jeepney': [2]
        }
        dummy_scenario_df = pd.DataFrame(dummy_scenario_data)
        OUTPUT_ROU_FILE = os.path.join(OUTPUT_ROU_DIR, "dummy_scenario.rou.xml")
        generate_route_file_from_scenario(
            scenario_df=dummy_scenario_df,
            net_file=NET_FILE,
            output_rou_file=OUTPUT_ROU_FILE,
            begin_time=0,
            end_time=300
        )
    else:
        print(f"Found {len(scenario_csv_files)} valid scenario CSVs. Generating route files...")
        for csv_file in scenario_csv_files:
            scenario_df = pd.read_csv(csv_file)
            base_name = os.path.basename(csv_file).replace(".csv", ".rou.xml")
            output_rou_file = os.path.join(OUTPUT_ROU_DIR, base_name)

            generate_route_file_from_scenario(
                scenario_df=scenario_df,
                net_file=NET_FILE,
                output_rou_file=output_rou_file,
                begin_time=0,
                end_time=60
            )
    print("All route file generation complete.")


