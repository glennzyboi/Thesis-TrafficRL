"""
Consolidate multiple route files from a bundle into a single route file for MARL training.
This fixes the SUMO issue where multiple route files can't define the same vehicle types.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path


def consolidate_bundle_routes(bundle_routes, output_file):
    """
    Consolidate multiple route files into a single route file
    
    Args:
        bundle_routes: List of route file paths to consolidate
        output_file: Output consolidated route file path
    """
    if not bundle_routes:
        print("ERROR: No route files to consolidate")
        return False
    
    # Create root element
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    
    # Track what we've added to avoid duplicates
    added_vtypes = set()
    route_counter = 0
    flow_counter = 0
    
    print(f"Consolidating {len(bundle_routes)} route files...")
    
    for i, route_file in enumerate(bundle_routes):
        if not os.path.exists(route_file):
            print(f"WARNING: Route file not found: {route_file}")
            continue
        
            print(f"   Processing: {os.path.basename(route_file)}")
        
        try:
            tree = ET.parse(route_file)
            file_root = tree.getroot()
            
            # Add vehicle types only from the first file to avoid duplicates
            if i == 0:
                for vtype in file_root.findall('vType'):
                    vtype_id = vtype.get('id')
                    if vtype_id not in added_vtypes:
                        root.append(vtype)
                        added_vtypes.add(vtype_id)
            print(f"     Added vehicle type: {vtype_id}")
            
            # Create mapping of original route IDs to new unique IDs
            route_id_mapping = {}
            
            # Add routes with unique IDs
            for route in file_root.findall('route'):
                original_id = route.get('id')
                unique_id = f"route_{route_counter}"
                route_id_mapping[original_id] = unique_id
                route.set('id', unique_id)
                root.append(route)
                route_counter += 1
            
            # Add flows with unique IDs and updated route references
            for flow in file_root.findall('flow'):
                # Update route reference to use new unique route ID
                original_route_id = flow.get('route')
                if original_route_id in route_id_mapping:
                    flow.set('route', route_id_mapping[original_route_id])
                
                # Set unique flow ID
                unique_flow_id = f"flow_{flow_counter}"
                flow.set('id', unique_flow_id)
                root.append(flow)
                flow_counter += 1
        
        except ET.ParseError as e:
            print(f"ERROR: Error parsing {route_file}: {e}")
            continue
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write consolidated file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)  # Pretty print
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    print(f"Consolidated route file created: {output_file}")
    print(f"   Vehicle types: {len(added_vtypes)}")
    print(f"   Routes: {route_counter}")
    print(f"   Flows: {flow_counter}")
    
    return True


def consolidate_all_bundles():
    """Consolidate route files for all available bundles"""
    scenarios_file = os.path.join("data", "processed", "scenarios_index.csv")
    
    if not os.path.exists(scenarios_file):
        print(f"ERROR: Scenarios index not found: {scenarios_file}")
        return
    
    import pandas as pd
    df = pd.read_csv(scenarios_file)
    
    consolidated_dir = os.path.join("data", "routes", "consolidated")
    os.makedirs(consolidated_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        day = row['Day']
        cycle = row['CycleNum']
        intersections = row['Intersections'].split(',')
        
        # Collect route files for this bundle
        bundle_routes = []
        bundle_dir = f"data/routes/{day}_cycle_{cycle}"
        
        for intersection in intersections:
            intersection = intersection.strip()
            route_file = os.path.join(bundle_dir, f"{intersection}_{day}_cycle{cycle}.rou.xml")
            
            if os.path.exists(route_file):
                bundle_routes.append(route_file)
            else:
                print(f"WARNING: Missing route file: {route_file}")
        
        if bundle_routes:
            # Create consolidated route file
            output_file = os.path.join(consolidated_dir, f"bundle_{day}_cycle_{cycle}.rou.xml")
            consolidate_bundle_routes(bundle_routes, output_file)
            print(f"Bundle {day} Cycle {cycle} consolidated\n")


if __name__ == "__main__":
    consolidate_all_bundles()