"""
Create simple working routes for testing D3QN training
"""
import os
import sys
import random

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

def create_test_routes():
    """Create simple test routes using available edges"""
    net_file = 'network/ThesisNetowrk.net.xml'
    
    print("🔍 Loading network...")
    net = sumolib.net.readNet(net_file)
    
    # Get all edges
    edges = [e.getID() for e in net.getEdges() if not e.getID().startswith(':')]
    print(f"   Found {len(edges)} edges")
    
    # Create route file with some basic flows
    route_content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    
    <!-- Vehicle types -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="55" guiShape="passenger"/>
    <vType id="bus" accel="1.2" decel="4.0" sigma="0.5" length="12" minGap="3.0" maxSpeed="30" guiShape="bus"/>
    
'''
    
    # Generate some flows between random edges
    flow_id = 0
    valid_routes_created = 0
    
    for _ in range(50):  # Try to create 50 routes
        source_edge = random.choice(edges)
        dest_edge = random.choice(edges)
        
        if source_edge != dest_edge:
            # Try to find a route
            try:
                route = net.getShortestPath(net.getEdge(source_edge), net.getEdge(dest_edge))
                if route[0] and len(route[0]) > 1:  # Valid route found
                    route_edges = [e.getID() for e in route[0]]
                    
                    # Create a flow
                    veh_type = random.choice(['car', 'bus'])
                    prob = random.uniform(0.1, 0.3)
                    
                    route_content += f'    <route id="route_{flow_id}" edges="{" ".join(route_edges)}"/>\n'
                    route_content += f'    <flow id="flow_{flow_id}" route="route_{flow_id}" begin="0" end="3600" probability="{prob:.2f}" type="{veh_type}"/>\n\n'
                    
                    flow_id += 1
                    valid_routes_created += 1
                    
                    if valid_routes_created >= 20:  # Stop after 20 valid routes
                        break
                        
            except Exception as e:
                continue  # Skip invalid routes
    
    route_content += '</routes>'
    
    # Save the route file
    os.makedirs('data/routes', exist_ok=True)
    route_file = 'data/routes/test_training_routes.rou.xml'
    
    with open(route_file, 'w') as f:
        f.write(route_content)
    
    print(f"✅ Created {valid_routes_created} valid routes in {route_file}")
    return route_file

if __name__ == '__main__':
    create_test_routes()
