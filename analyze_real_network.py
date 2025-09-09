"""
Analyze the real network to get actual edge IDs and traffic light information
"""
import os
import sys
import json

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

def analyze_network():
    """Analyze the network and extract traffic light information"""
    net_file = 'network/ThesisNetowrk.net.xml'
    
    print("🔍 Analyzing network file...")
    net = sumolib.net.readNet(net_file)
    
    # Get all traffic lights
    traffic_lights = net.getTrafficLights()
    print(f"   Found {len(traffic_lights)} traffic lights")
    
    # Get all edges
    edges = net.getEdges()
    print(f"   Found {len(edges)} edges")
    
    # Analyze traffic lights and their connections
    tl_info = {}
    
    for tl in traffic_lights:
        tl_id = tl.getID()
        print(f"\n🚦 Traffic Light: {tl_id}")
        
        # Get controlled lanes and derive edges from them
        controlled_lanes = tl.getControlledLanes()
        print(f"   Controls {len(controlled_lanes)} lanes")
        
        # Get incoming and outgoing edges
        incoming_edges = set()
        outgoing_edges = set()
        
        # Get edges from controlled lanes
        for lane_id in controlled_lanes:
            try:
                lane = net.getLane(lane_id)
                edge = lane.getEdge()
                edge_id = edge.getID()
                
                # For traffic lights, incoming edges are where vehicles approach
                # We'll consider all edges that have lanes controlled by this TL
                incoming_edges.add(edge_id)
                
                # Get outgoing connections from this lane
                outgoing_lanes = lane.getOutgoing()
                for out_lane in outgoing_lanes.values():
                    for target_lane in out_lane:
                        out_edge_id = target_lane.getEdge().getID()
                        outgoing_edges.add(out_edge_id)
                        
            except Exception as e:
                print(f"   Warning: Could not process lane {lane_id}: {e}")
                continue
        
        incoming_edges = list(incoming_edges)
        outgoing_edges = list(outgoing_edges)
        
        print(f"   Incoming edges: {incoming_edges[:5]}{'...' if len(incoming_edges) > 5 else ''}")
        print(f"   Outgoing edges: {outgoing_edges[:5]}{'...' if len(outgoing_edges) > 5 else ''}")
        
        tl_info[tl_id] = {
            'incoming_edges': incoming_edges,
            'outgoing_edges': outgoing_edges
        }
    
    # Update lane_map.json with real edge IDs
    lane_map = {}
    
    # Map intersection names to traffic light IDs (you may need to adjust this mapping)
    intersection_mapping = {
        'ECOLAND': list(tl_info.keys())[0] if len(tl_info) > 0 else None,
        'JOHNPAUL': list(tl_info.keys())[1] if len(tl_info) > 1 else None,
        'SANDAWA': list(tl_info.keys())[2] if len(tl_info) > 2 else None
    }
    
    for intersection_name, tl_id in intersection_mapping.items():
        if tl_id and tl_id in tl_info:
            lane_map[intersection_name] = {
                'traffic_light_id': tl_id,
                'inbound_edges': tl_info[tl_id]['incoming_edges'][:4],  # Take first 4
                'outbound_edges': tl_info[tl_id]['outgoing_edges'][:4]  # Take first 4
            }
    
    # Save updated lane_map.json
    with open('lane_map.json', 'w') as f:
        json.dump(lane_map, f, indent=2)
    
    print(f"\n✅ Updated lane_map.json with real edge IDs")
    print(f"   Mapped {len(lane_map)} intersections")
    
    # Show sample edge pairs for route generation
    print(f"\n📋 Sample edge pairs for route generation:")
    for intersection, info in lane_map.items():
        print(f"   {intersection}:")
        if info['inbound_edges'] and info['outbound_edges']:
            print(f"     Example route: {info['inbound_edges'][0]} -> {info['outbound_edges'][0]}")
    
    return lane_map

if __name__ == '__main__':
    analyze_network()
