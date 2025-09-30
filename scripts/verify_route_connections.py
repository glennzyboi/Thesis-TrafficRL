import os
import xml.etree.ElementTree as ET

def verify_route_connections():
    """Verify that routes properly connect to target edges"""
    print("🔍 VERIFYING ROUTE CONNECTIONS")
    print("=" * 50)
    
    route_file = "data/routes/consolidated/bundle_20250701_cycle_1.rou.xml"
    
    if not os.path.exists(route_file):
        print(f"❌ Route file not found: {route_file}")
        return
    
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Target edges to verify
    ecoland_targets = ["-1069919420", "-1069919422#1", "1102489115#0"]
    johnpaul_targets = ["106768827#0"]
    
    # Check Ecoland connections
    print("\n🏢 ECOLAND INTERSECTION:")
    print("Target edges: -1069919420, -1069919422#1, 1102489115#0")
    
    ecoland_routes = []
    for route in root.findall('route'):
        edges = route.get('edges')
        if any(target in edges for target in ecoland_targets):
            ecoland_routes.append(edges)
    
    print(f"✅ Found {len(ecoland_routes)} routes connecting to Ecoland targets")
    
    # Check specific connections
    for target in ecoland_targets:
        routes_to_target = [route for route in ecoland_routes if target in route]
        print(f"   • {target}: {len(routes_to_target)} routes")
        if routes_to_target:
            print(f"     Example: {routes_to_target[0]}")
    
    # Check JohnPaul connections
    print("\n🏢 JOHNPAUL INTERSECTION:")
    print("Target edges: 106768827#0")
    
    johnpaul_routes = []
    for route in root.findall('route'):
        edges = route.get('edges')
        if "106768827#0" in edges:
            johnpaul_routes.append(edges)
    
    print(f"✅ Found {len(johnpaul_routes)} routes connecting to JohnPaul targets")
    
    # Check specific connections
    for target in johnpaul_targets:
        routes_to_target = [route for route in johnpaul_routes if target in route]
        print(f"   • {target}: {len(routes_to_target)} routes")
        if routes_to_target:
            print(f"     Example: {routes_to_target[0]}")
    
    # Check 455558436#0 traffic reduction
    print("\n🚦 TRAFFIC REDUCTION CHECK:")
    routes_to_455558436 = []
    for route in root.findall('route'):
        edges = route.get('edges')
        if "455558436#0" in edges:
            routes_to_455558436.append(route.get('id'))
    
    print(f"Routes going to 455558436#0: {len(routes_to_455558436)}")
    
    # Check flow frequencies for 455558436#0
    high_freq_flows = 0
    for flow in root.findall('flow'):
        route_id = flow.get('route')
        route = root.find(f".//route[@id='{route_id}']")
        if route is not None and "455558436#0" in route.get('edges'):
            period = float(flow.get('period'))
            if period < 20.0:  # High frequency
                high_freq_flows += 1
    
    print(f"High frequency flows to 455558436#0: {high_freq_flows}")
    if high_freq_flows == 0:
        print("✅ Traffic to 455558436#0 has been reduced (should be rare)")
    else:
        print("⚠️ Still some high frequency traffic to 455558436#0")
    
    print("\n🎯 ROUTE CONNECTION VERIFICATION COMPLETE!")

if __name__ == "__main__":
    verify_route_connections()




