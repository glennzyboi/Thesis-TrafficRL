"""
Verification script to check processed data
"""

import pandas as pd
import os

def verify_processed_data():
    """Display summary of processed data"""
    
    print("D3QN Data Processing Verification")
    print("=" * 50)
    
    # Check master bundles
    master_file = "data/processed/master_bundles.csv"
    if os.path.exists(master_file):
        df = pd.read_csv(master_file)
        print(f"\n📊 Master Bundles Summary ({master_file})")
        print(f"   Rows: {len(df)}")
        print(f"   Intersections: {df['IntersectionID'].unique()}")
        print(f"   Days: {df['Day'].unique()}")
        print(f"   Cycles: {df['CycleNum'].unique()}")
        
        print(f"\n🚗 Vehicle Totals by Intersection:")
        for intersection in df['IntersectionID'].unique():
            idata = df[df['IntersectionID'] == intersection].iloc[0]
            print(f"   {intersection}:")
            print(f"     Total Vehicles: {idata['TotalVehicles']}")
            print(f"     Cycle Time: {idata['CycleTime_s']}s")
            print(f"     Cars: {idata['car_count']}, Motors: {idata['motor_count']}")
            print(f"     Jeepneys: {idata['jeepney_count']}, Buses: {idata['bus_count']}, Trucks: {idata['truck_count']}")
    else:
        print(f"❌ Master bundles file not found: {master_file}")
    
    # Check scenarios
    scenarios_file = "data/processed/scenarios_index.csv"
    if os.path.exists(scenarios_file):
        df_scen = pd.read_csv(scenarios_file)
        print(f"\n📁 Scenarios Summary ({scenarios_file})")
        print(f"   Available scenarios: {len(df_scen)}")
        for _, row in df_scen.iterrows():
            print(f"     {row['Day']} Cycle {row['CycleNum']}: {row['Intersections']}")
    else:
        print(f"❌ Scenarios index not found: {scenarios_file}")
    
    # Check routes
    routes_dir = "data/routes"
    if os.path.exists(routes_dir):
        route_files = [f for f in os.listdir(routes_dir) if f.endswith('.rou.xml')]
        print(f"\n🛣️ Generated Route Files ({routes_dir})")
        print(f"   Route files: {len(route_files)}")
        for rf in route_files:
            print(f"     {rf}")
    else:
        print(f"❌ Routes directory not found: {routes_dir}")
    
    print(f"\n✅ Data pipeline verification complete!")
    print(f"\nNext steps:")
    print(f"   1. Update lane_map.json with real SUMO edge IDs")
    print(f"   2. Test route files with SUMO simulation")
    print(f"   3. Integrate with D3QN training")

if __name__ == "__main__":
    verify_processed_data()
