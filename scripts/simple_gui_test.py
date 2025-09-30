"""
Simple GUI test to show improved traffic patterns
"""

import os
import sys
import traci
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def simple_gui_test():
    """
    Simple GUI test to show traffic patterns
    """
    print("SIMPLE GUI TEST")
    print("=" * 30)
    print("Showing improved traffic patterns:")
    print("• Multi-lane utilization")
    print("• Realistic traffic distribution")
    print("• PT vehicle restrictions")
    print()
    
    # SUMO configuration
    net_file = "network/ThesisNetowrk.net.xml"
    route_file = "data/routes/consolidated/bundle_20250701_cycle_1.rou.xml"
    
    # Start SUMO with GUI
    sumo_cmd = [
        "C:\\\\Program Files (x86)\\\\Eclipse\\\\Sumo\\\\bin\\\\sumo-gui.exe",
        "-n", net_file,
        "-r", route_file,
        "--step-length", "1.0",
        "--waiting-time-memory", "10000",
        "--time-to-teleport", "-1",
        "--no-warnings",
        "--quit-on-end",
        "--seed", "12345",
        "--start",
        "--delay", "100"
    ]
    
    print(f"Starting SUMO GUI...")
    print("Watch the traffic patterns in the GUI window")
    print("Test will run for 2 minutes")
    print("Press Ctrl+C to stop early")
    print()
    
    try:
        traci.start(sumo_cmd)
        
        # Run simulation for 120 seconds
        for step in range(120):
            traci.simulationStep()
            
            # Show progress every 20 seconds
            if step % 20 == 0:
                print(f"Step {step:3d}s: Simulation running...")
            
            time.sleep(0.1)
        
        print("GUI test completed!")
        
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        try:
            traci.close()
            print("SUMO connection closed")
        except:
            pass

if __name__ == "__main__":
    simple_gui_test()




