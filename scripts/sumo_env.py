
import os
import sys
import random

# Set SUMO_HOME environment variable if not already set
if 'SUMO_HOME' not in os.environ:
    # Common Windows installation paths
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
    else:
        print("⚠️ SUMO_HOME not found. Please set it manually or install SUMO.")
        print("   Example: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo")

# Add SUMO tools to the Python path
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

import traci
import sumolib

class SumoEnvironment:
    def __init__(self, net_file, rou_file, use_gui=False, num_seconds=3600):
        self.net_file = net_file
        self.rou_file = rou_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.sumo_cmd = self._build_sumo_cmd()
        self.current_step = 0

    def _build_sumo_cmd(self):
        sumo_binary = sumolib.checkBinary('sumo-gui') if self.use_gui else sumolib.checkBinary('sumo')
        command = [
            sumo_binary,
            '-n', self.net_file,
            '-r', self.rou_file,
            '--step-length', '1',
            '--waiting-time-memory', '10000',
            '--time-to-teleport', '-1',
            '--duration-log.statistics',
            '--statistic-output', '../logs/sumo_statistics.xml',
            '--tripinfo-output', '../logs/sumo_tripinfo.xml',
            '--no-warnings',
            '--quit-on-end',
            '--random',
            '--seed', str(random.randint(0, 100000))
        ]
        return command

    def start(self):
        print("Starting SUMO simulation...")
        traci.start(self.sumo_cmd)
        self.current_step = 0

    def step(self):
        traci.simulationStep()
        self.current_step += 1
        return self.current_step < self.num_seconds

    def close(self):
        print("Closing SUMO simulation...")
        traci.close()
        sys.stdout.flush()

    def get_lane_waiting_times(self):
        waiting_times = {}
        for lane_id in traci.lane.getIDList():
            waiting_times[lane_id] = traci.lane.getWaitingTime(lane_id)
        return waiting_times

    def get_traffic_light_state(self, tls_id):
        return traci.trafficlight.getPhase(tls_id)

    def set_traffic_light_phase(self, tls_id, phase_index):
        traci.trafficlight.setPhase(tls_id, phase_index)

    def get_traffic_light_ids(self):
        return traci.trafficlight.getIDList()

    def get_junction_ids(self):
        return traci.junction.getIDList()

    def get_lane_occupancy(self, lane_id):
        return traci.lane.getLastStepOccupancy(lane_id)

    def get_lane_mean_speed(self, lane_id):
        return traci.lane.getLastStepMeanSpeed(lane_id)

    def get_lane_vehicle_count(self, lane_id):
        return traci.lane.getLastStepVehicleNumber(lane_id)

    def get_all_lane_ids(self):
        return traci.lane.getIDList()


if __name__ == "__main__":
    NET_FILE = "../network/ThesisNetowrk.net.xml"
    ROU_FILE = "../traffic_files/generated_routes/dummy_scenario.rou.xml"

    env = SumoEnvironment(NET_FILE, ROU_FILE, use_gui=False, num_seconds=100)
    try:
        env.start()
        while env.step():
            if env.current_step % 10 == 0:
                print(f"Step: {env.current_step}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()


