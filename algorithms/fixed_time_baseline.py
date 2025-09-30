"""
Fixed-Time Traffic Signal Control Baseline for Performance Comparison
Implements standard fixed-time control as baseline for D3QN evaluation
Based on established SUMO+RL research methodologies
"""

import os
import sys
import numpy as np
import time
from collections import defaultdict

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

import traci
import sumolib


class FixedTimeController:
    """
    Fixed-time traffic signal controller for baseline comparison
    Implements standard urban traffic signal timing patterns
    """
    
    def __init__(self, net_file, rou_file, use_gui=True, num_seconds=300, 
                 warmup_time=30, step_length=1.0):
        """
        Initialize fixed-time controller
        
        Args:
            net_file: SUMO network file
            rou_file: SUMO route file  
            use_gui: Whether to show SUMO GUI
            num_seconds: Total simulation duration
            warmup_time: Warmup period before data collection
            step_length: Simulation step size
        """
        self.net_file = net_file
        self.rou_file = rou_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.warmup_time = warmup_time
        self.step_length = step_length
        
        # TRUE FIXED-TIME: No dynamic timing constraints needed
        # Fixed-time signals just follow their predetermined schedule
        print(f"   TRUE FIXED-TIME: Following predetermined timing plans only")
        
        # Fair cycling enforcement (matching D3QN fairness)
        self.cycle_tracking = {}  # Track complete cycles for each TL
        self.phase_timers = {}    # Track time in current phase
        self.current_phases = {}  # Track current phase
        self.last_phase_change = {}  # Track when phase was last changed
        
        # Fixed timing plans using ONLY working phases (from network analysis)
        # CRITICAL: These phases actually provide green lights in the SUMO network
        self.timing_plans = {
            'Ecoland_TrafficSignal': {
                'cycle_length': 90,  # seconds
                'phases': [
                    {'duration': 30, 'phase': 0},  # Working phase 0
                    {'duration': 30, 'phase': 2},  # Working phase 2
                    {'duration': 15, 'phase': 4},  # Working phase 4
                    {'duration': 15, 'phase': 6}   # Working phase 6
                ]
            },
            'JohnPaul_TrafficSignal': {
                'cycle_length': 120,  # Main arterial - longer cycle
                'phases': [
                    {'duration': 40, 'phase': 0},  # Working phase 0
                    {'duration': 40, 'phase': 5},  # Working phase 5
                    {'duration': 40, 'phase': 8}   # Working phase 8
                ]
            },
            'Sandawa_TrafficSignal': {
                'cycle_length': 75,   # Residential area - shorter cycle
                'phases': [
                    {'duration': 37, 'phase': 0},  # Working phase 0
                    {'duration': 38, 'phase': 2}   # Working phase 2
                ]
            }
        }
        
        # Performance metrics tracking
        self.metrics = {
            'total_waiting_time': 0,
            'total_vehicles': 0,
            'completed_trips': 0,
            'passenger_throughput': 0,
            'avg_speed': 0,
            'total_travel_time': 0,
            'total_fuel_consumption': 0,
            'queue_length_history': [],
            'waiting_time_history': [],
            'speed_history': [],
            'throughput_history': []
        }
        
        # Vehicle type to passenger capacity mapping (same as D3QN)
        self.passenger_capacity = {
            'car': 1.5,        # Average car occupancy
            'motor': 1.2,      # Motorcycle + passenger
            'jeepney': 10.0,   # Public utility vehicle
            'bus': 30.0,       # Public bus
            'truck': 1.0,      # Commercial vehicle
            'tricycle': 2.0    # Tricycle capacity
        }
        
        # Step-by-step data for detailed analysis
        self.step_data = []
        self.current_step = 0
        self.traffic_lights = []
        self.phase_timers = {}
        
        print(f"Fixed-Time Controller Initialized:")
        print(f"   Network: {os.path.basename(net_file)}")
        print(f"   Routes: {os.path.basename(rou_file)}")
        print(f"   GUI: {'Enabled' if use_gui else 'Disabled'}")
        print(f"   Duration: {num_seconds}s (Warmup: {warmup_time}s)")
        
        self._setup_sumo_config()
    
    def _setup_sumo_config(self):
        """Setup SUMO configuration"""
        sumo_binary = sumolib.checkBinary('sumo-gui') if self.use_gui else sumolib.checkBinary('sumo')
        
        self.sumo_cmd = [
            sumo_binary,
            '-n', self.net_file,
            '--step-length', str(self.step_length),
            '--waiting-time-memory', '10000',
            '--time-to-teleport', '-1',
            '--no-warnings',
            '--quit-on-end',
            '--seed', '42'  # Fixed seed for reproducible baseline
        ]
        
        # Handle multiple route files
        if isinstance(self.rou_file, list):
            route_files_str = ','.join(self.rou_file)
            self.sumo_cmd.extend(['-r', route_files_str])
        else:
            self.sumo_cmd.extend(['-r', self.rou_file])
        
        if self.use_gui:
            self.sumo_cmd.extend(['--start', '--delay', '100'])
    
    def run_simulation(self):
        """Run complete fixed-time simulation and collect metrics"""
        print(f"\nStarting Fixed-Time Baseline Simulation...")
        
        # Start SUMO
        traci.start(self.sumo_cmd)
        
        # Initialize traffic lights
        self.traffic_lights = traci.trafficlight.getIDList()
        for tl_id in self.traffic_lights:
            self.phase_timers[tl_id] = {
                'current_phase_index': 0,
                'phase_start_time': 0
            }
        
        print(f"   Found {len(self.traffic_lights)} traffic lights: {list(self.traffic_lights)}")
        
        # Warmup period
        print(f"   Warming up simulation ({self.warmup_time}s)...")
        for _ in range(int(self.warmup_time / self.step_length)):
            self._apply_fixed_timing()
            traci.simulationStep()
            self.current_step += 1
        
        print(f"   Warmup complete - {traci.vehicle.getIDCount()} vehicles active")
        
        # Main simulation with data collection
        print(f"   Collecting performance data...")
        data_collection_steps = 0
        
        while self.current_step * self.step_length < self.num_seconds:
            # Apply fixed timing control
            self._apply_fixed_timing()
            
            # Step simulation
            traci.simulationStep()
            self.current_step += 1
            data_collection_steps += 1
            
            # Collect metrics (only after warmup)
            if self.current_step * self.step_length >= self.warmup_time:
                self._collect_step_metrics()
            
            # Progress updates
            if data_collection_steps % 25 == 0:
                self._print_progress()
        
        # Final metrics calculation
        self._calculate_final_metrics()
        
        # Close simulation
        if traci.isLoaded():
            traci.close()
        
        return self.metrics
    
    def _apply_fixed_timing(self):
        """Apply fixed timing to all traffic lights with realistic constraints"""
        current_time = self.current_step * self.step_length
        
        # CRITICAL: During warmup (first 30 steps), set ALL lights to RED
        # This matches the D3QN agent's warmup behavior exactly
        if current_time < self.warmup_time:
            for tl_id in self.traffic_lights:
                # Set all traffic lights to RED during warmup (typically phase 1 or 3)
                # This ensures no vehicles move during warmup, just like D3QN agent
                try:
                    traci.trafficlight.setPhase(tl_id, 1)  # RED phase
                except:
                    traci.trafficlight.setPhase(tl_id, 0)  # Fallback
            return
        
        for tl_id in self.traffic_lights:
            # Initialize tracking if not exists
            if tl_id not in self.current_phases:
                self.current_phases[tl_id] = traci.trafficlight.getPhase(tl_id)
                self.phase_timers[tl_id] = 0
                self.last_phase_change[tl_id] = 0
                self.cycle_tracking[tl_id] = {'phases_completed': set(), 'cycle_start': 0}
            
            if tl_id in self.timing_plans:
                timing_plan = self.timing_plans[tl_id]
                cycle_length = timing_plan['cycle_length']
                
                # Calculate position in cycle (start timing AFTER warmup)
                cycle_position = (current_time - self.warmup_time) % cycle_length
                
                # TRUE FIXED-TIME: Follow the exact timing plan, NO dynamic constraints
                # This is how real fixed-time traffic signals work - they just follow a schedule
                
                # Calculate which phase should be active based on timing plan
                cumulative_time = 0
                desired_phase = 0  # Default phase
                
                for phase_info in timing_plan['phases']:
                    phase_duration = phase_info['duration']
                    phase_number = phase_info['phase']
                    
                    if cycle_position < cumulative_time + phase_duration:
                        desired_phase = phase_number
                        break
                    cumulative_time += phase_duration
                
                # Apply the phase directly - NO timing constraints, NO min/max checks!
                # This is TRUE fixed-time behavior
                traci.trafficlight.setPhase(tl_id, desired_phase)
                self.current_phases[tl_id] = desired_phase
    
    def _collect_step_metrics(self):
        """Collect detailed metrics for current step"""
        # Vehicle metrics
        total_vehicles = traci.vehicle.getIDCount()
        vehicle_ids = traci.vehicle.getIDList()
        
        # Speed and waiting time
        total_speed = 0
        total_waiting = 0
        total_queue = 0
        
        if total_vehicles > 0:
            speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in vehicle_ids]
            waiting_times = [traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids]
            total_speed = sum(speeds)
            total_waiting = sum(waiting_times)
        
        # Queue lengths
        for tl_id in self.traffic_lights:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            for lane_id in controlled_lanes:
                total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
        
        # Completed trips and passenger throughput (origin-based calculation for consistency with D3QN)
        arrived_vehicles = traci.simulation.getArrivedIDList()
        completed_this_step = len(arrived_vehicles)
        passenger_throughput_this_step = 0
        
        # Calculate passenger throughput based on vehicle origins (same as D3QN for fair comparison)
        intersection_entry_edges = {
            'Ecoland_TrafficSignal': ['106768821', '-794461797#2', '770761758#0'],
            'JohnPaul_TrafficSignal': ['1046997839#6', '869986417#1', '935563495#2'], 
            'Sandawa_TrafficSignal': ['1042538762#0', '934492020#7']
        }
        
        # Get all vehicles and count passengers by origin
        all_vehicles = traci.vehicle.getIDList()
        for veh_id in all_vehicles:
            try:
                route_id = traci.vehicle.getRouteID(veh_id)
                route_edges = traci.route.getEdges(route_id)
                
                if route_edges:
                    origin_edge = route_edges[0]
                    
                    # Check if this vehicle originated from any intersection
                    for tl_id, entry_edges in intersection_entry_edges.items():
                        if origin_edge in entry_edges:
                            veh_type = traci.vehicle.getTypeID(veh_id)
                            passenger_count = self.passenger_capacity.get(veh_type, 1.5)
                            passenger_throughput_this_step += passenger_count
                            break
            except:
                passenger_throughput_this_step += 1.5
        
        # Store step data
        step_metrics = {
            'step': self.current_step,
            'time': self.current_step * self.step_length,
            'vehicles': total_vehicles,
            'avg_speed': (total_speed / total_vehicles * 3.6) if total_vehicles > 0 else 0,  # km/h
            'avg_waiting': total_waiting / total_vehicles if total_vehicles > 0 else 0,
            'queue_length': total_queue,
            'completed_trips': completed_this_step,
            'throughput': self.metrics['completed_trips'] / max((self.current_step - self.warmup_time) * self.step_length / 3600, 0.01),  # veh/h cumulative (matching D3QN)
            'passenger_throughput': passenger_throughput_this_step  # passengers instantaneous - PRIMARY METRIC
        }
        
        self.step_data.append(step_metrics)
        
        # Update cumulative metrics
        self.metrics['completed_trips'] += completed_this_step
        self.metrics['passenger_throughput'] += passenger_throughput_this_step
        
    def _print_progress(self):
        """Print progress similar to D3QN training"""
        if self.step_data:
            latest = self.step_data[-1]
            current_time = latest['time'] - self.warmup_time
            print(f"     Time {current_time:3.0f}s: Vehicles={latest['vehicles']:3d} | "
                  f"Wait={latest['avg_waiting']:5.1f}s | Speed={latest['avg_speed']:4.1f}km/h | "
                  f"Queue={latest['queue_length']:2d} | Throughput={latest['throughput']:5.0f}veh/h")
    
    def _calculate_final_metrics(self):
        """Calculate final performance metrics"""
        if not self.step_data:
            return
        
        # Average metrics over data collection period
        total_steps = len(self.step_data)
        
        self.metrics['avg_waiting_time'] = sum(d['avg_waiting'] for d in self.step_data) / total_steps
        self.metrics['avg_speed'] = sum(d['avg_speed'] for d in self.step_data) / total_steps
        self.metrics['avg_queue_length'] = sum(d['queue_length'] for d in self.step_data) / total_steps
        self.metrics['max_queue_length'] = max(d['queue_length'] for d in self.step_data)
        
        # Throughput calculations
        simulation_duration_hours = (self.num_seconds - self.warmup_time) / 3600
        self.metrics['avg_throughput'] = self.metrics['completed_trips'] / simulation_duration_hours
        self.metrics['avg_passenger_throughput'] = self.metrics['passenger_throughput'] / simulation_duration_hours  # PRIMARY METRIC
        
        # Travel time efficiency (speed-based proxy)
        self.metrics['travel_time_index'] = 40.0 / max(self.metrics['avg_speed'], 1.0)  # Relative to 40 km/h
        
        print(f"\nFixed-Time Simulation Complete!")
        print(f"   Performance Summary:")
        print(f"     Completed Trips: {self.metrics['completed_trips']}")
        print(f"     Avg Vehicle Throughput: {self.metrics['avg_throughput']:.1f} veh/h")
        print(f"     Avg Passenger Throughput: {self.metrics['avg_passenger_throughput']:.1f} pass/h (PRIMARY METRIC)")
        print(f"     Avg Waiting Time: {self.metrics['avg_waiting_time']:.2f}s")
        print(f"     Avg Speed: {self.metrics['avg_speed']:.1f} km/h")
        print(f"     Avg Queue Length: {self.metrics['avg_queue_length']:.1f}")
        print(f"     Max Queue Length: {self.metrics['max_queue_length']}")


def run_fixed_time_baseline(route_file, episodes=1):
    """Run fixed-time baseline for given route file"""
    controller = FixedTimeController(
        net_file='network/ThesisNetowrk.net.xml',
        rou_file=route_file,
        use_gui=False,
        num_seconds=300,
        warmup_time=30,
        step_length=1.0
    )
    
    all_metrics = []
    
    for episode in range(episodes):
        print(f"\nFixed-Time Episode {episode + 1}/{episodes}")
        metrics = controller.run_simulation()
        all_metrics.append(metrics)
        
        if episodes > 1:
            time.sleep(2)  # Brief pause between episodes
    
    return all_metrics


if __name__ == "__main__":
    # Test with one route file
    route_file = "data/routes/consolidated/bundle_20250828_cycle_1.rou.xml"
    
    if os.path.exists(route_file):
        print("Running Fixed-Time Baseline Test")
        metrics = run_fixed_time_baseline(route_file, episodes=1)
        print(f"Baseline test completed!")
    else:
        print(f"ERROR: Route file not found: {route_file}")
