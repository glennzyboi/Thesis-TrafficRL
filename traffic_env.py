"""
Traffic Signal Control Environment for D3QN Training
Provides a complete RL environment wrapper around SUMO with state/action spaces
"""

import os
import sys
import numpy as np
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

class TrafficEnvironment:
    """
    Traffic Signal Control Environment for Reinforcement Learning
    
    State Space: Traffic metrics for each intersection (queue length, waiting time, etc.)
    Action Space: Traffic signal phase selection for each controlled intersection
    """
    
    def __init__(self, net_file, rou_file, use_gui=True, num_seconds=3600, 
                 warmup_time=300, step_length=1.0):
        """
        Initialize the traffic environment
        
        Args:
            net_file: Path to SUMO network file (.net.xml)
            rou_file: Path to SUMO route file (.rou.xml) or list of route files for MARL
            use_gui: Whether to show SUMO GUI for visualization
            num_seconds: Total simulation duration
            warmup_time: Time before agent control starts
            step_length: Simulation step size in seconds
        """
        self.net_file = net_file
        
        # Handle both single route file and multiple route files (for MARL)
        if isinstance(rou_file, list):
            self.rou_files = rou_file
            self.rou_file = rou_file[0] if rou_file else None  # For backward compatibility
        else:
            self.rou_files = [rou_file] if rou_file else []
            self.rou_file = rou_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.warmup_time = warmup_time
        self.step_length = step_length
        
        # Simulation state
        self.current_step = 0
        self.warmup_complete = False
        self.total_reward = 0
        
        # Traffic signal control
        self.traffic_lights = []
        self.tl_phases = {}  # Store available phases for each TL
        self.controlled_lanes = {}  # Lanes controlled by each TL
        
        # RL parameters
        self.state_size = None
        self.action_size = None
        
        # Performance metrics
        self.metrics = {
            'total_waiting_time': 0,
            'total_vehicles': 0,
            'avg_speed': 0,
            'completed_trips': 0,
            'passenger_throughput': 0
        }
        
        # Vehicle type to passenger capacity mapping (based on field data)
        self.passenger_capacity = {
            'car': 1.5,        # Average car occupancy
            'motor': 1.2,      # Motorcycle + passenger
            'jeepney': 10.0,   # Public utility vehicle
            'bus': 30.0,       # Public bus
            'truck': 1.0,      # Commercial vehicle
            'tricycle': 2.0    # Tricycle capacity
        }
        
        print(f"🚦 Traffic Environment Initialized:")
        print(f"   Network: {os.path.basename(net_file)}")
        if isinstance(rou_file, list):
            print(f"   Routes: {len(rou_file)} files for MARL")
            for rf in rou_file:
                print(f"     - {os.path.basename(rf)}")
        else:
            print(f"   Routes: {os.path.basename(rou_file)}")
        print(f"   GUI: {'Enabled' if use_gui else 'Disabled'}")
        print(f"   Duration: {num_seconds}s (Warmup: {warmup_time}s)")
        
        # Initialize SUMO configuration
        self._setup_sumo_config()
    
    def _setup_sumo_config(self):
        """Setup SUMO configuration and command"""
        sumo_binary = sumolib.checkBinary('sumo-gui') if self.use_gui else sumolib.checkBinary('sumo')
        
        self.sumo_cmd = [
            sumo_binary,
            '-n', self.net_file,
            '--step-length', str(self.step_length),
            '--waiting-time-memory', '10000',
            '--time-to-teleport', '-1',
            '--no-warnings', 
            '--quit-on-end',
            '--seed', str(random.randint(0, 100000))
        ]
        
        # Add route files (support multiple files for MARL)
        if self.rou_files:
            # SUMO expects comma-separated route files in a single -r option
            route_files_str = ','.join(self.rou_files)
            self.sumo_cmd.extend(['-r', route_files_str])
        
        # Add GUI-specific options
        if self.use_gui:
            self.sumo_cmd.extend([
                '--start',  # Start simulation immediately
                '--delay', '100'  # Delay between steps (ms) for better visualization
            ])
        
        # Debug: Print the SUMO command
        print(f"🔧 SUMO Command: {' '.join(self.sumo_cmd)}")
    
    def reset(self):
        """Reset the environment for a new episode"""
        # Close any existing simulation
        if traci.isLoaded():
            traci.close()
        
        # Start new simulation
        print(f"\n🔄 Starting new episode...")
        traci.start(self.sumo_cmd)
        
        # Reset episode variables
        self.current_step = 0
        self.warmup_complete = False
        self.total_reward = 0
        self.metrics = {key: 0 for key in self.metrics}
        
        # Initialize traffic lights
        self._initialize_traffic_lights()
        
        # Warm up the simulation
        self._warmup_simulation()
        
        # Get initial state
        initial_state = self._get_state()
        
        print(f"   Episode started - State size: {len(initial_state)}")
        return initial_state
    
    def _initialize_traffic_lights(self):
        """Initialize traffic light information"""
        self.traffic_lights = list(traci.trafficlight.getIDList())
        
        print(f"   Found {len(self.traffic_lights)} traffic lights: {self.traffic_lights}")
        
        # Get phases and controlled lanes for each traffic light
        for tl_id in self.traffic_lights:
            # Get all available phases
            program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            self.tl_phases[tl_id] = len(program.phases)
            
            # Get controlled lanes
            self.controlled_lanes[tl_id] = traci.trafficlight.getControlledLanes(tl_id)
        
        # Calculate state and action space sizes
        # State: [queue_length, waiting_time, avg_speed] per controlled lane
        total_lanes = sum(len(lanes) for lanes in self.controlled_lanes.values())
        self.state_size = total_lanes * 3  # 3 metrics per lane
        
        # Action: phase selection for each traffic light
        self.action_size = max(self.tl_phases.values()) if self.tl_phases else 4
        
        print(f"   State size: {self.state_size}, Action size: {self.action_size}")
    
    def _warmup_simulation(self):
        """Run simulation warmup period without agent control"""
        print(f"   🔥 Warming up simulation ({self.warmup_time}s)...")
        
        warmup_steps = int(self.warmup_time / self.step_length)
        for _ in range(warmup_steps):
            traci.simulationStep()
            self.current_step += 1
        
        self.warmup_complete = True
        print(f"   ✅ Warmup complete - {traci.vehicle.getIDCount()} vehicles active")
    
    def step(self, action):
        """
        Execute one environment step with the given action
        
        Args:
            action: Traffic light phase to set (integer)
            
        Returns:
            next_state: Next state observation
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information dict
        """
        # Apply action to traffic lights
        self._apply_action(action)
        
        # Step the simulation
        traci.simulationStep()
        self.current_step += 1
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Check if episode is done
        done = self._is_done()
        
        # Update metrics
        self._update_metrics()
        
        # Calculate detailed metrics for training visualization
        total_vehicles = traci.vehicle.getIDCount()
        total_waiting = 0
        total_queue_length = 0
        avg_speed = 0
        
        # Collect metrics from all lanes
        for tl_id in self.traffic_lights:
            for lane_id in self.controlled_lanes[tl_id]:
                total_waiting += traci.lane.getWaitingTime(lane_id)
                total_queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
        
        # Calculate average speed
        if total_vehicles > 0:
            speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in traci.vehicle.getIDList()]
            avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        # Enhanced info for debugging and visualization
        info = {
            'step': self.current_step,
            'vehicles': total_vehicles,
            'waiting_time': total_waiting / max(total_vehicles, 1),  # Average waiting per vehicle
            'avg_speed': avg_speed * 3.6,  # Convert m/s to km/h
            'queue_length': total_queue_length,
            'reward': reward,
            'total_reward': self.total_reward,
            'completed_trips': self.metrics.get('completed_trips', 0),
            'throughput': self.metrics.get('completed_trips', 0) / max(self.current_step * self.step_length / 3600, 0.01),  # Vehicles per hour
            'passenger_throughput': self.metrics.get('passenger_throughput', 0) / max(self.current_step * self.step_length / 3600, 0.01),  # Passengers per hour - PRIMARY METRIC
            'total_passenger_throughput': self.metrics.get('passenger_throughput', 0),  # Cumulative passengers
            'metrics': self.metrics.copy()
        }
        
        return next_state, reward, done, info
    
    def _apply_action(self, action):
        """Apply the action to traffic lights"""
        # For simplicity, apply same action to all traffic lights
        # In practice, you might want different actions for each intersection
        action = int(action) % self.action_size
        
        for tl_id in self.traffic_lights:
            # Make sure action is valid for this traffic light
            max_phase = self.tl_phases[tl_id] - 1
            phase = min(action, max_phase)
            traci.trafficlight.setPhase(tl_id, phase)
    
    def _get_state(self):
        """Get current state observation"""
        state = []
        
        for tl_id in self.traffic_lights:
            for lane_id in self.controlled_lanes[tl_id]:
                # Queue length (number of vehicles)
                queue_length = traci.lane.getLastStepVehicleNumber(lane_id)
                
                # Waiting time
                waiting_time = traci.lane.getWaitingTime(lane_id)
                
                # Average speed
                avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                
                # Normalize values
                queue_length = min(queue_length / 20.0, 1.0)  # Max 20 vehicles
                waiting_time = min(waiting_time / 300.0, 1.0)  # Max 5 minutes
                avg_speed = avg_speed / 13.89 if avg_speed > 0 else 0  # Normalize to 50 km/h
                
                state.extend([queue_length, waiting_time, avg_speed])
        
        # Pad state if needed
        while len(state) < self.state_size:
            state.append(0.0)
        
        return np.array(state[:self.state_size])
    
    def _calculate_reward(self):
        """
        Calculate optimized reward based on traffic signal control research
        Designed to outperform fixed-time control with balanced multi-objective optimization
        """
        total_vehicles = traci.vehicle.getIDCount()
        
        if total_vehicles == 0:
            return 0.5  # Higher baseline for empty traffic management
        
        # Collect comprehensive traffic metrics
        total_waiting = 0
        total_queue_length = 0
        lane_count = 0
        
        for tl_id in self.traffic_lights:
            for lane_id in self.controlled_lanes[tl_id]:
                waiting_time = traci.lane.getWaitingTime(lane_id)
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                total_waiting += waiting_time
                total_queue_length += queue_length
                lane_count += 1
        
        # Get vehicles that completed their journey this step and calculate passenger throughput
        arrived_vehicles = traci.simulation.getArrivedIDList()
        step_throughput = len(arrived_vehicles)  # Vehicle throughput
        step_passenger_throughput = 0            # Passenger throughput (primary objective)
        
        # Calculate passenger throughput for this step
        for veh_id in arrived_vehicles:
            try:
                veh_type = traci.vehicle.getTypeID(veh_id)
                passenger_count = self.passenger_capacity.get(veh_type, 1.0)
                step_passenger_throughput += passenger_count
            except:
                step_passenger_throughput += 1.5  # Average passenger fallback
        
        # Calculate average speed and collect individual speeds for variance
        speeds = []
        if total_vehicles > 0:
            for veh_id in traci.vehicle.getIDList():
                speed = traci.vehicle.getSpeed(veh_id)
                speeds.append(speed)
        
        avg_speed = np.mean(speeds) if speeds else 0
        speed_variance = np.var(speeds) if len(speeds) > 1 else 0
        
        # === OPTIMIZED REWARD COMPONENTS ===
        
        # 1. Waiting Time Component (Primary objective - minimize delays)
        if total_vehicles > 0:
            avg_waiting_per_vehicle = total_waiting / total_vehicles
            # Use exponential penalty for high waiting times (research-backed)
            waiting_penalty = -2.0 * (1 - np.exp(-avg_waiting_per_vehicle / 30.0))
        else:
            waiting_penalty = 0.0
        
        # 2. Queue Length Component (Congestion management)
        if lane_count > 0:
            avg_queue_per_lane = total_queue_length / lane_count
            # Progressive penalty that increases sharply with queue length
            queue_penalty = -1.5 * np.tanh(avg_queue_per_lane / 8.0)
        else:
            queue_penalty = 0.0
        
        # 3. Speed Component (Traffic flow efficiency)
        # Reward higher speeds but penalize speed variance (smoother flow)
        target_speed = 11.11  # 40 km/h in m/s
        speed_efficiency = min(avg_speed / target_speed, 1.0)
        speed_smoothness = 1.0 / (1.0 + speed_variance / 10.0)  # Penalize high variance
        speed_reward = 1.0 * speed_efficiency * speed_smoothness
        
        # 4. Passenger Throughput Component (PRIMARY OBJECTIVE)
        # Reward passenger throughput over vehicle throughput - this is our main research objective
        passenger_throughput_reward = step_passenger_throughput * 0.5  # High weight for passenger throughput
        vehicle_throughput_bonus = step_throughput * 0.1              # Lower weight for vehicle count
        
        # 5. System Efficiency Component (Overall network performance)
        # Reward balanced utilization across lanes
        if lane_count > 0 and total_queue_length > 0:
            queue_distribution = []
            for tl_id in self.traffic_lights:
                for lane_id in self.controlled_lanes[tl_id]:
                    queue = traci.lane.getLastStepHaltingNumber(lane_id)
                    queue_distribution.append(queue)
            
            queue_std = np.std(queue_distribution) if len(queue_distribution) > 1 else 0
            balance_reward = 0.3 * (1.0 / (1.0 + queue_std / 5.0))
        else:
            balance_reward = 0.3
        
        # 6. Phase Change Penalty (Discourage frequent switching)
        phase_change_penalty = 0.0
        if hasattr(self, 'last_actions') and hasattr(self, 'current_step'):
            if self.current_step > 0:
                # Small penalty for changing phases too frequently
                for tl_id in self.traffic_lights:
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    if hasattr(self, f'last_phase_{tl_id}'):
                        last_phase = getattr(self, f'last_phase_{tl_id}')
                        if current_phase != last_phase:
                            phase_change_penalty -= 0.1
                    setattr(self, f'last_phase_{tl_id}', current_phase)
        
        # === WEIGHTED COMBINATION (Passenger-Throughput Optimized) ===
        reward = (
            waiting_penalty * 0.25 +              # 25% - Waiting time (reduced)
            queue_penalty * 0.15 +                # 15% - Congestion control (reduced)
            speed_reward * 0.15 +                 # 15% - Flow efficiency (reduced)
            passenger_throughput_reward * 0.35 +  # 35% - PRIMARY OBJECTIVE: Passenger throughput
            vehicle_throughput_bonus * 0.05 +     # 5% - Vehicle throughput bonus
            balance_reward * 0.05 +               # 5% - System balance (reduced)
            phase_change_penalty                  # Variable penalty for stability
        )
        
        # Dynamic baseline that adapts to traffic density
        traffic_density = total_vehicles / max(lane_count, 1)
        adaptive_baseline = 0.2 + 0.1 * min(traffic_density / 10.0, 1.0)
        reward += adaptive_baseline
        
        # Store reward components for analysis
        if not hasattr(self, 'reward_components'):
            self.reward_components = []
        
        self.reward_components.append({
            'step': self.current_step,
            'waiting_penalty': waiting_penalty,
            'queue_penalty': queue_penalty,
            'speed_reward': speed_reward,
            'passenger_throughput_reward': passenger_throughput_reward,
            'vehicle_throughput_bonus': vehicle_throughput_bonus,
            'balance_reward': balance_reward,
            'total_reward': reward,
            'avg_waiting': avg_waiting_per_vehicle if total_vehicles > 0 else 0,
            'avg_queue': avg_queue_per_lane if lane_count > 0 else 0,
            'avg_speed': avg_speed,
            'vehicle_throughput': step_throughput,
            'passenger_throughput': step_passenger_throughput
        })
        
        return reward
    
    def _is_done(self):
        """Check if episode should terminate"""
        # Episode ends when simulation time is reached or no vehicles remain
        time_limit = self.current_step * self.step_length >= self.num_seconds
        no_vehicles = traci.vehicle.getIDCount() == 0 and self.current_step > self.warmup_time / self.step_length
        
        return time_limit or no_vehicles
    
    def _update_metrics(self):
        """Update performance metrics"""
        current_vehicles = traci.vehicle.getIDCount()
        self.metrics['total_vehicles'] = current_vehicles
        
        # Track completed trips (vehicles that have left the simulation)
        # Get list of vehicles that arrived at their destination this step
        arrived_vehicles = traci.simulation.getArrivedIDList()
        if arrived_vehicles:
            if 'completed_trips' not in self.metrics:
                self.metrics['completed_trips'] = 0
            if 'passenger_throughput' not in self.metrics:
                self.metrics['passenger_throughput'] = 0
                
            self.metrics['completed_trips'] += len(arrived_vehicles)
            
            # Calculate passenger throughput based on vehicle types
            for veh_id in arrived_vehicles:
                try:
                    # Get vehicle type from SUMO
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    # Add passenger capacity for this vehicle type
                    passenger_count = self.passenger_capacity.get(veh_type, 1.0)
                    self.metrics['passenger_throughput'] += passenger_count
                except:
                    # Fallback to average passenger count if vehicle type not found
                    self.metrics['passenger_throughput'] += 1.5
        
        if current_vehicles > 0:
            # Calculate average speed
            speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in traci.vehicle.getIDList()]
            self.metrics['avg_speed'] = sum(speeds) / len(speeds)
            
            # Calculate total waiting time
            waiting_times = 0
            for tl_id in self.traffic_lights:
                for lane_id in self.controlled_lanes[tl_id]:
                    waiting_times += traci.lane.getWaitingTime(lane_id)
            self.metrics['total_waiting_time'] = waiting_times
    
    def close(self):
        """Close the environment"""
        if traci.isLoaded():
            print("🛑 Closing SUMO simulation...")
            traci.close()
        sys.stdout.flush()
    
    def render(self):
        """Render environment (already handled by SUMO GUI if enabled)"""
        if not self.use_gui:
            print(f"Step {self.current_step}: {self.metrics['total_vehicles']} vehicles, "
                  f"Avg speed: {self.metrics['avg_speed']:.1f} m/s, "
                  f"Total reward: {self.total_reward:.2f}")
    
    def get_info(self):
        """Get current simulation information"""
        return {
            'current_step': self.current_step,
            'warmup_complete': self.warmup_complete,
            'traffic_lights': self.traffic_lights,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'metrics': self.metrics
        }
