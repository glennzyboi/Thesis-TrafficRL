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
                 warmup_time=300, step_length=1.0, min_phase_time=8, max_phase_time=90):
        """
        Initialize the traffic environment with realistic traffic signal constraints
        
        Args:
            net_file: Path to SUMO network file (.net.xml)
            rou_file: Path to SUMO route file (.rou.xml) or list of route files for MARL
            use_gui: Whether to show SUMO GUI for visualization
            num_seconds: Total simulation duration
            warmup_time: Time before agent control starts
            step_length: Simulation step size in seconds
            min_phase_time: Minimum green/red phase duration (10s - safety standard)
            max_phase_time: Maximum green/red phase duration (120s - efficiency standard)
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
        
        # Realistic traffic signal timing constraints (based on traffic engineering standards)
        self.min_phase_time = min_phase_time  # 10 seconds minimum (safety requirement)
        self.max_phase_time = max_phase_time  # 120 seconds maximum (efficiency requirement)
        
        # Simulation state
        self.current_step = 0
        self.warmup_complete = False
        self.total_reward = 0
        
        # Traffic signal control with timing constraints
        self.traffic_lights = []
        self.tl_phases = {}  # Store available phases for each TL
        self.controlled_lanes = {}  # Lanes controlled by each TL
        self.phase_timers = {}  # Track how long current phase has been active
        self.last_phase_change = {}  # Track when phase was last changed
        self.current_phases = {}  # Track current phase for each traffic light
        
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
        """Apply the action to traffic lights with realistic timing constraints"""
        # For simplicity, apply same action to all traffic lights
        # In practice, you might want different actions for each intersection
        action = int(action) % self.action_size
        
        for tl_id in self.traffic_lights:
            self._apply_action_to_tl(tl_id, action)
    
    def _apply_action_to_tl(self, tl_id, action):
        """Apply action to a specific traffic light with realistic timing constraints"""
        action = int(action) % self.action_size
        
        # Make sure action is valid for this traffic light
        max_phase = self.tl_phases[tl_id] - 1
        desired_phase = min(action, max_phase)
        
        # Initialize phase tracking if not exists
        if tl_id not in self.current_phases:
            self.current_phases[tl_id] = traci.trafficlight.getPhase(tl_id)
            self.phase_timers[tl_id] = 0
            self.last_phase_change[tl_id] = 0
        
        current_phase = self.current_phases[tl_id]
        time_in_current_phase = self.current_step - self.last_phase_change[tl_id]
        
        # Apply realistic timing constraints
        can_change_phase = True
        
        # Minimum phase time constraint (safety requirement)
        if time_in_current_phase < self.min_phase_time:
            can_change_phase = False
            
        # Maximum phase time constraint (efficiency requirement)
        # Force change if phase has been active too long
        if time_in_current_phase >= self.max_phase_time:
            can_change_phase = True
            # If agent wants same phase, force to next phase for efficiency
            if desired_phase == current_phase:
                desired_phase = (current_phase + 1) % (max_phase + 1)
        
        # Public transport priority: Check if buses/jeepneys are waiting
        if self._has_priority_vehicles_waiting(tl_id, desired_phase):
            # Override constraints for public transport (reduce minimum time)
            if time_in_current_phase >= max(5, self.min_phase_time // 2):
                can_change_phase = True
        
        # Apply phase change only if constraints allow
        if can_change_phase and desired_phase != current_phase:
            # Avoid giving green to empty lanes
            if not self._is_lane_empty_for_phase(tl_id, desired_phase):
                traci.trafficlight.setPhase(tl_id, desired_phase)
                self.current_phases[tl_id] = desired_phase
                self.last_phase_change[tl_id] = self.current_step
                self.phase_timers[tl_id] = 0
            else:
                # Find next non-empty phase or keep current
                for next_phase in range(max_phase + 1):
                    if not self._is_lane_empty_for_phase(tl_id, next_phase) and next_phase != current_phase:
                        traci.trafficlight.setPhase(tl_id, next_phase)
                        self.current_phases[tl_id] = next_phase
                        self.last_phase_change[tl_id] = self.current_step
                        self.phase_timers[tl_id] = 0
                        break
        
        # Update timer
        self.phase_timers[tl_id] += 1
    
    def _has_priority_vehicles_waiting(self, tl_id, phase):
        """Check if public transport vehicles (buses/jeepneys) are waiting for this phase"""
        try:
            # Get lanes controlled by this phase
            controlled_lanes = self.controlled_lanes.get(tl_id, [])
            
            for lane_id in controlled_lanes:
                # Get vehicles on this lane
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                for veh_id in vehicles:
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    # Priority for buses and jeepneys (public transport)
                    if veh_type in ['bus', 'jeepney']:
                        # Check if vehicle is stopped/slow (indicating waiting)
                        speed = traci.vehicle.getSpeed(veh_id)
                        if speed < 2.0:  # Speed less than 2 m/s indicates waiting
                            return True
            return False
        except:
            return False
    
    def _is_lane_empty_for_phase(self, tl_id, phase):
        """Check if lanes controlled by this phase are empty (no vehicles waiting)"""
        try:
            # Get lanes controlled by this phase
            controlled_lanes = self.controlled_lanes.get(tl_id, [])
            
            if not controlled_lanes:
                return True
            
            total_vehicles = 0
            for lane_id in controlled_lanes:
                # Count vehicles on approaching lanes
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                total_vehicles += len(vehicles)
            
            # Consider empty if less than 2 vehicles total
            return total_vehicles < 2
        except:
            return False
    
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
        # Use vehicle tracking from flow names to avoid querying departed vehicles
        for veh_id in arrived_vehicles:
            try:
                # Extract vehicle type from flow ID pattern (flow_vehicletype_X.Y)
                if 'flow_' in veh_id:
                    veh_id_lower = veh_id.lower()
                    if 'bus' in veh_id_lower:
                        step_passenger_throughput += 15.0  # Bus capacity
                    elif 'jeepney' in veh_id_lower:
                        step_passenger_throughput += 12.0  # Jeepney capacity
                    elif 'truck' in veh_id_lower:
                        step_passenger_throughput += 1.0   # Truck (goods)
                    elif 'motor' in veh_id_lower:
                        step_passenger_throughput += 1.2   # Motorcycle
                    elif 'car' in veh_id_lower:
                        step_passenger_throughput += 1.5   # Car
                    else:
                        step_passenger_throughput += 1.5   # Default fallback
                else:
                    step_passenger_throughput += 1.5  # Default fallback
            except:
                step_passenger_throughput += 1.5  # Safe fallback
        
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
            waiting_penalty = -4.0 * (1 - np.exp(-avg_waiting_per_vehicle / 20.0))  # More aggressive penalty
        else:
            waiting_penalty = 0.0
        
        # 2. Queue Length Component (Congestion management)
        if lane_count > 0:
            avg_queue_per_lane = total_queue_length / lane_count
            # Progressive penalty that increases sharply with queue length
            queue_penalty = -2.5 * np.tanh(avg_queue_per_lane / 6.0)  # More aggressive queue penalty
        else:
            queue_penalty = 0.0
        
        # 3. Speed Component (Traffic flow efficiency)
        # Reward higher speeds but penalize speed variance (smoother flow)
        target_speed = 11.11  # 40 km/h in m/s
        speed_efficiency = min(avg_speed / target_speed, 1.0)
        speed_smoothness = 1.0 / (1.0 + speed_variance / 10.0)  # Penalize high variance
        speed_reward = 1.0 * speed_efficiency * speed_smoothness
        
        # 4. Balanced Throughput Components (Vehicle + Passenger Optimization)
        # Balance passenger and vehicle throughput to address degradation issue
        passenger_throughput_reward = step_passenger_throughput * 0.3  # Reduced weight for passenger throughput
        
        # Enhanced vehicle throughput calculation
        base_vehicle_bonus = step_throughput * 0.5  # Increased base weight
        # Add throughput rate bonus (vehicles per time step)
        rate_bonus = min(step_throughput * 0.2, 5.0) if step_throughput > 0 else 0
        vehicle_throughput_bonus = base_vehicle_bonus + rate_bonus
        
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
        
        # === PUBLIC TRANSPORT PRIORITY BONUS ===
        public_transport_bonus = self._calculate_public_transport_bonus()
        
        # === BALANCED WEIGHTED COMBINATION (Vehicle + Passenger Optimized with PT Priority) ===
        # Rebalanced to address vehicle throughput degradation while maintaining passenger optimization
        reward = (
            waiting_penalty * 0.20 +              # 20% - Waiting time penalty
            queue_penalty * 0.15 +                # 15% - Congestion control
            speed_reward * 0.20 +                 # 20% - Flow efficiency
            passenger_throughput_reward * 0.20 +  # 20% - Passenger throughput (balanced)
            vehicle_throughput_bonus * 0.15 +     # 15% - Vehicle throughput (increased significantly)
            public_transport_bonus * 0.10 +       # 10% - Public transport priority (doubled weight)
            balance_reward * 0.00 +               # 0% - System balance (reduced complexity)
            phase_change_penalty                  # Variable penalty for stability
        )
        
        # Dynamic baseline that adapts to traffic density
        traffic_density = total_vehicles / max(lane_count, 1)
        adaptive_baseline = 0.2 + 0.1 * min(traffic_density / 10.0, 1.0)
        reward += adaptive_baseline
        
        # Store reward components for analysis
        if not hasattr(self, 'reward_components'):
            self.reward_components = []
        
        # Calculate public transport specific metrics
        pt_metrics = self._calculate_pt_performance_metrics()
        
        self.reward_components.append({
            'step': self.current_step,
            'waiting_penalty': waiting_penalty,
            'queue_penalty': queue_penalty,
            'speed_reward': speed_reward,
            'passenger_throughput_reward': passenger_throughput_reward,
            'vehicle_throughput_bonus': vehicle_throughput_bonus,
            'public_transport_bonus': public_transport_bonus,
            'balance_reward': balance_reward,
            'total_reward': reward,
            'avg_waiting': avg_waiting_per_vehicle if total_vehicles > 0 else 0,
            'avg_queue': avg_queue_per_lane if lane_count > 0 else 0,
            'avg_speed': avg_speed,
            'vehicle_throughput': step_throughput,
            'passenger_throughput': step_passenger_throughput,
            # Enhanced public transport metrics
            'buses_processed': pt_metrics['buses_processed'],
            'jeepneys_processed': pt_metrics['jeepneys_processed'],
            'pt_passenger_throughput': pt_metrics['pt_passenger_throughput'],
            'pt_avg_waiting': pt_metrics['pt_avg_waiting'],
            'pt_service_efficiency': pt_metrics['pt_service_efficiency']
        })
        
        return reward
    
    def _calculate_public_transport_bonus(self):
        """Calculate bonus reward for efficiently handling public transport vehicles"""
        bonus = 0.0
        
        try:
            # Get all vehicles in the simulation
            all_vehicles = traci.vehicle.getIDList()
            
            pt_vehicles_served = 0
            pt_vehicles_waiting = 0
            
            for veh_id in all_vehicles:
                try:
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    
                    # Focus on public transport (buses and jeepneys)
                    if veh_type in ['bus', 'jeepney']:
                        speed = traci.vehicle.getSpeed(veh_id)
                        waiting_time = traci.vehicle.getWaitingTime(veh_id)
                        
                        # Bonus for moving public transport
                        if speed > 5.0:  # Moving well (> 5 m/s = 18 km/h)
                            bonus += 2.0  # High bonus for moving PT
                        elif speed > 2.0:  # Moving moderately
                            bonus += 1.0  # Moderate bonus
                            pt_vehicles_served += 1
                        else:  # Waiting or slow
                            pt_vehicles_waiting += 1
                            # Penalty for long waiting times
                            if waiting_time > 30:  # More than 30 seconds waiting
                                bonus -= 1.5
                            elif waiting_time > 15:  # More than 15 seconds waiting
                                bonus -= 0.5
                
                except:
                    continue
            
            # Additional bonus for ratio of served vs waiting PT vehicles
            total_pt = pt_vehicles_served + pt_vehicles_waiting
            if total_pt > 0:
                service_ratio = pt_vehicles_served / total_pt
                bonus += service_ratio * 3.0  # Up to 3.0 bonus for high service ratio
            
            # Normalize bonus to reasonable range
            bonus = max(-5.0, min(10.0, bonus))
            
        except:
            bonus = 0.0
        
        return bonus
    
    def _calculate_pt_performance_metrics(self):
        """
        Calculate detailed public transport performance metrics for evaluation
        Based on research standards for MARL traffic control evaluation
        """
        metrics = {
            'buses_processed': 0,
            'jeepneys_processed': 0,
            'pt_passenger_throughput': 0.0,
            'pt_avg_waiting': 0.0,
            'pt_service_efficiency': 0.0
        }
        
        try:
            all_vehicles = traci.vehicle.getIDList()
            departed_vehicles = traci.simulation.getDepartedIDList()
            
            # Count currently active PT vehicles
            active_buses = 0
            active_jeepneys = 0
            total_pt_waiting = 0.0
            pt_count = 0
            
            for veh_id in all_vehicles:
                try:
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    if veh_type == 'bus':
                        active_buses += 1
                        pt_count += 1
                        total_pt_waiting += traci.vehicle.getWaitingTime(veh_id)
                    elif veh_type == 'jeepney':
                        active_jeepneys += 1
                        pt_count += 1
                        total_pt_waiting += traci.vehicle.getWaitingTime(veh_id)
                except:
                    continue
            
            # Count completed PT trips this step
            step_buses_completed = 0
            step_jeepneys_completed = 0
            step_pt_passengers = 0.0
            
            for veh_id in departed_vehicles:
                try:
                    # Extract vehicle type from flow ID pattern
                    if '_bus_' in veh_id:
                        step_buses_completed += 1
                        step_pt_passengers += 40.0  # Average bus capacity
                    elif '_jeepney_' in veh_id:
                        step_jeepneys_completed += 1
                        step_pt_passengers += 16.0  # Average jeepney capacity
                except:
                    continue
            
            # Update cumulative metrics
            if not hasattr(self, 'cumulative_pt_metrics'):
                self.cumulative_pt_metrics = {
                    'total_buses': 0,
                    'total_jeepneys': 0,
                    'total_pt_passengers': 0.0
                }
            
            self.cumulative_pt_metrics['total_buses'] += step_buses_completed
            self.cumulative_pt_metrics['total_jeepneys'] += step_jeepneys_completed
            self.cumulative_pt_metrics['total_pt_passengers'] += step_pt_passengers
            
            # Calculate efficiency: moving vehicles / total vehicles
            moving_pt = sum(1 for veh_id in all_vehicles 
                           if traci.vehicle.getTypeID(veh_id) in ['bus', 'jeepney'] 
                           and traci.vehicle.getSpeed(veh_id) > 2.0)
            
            pt_service_efficiency = moving_pt / max(pt_count, 1) if pt_count > 0 else 1.0
            
            metrics = {
                'buses_processed': self.cumulative_pt_metrics['total_buses'],
                'jeepneys_processed': self.cumulative_pt_metrics['total_jeepneys'], 
                'pt_passenger_throughput': self.cumulative_pt_metrics['total_pt_passengers'],
                'pt_avg_waiting': total_pt_waiting / max(pt_count, 1),
                'pt_service_efficiency': pt_service_efficiency
            }
            
        except:
            pass
        
        return metrics
    
    def _is_done(self):
        """Check if episode should terminate"""
        # Episode ends when simulation time is reached or no vehicles remain
        # FIXED: Account for warmup time in total episode duration
        total_episode_time = self.warmup_time + self.num_seconds
        time_limit = self.current_step * self.step_length >= total_episode_time
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
            # Use safer vehicle type estimation from flow names
            for veh_id in arrived_vehicles:
                try:
                    # Extract vehicle type from flow ID pattern (flow_vehicletype_X.Y)
                    if 'flow_' in veh_id:
                        veh_id_lower = veh_id.lower()
                        if 'bus' in veh_id_lower:
                            self.metrics['passenger_throughput'] += 15.0  # Bus capacity
                        elif 'jeepney' in veh_id_lower:
                            self.metrics['passenger_throughput'] += 12.0  # Jeepney capacity
                        elif 'truck' in veh_id_lower:
                            self.metrics['passenger_throughput'] += 1.0   # Truck (goods)
                        elif 'motor' in veh_id_lower:
                            self.metrics['passenger_throughput'] += 1.2   # Motorcycle
                        elif 'car' in veh_id_lower:
                            self.metrics['passenger_throughput'] += 1.5   # Car
                        else:
                            self.metrics['passenger_throughput'] += 1.5   # Default
                    else:
                        self.metrics['passenger_throughput'] += 1.5   # Default fallback
                except:
                    # Safe fallback to average passenger count
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
    
    # MARL Methods
    def get_marl_states(self):
        """Get states for all agents in MARL setup"""
        states = {}
        for tl_id in self.traffic_lights:
            states[tl_id] = self._get_agent_state(tl_id)
        return states
    
    def _get_agent_state(self, tl_id):
        """Get state for a specific agent/intersection"""
        state = []
        
        # Get controlled lanes for this traffic light
        controlled_lanes = self.controlled_lanes.get(tl_id, [])
        
        # Traffic metrics for each lane
        for lane_id in controlled_lanes:
            try:
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                waiting_time = traci.lane.getWaitingTime(lane_id) / 100.0  # Normalize
                occupancy = traci.lane.getLastStepOccupancy(lane_id)
                avg_speed = traci.lane.getLastStepMeanSpeed(lane_id) / 13.89  # Normalize to 0-1
                
                state.extend([queue_length, waiting_time, occupancy, avg_speed])
            except:
                state.extend([0, 0, 0, 0])
        
        # Current signal phase
        try:
            current_phase = traci.trafficlight.getPhase(tl_id)
            phase_duration = traci.trafficlight.getPhaseDuration(tl_id)
            
            # One-hot encode phase (assuming max 11 phases)
            phase_encoded = [0] * 11
            if current_phase < 11:
                phase_encoded[current_phase] = 1
            
            state.extend(phase_encoded)
            state.append(phase_duration / 60.0)  # Normalize phase duration
        except:
            state.extend([0] * 12)
        
        # Pad state to fixed size (159 features)
        while len(state) < 159:
            state.append(0)
        
        return np.array(state[:159], dtype=np.float32)
    
    def step_marl(self, actions):
        """Step with MARL actions"""
        # Apply actions for each traffic light
        for tl_id, action in actions.items():
            if tl_id in self.traffic_lights:
                self._apply_action_to_tl(tl_id, action)
        
        # Step simulation
        traci.simulationStep()
        self.current_step += 1
        
        # Calculate rewards and get new states
        states = self.get_marl_states()
        rewards = self._calculate_marl_rewards()
        
        # Check if episode is done
        # FIXED: Account for warmup time in total episode duration
        total_episode_steps = (self.warmup_time + self.num_seconds) / self.step_length
        done = (self.current_step >= total_episode_steps or 
                traci.simulation.getMinExpectedNumber() <= 0)
        
        # Update metrics
        self._update_metrics()
        
        # Enhanced info for MARL training compatibility
        info = {
            'step': self.current_step,
            'vehicles': self.metrics.get('total_vehicles', 0),  # Match single-agent format
            'completed_trips': self.metrics.get('completed_trips', 0),  # Match single-agent format
            'passenger_throughput': self.metrics.get('passenger_throughput', 0),  # Cumulative passengers
            'total_vehicles': self.metrics.get('total_vehicles', 0),
            'avg_speed': self.metrics.get('avg_speed', 0),
            'throughput': self.metrics.get('completed_trips', 0) / max(self.current_step * self.step_length / 3600, 0.01),  # Vehicles per hour
        }
        
        return states, rewards, done, info
    
    def _calculate_marl_rewards(self):
        """Calculate rewards for each agent"""
        rewards = {}
        
        # Calculate passenger throughput based on vehicle origins to avoid double counting
        passenger_throughput = 0
        intersection_entry_edges = {
            'Ecoland_TrafficSignal': ['106768821', '-794461797#2', '770761758#0', '-24224169#2'],
            'JohnPaul_TrafficSignal': ['1046997839#6', '869986417#1', '935563495#2'], 
            'Sandawa_TrafficSignal': ['1042538762#0', '934492020#7']
        }
        
        all_vehicles = traci.vehicle.getIDList()
        for veh_id in all_vehicles:
            try:
                route_id = traci.vehicle.getRouteID(veh_id)
                route_edges = traci.route.getEdges(route_id)
                if route_edges:
                    origin_edge = route_edges[0]
                    for tl_id, entry_edges in intersection_entry_edges.items():
                        if origin_edge in entry_edges:
                            veh_type = traci.vehicle.getTypeID(veh_id)
                            passenger_count = self.passenger_capacity.get(veh_type, 1.5)
                            passenger_throughput += passenger_count
                            break
            except:
                passenger_throughput += 1.5
        
        # Calculate rewards for each traffic light
        for tl_id in self.traffic_lights:
            controlled_lanes = self.controlled_lanes.get(tl_id, [])
            
            # Calculate local metrics
            local_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes)
            local_waiting = sum(traci.lane.getWaitingTime(lane) for lane in controlled_lanes)
            local_speed = np.mean([traci.lane.getLastStepMeanSpeed(lane) for lane in controlled_lanes])
            
            # Multi-component reward
            queue_reward = -local_queue * 0.1
            waiting_reward = -local_waiting * 0.01  
            speed_reward = local_speed * 0.1
            throughput_reward = passenger_throughput * 0.001
            
            total_reward = queue_reward + waiting_reward + speed_reward + throughput_reward
            rewards[tl_id] = total_reward
        
        return rewards
    
    def get_agent_configs(self):
        """Get configuration for each agent"""
        configs = {}
        for tl_id in self.traffic_lights:
            configs[tl_id] = {
                'state_size': 159,
                'action_size': 11,
                'controlled_lanes': self.controlled_lanes.get(tl_id, [])
            }
        return configs
