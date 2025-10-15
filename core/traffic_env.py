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
        print("SUMO_HOME not found. Please set it manually or install SUMO.")
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
        
        # Fair lane access enforcement to prevent agent from exploiting single lanes
        self.cycle_tracking = {}  # Track phase cycles for fairness
        self.steps_since_last_cycle = {}  # Steps since complete cycle
        self.max_steps_per_cycle = 200  # Maximum steps before forced cycle completion
        # Phase-change stability controls
        self.phase_cooldown_steps = 3  # Minimal cooldown to reduce oscillations
        self.max_phase_change_penalty = 0.5  # Cap per-step penalty magnitude
        
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
        
        # DAVAO CITY-SPECIFIC PASSENGER CAPACITIES
        # Based on LTFRB standards, DOTr studies, and local transport research
        # References: JICA Davao Study (2019), LTFRB MC 2015-034, DOTr PUVMP (2017)
        self.passenger_capacity = {
            'car': 1.3,           # Davao City average (JICA 2019) - lower than national
            'motor': 1.4,         # Motorcycle + backride average (LTO data)
            'truck': 1.1,         # Driver + occasional helper
            'jeepney': 14.0,      # Traditional PUJ (LTFRB + Davao survey)
            'modern_jeepney': 22.0,  # PUVMP compliant vehicles (DOTr 2017)
            'bus': 35.0,          # City bus (Davao-specific, lower than Manila)
            'tricycle': 2.5,      # Driver + 1-2 passengers average (LTFRB)
            'default': 1.5        # Fallback for unknown types
        }
        
        print(f"Traffic Environment Initialized:")
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
        print(f"SUMO Command: {' '.join(self.sumo_cmd)}")
    
    def reset(self):
        """Reset the environment for a new episode"""
        # Close any existing simulation
        if traci.isLoaded():
            traci.close()
        
        # Start new simulation
        print(f"\nStarting new episode...")
        traci.start(self.sumo_cmd)
        
        # Reset episode variables
        self.current_step = 0
        self.warmup_complete = False
        self.total_reward = 0
        self.cumulative_throughput = 0  # CRITICAL FIX: Reset cumulative throughput
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
        # SIMPLIFIED STATE: 3 metrics per lane + 3 per intersection + 5 global = ~50 dimensions
        total_lanes = sum(len(lanes) for lanes in self.controlled_lanes.values())
        total_intersections = len(self.traffic_lights)
        self.state_size = total_lanes * 3 + total_intersections * 3 + 5  # Simplified state representation
        
        # ENHANCED ACTION SPACE: Granular traffic control
        # Two phases per traffic light (3 traffic lights x 2 phases = 6 actions)
        self.action_size = 6
        
        print(f"   State size: {self.state_size}, Action size: {self.action_size}")
        
        # Initialize phase tracking dictionaries
        self.phase_timers = {}
        self.last_phase_change = {}
        self.cycle_tracking = {}
        self.steps_since_last_cycle = {}
        self.current_phases = {}
        
        # Initialize for all traffic lights
        for tl_id in self.traffic_lights:
            self.phase_timers[tl_id] = 0
            self.last_phase_change[tl_id] = 0
            self.cycle_tracking[tl_id] = {'phases_used': set(), 'current_cycle_start': 0}
            self.steps_since_last_cycle[tl_id] = 0
            self.current_phases[tl_id] = 0
    
    def _warmup_simulation(self):
        """Run simulation warmup period without agent control"""
        print(f"   Warming up simulation ({self.warmup_time}s)...")
        print(f"   Calibration phase: All lights RED for realistic vehicle loading...")
        
        # Phase 1: Set all lights to RED for vehicle loading (first 1/3 of warmup)
        warmup_steps = int(self.warmup_time / self.step_length)
        calibration_steps = warmup_steps // 3
        
        for tl_id in self.traffic_lights:
            # Find red phase (usually phase 0 or similar)
            red_phase = self._find_red_phase(tl_id)
            traci.trafficlight.setPhase(tl_id, red_phase)
        
        for step in range(calibration_steps):
            traci.simulationStep()
            self.current_step += 1
            if step % (calibration_steps // 5) == 0:  # Progress indicator
                vehicles = len(traci.vehicle.getIDList())
                print(f"     Loading: {vehicles} vehicles in network")
        
        # Phase 2: Normal warmup with traffic light operation (remaining 2/3)
        print(f"   Traffic light operation warmup...")
        for step in range(warmup_steps - calibration_steps):
            traci.simulationStep()
            self.current_step += 1
        
        self.warmup_complete = True
        print(f"   Warmup complete - {traci.vehicle.getIDCount()} vehicles active")
    
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
        # Build per-lane metrics for realism verification
        lane_metrics = {}
        try:
            for tl_id in self.traffic_lights:
                for lane_id in self.controlled_lanes[tl_id]:
                    lane_metrics[lane_id] = {
                        'waiting_time': float(traci.lane.getWaitingTime(lane_id)),
                        'queue': int(traci.lane.getLastStepHaltingNumber(lane_id)),
                        'speed_kmh': float(traci.lane.getLastStepMeanSpeed(lane_id) * 3.6)
                    }
        except Exception:
            pass

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
            'metrics': self.metrics.copy(),
            'lane_metrics': lane_metrics
        }
        
        return next_state, reward, done, info
    
    def _apply_action(self, action):
        """
        Apply enhanced action with intelligent traffic control
        
        Enhanced Action Space (6 actions):
        Actions 0-1: Ecoland_TrafficSignal (phase 0, phase 2)
        Actions 2-3: JohnPaul_TrafficSignal (phase 0, phase 5)  
        Actions 4-5: Sandawa_TrafficSignal (phase 0, phase 2)
        
        Enhanced with intelligent phase selection and coordination
        """
        action = int(action) % self.action_size
        
        # Map action to traffic light and phase
        tl_ids = list(self.traffic_lights)
        
        if action < 2:  # Ecoland_TrafficSignal
            tl_id = tl_ids[0]
            target_phase = 0 if action == 0 else 2
        elif action < 4:  # JohnPaul_TrafficSignal
            tl_id = tl_ids[1]
            target_phase = 0 if action == 2 else 5
        else:  # Sandawa_TrafficSignal
            tl_id = tl_ids[2]
            target_phase = 0 if action == 4 else 2
        
        # Apply intelligent phase change with coordination
        self._apply_intelligent_phase_change(tl_id, target_phase)
    
    def _apply_intelligent_phase_change(self, tl_id, target_phase):
        """Apply intelligent phase change with coordination and PT priority"""
        if tl_id not in self.traffic_lights:
            return
        
        # Check minimum phase time constraint
        if self.phase_timers[tl_id] < self.min_phase_time:
            return  # Don't change phase too quickly
        # Cooldown to prevent rapid oscillation
        time_since_change = self.current_step - self.last_phase_change.get(tl_id, 0)
        if time_since_change < self.phase_cooldown_steps:
            return
        
        # Check for PT vehicles and prioritize
        if self._has_pt_vehicles_waiting(tl_id):
            # Find best phase for PT vehicles
            best_phase = self._find_best_pt_phase(tl_id)
            if best_phase is not None:
                target_phase = best_phase
        
        # Apply coordination with nearby intersections
        self._apply_coordination_effects(tl_id, target_phase)
        
        # Set the phase
        traci.trafficlight.setPhase(tl_id, target_phase)
        self.current_phases[tl_id] = target_phase
        self.phase_timers[tl_id] = 0
    
    def _find_best_pt_phase(self, tl_id):
        """Find the best phase for PT vehicles at this intersection"""
        best_phase = None
        max_pt_benefit = 0
        
        for phase in range(self.tl_phases[tl_id]):
            pt_benefit = self._calculate_pt_phase_benefit(tl_id, phase)
            if pt_benefit > max_pt_benefit:
                max_pt_benefit = pt_benefit
                best_phase = phase
        
        return best_phase
    
    def _calculate_pt_phase_benefit(self, tl_id, phase):
        """Calculate how much PT vehicles benefit from this phase"""
        benefit = 0
        for lane_id in self.controlled_lanes[tl_id]:
            try:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                for veh_id in vehicle_ids:
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    if veh_type in ['bus', 'jeepney']:
                        # Higher benefit for vehicles waiting longer
                        waiting_time = traci.vehicle.getWaitingTime(veh_id)
                        benefit += waiting_time * 2  # Double weight for PT vehicles
            except:
                pass
        return benefit
    
    def _apply_coordination_effects(self, tl_id, target_phase):
        """Apply coordination effects with nearby intersections"""
        # Simple coordination: if changing to green, check if nearby intersections
        # should also change to avoid conflicts
        
        # Get current phases of other intersections
        other_phases = {}
        for other_tl_id in self.traffic_lights:
            if other_tl_id != tl_id:
                other_phases[other_tl_id] = traci.trafficlight.getPhase(other_tl_id)
        
        # Simple coordination logic: if this intersection is going green,
        # ensure nearby intersections don't conflict
        if target_phase in [0, 2, 5]:  # Green phases
            for other_tl_id, other_phase in other_phases.items():
                # If other intersection is also green, consider coordination
                if other_phase in [0, 2, 5]:
                    # Simple delay to avoid simultaneous green changes
                    if self.phase_timers[other_tl_id] > 5:  # Only if other has been green for a while
                        pass  # Allow the change
                    # Could add more sophisticated coordination here
    
    def _apply_phase_change(self, tl_id, target_phase=None):
        """Apply phase change to specific traffic light with PT priority"""
        if tl_id not in self.traffic_lights:
            return
        
        # Check for PT vehicles and prioritize their phase
        if self._has_pt_vehicles_waiting(tl_id):
            # Find phase that serves PT vehicles
            for phase in range(self.tl_phases[tl_id]):
                if self._pt_vehicles_in_phase(tl_id, phase):
                    traci.trafficlight.setPhase(tl_id, phase)
                    self.current_phases[tl_id] = phase
                    self.phase_timers[tl_id] = 0
                    return
        
        # Use target phase if provided, otherwise normal phase change
        if target_phase is not None:
            traci.trafficlight.setPhase(tl_id, target_phase)
            self.current_phases[tl_id] = target_phase
        else:
            # Normal phase change
            current_phase = traci.trafficlight.getPhase(tl_id)
            next_phase = (current_phase + 1) % self.tl_phases[tl_id]
            traci.trafficlight.setPhase(tl_id, next_phase)
            self.current_phases[tl_id] = next_phase
        
        self.phase_timers[tl_id] = 0
    
    def _has_pt_vehicles_waiting(self, tl_id):
        """Check if PT vehicles are waiting at this traffic light"""
        for lane_id in self.controlled_lanes[tl_id]:
            try:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                for veh_id in vehicle_ids:
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    if veh_type in ['bus', 'jeepney']:
                        return True
            except:
                pass
        return False
    
    def _pt_vehicles_in_phase(self, tl_id, phase):
        """Check if PT vehicles are in the given phase"""
        # This is a simplified check - in practice, you'd need to map phases to lanes
        return self._has_pt_vehicles_waiting(tl_id)
    
    def _apply_phase_control_action(self, action):
        """Apply phase control action to all traffic lights"""
        for tl_id in self.traffic_lights:
            if action == 0:  # extend_current_phase
                self._extend_current_phase(tl_id)
            elif action == 1:  # normal_phase_change
                self._normal_phase_change(tl_id)
            elif action == 2:  # quick_phase_change
                self._quick_phase_change(tl_id)
            elif action == 3:  # skip_to_next_phase
                self._skip_to_next_phase(tl_id)
            elif action == 4:  # emergency_clearance
                self._emergency_clearance(tl_id)
            elif action == 5:  # maintain_current_phase
                self._maintain_current_phase(tl_id)
    
    def _apply_coordination_action(self, coord_action):
        """Apply coordination action between intersections"""
        if coord_action == 0:  # coordinate_ecoland_johnpaul
            self._coordinate_intersections(['Ecoland_TrafficSignal', 'JohnPaul_TrafficSignal'])
        elif coord_action == 1:  # coordinate_johnpaul_sandawa
            self._coordinate_intersections(['JohnPaul_TrafficSignal', 'Sandawa_TrafficSignal'])
        elif coord_action == 2:  # coordinate_ecoland_sandawa
            self._coordinate_intersections(['Ecoland_TrafficSignal', 'Sandawa_TrafficSignal'])
        elif coord_action == 3:  # independent_control
            self._independent_control()
        elif coord_action == 4:  # priority_flow_direction
            self._priority_flow_direction()
        elif coord_action == 5:  # congestion_relief_mode
            self._congestion_relief_mode()
    
    def _extend_current_phase(self, tl_id):
        """Extend current phase if beneficial (longer green for busy lanes)"""
        current_phase = traci.trafficlight.getPhase(tl_id)
        # Check if current phase has vehicles and extend if so
        if not self._is_lane_empty_for_phase(tl_id, current_phase):
            # Extend by keeping current phase longer
            self.phase_timers[tl_id] = max(0, self.phase_timers[tl_id] - 5)  # Reset timer
    
    def _normal_phase_change(self, tl_id):
        """Normal phase change with standard timing"""
        self._apply_action_to_tl(tl_id, 1)  # Standard phase change
    
    def _quick_phase_change(self, tl_id):
        """Quick phase change for congestion relief"""
        current_phase = traci.trafficlight.getPhase(tl_id)
        next_phase = (current_phase + 1) % (self.tl_phases[tl_id])
        traci.trafficlight.setPhase(tl_id, next_phase)
        self.current_phases[tl_id] = next_phase
        self.phase_timers[tl_id] = 0
    
    def _skip_to_next_phase(self, tl_id):
        """Skip to next phase for congestion relief"""
        current_phase = traci.trafficlight.getPhase(tl_id)
        next_phase = (current_phase + 2) % (self.tl_phases[tl_id])
        traci.trafficlight.setPhase(tl_id, next_phase)
        self.current_phases[tl_id] = next_phase
        self.phase_timers[tl_id] = 0
    
    def _emergency_clearance(self, tl_id):
        """Emergency clearance for PT vehicles"""
        # Check for PT vehicles and prioritize their phase
        for phase in range(self.tl_phases[tl_id]):
            if self._has_priority_vehicles_waiting(tl_id, phase):
                traci.trafficlight.setPhase(tl_id, phase)
                self.current_phases[tl_id] = phase
                self.phase_timers[tl_id] = 0
                break
    
    def _maintain_current_phase(self, tl_id):
        """Maintain current phase for stability"""
        # Do nothing - keep current phase
        pass
    
    def _coordinate_intersections(self, tl_ids):
        """Coordinate specific intersections"""
        # Set coordination mode
        self.coordination_mode = True
        
        # Apply synchronized phases
        for i, tl_id in enumerate(tl_ids):
            if tl_id in self.traffic_lights:
                # Use offset phases for coordination
                phase_offset = i * 2
                desired_phase = phase_offset % self.tl_phases[tl_id]
                traci.trafficlight.setPhase(tl_id, desired_phase)
                self.current_phases[tl_id] = desired_phase
                self.phase_timers[tl_id] = 0
    
    def _independent_control(self):
        """Independent control for each intersection"""
        self.coordination_mode = False
        # Apply normal phase change to each intersection
        for tl_id in self.traffic_lights:
            self._normal_phase_change(tl_id)
    
    def _priority_flow_direction(self):
        """Priority flow direction for PT vehicles"""
        # Find intersection with most PT vehicles and prioritize it
        max_pt_vehicles = 0
        priority_tl = None
        
        for tl_id in self.traffic_lights:
            pt_count = 0
            for lane_id in self.controlled_lanes[tl_id]:
                try:
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in vehicle_ids:
                        if traci.vehicle.getTypeID(veh_id) in ['bus', 'jeepney']:
                            pt_count += 1
                except:
                    pass
            
            if pt_count > max_pt_vehicles:
                max_pt_vehicles = pt_count
                priority_tl = tl_id
        
        if priority_tl:
            self._emergency_clearance(priority_tl)
    
    def _congestion_relief_mode(self):
        """System-wide congestion relief"""
        # Find most congested intersection and apply quick change
        max_queue = 0
        congested_tl = None
        
        for tl_id in self.traffic_lights:
            total_queue = 0
            for lane_id in self.controlled_lanes[tl_id]:
                total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
            
            if total_queue > max_queue:
                max_queue = total_queue
                congested_tl = tl_id
        
        if congested_tl:
            self._quick_phase_change(congested_tl)
    
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
            self.cycle_tracking[tl_id] = {'phases_used': set(), 'current_cycle_start': 0}
            self.steps_since_last_cycle[tl_id] = 0
        
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
        
        # Fair lane access enforcement - prevent exploiting single lanes
        cycle_info = self.cycle_tracking[tl_id]
        self.steps_since_last_cycle[tl_id] += 1
        
        # Check if agent is trying to exploit by not cycling through phases
        if self.steps_since_last_cycle[tl_id] > self.max_steps_per_cycle:
            # Force completion of cycle by going to next unused phase
            unused_phases = set(range(max_phase + 1)) - cycle_info['phases_used']
            if unused_phases:
                desired_phase = min(unused_phases)  # Go to lowest unused phase
                can_change_phase = True
                print(f"   Forcing cycle completion for {tl_id} - Phase {desired_phase}")
            else:
                # Reset cycle if all phases used
                cycle_info['phases_used'] = set()
                self.steps_since_last_cycle[tl_id] = 0
                cycle_info['current_cycle_start'] = self.current_step
        
        # Public transport priority: Check if buses/jeepneys are waiting
        if self._has_priority_vehicles_waiting(tl_id, desired_phase):
            # Override constraints for public transport (reduce minimum time)
            if time_in_current_phase >= max(5, self.min_phase_time // 2):
                can_change_phase = True
        
        # Apply phase change only if constraints allow
        if can_change_phase and desired_phase != current_phase:
            # Additional cooldown constraint to reduce oscillations
            if time_in_current_phase < self.phase_cooldown_steps:
                pass
            else:
            # Avoid giving green to empty lanes
                if not self._is_lane_empty_for_phase(tl_id, desired_phase):
                    traci.trafficlight.setPhase(tl_id, desired_phase)
                    self.current_phases[tl_id] = desired_phase
                    self.last_phase_change[tl_id] = self.current_step
                    self.phase_timers[tl_id] = 0
                
                    # Track phase usage for fairness
                    cycle_info['phases_used'].add(desired_phase)
                
                    # Reset cycle if all phases have been used
                    if len(cycle_info['phases_used']) >= max_phase + 1:
                        cycle_info['phases_used'] = set()
                        self.steps_since_last_cycle[tl_id] = 0
                        cycle_info['current_cycle_start'] = self.current_step
                else:
                    # Find next non-empty phase or keep current
                    for next_phase in range(max_phase + 1):
                        if not self._is_lane_empty_for_phase(tl_id, next_phase) and next_phase != current_phase:
                            traci.trafficlight.setPhase(tl_id, next_phase)
                            self.current_phases[tl_id] = next_phase
                            self.last_phase_change[tl_id] = self.current_step
                            self.phase_timers[tl_id] = 0
                            
                            # Track phase usage for fairness
                            cycle_info['phases_used'].add(next_phase)
                            
                            # Reset cycle if all phases have been used
                            if len(cycle_info['phases_used']) >= max_phase + 1:
                                cycle_info['phases_used'] = set()
                                self.steps_since_last_cycle[tl_id] = 0
                                cycle_info['current_cycle_start'] = self.current_step
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
    
    def _find_red_phase(self, tl_id):
        """Find the red phase for a traffic light (usually phase 0)"""
        try:
            # Most SUMO networks have red phase as 0, but let's check
            phases = traci.trafficlight.getAllProgramLogics(tl_id)
            if phases:
                for i, phase in enumerate(phases[0].phases):
                    # Red phases typically have all 'r' or 'R' states
                    if all(state.lower() == 'r' for state in phase.state):
                        return i
            # Default to phase 0 if no pure red phase found
            return 0
        except Exception:
            return 0
    
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
        """
        SIMPLIFIED STATE REPRESENTATION
        Focus on essential traffic metrics that directly impact performance
        Reduced from 208 to ~50 dimensions for better learning
        """
        state = []
        
        # === CORE LANE METRICS (3 per lane) ===
        for tl_id in self.traffic_lights:
            for lane_id in self.controlled_lanes[tl_id]:
                # 1. Queue length (normalized 0-1)
                queue_length = min(traci.lane.getLastStepHaltingNumber(lane_id) / 20.0, 1.0)
                
                # 2. Waiting time (normalized 0-1)
                waiting_time = min(traci.lane.getWaitingTime(lane_id) / 100.0, 1.0)
                
                # 3. Speed efficiency (normalized 0-1)
                speed = traci.lane.getLastStepMeanSpeed(lane_id)
                speed_efficiency = min(speed / 15.0, 1.0) if speed > 0 else 0
                
                state.extend([queue_length, waiting_time, speed_efficiency])
        
        # === INTERSECTION-LEVEL METRICS (3 per intersection) ===
        for tl_id in self.traffic_lights:
            # 1. Total waiting time at intersection
            intersection_waiting = 0
            intersection_queue = 0
            for lane_id in self.controlled_lanes[tl_id]:
                intersection_waiting += traci.lane.getWaitingTime(lane_id)
                intersection_queue += traci.lane.getLastStepHaltingNumber(lane_id)
            
            # Normalize intersection metrics
            intersection_waiting = min(intersection_waiting / 200.0, 1.0)
            intersection_queue = min(intersection_queue / 50.0, 1.0)
            
            # 2. Phase efficiency (time since last phase change)
            phase_timer = self.phase_timers.get(tl_id, 0)
            phase_efficiency = min(phase_timer / 30.0, 1.0)  # Normalize to 30 steps
            
            state.extend([intersection_waiting, intersection_queue, phase_efficiency])
        
        # === GLOBAL SYSTEM METRICS (5 total) ===
        
        # 1. Total vehicles in network
        total_vehicles = traci.vehicle.getIDCount()
        system_load = min(total_vehicles / 300.0, 1.0)
        
        # 2. System throughput (vehicles completed this step)
        step_throughput = len(traci.simulation.getArrivedIDList())
        throughput_rate = min(step_throughput / 10.0, 1.0)
        
        # 3. Time of day (for traffic patterns)
        time_of_day = (self.current_step % 300) / 300.0
        
        # 4. System speed (average across all vehicles)
        speeds = []
        for veh_id in traci.vehicle.getIDList():
            speeds.append(traci.vehicle.getSpeed(veh_id))
        avg_system_speed = np.mean(speeds) if speeds else 0
        system_speed = min(avg_system_speed / 15.0, 1.0)
        
        # 5. Traffic density (vehicles per lane)
        total_lanes = sum(len(lanes) for lanes in self.controlled_lanes.values())
        density = min(total_vehicles / max(total_lanes, 1) / 5.0, 1.0)
        
        state.extend([system_load, throughput_rate, time_of_day, system_speed, density])
        
        # Convert to numpy array and ensure correct size
        state_array = np.array(state, dtype=np.float32)
        
        # Pad or truncate to expected state size
        if len(state_array) < self.state_size:
            padding = np.zeros(self.state_size - len(state_array), dtype=np.float32)
            state_array = np.concatenate([state_array, padding])
        elif len(state_array) > self.state_size:
            state_array = state_array[:self.state_size]
        
        return state_array
    
    def _extract_date_pattern(self):
        """
        Extract date pattern for LSTM temporal learning
        This helps the LSTM learn traffic patterns based on dates
        """
        # Extract date from current scenario if available
        if hasattr(self, 'current_scenario') and self.current_scenario:
            try:
                # Extract date from scenario name (e.g., "Day 20250701, Cycle 1")
                if 'Day' in self.current_scenario:
                    date_str = self.current_scenario.split('Day ')[1].split(',')[0]
                    # Convert to day of week pattern (0-6)
                    from datetime import datetime
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                    day_of_week = date_obj.weekday()  # 0=Monday, 6=Sunday
                    return day_of_week / 6.0  # Normalize to 0-1
            except:
                pass
        
        # Default: use current step as pattern
        return (self.current_step % 7) / 6.0  # Weekly pattern
    
    def _calculate_reward(self):
        """
        PERFORMANCE-ALIGNED REWARD FUNCTION
        Directly correlates rewards with actual traffic performance metrics
        Designed to beat fixed-time baseline across all key metrics
        """
        total_vehicles = traci.vehicle.getIDCount()
        
        if total_vehicles == 0:
            return 0.1  # Minimal reward for empty network
        
        # === CORE TRAFFIC METRICS ===
        total_waiting = 0
        total_queue_length = 0
        lane_count = 0
        speeds = []
        
        for tl_id in self.traffic_lights:
            for lane_id in self.controlled_lanes[tl_id]:
                waiting_time = traci.lane.getWaitingTime(lane_id)
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                total_waiting += waiting_time
                total_queue_length += queue_length
                lane_count += 1
        
        # Calculate average metrics
        avg_waiting = total_waiting / max(total_vehicles, 1)
        avg_queue = total_queue_length / max(lane_count, 1)
        
        # Speed calculation
        if total_vehicles > 0:
            for veh_id in traci.vehicle.getIDList():
                speed = traci.vehicle.getSpeed(veh_id)
                speeds.append(speed)
        avg_speed = np.mean(speeds) if speeds else 0
        
        # Throughput calculation
        arrived_vehicles = traci.simulation.getArrivedIDList()
        step_throughput = len(arrived_vehicles)
        
        # Calculate passenger throughput using Davao City-specific capacities
        step_passenger_throughput = 0
        for veh_id in arrived_vehicles:
            try:
                if 'flow_' in veh_id:
                    veh_id_lower = veh_id.lower()
                    # Map vehicle ID to Davao City passenger capacity
                    if 'bus' in veh_id_lower:
                        step_passenger_throughput += 35.0       # Davao city bus
                    elif 'jeepney' in veh_id_lower or 'jeep' in veh_id_lower:
                        step_passenger_throughput += 14.0       # Traditional PUJ
                    elif 'modern' in veh_id_lower:
                        step_passenger_throughput += 22.0       # Modern jeepney
                    elif 'motor' in veh_id_lower or 'motorcycle' in veh_id_lower:
                        step_passenger_throughput += 1.4        # Motorcycle + backride
                    elif 'truck' in veh_id_lower:
                        step_passenger_throughput += 1.1        # Driver + helper
                    elif 'tricycle' in veh_id_lower or 'trike' in veh_id_lower:
                        step_passenger_throughput += 2.5        # Tricycle
                    elif 'car' in veh_id_lower:
                        step_passenger_throughput += 1.3        # Private car
                    else:
                        step_passenger_throughput += 1.5        # Default
                else:
                    step_passenger_throughput += 1.5           # Default for unknown
            except:
                step_passenger_throughput += 1.5               # Fallback
        
        # === PERFORMANCE-ALIGNED REWARD COMPONENTS ===
        # SCOPE-ALIGNED: Prioritize throughput (thesis goal) while maintaining other metrics
        
        # 1. WAITING TIME REWARD (25% weight) - Reduced to prioritize throughput
        # Target: < 6 seconds (fixed-time baseline: 5.79s)
        if avg_waiting <= 3.0:
            waiting_reward = 10.0  # Excellent performance
        elif avg_waiting <= 6.0:
            waiting_reward = 5.0 * (6.0 - avg_waiting) / 3.0  # Good performance
        elif avg_waiting <= 10.0:
            waiting_reward = 2.0 * (10.0 - avg_waiting) / 4.0  # Acceptable
        else:
            waiting_reward = -5.0  # Poor performance
        
        # 2. THROUGHPUT REWARD (hybrid approach) - Balance immediate and cumulative throughput
        # Target: > 5,000 veh/h (fixed-time baseline: 5,328 veh/h)
        # Strategy: Use both immediate step throughput and cumulative average for responsiveness
        
        if not hasattr(self, 'cumulative_throughput'):
            self.cumulative_throughput = 0
        
        self.cumulative_throughput += step_throughput
        cumulative_rate = (self.cumulative_throughput / max(self.current_step, 1)) * 12  # Average hourly rate
        
        # Immediate throughput rate (last 10 steps for responsiveness)
        if not hasattr(self, 'recent_throughput'):
            self.recent_throughput = []
        self.recent_throughput.append(step_throughput)
        if len(self.recent_throughput) > 10:
            self.recent_throughput.pop(0)
        immediate_rate = np.mean(self.recent_throughput) * 12  # Immediate hourly rate
        
        # Hybrid throughput rate (70% cumulative, 30% immediate for balance)
        throughput_rate = 0.7 * cumulative_rate + 0.3 * immediate_rate
        
        # Enhanced normalization with better gradient
        throughput_norm = min(throughput_rate / 5500.0, 1.0)
        # More aggressive reward scaling for throughput
        throughput_reward = (throughput_norm * 12.0) - 3.0  # Range: [-3, +9]
        
        # 3. SPEED REWARD (20% weight) - Traffic flow efficiency
        # Target: > 20 km/h (fixed-time baseline: 20.60 km/h)
        speed_kmh = avg_speed * 3.6  # Convert m/s to km/h
        if speed_kmh >= 25.0:
            speed_reward = 3.0  # Excellent
        elif speed_kmh >= 20.0:
            speed_reward = 2.0 * (speed_kmh - 20.0) / 5.0  # Good
        elif speed_kmh >= 15.0:
            speed_reward = 1.0 * (speed_kmh - 15.0) / 5.0  # Acceptable
        else:
            speed_reward = -1.0  # Poor
        
        # 4. QUEUE MANAGEMENT REWARD (15% weight) - Congestion control
        # Target: < 80 vehicles (fixed-time baseline: 70.21)
        if avg_queue <= 50:
            queue_reward = 2.0  # Excellent
        elif avg_queue <= 80:
            queue_reward = 1.0 * (80 - avg_queue) / 30  # Good
        elif avg_queue <= 120:
            queue_reward = 0.5 * (120 - avg_queue) / 40  # Acceptable
        else:
            queue_reward = -1.0  # Poor
        
        # 5. PASSENGER THROUGHPUT BONUS (5% weight) - Thesis focus
        passenger_rate = step_passenger_throughput * 12  # Hourly rate
        if passenger_rate >= 6000:
            passenger_bonus = 2.0
        elif passenger_rate >= 5000:
            passenger_bonus = 1.0 * (passenger_rate - 5000) / 1000
        else:
            passenger_bonus = 0.5 * passenger_rate / 5000
        
        # 6. PRESSURE TERM (5-10% weight) - Relieve imbalance, reduce oscillations
        # Proxy: penalize queues and reward throughput trend using a bounded function with EMA smoothing
        pressure_proxy = -avg_queue + (throughput_rate / 5000.0)
        if not hasattr(self, 'pressure_ema'):
            self.pressure_ema = pressure_proxy
        else:
            self.pressure_ema = 0.7 * self.pressure_ema + 0.3 * pressure_proxy
        pressure_term = float(np.tanh(self.pressure_ema))  # Bound to [-1, 1]

        # === COMBINE REWARDS ===
        # THROUGHPUT-OPTIMIZED REWARD BALANCE
        # Analysis shows: Waiting time +30.6% (excellent), Throughput -33.8% (critical failure)
        # Strategy: Prioritize throughput while maintaining waiting time gains
        
        # 6. THROUGHPUT BONUS - Additional reward for high vehicle processing
        throughput_bonus = 0.0
        if step_throughput >= 3:  # High throughput step
            throughput_bonus = 2.0
        elif step_throughput >= 2:  # Medium throughput step
            throughput_bonus = 1.0
        elif step_throughput >= 1:  # Low throughput step
            throughput_bonus = 0.5
        
        # 7. VEHICLE DENSITY BONUS - Reward for processing more vehicles
        density_bonus = 0.0
        if total_vehicles >= 200:  # High density
            density_bonus = 1.5
        elif total_vehicles >= 100:  # Medium density
            density_bonus = 1.0
        elif total_vehicles >= 50:  # Low density
            density_bonus = 0.5
        
        # MODERATE REBALANCING FOR THROUGHPUT PRIORITIZATION WITH STABILITY
        # Balances throughput improvement (+6.3% achieved) with training stability (loss +209% → +50-100%)
        # Total throughput focus: 65% (throughput_reward 50% + throughput_bonus 15%)
        # Goldilocks approach: Between conservative 57% and aggressive 75%
        reward = (
            waiting_reward * 0.22 +      # 22% - Moderate (was 28% conservative, 15% aggressive)
            throughput_reward * 0.50 +   # 50% - Primary focus (was 45% conservative, 55% aggressive)
            speed_reward * 0.12 +        # 12% - Balanced (was 15% conservative, 10% aggressive)
            queue_reward * 0.08 +        # 8% - Moderate (was 10% conservative, 5% aggressive)
            pressure_term * 0.05 +       # 5% - Maintained (stability)
            throughput_bonus * 0.15      # 15% - Moderate bonus (was 12% conservative, 20% aggressive)
            # passenger_bonus removed - consolidated into throughput metrics
        )

        # 7. Density guardrail: discourage policies that trap vehicles (very high load)
        try:
            # Mild spillback penalty: if queues are very high relative to lanes
            total_vehicles_now = traci.vehicle.getIDCount()
            total_lanes = max(sum(len(lanes) for lanes in self.controlled_lanes.values()), 1)
            vehicles_per_lane = total_vehicles_now / total_lanes
            if vehicles_per_lane > 25:   # conservative threshold
                reward -= 0.1
        except Exception:
            pass
        
        # === STABILITY PENALTIES ===
        # Phase change penalty (prevent excessive switching)
        phase_change_penalty = 0.0
        if hasattr(self, 'last_actions') and self.current_step > 0:
            for tl_id in self.traffic_lights:
                current_phase = traci.trafficlight.getPhase(tl_id)
                if hasattr(self, f'last_phase_{tl_id}'):
                    last_phase = getattr(self, f'last_phase_{tl_id}')
                    if current_phase != last_phase:
                        phase_change_penalty -= 0.2
                setattr(self, f'last_phase_{tl_id}', current_phase)
        
        # Cap total phase-change penalty per step
        reward += max(-self.max_phase_change_penalty, phase_change_penalty)

        # Reward clipping for numerical stability
        reward = float(np.clip(reward, -10.0, 10.0))
        
        # === STORE METRICS FOR ANALYSIS ===
        if not hasattr(self, 'reward_components'):
            self.reward_components = []
        
        self.reward_components.append({
            'step': self.current_step,
            'waiting_reward': waiting_reward,
            'throughput_reward': throughput_reward,
            'speed_reward': speed_reward,
            'queue_reward': queue_reward,
            'passenger_bonus': passenger_bonus,
            'pressure_term': pressure_term,
            'throughput_bonus': throughput_bonus,
            'density_bonus': density_bonus,
            'phase_change_penalty': phase_change_penalty,
            'total_reward': reward,
            'avg_waiting': avg_waiting,
            'avg_queue': avg_queue,
            'avg_speed': speed_kmh,
            'throughput_rate': throughput_rate,
            'throughput_norm': throughput_norm,
            'passenger_rate': passenger_rate,
            'step_throughput': step_throughput,
            'total_vehicles': total_vehicles,
            'reward_weights': {
                'waiting': 0.15,          # Reduced from 0.28
                'throughput': 0.55,       # Increased from 0.45
                'speed': 0.10,            # Reduced from 0.15
                'queue': 0.05,            # Reduced from 0.10
                'passenger': 0.0,         # Removed - consolidated
                'pressure': 0.05,         # Maintained
                'throughput_bonus': 0.20  # Increased from 0.12
            }
        })
        
        return reward
    
    def _calculate_pt_priority_bonus(self):
        """Calculate immediate reward for PT vehicle priority"""
        bonus = 0.0
        for tl_id in self.traffic_lights:
            for lane_id in self.controlled_lanes[tl_id]:
                try:
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in vehicle_ids:
                        veh_type = traci.vehicle.getTypeID(veh_id)
                        if veh_type in ['bus', 'jeepney']:
                            # Immediate bonus for PT vehicles
                            bonus += 2.0
                            # Additional bonus if PT vehicle is moving (not stuck)
                            speed = traci.vehicle.getSpeed(veh_id)
                            if speed > 5.0:  # Moving at reasonable speed
                                bonus += 1.0
                except:
                    pass
        return bonus
    
    def _calculate_emergency_bonus(self):
        """Calculate bonus for emergency vehicle clearance"""
        bonus = 0.0
        for tl_id in self.traffic_lights:
            for lane_id in self.controlled_lanes[tl_id]:
                try:
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in vehicle_ids:
                        veh_type = traci.vehicle.getTypeID(veh_id)
                        if veh_type in ['emergency']:
                            bonus += 5.0  # High bonus for emergency vehicles
                except:
                    pass
        return bonus
    
    def _calculate_safety_penalty(self):
        """Calculate penalty for safety violations"""
        penalty = 0.0
        # Check for vehicles running red lights (simplified)
        for tl_id in self.traffic_lights:
            current_phase = traci.trafficlight.getPhase(tl_id)
            # This is a simplified safety check - in practice, you'd need more sophisticated detection
            if current_phase == 0:  # Assuming phase 0 is red
                penalty -= 1.0  # Small penalty for red phase
        return penalty
    
    def _calculate_throughput_reward(self):
        """Calculate reward for vehicle throughput"""
        arrived_vehicles = traci.simulation.getArrivedIDList()
        step_throughput = len(arrived_vehicles)
        
        # Logarithmic reward to prevent over-optimization
        if step_throughput > 0:
            return 3.0 * np.log(1 + step_throughput)
        return 0.0
    
    def _calculate_queue_penalty(self):
        """Calculate penalty for queue length (congestion)"""
        total_queue = 0
        lane_count = 0
        
        for tl_id in self.traffic_lights:
            for lane_id in self.controlled_lanes[tl_id]:
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                total_queue += queue_length
                lane_count += 1
        
        if lane_count > 0:
            avg_queue = total_queue / lane_count
            return -2.0 * np.tanh(avg_queue / 8.0)  # Progressive penalty
        return 0.0
    
    def _calculate_speed_bonus(self):
        """Calculate bonus for speed efficiency"""
        speeds = []
        total_vehicles = traci.vehicle.getIDCount()
        
        if total_vehicles > 0:
            for veh_id in traci.vehicle.getIDList():
                speed = traci.vehicle.getSpeed(veh_id)
                speeds.append(speed)
        
        if speeds:
            avg_speed = np.mean(speeds)
            target_speed = 11.11  # 40 km/h in m/s
            efficiency = min(avg_speed / target_speed, 1.0)
            return 2.0 * efficiency
        return 0.0
    
    def _calculate_phase_change_penalty(self):
        """Calculate penalty for frequent phase changes"""
        penalty = 0.0
        if hasattr(self, 'current_step') and self.current_step > 0:
            # Small penalty for phase changes to encourage stability
            for tl_id in self.traffic_lights:
                current_phase = traci.trafficlight.getPhase(tl_id)
                if hasattr(self, f'last_phase_{tl_id}'):
                    last_phase = getattr(self, f'last_phase_{tl_id}')
                    if current_phase != last_phase:
                        penalty -= 0.1
                setattr(self, f'last_phase_{tl_id}', current_phase)
        return penalty
    
    def _calculate_coordination_bonus(self):
        """Calculate bonus for intersection coordination"""
        if hasattr(self, 'coordination_mode') and self.coordination_mode:
            return 1.0  # Bonus for coordinated control
        return 0.0
    
    def _calculate_balance_reward(self):
        """Calculate reward for balanced system utilization"""
        queue_distribution = []
        for tl_id in self.traffic_lights:
            for lane_id in self.controlled_lanes[tl_id]:
                queue = traci.lane.getLastStepHaltingNumber(lane_id)
                queue_distribution.append(queue)
        
        if len(queue_distribution) > 1:
            queue_std = np.std(queue_distribution)
            return 0.5 * (1.0 / (1.0 + queue_std / 5.0))  # Reward balanced distribution
        return 0.5
    
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
    
    def _initialize_flow_type_mapping(self):
        """
        Initialize mapping from flow IDs to vehicle types by parsing the current route file.
        This allows us to determine vehicle types for departed vehicles.
        """
        try:
            import xml.etree.ElementTree as ET
            
            # Get current route file from environment
            route_file = getattr(self, 'rou_file', None)
            if not route_file:
                # Fallback - create empty mapping
                self.flow_to_type_mapping = {}
                return
            
            # Parse the route file to extract flow-to-type mapping
            self.flow_to_type_mapping = {}
            
            tree = ET.parse(route_file)
            root = tree.getroot()
            
            # Find all flow elements and extract their ID and type
            for flow_elem in root.findall('flow'):
                flow_id = flow_elem.get('id')
                flow_type = flow_elem.get('type')
                if flow_id and flow_type:
                    self.flow_to_type_mapping[flow_id] = flow_type
            
            print(f"Initialized flow-to-type mapping: {len(self.flow_to_type_mapping)} flows loaded")
            # Print PT flows for debugging
            pt_flows = {k: v for k, v in self.flow_to_type_mapping.items() if v in ['bus', 'jeepney']}
            print(f"PT flows detected: {pt_flows}")
            
        except Exception as e:
            print(f"WARNING: Could not initialize flow mapping: {e}")
            self.flow_to_type_mapping = {}

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
                    # Map flow ID to vehicle type using flow-to-type mapping
                    # Vehicle IDs are like "flow_21.0", we need to extract the flow number
                    if 'flow_' in veh_id:
                        # Get the flow ID (before the dot if it exists)
                        flow_id = veh_id.split('.')[0]
                        
                        # Get vehicle type from TraCI or use mapping
                        # For departed vehicles, we need to use our stored mapping
                        # Create flow-to-type mapping from route file data
                        flow_to_type = getattr(self, 'flow_to_type_mapping', {})
                        
                        if not flow_to_type:
                            # Initialize mapping on first use - parse from current route
                            self._initialize_flow_type_mapping()
                            flow_to_type = getattr(self, 'flow_to_type_mapping', {})
                        
                        veh_type = flow_to_type.get(flow_id, 'unknown')
                        
                        if veh_type == 'bus':
                            step_buses_completed += 1
                            step_pt_passengers += 40.0  # Average bus capacity
                        elif veh_type == 'jeepney':
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
            print("Closing SUMO simulation...")
            traci.close()
        try:
            sys.stdout.flush()
        except Exception:
            pass
    
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