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
            rou_file: Path to SUMO route file (.rou.xml)
            use_gui: Whether to show SUMO GUI for visualization
            num_seconds: Total simulation duration
            warmup_time: Time before agent control starts
            step_length: Simulation step size in seconds
        """
        self.net_file = net_file
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
            'completed_trips': 0
        }
        
        print(f"🚦 Traffic Environment Initialized:")
        print(f"   Network: {os.path.basename(net_file)}")
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
            '-r', self.rou_file,
            '--step-length', str(self.step_length),
            '--waiting-time-memory', '10000',
            '--time-to-teleport', '-1',
            '--no-warnings', 
            '--quit-on-end',
            '--seed', str(random.randint(0, 100000))
        ]
        
        # Add GUI-specific options
        if self.use_gui:
            self.sumo_cmd.extend([
                '--start',  # Start simulation immediately
                '--delay', '100'  # Delay between steps (ms) for better visualization
            ])
    
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
        
        # Info for debugging
        info = {
            'step': self.current_step,
            'vehicles': traci.vehicle.getIDCount(),
            'reward': reward,
            'total_reward': self.total_reward,
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
        """Calculate reward based on traffic metrics"""
        # Calculate total waiting time for all vehicles
        total_waiting = 0
        total_vehicles = traci.vehicle.getIDCount()
        
        if total_vehicles == 0:
            return 0.0
        
        # Sum waiting times for all lanes
        for tl_id in self.traffic_lights:
            for lane_id in self.controlled_lanes[tl_id]:
                total_waiting += traci.lane.getWaitingTime(lane_id)
        
        # Reward is negative waiting time (minimize waiting)
        reward = -total_waiting / max(total_vehicles, 1)
        
        # Bonus for maintaining traffic flow
        if total_vehicles > 0:
            avg_speed = sum(traci.vehicle.getSpeed(veh_id) 
                          for veh_id in traci.vehicle.getIDList()) / total_vehicles
            speed_bonus = avg_speed / 13.89  # Normalize to 50 km/h
            reward += speed_bonus * 0.1
        
        return reward
    
    def _is_done(self):
        """Check if episode should terminate"""
        # Episode ends when simulation time is reached or no vehicles remain
        time_limit = self.current_step * self.step_length >= self.num_seconds
        no_vehicles = traci.vehicle.getIDCount() == 0 and self.current_step > self.warmup_time / self.step_length
        
        return time_limit or no_vehicles
    
    def _update_metrics(self):
        """Update performance metrics"""
        self.metrics['total_vehicles'] = traci.vehicle.getIDCount()
        
        if self.metrics['total_vehicles'] > 0:
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
