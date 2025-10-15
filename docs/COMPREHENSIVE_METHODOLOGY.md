# Comprehensive Methodology: D3QN-based Multi-Agent Traffic Signal Control

**Study:** Dueling Double Deep Q-Network for Traffic Signal Control in Davao City  
**Focus:** Multi-Agent Reinforcement Learning with Architecture Comparison  
**Date:** October 7, 2025  

---

## Table of Contents
1. [Study Overview](#study-overview)
2. [Problem Statement](#problem-statement)
3. [Research Objectives](#research-objectives)
4. [Data Collection & Processing Pipeline](#data-collection--processing-pipeline)
5. [Traffic Simulation Setup & Realism](#traffic-simulation-setup--realism)
6. [Anti-Cheating Policies & Constraints](#anti-cheating-policies--constraints)
7. [System Architecture](#system-architecture)
8. [Agent Architecture Design](#agent-architecture-design)
9. [State and Action Space](#state-and-action-space)
10. [Reward Function Design](#reward-function-design)
11. [Training Protocol](#training-protocol)
12. [Experimental Setup](#experimental-setup)
13. [Performance Metrics](#performance-metrics)
14. [Statistical Validation](#statistical-validation)
15. [Davao City Adaptations](#davao-city-adaptations)

---

## 1. Study Overview

### 1.1 Research Context
This study develops a **Dueling Double Deep Q-Network (D3QN)** based Multi-Agent Reinforcement Learning (MARL) system for adaptive traffic signal control in Davao City. The research addresses the critical challenge of optimizing traffic flow in multi-intersection networks while maintaining computational efficiency and data-driven decision making.

### 1.2 Study Scope
- **Geographic Focus:** Davao City, Philippines (three major intersections)
- **Network Complexity:** 3 traffic lights, 40+ lanes, multiple vehicle types
- **Traffic Data:** Real-world traffic patterns from Davao City (2025)
- **Simulation Platform:** SUMO (Simulation of Urban Mobility)
- **Control Approach:** Multi-Agent Reinforcement Learning

### 1.3 Novel Contributions
1. **Architecture Comparison:** First systematic comparison of LSTM vs Non-LSTM D3QN for traffic control with limited data
2. **Data-Driven Architecture Selection:** Methodology for selecting optimal architecture based on data availability
3. **Davao City Adaptation:** Context-specific passenger capacity modeling for local vehicle types
4. **Aggressive Reward Rebalancing:** Novel approach to balance multi-objective optimization with throughput prioritization

---

## 2. Problem Statement

### 2.1 Research Problem
Traditional fixed-time traffic signal control systems in Davao City exhibit:
- **Inflexible timing plans** that cannot adapt to varying traffic conditions
- **Suboptimal throughput** during peak and off-peak hours
- **Increased waiting times** and congestion
- **Limited coordination** between adjacent intersections

### 2.2 Research Gap
While Deep Reinforcement Learning shows promise for traffic control, existing research has limitations:
- Most studies use **synthetic traffic data** rather than real-world patterns
- Limited exploration of **architecture selection** based on data availability
- Insufficient consideration of **local vehicle type distributions**
- Lack of systematic comparison between **temporal (LSTM) vs spatial (Dense) architectures**

### 2.3 Research Questions
1. Can D3QN-based MARL improve traffic performance over fixed-time control?
2. How does LSTM architecture compare to dense-only networks with limited training data?
3. What reward function configuration optimally balances multiple traffic objectives?
4. How can the system be adapted to Davao City's specific traffic characteristics?

---

## 3. Research Objectives

### 3.1 Primary Objectives
1. **Develop D3QN-based MARL system** for adaptive traffic signal control
2. **Compare LSTM vs Non-LSTM architectures** systematically
3. **Optimize reward function** for throughput improvement
4. **Validate performance** against fixed-time baseline

### 3.2 Performance Targets
- **Throughput:** ≤10% degradation vs fixed-time (stretch goal), ≤25% acceptable
- **Waiting Time:** ≥15% improvement over fixed-time
- **Training Stability:** Convergent learning within 200 episodes
- **Statistical Significance:** p < 0.05 for all major metrics

### 3.3 Success Criteria
- Demonstrable learning progress over training episodes
- Statistically significant improvements in multiple traffic metrics
- Academically defensible results with clear research contributions
- Reproducible methodology with proper validation

---

## 4. Data Collection & Processing Pipeline

### 4.1 Real-World Traffic Data Collection

**Data Source:** Davao City Traffic and Traffic Management Office (CTTMO)

**Collection Method:**
- **Location:** Three major intersections (Ecoland, JohnPaul, Sandawa)
- **Period:** July 2025 - August 2025 (peak summer traffic)
- **Duration:** 8-hour cycles per day (6 AM - 2 PM coverage)
- **Cycle Definition:** 5-minute observation windows
- **Total Scenarios:** 68 unique traffic scenarios (Day × Cycle combinations)

**Data Collection Points:**
```
Intersection Coverage:
├── Ecoland Intersection: 4 approach lanes
├── JohnPaul Intersection: 6 approach lanes  
└── Sandawa Intersection: 3 approach lanes

Vehicle Classification:
├── Cars (private vehicles)
├── Motorcycles
├── Jeepneys (Traditional PUJ)
├── Buses (City transit)
├── Trucks (Commercial)
└── Tricycles (Local transport)
```

**Raw Data Format:**
```csv
Day, CycleNum, IntersectionID, TotalVehicles, car_count, motor_count, 
jeepney_count, bus_count, truck_count, CycleTime_s
```

### 4.2 Data Preprocessing Pipeline

**Step 1: Data Cleaning**
```python
# Remove incomplete observations
- Filter out cycles with missing vehicle counts
- Validate total vehicles = sum(individual counts)
- Remove anomalous data points (outliers beyond 3 standard deviations)

# Quality control
- Minimum 50 vehicles per cycle (ensures sufficient traffic)
- Maximum 500 vehicles per cycle (prevents unrealistic congestion)
- Valid cycle time: 240-360 seconds (4-6 minutes)
```

**Step 2: Scenario Indexing**
```python
# Create scenarios_index.csv
# Links each scenario to its route files
Columns: Day, CycleNum, Intersections, ScenarioPath

# Example:
20250701, 1, "ECOLAND,JOHNPAUL,SANDAWA", ../../out/scenarios/20250701/cycle_1
```

**Step 3: Route File Generation**
```python
# For each scenario, generate SUMO route files
# Input: Traffic counts per intersection per vehicle type
# Output: SUMO .rou.xml files with vehicle flows

Process:
1. Map intersection entry/exit points to SUMO network edges
2. Calculate probability of each vehicle type appearing
3. Generate shortest-path routes through the network
4. Create flow definitions with realistic inter-arrival times
```

**Vehicle Flow Calculation:**
```python
# Convert vehicle counts to SUMO flow probabilities
for vehicle_type in ['car', 'motor', 'jeepney', 'bus', 'truck']:
    count = traffic_data[intersection][vehicle_type]
    cycle_time = traffic_data[intersection]['cycle_time']
    
    # Probability of spawning per second
    probability = count / cycle_time  # vehicles per second
    
    # SUMO flow definition
    flow = {
        'begin': 0,
        'end': episode_duration,
        'probability': probability,
        'type': vehicle_type,
        'departLane': 'random',  # Realistic lane selection
        'departSpeed': 'random'  # Realistic speed variation
    }
```

### 4.3 Route Consolidation for Multi-Agent Learning

**Challenge:** SUMO doesn't allow duplicate vehicle type definitions across multiple route files

**Solution:** Consolidate multiple intersection route files into single bundle files

```python
# scripts/consolidate_bundle_routes.py
def consolidate_bundle_routes(bundle_routes, output_file):
    """
    Consolidate routes from 3 intersections into 1 file
    - Add vehicle types only once (from first file)
    - Assign unique route IDs (route_0, route_1, ...)
    - Assign unique flow IDs (flow_0, flow_1, ...)
    - Preserve all flow definitions and parameters
    """
    # Collect from all intersection files
    for route_file in ['ECOLAND_*.rou.xml', 'JOHNPAUL_*.rou.xml', 'SANDAWA_*.rou.xml']:
        # Extract vehicle types (once)
        # Extract routes (unique IDs)
        # Extract flows (update route references)
    
    # Output: bundle_YYYYMMDD_cycle_N.rou.xml
```

**Result:** 68 consolidated route files, one per scenario

### 4.4 Train/Validation/Test Split

**Splitting Strategy:** Scenario-based (not random)

```python
Split Configuration:
├── Training Set: 70% (48 scenarios)
│   └── Used for agent learning
├── Validation Set: 20% (13 scenarios)
│   └── Used for hyperparameter tuning, checkpoint selection
└── Test Set: 10% (7 scenarios)
    └── Used for final evaluation only (never seen during training)

# Rationale for scenario-based splitting:
- Prevents data leakage (same traffic patterns in train/test)
- Ensures generalization to unseen conditions
- Maintains temporal independence
```

**Split Method:**
```python
import pandas as pd
import numpy as np

def split_scenarios(scenarios_df, random_seed=42):
    np.random.seed(random_seed)
    
    # Shuffle scenarios
    shuffled = scenarios_df.sample(frac=1, random_state=random_seed)
    
    n_scenarios = len(shuffled)
    train_end = int(n_scenarios * 0.70)
    val_end = int(n_scenarios * 0.90)
    
    train = shuffled[:train_end]
    validation = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    return train, validation, test
```

### 4.5 Data Characteristics & Statistics

**Training Data Summary:**
```
Total Scenarios: 68
├── Different Days: 17 unique dates
├── Cycles per Day: 1-3 cycles (morning, midday, afternoon)
└── Total Observations: ~200,000 vehicle movements

Vehicle Type Distribution (averaged):
├── Cars: 45% (dominant private transport)
├── Motorcycles: 30% (very high in Davao)
├── Jeepneys: 15% (primary public transport)
├── Buses: 5% (limited city bus service)
├── Trucks: 3% (commercial traffic)
└── Tricycles: 2% (local short trips)

Traffic Volume Range:
├── Minimum: 280 vehicles per 5-minute cycle
├── Maximum: 450 vehicles per 5-minute cycle
├── Average: 355 ± 45 vehicles per cycle
└── Peak Hours: 7-9 AM, 11 AM-1 PM
```

**Data Quality Assurance:**
- All scenarios validated for completeness
- Vehicle counts cross-checked with video footage
- Cycle times verified against intersection timing plans
- Route feasibility validated in SUMO network

---

## 5. Traffic Simulation Setup & Realism

### 5.1 SUMO Configuration for Realistic Traffic

**Simulation Parameters:**
```python
SUMO Configuration:
├── Step Length: 1.0 second (real-time scale)
├── Waiting Time Memory: 10,000 seconds (long memory for queue tracking)
├── Time-to-Teleport: -1 (DISABLED - prevents unrealistic vehicle removal)
├── Random Seed: Variable per episode (scenario-specific reproducibility)
├── Warnings: Disabled (clean output for logging)
└── Quit-on-End: Enabled (automatic termination)
```

**Critical Anti-Exploitation Settings:**
```python
# Prevent agent from "cheating" by removing stuck vehicles
'--time-to-teleport', '-1'  # Vehicles NEVER teleport
# Forces agent to actually solve congestion, not wait for SUMO to remove vehicles

# Long waiting time memory
'--waiting-time-memory', '10000'  # Track waiting for entire episode
# Ensures accurate waiting time calculations
```

### 5.2 Vehicle Type Modeling (Davao-Specific)

**Realistic Vehicle Dynamics:**
```python
VEHICLE_TYPES = {
    "car": {
        "accel": "2.6 m/s²",      # Standard acceleration
        "decel": "4.5 m/s²",      # Comfortable braking
        "sigma": "0.5",           # Driver imperfection (50%)
        "length": "5 m",          # Average car length
        "minGap": "2.5 m",        # Safe following distance
        "maxSpeed": "11.11 m/s",  # 40 km/h (city speed limit)
        "guiShape": "passenger"
    },
    "jeepney": {
        "accel": "1.8 m/s²",      # Slower acceleration (heavier)
        "decel": "4.2 m/s²",      # Good braking system
        "sigma": "0.5",           # Driver imperfection
        "length": "8 m",          # Traditional PUJ length
        "minGap": "2.8 m",        # Larger following distance
        "maxSpeed": "11.11 m/s",  # Same city speed limit
        "guiShape": "bus"
    },
    "bus": {
        "accel": "1.2 m/s²",      # Slowest acceleration (largest vehicle)
        "decel": "4.0 m/s²",      # Heavy vehicle braking
        "sigma": "0.5",           # Driver imperfection
        "length": "12 m",         # City bus length
        "minGap": "3.0 m",        # Largest following distance
        "maxSpeed": "11.11 m/s",  # Same city speed limit
        "guiShape": "bus"
    },
    "motor": {
        "accel": "3.0 m/s²",      # Fastest acceleration
        "decel": "5.0 m/s²",      # Best braking
        "sigma": "0.3",           # Less imperfection (agile)
        "length": "2 m",          # Motorcycle length
        "minGap": "1.5 m",        # Smallest following distance
        "maxSpeed": "11.11 m/s",  # Same city speed limit
        "guiShape": "motorcycle"
    },
    "truck": {
        "accel": "1.5 m/s²",      # Slow acceleration (heavy load)
        "decel": "3.5 m/s²",      # Moderate braking
        "sigma": "0.4",           # Careful driving
        "length": "10 m",         # Commercial truck length
        "minGap": "3.5 m",        # Large following distance
        "maxSpeed": "11.11 m/s",  # Same city speed limit
        "guiShape": "truck"
    }
}
```

**Rationale for Parameters:**
- **All vehicles limited to 40 km/h:** Enforces realistic city speed limit
- **Different acceleration/deceleration:** Models actual vehicle performance differences
- **Sigma (driver imperfection):** Adds realistic driving variability
- **Vehicle lengths:** Based on actual measurements in Davao
- **Following distances:** Based on traffic engineering safety standards

### 5.3 Route Generation from Real Data

**Process Flow:**
```
Real Traffic Data → Route Generation → SUMO Simulation

Step 1: Load traffic counts
├── Input: master_bundles.csv
├── Contains: Vehicle counts per type per intersection
└── Format: Day, Cycle, Intersection, car_count, motor_count, etc.

Step 2: Map to network edges
├── Entry Edges: Where vehicles enter (intersection approaches)
├── Exit Edges: Where vehicles leave (intersection exits)
└── Use shortest-path algorithm to connect entry → exit

Step 3: Calculate flow probabilities
├── probability = vehicle_count / cycle_time
├── Accounts for realistic inter-arrival times
└── Preserves traffic volume characteristics

Step 4: Generate SUMO flow definitions
├── Begin time: 0 seconds
├── End time: Episode duration (300s)
├── Department: random lane, random speed
└── Creates realistic traffic loading
```

**Example Route Generation:**
```xml
<!-- Generated from real data: 45 cars observed in 5-minute cycle -->
<vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="11.11" guiShape="passenger"/>

<route id="route_0" edges="106768821 -1069919419 -794461795 455558436#0"/>

<flow id="flow_0" route="route_0" begin="0" end="300" probability="0.15" type="car" departLane="random" departSpeed="random"/>
<!-- probability = 45 vehicles / 300 seconds = 0.15 vehicles/second -->
```

### 5.4 Simulation Warmup Protocol

**Purpose:** Ensure realistic traffic loading before agent control begins

**Warmup Process (30 seconds):**
```python
# Phase 1: Calibration (10 seconds)
- Set ALL traffic lights to RED
- Allow vehicles to enter network without clearing
- Purpose: Build realistic queue conditions

# Phase 2: Traffic Light Warmup (10 seconds)
- Enable normal traffic light cycling
- Allow some vehicles to clear
- Purpose: Establish dynamic traffic flow

# Phase 3: Final Loading (10 seconds)
- Continue traffic light operation
- Stabilize vehicle distribution
- Purpose: Reach steady-state before agent takes control

# After warmup:
- Record initial vehicle count
- Initialize agent state
- Begin agent-controlled episode
```

**Rationale:**
- **Prevents "cold start" bias:** Agent starts with realistic traffic conditions
- **Ensures comparability:** All episodes start from similar network states
- **Models reality:** Real traffic signals never start with empty roads

---

## 6. Anti-Cheating Policies & Constraints

### 6.1 Research Motivation

Reinforcement learning agents can exploit simulation loopholes to achieve artificially high performance. To ensure academically valid results, we implement comprehensive anti-cheating policies.

### 6.2 Critical Anti-Exploitation Mechanisms

**1. No Vehicle Teleportation**
```python
'--time-to-teleport', '-1'  # DISABLED
```

**Problem:** SUMO's default behavior teleports vehicles stuck >300 seconds  
**Exploitation:** Agent could ignore congestion, waiting for SUMO to remove vehicles  
**Our Solution:** Disable teleportation - agent MUST solve congestion  
**Impact:** Forces agent to learn proper traffic management

**2. Long Waiting Time Memory**
```python
'--waiting-time-memory', '10000'  # 10,000 seconds
```

**Problem:** Short memory allows agent to ignore long-waiting vehicles  
**Exploitation:** Agent could focus only on recent vehicles  
**Our Solution:** Track waiting time for entire episode  
**Impact:** Penalizes policies that create sustained queues

### 6.3 Phase Timing Constraints

**Minimum Phase Time (Safety)**
```python
self.min_phase_time = 12 seconds  # HARD CONSTRAINT
```

**Rationale:**
- **Safety Standard:** Pedestrian crossing requires minimum 10-12 seconds
- **Driver Reaction:** Vehicles need time to respond to green light
- **Queue Clearing:** Minimum time to process waiting vehicles
- **Prevents Rapid Oscillation:** Agent cannot rapidly flip phases for reward exploitation

**Maximum Phase Time (Efficiency)**
```python
self.max_phase_time = 120 seconds  # HARD CONSTRAINT
```

**Rationale:**
- **Fairness:** Prevents one direction from monopolizing green time
- **Network Efficiency:** Forces consideration of all approaches
- **Realistic Practice:** Based on traffic engineering standards
- **Prevents Phase Locking:** Agent cannot hold one phase indefinitely

**Phase Change Cooldown**
```python
self.phase_cooldown_steps = 5 seconds  # SOFT CONSTRAINT
```

**Rationale:**
- **Mechanical Limitation:** Real traffic signals have actuation delays
- **Stability:** Reduces action oscillation (thrashing)
- **Driver Safety:** Prevents confusing rapid signal changes
- **Realistic Operation:** Models actual traffic signal controllers

### 6.4 Forced Cycle Completion

**Policy:**
```python
# If agent hasn't completed a full cycle in 200 steps (200 seconds)
if steps_since_last_cycle[tl_id] > 200:
    # Force transition to next phase in cycle
    # Ensures all lanes get green time eventually
```

**Rationale:**
- **Fairness:** Prevents agent from favoring specific lanes/directions
- **Prevents Starvation:** Ensures all traffic gets served
- **Models Real Constraints:** Actual signals must serve all approaches
- **Academic Validity:** Agent cannot exploit by ignoring certain lanes

### 6.5 Public Transport Priority (Realistic Policy)

**Policy:**
```python
if bus or jeepney waiting at intersection:
    # Allow phase change even if below min_phase_time
    # Reduced minimum to min_phase_time // 2 (6 seconds)
```

**Rationale:**
- **Davao City Policy:** Public transport has priority (LTFRB mandate)
- **High Passenger Capacity:** 14-35 passengers per vehicle
- **Service Quality:** Maintains public transport reliability
- **Realistic Constraint:** Agent must model actual traffic policies

### 6.6 Action Space Limitations

**Constrained Action Set:**
```
Action 0: Extend Current Phase (+5 seconds if beneficial)
Action 1: Normal Phase Change (standard transition)
Action 2: Quick Phase Change (yellow → red → green sequence)
Action 3: Skip to Next Phase (emergency override, limited use)
Action 4: Emergency Clearance (high queue only)
Action 5: Maintain Current Phase (no change)
```

**Action Constraints:**
- **No "Do Nothing" Forever:** Maintain action limited to max_phase_time
- **No Direct Phase Selection:** Must follow logical phase sequences
- **No Traffic Light Disabling:** All signals must operate continuously
- **No Priority Lane Bias:** Forced cycle completion prevents favoritism

### 6.7 State Observation Realism

**Perfect Information Assumption:**
```python
# Assume 100% sensor accuracy (loop detectors, cameras)
# Justified by: Modern traffic systems have >95% detection accuracy
# Limitation acknowledged: Real-world would have sensor noise
```

**No Future Information:**
```python
# Agent sees ONLY current state:
- Current lane queues
- Current vehicle speeds
- Current waiting times
- Current phase timers

# Agent does NOT see:
- Future vehicle arrivals (no cheating)
- Planned routes of individual vehicles
- Traffic demand forecasts
```

### 6.8 Evaluation Protocol Anti-Bias

**1. Fixed Random Seeds**
```python
# Each scenario gets consistent random seed
# Ensures D3QN and Fixed-Time see identical traffic
seed = hash(f"{day}_{cycle}_{run_number}") % 100000
```

**2. Identical Simulation Conditions**
```python
D3QN Evaluation:
- Same network, routes, warmup protocol
- Same episode duration, step length
- Only difference: Signal control logic
```

**3. Multiple Evaluation Runs**
```python
# Run 7-25 episodes per agent type
# Provides statistical power
# Accounts for stochastic variability
```

---

## 7. System Architecture

### 7.1 Multi-Agent System Design

**Architecture Type:** Decentralized Execution with Centralized Training (CTDE variant)

Each traffic light operates as an **independent agent** with:
- Local state observation (surrounding lanes)
- Individual action selection (phase control)
- Shared reward function (system-wide optimization)
- Coordinated learning (experience sharing)

**Coordination Mechanism:**
```
Agent Communication:
├── State Sharing: Each agent observes neighboring states
├── Pressure Balancing: EMA-smoothed pressure term
├── Phase Coordination: Conflict avoidance logic
└── Experience Replay: Shared memory buffer
```

### 4.2 System Components

```
D3QN MARL System
├── Traffic Environment (SUMO + TraCI)
│   ├── Network: Davao City road network
│   ├── Routes: Real-world traffic flows
│   └── Metrics: Real-time performance tracking
│
├── Multi-Agent Controller
│   ├── Agent 1: Ecoland Traffic Signal
│   ├── Agent 2: JohnPaul Traffic Signal
│   └── Agent 3: Sandawa Traffic Signal
│
├── D3QN Neural Network
│   ├── Input: Traffic state (167 dimensions)
│   ├── Hidden: Dense/LSTM layers
│   ├── Output: Dueling Q-values (6 actions)
│   └── Target Network: Stabilized learning
│
└── Experience Replay
    ├── Buffer Size: 75,000 experiences
    ├── Batch Size: 128 samples
    └── Sampling: Uniform random
```

---

## 5. Agent Architecture Design

### 5.1 Architecture Comparison Rationale

A **critical research contribution** is the systematic comparison of two architectures:

**Research Hypothesis:** With limited training data (<200 episodes), dense-only networks outperform LSTM-based networks due to:
1. **Data Efficiency:** LSTM requires 5-10× more data for effective temporal learning
2. **Immediate Response:** Traffic signals need current state prioritization over history
3. **Spatial Focus:** Lane-specific features more important than temporal sequences
4. **Training Stability:** Dense networks show less variance with limited data

### 5.2 LSTM-based D3QN Agent

**Architecture:**
```python
Input Layer: (sequence_length=10, state_size=167)
├── LSTM Layer 1: 128 units, return_sequences=True
│   ├── Dropout: 0.3
│   └── Recurrent Dropout: 0.2
├── LSTM Layer 2: 64 units, return_sequences=False
│   ├── Dropout: 0.3
│   └── Recurrent Dropout: 0.2
├── Dense Layer 1: 128 units, ReLU activation
│   ├── L2 Regularization: 0.001
│   └── Dropout: 0.3
├── Dense Layer 2: 64 units, ReLU activation
│   └── L2 Regularization: 0.001
├── Value Stream: 32 units → 1 output
└── Advantage Stream: 32 units → 6 outputs
    └── Dueling Combination: Q(s,a) = V(s) + A(s,a) - mean(A)

Total Parameters: 146,597
```

**Design Rationale:**
- **Temporal Memory:** Maintains 10-step history for pattern recognition
- **Dropout Layers:** Prevent overfitting with limited data
- **L2 Regularization:** Weight decay for generalization
- **Dueling Architecture:** Separate value and advantage estimation

### 5.3 Non-LSTM D3QN Agent (Recommended)

**Architecture:**
```python
Input Layer: (state_size=167)
├── Dense Layer 1: 512 units, ReLU activation
│   ├── L2 Regularization: 0.001
│   └── Dropout: 0.3
├── Dense Layer 2: 256 units, ReLU activation
│   ├── L2 Regularization: 0.001
│   └── Dropout: 0.3
├── Dense Layer 3: 128 units, ReLU activation
│   ├── L2 Regularization: 0.001
│   └── Dropout: 0.3
├── Dense Layer 4: 64 units, ReLU activation
│   └── L2 Regularization: 0.001
├── Value Stream: 32 units → 1 output
└── Advantage Stream: 32 units → 6 outputs
    └── Dueling Combination: Q(s,a) = V(s) + A(s,a) - mean(A)

Total Parameters: 262,887
```

**Design Rationale:**
- **Spatial Feature Extraction:** Larger dense layers focus on current state
- **No Temporal Dependency:** Each state processed independently
- **Better Parameter Utilization:** More parameters on spatial features
- **Stable Gradients:** No vanishing gradient issues from LSTM

### 5.4 Training Hyperparameters (Both Agents)

```python
Hyperparameter Configuration:
├── Learning Rate: 0.0005 (Adam optimizer)
├── Epsilon: 1.0 → 0.01 (decay: 0.9995)
├── Discount Factor (γ): 0.95
├── Target Update (τ): 0.005 (Polyak averaging)
├── Batch Size: 128
├── Memory Buffer: 75,000 experiences
├── Gradient Clipping: 5.0
└── Loss Function: Huber Loss (δ=1.0)
```

**Optimization Techniques:**
1. **Soft Target Updates:** Polyak averaging (τ=0.005) for stability
2. **Gradient Clipping:** Prevent exploding gradients
3. **Huber Loss:** Robust to outliers vs MSE
4. **Experience Replay:** Break temporal correlations

---

## 6. State and Action Space

### 6.1 State Representation

**State Vector Dimensions:** 167 features per timestep

**State Components:**
```
Traffic State (167 dimensions):
├── Lane-Level Features (40 lanes × 4 metrics = 160)
│   ├── Queue Length: Vehicles waiting per lane
│   ├── Average Speed: Mean velocity (m/s)
│   ├── Occupancy: Lane utilization (0-1)
│   └── Waiting Time: Cumulative wait per lane
│
├── Intersection-Level Features (3 signals × 2 metrics = 6)
│   ├── Current Phase: One-hot encoded
│   └── Phase Duration: Time in current phase
│
└── Global Context (1 metric)
    └── Total System Load: Network-wide vehicles
```

**State Preprocessing:**
- **Normalization:** Min-max scaling (0-1 range)
- **Missing Values:** Zero-filling for empty lanes
- **Temporal Handling:** 
  - LSTM: 10-step sequence buffer
  - Non-LSTM: Single timestep only

### 6.2 Action Space

**Action Type:** Discrete (6 actions per agent)

**Action Definitions:**
```
Traffic Signal Control Actions:
├── Action 0: Extend Current Phase (maintain +5s)
├── Action 1: Normal Phase Change (standard transition)
├── Action 2: Quick Phase Change (immediate transition)
├── Action 3: Skip to Next Phase (emergency override)
├── Action 4: Emergency Clearance (priority clearing)
└── Action 5: Maintain Current Phase (no change)
```

**Action Constraints:**
- **Minimum Green Time:** 12 seconds (safety)
- **Maximum Green Time:** 120 seconds (fairness)
- **Phase Change Cooldown:** 5 seconds (stability)
- **Yellow Phase:** Automatic 3-second transition

**Multi-Agent Action Coordination:**
- Each agent selects actions independently
- Coordination through shared state visibility
- Conflict resolution via pressure balancing
- Emergency override for public transport priority

---

## 7. Reward Function Design

### 7.1 Reward Function Evolution

The reward function underwent **iterative refinement** based on training results:

**Version 1 (Initial):** Balanced multi-objective
- Throughput degradation: -32%
- Waiting time improvement: +25%
- **Issue:** Insufficient throughput focus

**Version 2 (Conservative Rebalancing):** Moderate throughput increase
- Throughput degradation: -27%
- Waiting time improvement: +33%
- **Issue:** Waiting time over-performing, throughput still poor

**Version 3 (Aggressive Rebalancing - Current):** Heavy throughput prioritization
- **Target:** Throughput degradation ≤-20%
- **Expected:** Waiting time improvement +18-22%

### 7.2 Final Reward Function (Version 3)

**Mathematical Formulation:**

\[
R_{total} = \sum_{i=1}^{6} w_i \cdot R_i
\]

Where:
- \( w_i \) = weight for component \( i \)
- \( R_i \) = normalized reward component \( i \)

**Component Breakdown:**

```python
reward = (
    R_waiting * 0.15 +        # 15% - Waiting time minimization
    R_throughput * 0.55 +     # 55% - Throughput maximization (PRIMARY)
    R_speed * 0.10 +          # 10% - Speed efficiency
    R_queue * 0.05 +          # 5% - Queue management
    R_pressure * 0.05 +       # 5% - Pressure stabilization
    R_bonus * 0.20            # 20% - Throughput bonus (REINFORCEMENT)
)

Total Throughput Focus: 75% (55% + 20%)
```

### 7.3 Reward Component Details

**1. Waiting Time Reward (15%)**
```python
# Penalize long waiting times
avg_waiting = sum(vehicle_waiting_times) / num_vehicles
if avg_waiting <= 5.0:
    R_waiting = 3.0  # Excellent
elif avg_waiting <= 10.0:
    R_waiting = 2.0 - (avg_waiting - 5.0) / 5.0  # Good to acceptable
else:
    R_waiting = -2.0  # Poor (heavy penalty)
```

**2. Throughput Reward (55% - PRIMARY)**
```python
# Hybrid approach: 70% cumulative + 30% immediate
cumulative_rate = (cumulative_throughput / time_steps) * 3600
immediate_rate = mean(recent_10_steps) * 3600
throughput_rate = 0.7 * cumulative_rate + 0.3 * immediate_rate

# Normalize and scale
throughput_norm = min(throughput_rate / 5500.0, 1.0)
R_throughput = (throughput_norm * 12.0) - 3.0  # Range: [-3, +9]
```

**3. Speed Reward (10%)**
```python
# Encourage high average speeds
speed_kmh = avg_speed * 3.6
if speed_kmh >= 25.0:
    R_speed = 3.0
elif speed_kmh >= 20.0:
    R_speed = 2.0 * (speed_kmh - 20.0) / 5.0
else:
    R_speed = (speed_kmh / 20.0) - 1.0  # Penalty below 20 km/h
```

**4. Queue Reward (5%)**
```python
# Minimize queue lengths
avg_queue = sum(lane_queues) / num_lanes
if avg_queue <= 20:
    R_queue = 2.0
elif avg_queue <= 40:
    R_queue = 1.0 * (40 - avg_queue) / 20
else:
    R_queue = -1.0  # Heavy queues penalty
```

**5. Pressure Term (5%)**
```python
# Balance traffic flow between intersections
pressure_proxy = mean([outflow - inflow for each junction])
pressure_ema = 0.7 * pressure_ema_prev + 0.3 * pressure_proxy
R_pressure = -abs(pressure_ema) / 10.0  # Penalize imbalance
```

**6. Throughput Bonus (20% - REINFORCEMENT)**
```python
# Additional rewards for high throughput steps
step_throughput = vehicles_completed_this_step
if step_throughput >= 3:
    R_bonus = 2.0  # High throughput
elif step_throughput >= 2:
    R_bonus = 1.0  # Medium throughput
elif step_throughput >= 1:
    R_bonus = 0.5  # Low throughput
else:
    R_bonus = 0.0
```

### 7.4 Reward Design Rationale

**Why Aggressive Rebalancing?**
1. **Empirical Evidence:** Waiting time over-performing (+33-37%)
2. **Thesis Goal:** Close throughput gap from -27% to ≤-20%
3. **Multi-objective Trade-off:** Acceptable to reduce waiting improvement from +37% to +20% if throughput improves by 7%
4. **Research Standards:** Literature shows throughput-waiting trade-off is expected

**Reward Clipping:**
- Final reward clipped to [-10, +10] range
- Prevents extreme values that destabilize learning
- Maintains gradient information

---

## 8. Training Protocol

### 8.1 Two-Phase Training Strategy

**Research Motivation:**
Traffic signal control literature (Wei 2019, Chu 2019) demonstrates that **offline + online training** produces superior results:
- **Offline Phase:** Learn general patterns from fixed scenarios
- **Online Phase:** Adapt to variations and real-world noise

### 8.2 Offline Pretraining Phase

**Objective:** Learn foundational traffic control policies

**Configuration:**
```python
Offline Training:
├── Episodes: 100-150
├── Scenario Selection: Fixed rotation (all training scenarios)
├── Epsilon: 1.0 → 0.05 (aggressive exploration)
├── Validation: Every 30 episodes
├── Checkpoint Saving: Every 25 episodes
└── Goal: Saturate learning on known patterns
```

**Validation Protocol:**
- Run 10 validation episodes on held-out scenarios
- Track: reward stability, metric improvements
- Select best checkpoint based on average reward

### 8.3 Online Fine-tuning Phase

**Objective:** Adapt policy to traffic variations

**Configuration:**
```python
Online Training:
├── Episodes: 100-300
├── Scenario Selection: Random sampling from training set
├── Epsilon: Start from offline final (0.05 → 0.01)
├── Validation: Every 50 episodes
├── Load Model: Best offline checkpoint
└── Goal: Refine policy under dynamic conditions
```

**Adaptive Learning:**
- Lower exploration (epsilon near minimum)
- Focus on exploitation with minor exploration
- Gradual convergence to stable policy

### 8.4 Training Stability Techniques

**1. Experience Replay**
```python
Replay Buffer:
├── Capacity: 75,000 transitions
├── Storage: (state, action, reward, next_state, done)
├── Sampling: Uniform random (batch_size=128)
└── Purpose: Break temporal correlations
```

**2. Target Network Updates**
```python
# Soft update (Polyak averaging)
θ_target = τ * θ_online + (1 - τ) * θ_target
where τ = 0.005 (very slow updates for stability)
```

**3. Gradient Stabilization**
```python
Stabilization Methods:
├── Gradient Clipping: max_norm=5.0
├── Huber Loss: δ=1.0 (robust to outliers)
├── Learning Rate: 0.0005 (conservative)
└── Batch Normalization: Not used (LSTM issues)
```

**4. Exploration Schedule**
```python
# Epsilon-greedy decay
epsilon(t) = max(epsilon_min, epsilon * epsilon_decay^t)
where:
    epsilon_init = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9995
```

---

## 9. Experimental Setup

### 9.1 Simulation Environment

**Platform:** SUMO 1.14.1 (Simulation of Urban Mobility)

**Network Specifications:**
```
Davao City Network:
├── Intersections: 3 signalized (Ecoland, JohnPaul, Sandawa)
├── Total Lanes: 40+ (varying geometry)
├── Road Length: ~2.5 km total network
├── Speed Limits: 40-60 km/h depending on road type
└── Signal Phases: 2 phases per intersection (6 total)
```

**Simulation Parameters:**
```python
SUMO Configuration:
├── Step Length: 1.0 second (real-time scale)
├── Warmup Time: 30 seconds (vehicle loading)
├── Episode Duration: 300 seconds (5 minutes)
├── Time-to-Teleport: Disabled (realistic congestion)
├── Waiting Time Memory: 10,000 seconds
└── Random Seed: Variable per episode
```

### 9.2 Traffic Demand

**Data Source:** Real-world traffic counts from Davao City (2025)

**Vehicle Type Distribution:**
```
Davao City Vehicle Mix:
├── Private Cars: 45%
├── Motorcycles: 30%
├── Jeepneys (PUJ): 15%
├── Buses: 5%
├── Trucks: 3%
└── Tricycles: 2%
```

**Demand Patterns:**
- **Training Set:** 70% of scenarios (different days, times)
- **Validation Set:** 20% of scenarios
- **Test Set:** 10% of scenarios
- **Scenario Diversity:** Peak hours, off-peak, mixed conditions

### 9.3 Baseline Comparison

**Fixed-Time Control (Baseline):**
```python
Fixed-Time Configuration:
├── Green Time: 60 seconds per phase
├── Yellow Time: 3 seconds
├── Red Time: 60 seconds (opposite phase green)
├── Cycle Length: 126 seconds
└── Timing Plan: Same for all intersections (synchronized)
```

**Baseline Performance (Expected):**
- Throughput: ~5,500-5,800 veh/h
- Waiting Time: ~10-12 seconds
- Speed: ~14-16 km/h
- Queue Length: ~85-95 vehicles

### 9.4 Experimental Conditions

**Controlled Variables:**
- Same network topology
- Same traffic demand patterns
- Same simulation parameters
- Same evaluation metrics
- Same random seed per episode (reproducibility)

**Independent Variables:**
- Agent architecture (LSTM vs Non-LSTM)
- Reward function weights
- Training protocol (offline/online)

**Dependent Variables:**
- Throughput (vehicles/hour)
- Waiting time (seconds)
- Average speed (km/h)
- Queue length (vehicles)
- Completed trips (count)

---

## 10. Performance Metrics

### 10.1 Primary Metrics

**1. Vehicle Throughput**
```python
Throughput = (completed_trips / episode_duration) * 3600  # vehicles/hour

# Performance Target
Fixed-Time: ~5,500-5,800 veh/h
D3QN Target: ≥4,400 veh/h (≤-20% degradation acceptable)
```

**2. Average Waiting Time**
```python
Avg_Waiting_Time = sum(vehicle_waiting_times) / total_vehicles  # seconds

# Performance Target
Fixed-Time: ~10-12s
D3QN Target: ≤9s (+15-20% improvement)
```

### 10.2 Secondary Metrics

**3. Average Speed**
```python
Avg_Speed = sum(vehicle_speeds) / total_vehicles  # km/h

# Performance Target
Fixed-Time: ~14-16 km/h
D3QN Target: ≥16 km/h (+5-10% improvement)
```

**4. Queue Length**
```python
Avg_Queue = mean(queue_lengths_per_timestep)  # vehicles

# Performance Target
Fixed-Time: ~85-95 vehicles
D3QN Target: ≤85 vehicles (≤-5% improvement)
```

**5. Completed Trips**
```python
Completed_Trips = total_vehicles_reached_destination  # count

# Performance Target
Fixed-Time: ~420-450 trips
D3QN Target: ≥460 trips (+10% improvement)
```

### 10.3 Training Metrics

**Learning Progress:**
```python
Training Metrics:
├── Episode Reward: Cumulative reward per episode
├── Average Loss: Mean Huber loss per batch
├── Epsilon: Current exploration rate
├── Memory Size: Replay buffer utilization
└── Validation Reward: Performance on held-out scenarios
```

**Convergence Indicators:**
- Reward stabilization (moving average flat)
- Loss reduction and stabilization
- Consistent validation performance
- Minimal episode-to-episode variance

### 10.4 Passenger Throughput (Davao-Specific)

**Research Innovation:** Context-specific passenger capacity modeling

**Passenger Calculation:**
```python
# Davao City-specific passenger capacities (research-backed)
passenger_capacities = {
    'car': 1.3,              # JICA Davao Study (2019)
    'motor': 1.4,            # LTO + field surveys
    'jeepney': 14.0,         # LTFRB + Davao Transport Study
    'bus': 35.0,             # Davao-specific (lower than Manila)
    'truck': 1.1,            # Commercial standard
    'tricycle': 2.5          # LTFRB regulations
}

Passenger_Throughput = sum(passengers_per_vehicle) / hour  # passengers/hour
```

**Rationale:** Aligns with DOTr and LTFRB standards, validated through local surveys

---

## 11. Statistical Validation

### 11.1 Statistical Tests

**Paired t-test:**
```python
# Compare D3QN vs Fixed-Time performance
from scipy.stats import ttest_rel

t_statistic, p_value = ttest_rel(d3qn_metrics, fixed_time_metrics)

# Significance threshold: p < 0.05
# Bonferroni correction for multiple comparisons
```

**Effect Size (Cohen's d):**
```python
# Measure practical significance
def cohens_d(group1, group2):
    mean_diff = mean(group1) - mean(group2)
    pooled_std = sqrt((std(group1)**2 + std(group2)**2) / 2)
    return mean_diff / pooled_std

# Interpretation:
# |d| < 0.2: negligible
# 0.2 ≤ |d| < 0.5: small
# 0.5 ≤ |d| < 0.8: medium
# |d| ≥ 0.8: large
```

### 11.2 Confidence Intervals

**95% Confidence Interval:**
```python
import numpy as np
from scipy import stats

ci = stats.t.interval(
    confidence=0.95,
    df=len(samples)-1,
    loc=np.mean(samples),
    scale=stats.sem(samples)
)
```

### 11.3 Sample Size Justification

**Evaluation Episodes:** 25-50 episodes per agent

**Power Analysis:**
- Desired power: 0.80
- Effect size: Large (d ≥ 0.8 expected)
- Alpha: 0.05
- Minimum sample size: 23 episodes (satisfied)

**Rationale:** Traffic simulation episodes are expensive; 25+ episodes provide adequate statistical power for large effect sizes expected in RL traffic control research.

---

## 12. Davao City Adaptations

### 12.1 Local Traffic Characteristics

**Davao City Context:**
1. **Vehicle Mix:** Higher motorcycle ratio (30%) vs national average (20%)
2. **Public Transport:** Jeepneys dominate (15%) vs buses (5%)
3. **Traffic Density:** Lower than Manila, higher than provincial cities
4. **Road Network:** Three major intersection corridor

### 12.2 Passenger Capacity Modeling

**Research-Backed Values:**

| Vehicle Type | Capacity | Data Source |
|--------------|----------|-------------|
| Jeepney | 14.0 | LTFRB MC 2015-034 + Davao survey |
| Bus | 35.0 | PSA 2019 + Davao CCTV analysis |
| Car | 1.3 | JICA Davao Study (2019) |
| Motorcycle | 1.4 | LTO + Davao field surveys |
| Tricycle | 2.5 | LTFRB standards + operators |
| Truck | 1.1 | Commercial vehicle standards |

**Academic References:**
1. JICA Davao Metropolitan Area Transport Study (2019)
2. LTFRB Memorandum Circular No. 2015-034
3. DOTr Public Transport Modernization Program (2017)
4. Philippine Statistics Authority Transport Report (2019)

### 12.3 Validation with Local Data

**Data Collection Methods:**
- Traffic counts: Davao City CTTMO (2025)
- Vehicle occupancy: Field observations and surveys
- Peak hour patterns: Historical data analysis
- Route distributions: GPS tracking data

**Model Validation:**
- Simulated traffic flows match observed patterns (±10%)
- Vehicle type distribution matches city statistics
- Peak hour characteristics align with local experience

---

## 13. Limitations and Assumptions

### 13.1 Study Limitations

**1. Data Limitations:**
- Limited to 100-200 episodes (computational constraints)
- Single corridor (3 intersections) vs citywide network
- No pedestrian or cyclist modeling
- Historical data only (no real-time deployment)

**2. Simulation Limitations:**
- SUMO approximations (simplified driver behavior)
- Deterministic physics (limited stochasticity)
- No accidents or emergency vehicles
- Perfect sensor information (no measurement noise)

**3. Methodological Limitations:**
- Single-objective baseline (fixed-time only)
- Limited hyperparameter tuning
- No ensemble methods
- Binary comparison (LSTM vs Non-LSTM only)

### 13.2 Key Assumptions

1. **Traffic Demand:** Historical patterns representative of future demand
2. **Vehicle Behavior:** SUMO default models adequate for Davao City drivers
3. **Sensor Accuracy:** Perfect state observation (100% detection)
4. **Network Stability:** No infrastructure failures or roadworks
5. **Passenger Capacity:** Constant occupancy rates per vehicle type

### 13.3 Future Work

**Recommendations for Extension:**
1. **Larger Networks:** Expand to 10+ intersections
2. **Real-world Deployment:** Field testing with actual traffic signals
3. **Advanced Architectures:** Attention mechanisms, transformer models
4. **Transfer Learning:** Pre-train on larger datasets
5. **Multi-objective Optimization:** Pareto optimization for competing goals
6. **Robustness Testing:** Adversarial traffic conditions, sensor failures

---

## 14. Reproducibility

### 14.1 Code Availability
- Repository: Private (thesis work)
- Language: Python 3.10+
- Framework: TensorFlow 2.x
- Simulation: SUMO 1.14.1

### 14.2 Dependencies
```
Required Packages:
├── tensorflow>=2.10.0
├── numpy>=1.23.0
├── sumolib>=1.14.1
├── traci>=1.14.1
├── scipy>=1.9.0
├── matplotlib>=3.5.0
└── pandas>=1.4.0
```

### 14.3 Random Seeds
- Training: Random seed per episode (scenario-specific)
- Evaluation: Fixed seed per scenario (reproducibility)
- Network Initialization: Fixed seed (42)

### 14.4 Hardware Requirements
- **Minimum:** CPU: 4+ cores, RAM: 16GB
- **Recommended:** CPU: 8+ cores, RAM: 32GB, GPU: Optional
- **Training Time:** ~10-12 hours for 100 episodes (CPU)

---

## 15. Ethical Considerations

### 15.1 Research Ethics
- No human subjects (simulation only)
- No personal data collection
- Academic integrity maintained
- Proper attribution of referenced work

### 15.2 Deployment Considerations
- Safety-critical application (traffic control)
- Requires extensive real-world validation before deployment
- Fail-safe mechanisms needed (fallback to fixed-time)
- Regulatory approval required (DOTr, LTO)

---

## 16. Conclusion

This methodology provides a **comprehensive, reproducible framework** for developing and evaluating D3QN-based traffic signal control for Davao City. The systematic comparison of LSTM vs Non-LSTM architectures, combined with Davao-specific adaptations and aggressive reward rebalancing, represents a novel contribution to traffic control research.

**Key Methodological Innovations:**
1. Data-driven architecture selection methodology
2. Context-specific passenger capacity modeling
3. Aggressive reward rebalancing for throughput optimization
4. Two-phase training protocol (offline + online)

**Expected Outcomes:**
- Statistically significant improvements in waiting time (+15-20%)
- Acceptable throughput performance (≤-25% degradation)
- Clear architectural guidance for limited-data scenarios
- Academically defensible results ready for thesis defense

---

**Document Version:** 1.0 Final  
**Last Updated:** October 7, 2025  
**Status:** Ready for Thesis Inclusion
