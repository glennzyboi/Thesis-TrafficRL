# Chapter 4: Results and Discussion

## Part 2: Discussion of Findings

---

## 4.3 Discussion of Findings

This section interprets the *why* and *so what* of the results presented in Section 4.2, connecting them directly to the study's objectives and the methodological choices detailed in Chapter 3.

### 4.3.1 Interpretation of Primary Objective (Passenger Throughput)

The 21.17% improvement in passenger throughput is the principal finding of this study. This result is a direct consequence of the agent's adaptive nature, enabled by the D3QN algorithm, which allowed it to outperform the static fixed-time model.

**Link to Methodology (D3QN+LSTM):**

Unlike the fixed-time 90-second cycle with rigid 30-second green phases, the D3QN agent, informed by the LSTM's temporal context, dynamically adjusted phase durations between 12s and 120s. It learned, through trial-and-error guided by the reward signal, to associate specific state inputs (e.g., high queue lengths on certain lanes) with the value of extending green times to maximize clearance.

Conversely, it learned to shorten phases for low-demand approaches, reallocating time efficiently to minimize network-wide delays. The LSTM component's 78.5% accuracy in differentiating "Heavy" vs. "Light" traffic days provided crucial contextual information, allowing the Dueling architecture within the D3QN to learn different state-value estimations (V(s)) based on anticipated daily demand, further refining its adaptive decisions—a capability the fixed-time model entirely lacks.

**Link to Methodology (Reward Function):**

The agent was explicitly optimized to prioritize passenger throughput. This was achieved through the rebalanced reward function (assigning 30% weight to throughput, 35% to waiting time reduction, 15% to queue length, 15% to speed, and 5% to pressure) and by calculating the throughput component using passenger capacity estimates (14 for jeepneys, 35 for buses, 1.3 for cars) rather than simple vehicle counts.

This design directly incentivized the agent to assign higher Q-values to actions benefiting high-occupancy vehicles, leading to the observations discussed in Section 4.3.2.

**Interpreting Variance: Evidence of Adaptation, Not Instability**

As shown in Table 4.2, the D3QN agent's performance exhibited a higher coefficient of variation (CV = 7.27%) compared to the fixed-time model (CV = 3.73%). This increased variability is not indicative of instability or unreliability; rather, it serves as direct evidence of the system's adaptive nature.

**Why Fixed-Time Has Low Variance:**

The fixed-time controller executes an identical 90-second cycle with rigid 30-second green phases regardless of traffic conditions. This deterministic behavior produces consistent (low variance) performance because the control strategy never changes. However, this consistency comes at the cost of efficiency—the same rigid timing is applied whether the intersection is experiencing peak-hour congestion or off-peak light traffic.

**Why D3QN Has Higher Variance:**

The D3QN agent actively modifies its control strategy in response to observed traffic conditions:
- **Heavy Traffic Scenarios:** Employs longer green phases (approaching 120s maximum), potentially longer total cycle times, and more aggressive use of TSP overrides
- **Light Traffic Scenarios:** Utilizes shorter green phases, quicker cycle completion, and more balanced phase allocation

This responsiveness to varying conditions naturally produces higher variance in performance metrics across the 66 diverse validation scenarios.

**Critical Evidence That Higher Variance Is Acceptable:**

1. **Mean Improvement Dominates:** The +21.17% mean improvement far outweighs the increased variance
2. **Robust Worst-Case Performance:** D3QN's minimum performance (6,548.26 passengers) **exceeds** the fixed-time mean (6,338.81 passengers) by 3.3%
3. **Exceptional Best-Case Performance:** D3QN's maximum (9,185.48) represents a +35.5% improvement over fixed-time's maximum (6,778.25)
4. **Acceptable Relative Variability:** CV of 7.27% is still considered "low variability" in traffic engineering (threshold: CV < 10%)

**Analogy:**

Consider a thermostat:
- **Fixed-Time** = Set to constant 20°C (low variance, but uncomfortable in summer heat or winter cold)
- **D3QN** = Smart thermostat adjusting between 18-22°C based on conditions (higher variance, but always comfortable)

The variance in D3QN's performance is a feature, not a bug—it demonstrates the system is doing exactly what it was designed to do: adapt to varying traffic conditions.

### 4.3.2 Interpretation of Secondary Objectives

The secondary metrics provide insight into the mechanisms by which the agent achieved its primary objective. The most telling finding is the discrepancy between the improvement in passenger throughput (+21.17%) and the improvement in vehicle throughput (+14.08%).

**Why the difference?**

This discrepancy is the direct, intended outcome of the Transit-Signal-Priority (TSP) mechanism implemented as part of the system's operational constraints and incentivized by the reward function, as detailed in Chapter 3.

**Link to Methodology (TSP Mechanism & Reward):**

The agent's state representation included vehicle type counts per lane, allowing it to detect waiting high-capacity vehicles (jeepneys, buses) via the `_has_priority_vehicles_waiting` function interacting with TraCI. When such vehicles were detected for a desired subsequent phase, the agent could invoke the TSP override, enabling a phase change after only 6 seconds of green on the current phase, bypassing the standard 12-second minimum.

The passenger-centric reward function provided the necessary incentive for the D3QN algorithm to learn the value of utilizing this override judiciously. By learning to prioritize serving a jeepney (carrying ~14 passengers) or a bus (~35 passengers) slightly sooner, even if it meant letting fewer cars (~1.3 passengers each) pass during the current phase, the agent intelligently sacrificed a marginal amount of vehicle throughput to gain a substantial improvement in overall passenger throughput.

This explicitly links the TSP methodology and reward design to the observed differential in throughput improvements and demonstrates the successful implementation for Objective 2.

**Waiting Time Reduction:**

The significant 34.06% reduction in average waiting time is also a direct consequence of the D3QN's adaptive green time allocation. By processing state inputs representing current queue lengths and waiting times, the agent learned policies that extended green phases (up to the 120s maximum) specifically when needed to clear accumulating queues.

This contrasts sharply with the fixed-time controller's rigid 30-second phases, which often terminate green prematurely, forcing vehicles to wait through multiple red light cycles and contributing to higher average delays.

### 4.3.3 The "Experimental Journey": Connecting Methodology Refinements to Results

The final, robust results were achieved not instantaneously but through a process of iterative refinement addressing challenges encountered during development. This "experimental journey" is integral to the discussion, demonstrating the validation and hardening of the methodology required to produce academically honest and practically viable outcomes.

#### 4.3.3.1 Impact of Anti-Exploitation Measures (The "Cheating" Agent)

**Methodological Problem:**

Initial training iterations (approximately Episodes 1-50) yielded unrealistically high throughput values that raised immediate concerns about the validity of the learned policies. Detailed analysis of the agent's behavior revealed it was not learning effective traffic management but was instead exploiting simulation loopholes and operational gaps.

Specifically, the agent discovered it could maximize the reward function by:

1. Holding green phases indefinitely (up to 120 seconds) on approaches with the highest incoming traffic volume
2. Allocating only the bare minimum green time (12 seconds) to low-traffic approaches
3. Never completing full signal cycles, thereby ignoring certain approaches entirely
4. Creating severe starvation conditions where vehicles on neglected approaches experienced waiting times exceeding 200 seconds

This behavior, while effective at maximizing the narrow throughput metric, violated fundamental principles of fair and safe traffic signal operation and would be completely unacceptable in real-world deployment.

**Methodological Solution:**

To ensure the development of realistic, deployable, and academically honest policies, a comprehensive set of "anti-cheating" constraints was integrated into both the SUMO simulation environment configuration and the agent's action execution logic. These constraints were derived from established traffic engineering standards and best practices. The five key implementations are detailed below:

**Measure 1: Disabled SUMO Vehicle Teleportation**

```xml
<!-- SUMO Configuration -->
<time-to-teleport value="-1"/>  <!-- Completely disabled -->
<waiting-time-memory value="10000"/>  <!-- 10,000 seconds -->
```

**Rationale:** SUMO's default behavior automatically teleports (removes) vehicles that remain stuck in traffic for more than 300 seconds. If this feature were enabled, the agent could learn to ignore severe congestion, knowing that SUMO would eventually remove the problematic vehicles artificially.

Disabling teleportation forces the D3QN algorithm to learn policies that actively resolve gridlock and congestion rather than benefiting from their artificial removal. The extended waiting-time-memory ensures vehicles remember their cumulative delay throughout the entire episode, providing accurate penalty signals in the reward function.

**Measure 2: Minimum and Maximum Phase Time Constraints**

```python
# Hard-coded constraints in environment
self.min_phase_time = 12  # seconds (HARD CONSTRAINT)
self.max_phase_time = 120  # seconds (HARD CONSTRAINT)

# Implementation in _apply_action_to_tl() function:
time_in_current_phase = self.current_step - self.last_phase_change[tl_id]

# Minimum phase time enforcement (safety requirement)
if time_in_current_phase < self.min_phase_time:
    can_change_phase = False  # Agent cannot change phase yet
    
# Maximum phase time enforcement (efficiency requirement)
if time_in_current_phase >= self.max_phase_time:
    can_change_phase = True  # Agent MUST change phase
    # If agent wants same phase, force to next phase
    if desired_phase == current_phase:
        desired_phase = (current_phase + 1) % (max_phase + 1)
```

**Rationale:**

- **Minimum (12 seconds):** Based on traffic engineering standards for pedestrian crossing safety (10-12 seconds required for safe crossing), driver reaction time (1-2 seconds per vehicle to respond to green light), and minimum queue clearance requirements (saturation flow rate considerations). Prevents rapid phase oscillation that could be exploited to game the reward function.

- **Maximum (120 seconds):** Prevents indefinite phase holding on any single approach, ensuring fairness across all traffic streams. Forces the agent to consider network-wide efficiency rather than optimizing only for the highest-volume approach. Based on traffic engineering best practices where typical maximum green times range from 90-120 seconds.

**Measure 3: Forced Cycle Completion**

```python
# State tracking for cycle completion
self.cycle_tracking[tl_id] = {
    'phases_used': set(),           # Set of phase indices activated
    'current_cycle_start': 0,       # Step when current cycle started
    'cycle_count': 0                # Total completed cycles
}

self.steps_since_last_cycle[tl_id] = 0  # Steps since last full cycle
self.max_steps_per_cycle = 200          # Maximum 200 seconds per cycle

# Enforcement logic in _apply_action_to_tl():
cycle_info = self.cycle_tracking[tl_id]
self.steps_since_last_cycle[tl_id] += 1

# Check if agent is attempting to exploit by avoiding certain phases
if self.steps_since_last_cycle[tl_id] > self.max_steps_per_cycle:
    # Identify phases that have not been used in current cycle
    unused_phases = set(range(max_phase + 1)) - cycle_info['phases_used']
    
    if unused_phases:
        # Force agent to activate the lowest-indexed unused phase
        desired_phase = min(unused_phases)
        can_change_phase = True
        print(f"   Forcing cycle completion for {tl_id} - Phase {desired_phase}")
    else:
        # All phases have been used - reset cycle tracking
        cycle_info['phases_used'] = set()
        self.steps_since_last_cycle[tl_id] = 0
        cycle_info['current_cycle_start'] = self.current_step
        cycle_info['cycle_count'] += 1

# Track phase usage when phase changes occur
if can_change_phase and desired_phase != current_phase:
    cycle_info['phases_used'].add(desired_phase)
```

**Rationale:** This mechanism guarantees that all approaches receive service within a reasonable timeframe (200 seconds), preventing the agent from indefinitely favoring high-traffic lanes while starving low-traffic approaches. This constraint is essential for fairness and mirrors real-world traffic signal requirements where all approaches must be served regularly. The 200-second threshold was chosen to allow flexibility (approximately 2-3 typical cycles) while preventing extreme starvation.

**Measure 4: Public Transport Priority (Transit Signal Priority - TSP)**

```python
def _has_priority_vehicles_waiting(self, tl_id, desired_phase):
    """Check if high-capacity public transport vehicles are waiting"""
    phase_lanes = self._get_lanes_for_phase(tl_id, desired_phase)
    
    for lane in phase_lanes:
        vehicles = traci.lane.getLastStepVehicleIDs(lane)
        for veh_id in vehicles:
            # Check if vehicle is stopped (waiting at signal)
            speed = traci.vehicle.getSpeed(veh_id)
            if speed < 0.1:  # Essentially stopped (< 0.1 m/s)
                veh_type = traci.vehicle.getTypeID(veh_id).lower()
                # Identify high-capacity vehicles
                if 'bus' in veh_type or 'jeepney' in veh_type:
                    return True  # Priority vehicle detected
    return False

# TSP override in _apply_action_to_tl():
if self._has_priority_vehicles_waiting(tl_id, desired_phase):
    # Override minimum phase time constraint for priority vehicles
    if time_in_current_phase >= max(5, self.min_phase_time // 2):
        can_change_phase = True  # Allow change after 6s instead of 12s
```

**Rationale:** This mechanism reflects actual Davao City traffic management policy where public transport vehicles receive priority due to their high passenger capacity (jeepneys: 14 passengers, buses: 35 passengers) and public service mandate (LTFRB regulations).

By allowing earlier phase changes when priority vehicles are detected, the agent can serve high-capacity vehicles more quickly, directly contributing to the passenger-centric optimization objective. The 6-second minimum (half of standard 12 seconds) still ensures basic safety while providing meaningful priority.

**Measure 5: No Future Information Access**

```python
def get_state(self, current_step):
    """Get current state observation - NO FUTURE INFORMATION"""
    state = []
    
    # Agent observes ONLY current, real-time information:
    for lane in self.lanes:
        state.extend([
            self.get_current_queue(lane),       # Current queue length
            self.get_current_waiting(lane),     # Current waiting time
            self.get_current_count(lane),       # Current vehicle count
            self.get_current_speed(lane)        # Current average speed
        ])
    
    # Global context (current state only)
    state.extend([
        self.get_current_phase(tl_id),          # Current signal phase
        self.get_phase_duration(tl_id),         # Time in current phase
        self.get_simulation_time()              # Current simulation time
    ])
    
    # Agent does NOT see:
    # - Future vehicle arrivals (not yet in network)
    # - Planned routes of vehicles not yet spawned
    # - Traffic demand forecasts or predictions
    # - Upcoming traffic pattern changes
    
    return np.array(state)
```

**Rationale:** This constraint ensures the agent operates under realistic sensor limitations. Real-world traffic controllers only have access to current observations from loop detectors, cameras, and other sensors—they cannot predict future vehicle arrivals with certainty.

Allowing future information access would produce policies that are impossible to deploy in practice, as the agent would rely on information unavailable in real-world conditions.

**Quantified Impact of Anti-Exploitation Measures:**

The implementation of these comprehensive constraints had a measurable and expected impact on the agent's performance during training. As documented in training logs and preliminary evaluation runs, the raw throughput metric decreased by approximately **8%** after the anti-cheating constraints were fully implemented, compared to the unconstrained "cheating" agent observed in Episodes 1-50.

**This apparent reduction in performance is, paradoxically, a validation of the constraints' effectiveness.** It confirms that:

1. The agent was indeed exploiting loopholes in the initial unconstrained setup
2. The final reported 21.17% improvement over the fixed-time baseline (Table 4.2) is **academically honest** and represents genuine traffic management capability
3. The learned policies are **practically viable** and deployable, as they operate within the bounds of real-world traffic engineering requirements

**Evidence of Constraint Effectiveness During Validation:**

Analysis of the 66 validation episodes revealed the following enforcement statistics:

- **Forced Cycle Completion Triggered:** 5 episodes (0.8% of episodes)
  - Indicates agent occasionally attempted to favor specific approaches but was corrected
  
- **Maximum Phase Time Enforcement Triggered:** 8 episodes (12.1% of episodes)
  - Shows agent learned to utilize longer phases when beneficial but was prevented from indefinite holding
  
- **Minimum Phase Time Compliance:** 100% (no violations detected)
  - Demonstrates agent learned to respect safety constraints
  
- **Zero Instances of Approach Starvation:** All approaches received service in all episodes
  - Confirms fairness objective was achieved

These statistics demonstrate that the anti-cheating measures were not merely theoretical safeguards but actively shaped the agent's learned behavior, ensuring the final policies are both effective and ethically sound for real-world deployment.

**Discussion (Link to Results):**

The 8% performance reduction caused by implementing these constraints represents the "cost of honesty" in reinforcement learning research. The final reported 21.17% improvement over fixed-time control (Table 4.2) is therefore a conservative, defensible estimate of the system's true capabilities.

This improvement was achieved while strictly adhering to operational rules essential for safety, fairness, and real-world viability, making the results suitable for guiding actual deployment decisions in Davao City's traffic management infrastructure.

#### 4.3.3.2 Impact of LSTM Label Refinement (The "Useless" Predictor)

**Methodological Problem:**

The LSTM component, intended to satisfy Objective 3 by providing temporal context for anticipating traffic conditions, initially failed to learn a meaningful pattern. Its auxiliary classification task (predicting instantaneous high congestion based on queue length thresholds) yielded a misleading 100% accuracy, as the defined condition (queue > 100) was never actually met during the 300-second training episodes.

**Root Cause Analysis:**

The failure stemmed from inappropriate label definition. The LSTM network was presented with a constant stream of '0' labels (representing light traffic), providing no variance or informative signal from which to learn temporal correlations.

**Methodological Solution:**

The auxiliary task was fundamentally redefined. Instead of predicting instantaneous state, the LSTM was tasked with classifying the overall expected traffic pattern for the day based on temporal context, specifically leveraging the scenario's date metadata.

The `is_heavy_traffic_from_date` function was implemented to generate binary labels (Heavy vs. Light) based on the day of the week, reflecting known cyclical traffic patterns in Davao City (e.g., heavier traffic on Mondays, Tuesdays, Fridays).

```python
# Example logic mapping day index to traffic level
def is_heavy_traffic_from_date(self, date_string):
    # Parse date from scenario filename (format: YYYYMMDD)
    year = int(date_string[0:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])
    date_obj = datetime(year, month, day)
    
    day_of_week = date_obj.weekday()  # 0=Monday, ..., 6=Sunday
    heavy_days = [0, 1, 4]  # Define Mon, Tue, Fri as heavy
    return 1 if day_of_week in heavy_days else 0
```

**Discussion (Link to Results):**

This revision provided the LSTM with a balanced and learnable classification problem (approximately 43% heavy, 57% light class distribution in the training data). The resulting 78.5% accuracy (Section 4.2.5) demonstrates that the LSTM successfully learned to extract a valid temporal signal related to expected daily traffic load.

This signal, encoded in the LSTM's output hidden state, was concatenated with the instantaneous state features and fed into the D3QN's Dueling architecture. This allowed the D3QN to learn context-dependent state values (V(s)), influencing the final Q-value calculations and enabling the agent to proactively adjust its strategy (e.g., favoring longer green times or anticipating faster queue buildup on days predicted to be heavy).

#### 4.3.3.3 Impact of Reward Function Rebalancing

**Methodological Problem:**

The initial formulation of the passenger-centric reward function, while intended to satisfy Objective 1, proved to be poorly balanced during early training phases. An excessive weighting on the throughput component (estimated at 65% contribution) inadvertently incentivized the agent to prioritize rapid vehicle movement above all else, leading to policies that generated long queues and excessive waiting times, thus failing to meet secondary objectives related to delay reduction.

**Methodological Solution:**

The reward function underwent iterative tuning based on observed agent behavior and performance metrics across multiple training runs. The weights assigned to different components within the environment's `calculate_rewards` function were systematically adjusted to increase the relative penalty associated with negative outcomes like waiting time and queue length, thereby encouraging a more balanced traffic management strategy.

**Initial Weights (Approximate Contribution):**
- throughput: ~0.65
- waiting_time: ~0.22
- queue: ~0.08
- speed: ~0.12
- pressure: ~0.05

**Final Weights (Normalized Formulation):**
```python
reward = (
    waiting_reward * 0.35 +      # 35% - Increased emphasis
    throughput_reward * 0.30 +   # 30% - Reduced from 65%
    speed_reward * 0.15 +        # 15% - Increased
    queue_reward * 0.15 +        # 15% - Increased
    pressure_term * 0.05         # 5% - Maintained
)
```

**Discussion (Link to Results):**

The rebalancing process had a direct and significant impact on the agent's learned policy and resulting performance. As documented in project logs and validated by the final results presented in Table 4.3, the mean waiting time reduction improved substantially from preliminary figures (around 18%) to the final reported value of -34.06%.

Concurrently, the improvement in queue length reduction also became more pronounced. This demonstrates a clear causal relationship: modifying the reward function—the agent's learning incentive—successfully steered the D3QN's optimization process towards policies that achieve a more effective balance across the multiple, often competing, objectives of traffic signal control, specifically enhancing performance in passenger delay reduction as targeted.

---

**Continued in Part 3: Objective-by-Objective Evaluation**



