# D3QN Multi-Agent Traffic Signal Control System

A complete Multi-Agent Reinforcement Learning (MARL) system for traffic signal control using Dueling Double Deep Q-Networks (D3QN) with real traffic data integration.

## 🚀 Overview

This system trains AI agents to control traffic signals at multiple intersections simultaneously using **real field data**. The agents learn to coordinate signal timing across intersections to optimize traffic flow, reduce waiting times, and improve overall network performance.

### Key Features

- **🤖 Multi-Agent Reinforcement Learning**: D3QN agents control multiple intersections
- **📊 Real Data Integration**: Train on actual traffic observations from field studies
- **🔄 Bundle-Based Training**: Synchronized scenarios across all intersections
- **🖥️ SUMO Integration**: Full traffic simulation with GUI visualization
- **📈 Performance Tracking**: Comprehensive training metrics and model saving

## Project Structure

```
D3QN/
├── data/
│   ├── raw/                        # Raw Excel files from field observations
│   ├── processed/                  # Processed CSV bundles and scenarios
│   └── routes/
│       └── consolidated/           # MARL-ready route files (bundle_day_cycle.rou.xml)
├── scripts/
│   ├── compile_bundles.py          # Process Excel → CSV bundles
│   ├── generate_scenario_routes.py # Generate routes from real data
│   └── consolidate_bundle_routes.py # Merge routes for MARL
├── network/                        # SUMO network files (.net.xml)
├── models/                         # Trained D3QN models
├── *.py                           # Core training and environment files
├── train_bundle_d3qn.py           # Main MARL training script
├── train_d3qn.py                  # Single-agent training script
├── traffic_env.py                 # SUMO environment wrapper
├── d3qn_agent.py                  # D3QN agent implementation
└── requirements.txt               # Python dependencies
```

## 🔧 Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install SUMO

Download and install SUMO from [https://www.eclipse.org/sumo/](https://www.eclipse.org/sumo/)

Ensure `SUMO_HOME` environment variable is set (the system will auto-detect common installation paths).

## 📊 Data Pipeline

### 1. Raw Data Collection

Place Excel files with traffic observations in `data/raw/`:
- Format: `INTERSECTION_YYYYMMDD_cycleN.xlsx`
- Contains vehicle counts by type (car, motor, jeepney, bus, truck, tricycle)
- Includes cycle times and passenger throughput data

### 2. Process Raw Data

```bash
python scripts/compile_bundles.py
```

This creates:
- `data/processed/master_bundles.csv` - Combined dataset
- `data/processed/scenarios_index.csv` - Bundle index
- `out/scenarios/` - Individual scenario CSV files

### 3. Generate Routes

```bash
python scripts/generate_scenario_routes.py
```

Creates intersection-specific route files from real traffic data.

### 4. Consolidate for MARL

```bash
python scripts/consolidate_bundle_routes.py
```

Merges individual intersection routes into synchronized bundle files for true MARL training.

## 🤖 Training

### Multi-Agent Training (Recommended)

Train agents on synchronized traffic scenarios across all intersections:

```bash
python train_bundle_d3qn.py --episodes 100
```

Features:
- **True MARL**: All intersections active simultaneously
- **Bundle Selection**: Random sampling from real traffic scenarios
- **Synchronized Traffic**: Agents learn to coordinate across intersections
- **GUI Visualization**: Watch agents learn in real-time

### Single-Agent Training

Train on individual scenarios:

```bash
python train_d3qn.py --episodes 50
```

## 🎯 System Architecture

### Multi-Agent Reinforcement Learning

```
Bundle Selection → Consolidated Route File → SUMO Simulation → Agent Actions
     ↓                       ↓                      ↓              ↓
Real Traffic Data → All Intersections → State Observation → Signal Control
     ↑                       ↑                      ↑              ↑
Field Observations ← Traffic Lights ← Reward Calculation ← Performance Metrics
```

### State Space (per intersection)
- Queue lengths per lane
- Waiting times per vehicle
- Current signal phase
- Phase duration
- Traffic flow rates

### Action Space
- Signal phase selection for each intersection
- Coordinated timing decisions across network

### Reward Function
- Minimizes total waiting time
- Reduces queue lengths
- Optimizes passenger throughput
- Balances network-wide performance

## 📈 Current Performance

### Traffic Scenarios
Your system currently includes:
- **6 Traffic Bundles**: 2 days × 3 cycles per day
- **3 Intersections**: ECOLAND, JOHNPAUL, SANDAWA
- **Realistic Traffic**: 20-160 vehicles per scenario
- **Synchronized Data**: True multi-intersection coordination

### Vehicle Types & Parameters
- **Car**: Length 5m, Max Speed 40 km/h
- **Motor**: Length 2m, Max Speed 40 km/h  
- **Jeepney**: Length 8m, Max Speed 40 km/h
- **Bus**: Length 12m, Max Speed 40 km/h
- **Truck**: Length 10m, Max Speed 40 km/h

## 🎮 Usage Examples

### Quick Training Run
```bash
# Train for 5 episodes with visualization
python train_bundle_d3qn.py --episodes 5
```

### Production Training
```bash
# Train for 100+ episodes for optimal performance
python train_bundle_d3qn.py --episodes 200
```

### Add New Traffic Data
1. Place Excel files in `data/raw/`
2. Run `python scripts/compile_bundles.py`
3. Run `python scripts/generate_scenario_routes.py`
4. Run `python scripts/consolidate_bundle_routes.py`
5. Train: `python train_bundle_d3qn.py --episodes N`

## 🔬 Key Innovation: True MARL

This system implements **true Multi-Agent Reinforcement Learning** by:

1. **Synchronized Training**: All intersections use traffic data from the same day/cycle
2. **Coordinated Decision Making**: Agents observe and influence each other's performance
3. **Realistic Interactions**: Traffic flows naturally between intersections
4. **Bundle-Based Episodes**: Each training episode represents a complete traffic scenario

Unlike systems that train individual intersections separately, this approach learns **network-wide coordination** patterns from real traffic data.

## 📊 Data Sources

- **Field Observations**: Manual vehicle counting at intersections
- **Traffic Patterns**: Real demand variations by time and location  
- **Vehicle Mix**: Authentic Philippine traffic composition
- **Cycle Times**: Actual signal timing from field measurements

## 🚀 Future Enhancements

- **Expand Network**: Add more intersections to the MARL system
- **Temporal Learning**: Include time-of-day and day-of-week patterns
- **Adaptive Signals**: Real-time adjustment to traffic conditions
- **Performance Analysis**: Compare against fixed-time and actuated control
- **Deployment**: Integration with real traffic management systems

## 📄 License

This project implements research methodologies for traffic signal optimization using reinforcement learning and real traffic data integration.