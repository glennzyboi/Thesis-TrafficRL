# 🚦 D3QN Traffic Signal Control with Public Transport Priority

A comprehensive implementation of enhanced Dueling Double Deep Q-Network (D3QN) with LSTM for **public transport priority** traffic signal control using SUMO. This system provides an academically rigorous framework for reinforcement learning-based traffic optimization with specific focus on maximizing passenger throughput through bus and jeepney priority.

## 📁 Current Project Structure

```
D3QN/
├── algorithms/                    # Core RL algorithms and baselines
│   ├── __init__.py               # Package initialization
│   ├── d3qn_agent.py            # Main D3QN+LSTM agent implementation
│   └── fixed_time_baseline.py   # Fixed-time baseline controller
├── core/                         # Core environment and simulation
│   ├── __init__.py              # Package initialization
│   └── traffic_env.py           # SUMO traffic environment wrapper
├── evaluation/                   # Performance analysis and validation
│   ├── __init__.py              # Package initialization
│   ├── performance_comparison.py # Enhanced statistical comparison framework
│   ├── results_analysis.py      # Comprehensive result analysis
│   ├── hyperparameter_validation.py # Hyperparameter optimization
│   └── reward_function_validation.py # Reward function validation
├── experiments/                  # Training scripts and configurations
│   ├── __init__.py              # Package initialization
│   ├── comprehensive_training.py # Main hybrid training orchestrator
│   └── train_d3qn.py            # Basic training implementation
├── utils/                        # Utilities and supporting functions
│   ├── __init__.py              # Package initialization
│   └── production_logger.py     # Production-grade logging system
├── scripts/                      # Data processing and route generation
│   ├── compile_bundles.py       # Process Excel → CSV bundles
│   ├── generate_scenario_routes.py # Generate SUMO routes from data
│   └── consolidate_bundle_routes.py # Merge routes for training
├── data/                         # Training data and scenarios
│   ├── raw/                     # Original Excel data files (108 files)
│   ├── processed/               # Processed CSV scenarios
│   └── routes/                  # Generated SUMO route files (24 scenarios)
├── docs/                         # Comprehensive documentation
│   ├── comprehensive_methodology.md    # Complete methodology
│   ├── DEFENSE_PREPARATION_COMPLETE.md # Academic defense prep
│   ├── TECHNICAL_IMPLEMENTATION_GUIDE.md # Technical details
│   └── [other documentation files]
├── network/                      # SUMO network definition
│   └── ThesisNetowrk.net.xml    # Main intersection network
├── comprehensive_results/        # Training results and analysis
├── models/                       # Trained model checkpoints
├── production_logs/             # Production logging outputs
├── requirements.txt             # Python dependencies (enhanced)
└── __init__.py                  # Package root initialization
```

## 🚀 Enhanced Key Features

### **🚌 Public Transport Priority System (Ma et al., 2020)**
- **Lane-Level PT Detection**: Real-time counting of buses and jeepneys in each lane
- **Passenger Load Estimation**: Dynamic capacity calculation (Bus: 40, Jeepney: 14 passengers)
- **Priority Urgency Metrics**: PT vehicle waiting times for intelligent priority decisions
- **System-Wide PT Context**: Global public transport load monitoring

### **🧠 Advanced D3QN + LSTM Architecture**
- **Enhanced State Space**: 12 features per lane (638 total dimensions) with PT priority
- **Temporal Dynamics**: Velocity features capture traffic buildup/dissipation patterns (Liang et al., 2019)
- **Granular Control**: 18-action space with phase selection + timing variations
- **LSTM Sequence Learning**: 10-timestep temporal memory for pattern recognition

### **🎓 Academic Standards & Research Foundation**
- **Realistic Timing**: FHWA/Webster standards (15-75s) based on traffic engineering research
- **Literature Implementation**: Genders & Razavi (2016), Wei et al. (2019), Ma et al. (2020)
- **Statistical Rigor**: Academic-grade validation with proper data splits and significance testing
- **Working Phase Analysis**: Network debugging identified functional vs. non-functional phases

### **🔬 Production-Grade Implementation**
- **Hybrid Training**: Research-optimized 70-30 offline/online split for extended learning
- **Overfitting Prevention**: Early stopping, reward stability monitoring, enhanced regularization
- **Real-World Data**: 198 raw Excel files → 66 processed traffic scenarios from Davao City
- **Defense-Ready**: Comprehensive logging, reproducibility, and academic documentation

## 🔬 Academic Framework

### Statistical Methodology
- **Power Analysis**: Ensures adequate sample sizes (n ≥ 20)
- **Effect Size Calculation**: Cohen's d with magnitude interpretation
- **Confidence Intervals**: 95% CI for mean differences
- **Multiple Comparison Correction**: Bonferroni correction for multiple metrics
- **Assumption Testing**: Normality (Shapiro-Wilk) and equal variance (Levene's test)
- **Non-parametric Alternatives**: Wilcoxon signed-rank when assumptions violated

### Training Paradigms
1. **Offline Pre-training** (70% episodes): Stable learning from replay buffer
2. **Online Fine-tuning** (30% episodes): Real-time adaptation to new scenarios
3. **Hybrid Approach**: Best of both worlds for robust performance

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- SUMO (Simulation of Urban Mobility)
- TensorFlow 2.x

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set SUMO_HOME environment variable
export SUMO_HOME="/path/to/sumo"
```

### Basic Training
```bash
# Run comprehensive training with hybrid mode (default)
python experiments/comprehensive_training.py --experiment_name test_run --episodes 5

# Run full training with hybrid mode
python experiments/comprehensive_training.py --experiment_name my_experiment --episodes 500

# Run statistical comparison (requires 20+ episodes for validity)
python evaluation/performance_comparison.py --num_episodes 25
```

### Training Modes

**Hybrid Training (Recommended - matches study framework)**:
```bash
python experiments/comprehensive_training.py --experiment_name hybrid_500ep --episodes 500
```

**Pure Online Learning**:
```bash
python experiments/comprehensive_training.py --experiment_name online_500ep --episodes 500 --training_mode online
```

**Pure Offline Learning**:
```bash
python experiments/comprehensive_training.py --experiment_name offline_500ep --episodes 500 --training_mode offline
```

**Quick Test (5 episodes)**:
```bash
python experiments/comprehensive_training.py --experiment_name test_5ep --episodes 5
```

## 📊 Performance Evaluation

The system includes comprehensive evaluation tools:

- **Statistical Comparison**: Rigorous paired t-tests with effect sizes
- **Training Analysis**: Overfitting detection and convergence monitoring
- **Performance Visualization**: Publication-ready plots and charts
- **Academic Reporting**: Detailed statistical summaries

## 🏛️ Academic Validation

This implementation follows best practices from traffic signal control literature:

- **Sample Size**: Minimum 20 episodes for statistical power
- **Data Splitting**: Temporal train/validation/test splits (70/20/10)
- **Multiple Metrics**: Throughput, waiting time, speed, queue length
- **Baseline Comparison**: Fixed-time controllers with realistic timing
- **Real-World Data**: Field-collected traffic counts from Davao City

## 📖 Documentation

Complete documentation is available in the `docs/` folder:

- `comprehensive_methodology.md`: Detailed methodology documentation
- `DEFENSE_PREPARATION_COMPLETE.md`: Academic defense preparation
- `TECHNICAL_IMPLEMENTATION_GUIDE.md`: Technical implementation details

## 🔧 Current Stable Configuration

The system has been stabilized with the following production-ready configuration:

### **Traffic Light Control**
```python
# Working phases only (fixes SUMO network issues)
working_phases = {
    'Ecoland_TrafficSignal': [0, 2, 4, 6],      # 4 functional phases
    'JohnPaul_TrafficSignal': [0, 5, 8],        # 3 functional phases  
    'Sandawa_TrafficSignal': [0, 2]             # 2 functional phases
}

# Enhanced action space: Granular control with timing variations
actions_per_tl = {
    'Ecoland_TrafficSignal': 8,      # 4 phases + 4 timing variations
    'JohnPaul_TrafficSignal': 6,     # 3 phases + 3 timing variations  
    'Sandawa_TrafficSignal': 4       # 2 phases + 2 timing variations
}

# Timing constraints (Academic Standards)
min_phase_time = 15  # seconds (FHWA safety standard)
max_phase_time = 75  # seconds (Webster's optimal range)
```

### **Training Parameters**
```python
config = {
    'training_mode': 'hybrid',          # 70% offline + 30% online
    'episodes': 200,                    # Recommended for stability
    'learning_rate': 0.0001,            # Reduced for stability
    'memory_size': 50000,               # Experience replay buffer
    'batch_size': 64,                   # Balanced batch size
    'sequence_length': 10,              # LSTM temporal window
    'epsilon_decay': 0.999,             # Gradual exploration decay
}
```

### **Enhanced State Representation (Public Transport Priority)**
```python
# Based on Ma et al. (2020) and transit priority literature
state_features_per_lane = [
    # Core traffic metrics (5 features)
    'queue_length', 'waiting_time', 'avg_speed', 'flow_rate', 'occupancy',
    
    # PUBLIC TRANSPORT PRIORITY (3 features) - CRITICAL ENHANCEMENT
    'pt_vehicles_count',     # Buses/jeepneys in lane
    'pt_passenger_load',     # Estimated passenger capacity  
    'pt_waiting_time',       # Priority urgency metric
    
    # TEMPORAL DYNAMICS (4 features) - Liang et al. (2019)
    'vel_queue',             # Rate of change in queue length
    'vel_waiting',           # Rate of change in waiting time
    'vel_flow',              # Rate of change in flow rate
    'vel_pt'                 # Rate of change in PT vehicles
]

# Global context: time_context + pt_system_load
# Total: 53 lanes × 12 features + 2 global = 638 state dimensions
```

### **🕒 Temporal Dynamics Explained**

**Academic Foundation**: Based on Liang et al. (2019) "Temporal traffic pattern learning for urban signal control"

**What are Temporal Dynamics?**
- **Definition**: Rate of change (velocity) of traffic metrics over time
- **Purpose**: Capture traffic buildup and dissipation patterns
- **Benefit**: Agent learns trends, not just current state

**Implementation**:
```python
# Example: Queue length temporal dynamics
current_queue = 15 vehicles
previous_queue = 10 vehicles
vel_queue = current_queue - previous_queue = +5 vehicles/step

# Interpretation:
# +5: Queue building up (congestion forming)
# -3: Queue dissipating (traffic clearing)  
#  0: Stable queue (equilibrium state)
```

**Real-World Application**:
- **Predictive Control**: Agent anticipates traffic buildup before it becomes severe
- **Proactive Decisions**: Changes signals before congestion peaks
- **Pattern Recognition**: LSTM learns recurring traffic patterns (rush hour, etc.)
- **PT Priority**: Detects when PT vehicles are accumulating, triggering priority

### **Reward Function Weights**
```python
reward_weights = {
    'waiting_penalty': 0.20,        # 20% - Minimize delays
    'queue_penalty': 0.15,          # 15% - Congestion control  
    'speed_reward': 0.20,           # 20% - Flow efficiency
    'passenger_throughput': 0.20,   # 20% - Urban planning priority
    'vehicle_throughput': 0.15,     # 15% - Enhanced for balance
    'public_transport_bonus': 0.10, # 10% - PT priority
}
```

## 📄 License

This project is developed for academic research purposes. Please cite appropriately if used in academic work.

## 🤝 Contributing

This is a research implementation. For questions or collaboration opportunities, please refer to the documentation or contact the development team.