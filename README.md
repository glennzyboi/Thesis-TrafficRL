# 🚦 D3QN Traffic Signal Control System

A comprehensive implementation of Dueling Double Deep Q-Network (D3QN) with LSTM for intelligent traffic signal control using SUMO. This system provides an academically rigorous framework for reinforcement learning-based traffic optimization.

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

## 🚀 Key Features

- **Advanced D3QN + LSTM**: Temporal pattern learning for traffic control
- **Hybrid Training**: Combines offline pre-training with online fine-tuning
- **Statistical Rigor**: Academic-grade statistical validation (power analysis, effect sizes, confidence intervals)
- **Multi-Agent Support**: Distributed intersection control
- **Production Logging**: Comprehensive performance tracking
- **Real-World Data**: Validated on Davao City traffic patterns

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

## 🔧 Configuration

Key configuration options in training scripts:

```python
config = {
    'training_mode': 'hybrid',      # hybrid/online/offline
    'episodes': 500,                # Total training episodes
    'learning_rate': 0.0005,        # Optimized learning rate
    'memory_size': 50000,           # Experience replay buffer
    'sequence_length': 10,          # LSTM temporal window
    'validation_freq': 25,          # Validation interval
}
```

## 📄 License

This project is developed for academic research purposes. Please cite appropriately if used in academic work.

## 🤝 Contributing

This is a research implementation. For questions or collaboration opportunities, please refer to the documentation or contact the development team.