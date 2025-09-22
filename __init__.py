"""
D3QN Traffic Signal Control System

A comprehensive implementation of Dueling Double Deep Q-Network (D3QN) with LSTM 
for intelligent traffic signal control using SUMO simulation.

Academic Framework:
- Hybrid offline/online training paradigms
- Statistical validation with power analysis
- Production-grade logging and evaluation
- Real-world traffic data integration

Package Structure:
- algorithms: Core RL algorithms and baselines
- core: Environment and simulation components  
- evaluation: Performance analysis and validation
- experiments: Training scripts and configurations
- utils: Supporting utilities and logging
"""

__version__ = "1.0.0"
__author__ = "D3QN Research Team"
__description__ = "Academic-grade D3QN traffic signal control system"

# Core imports for package-level access
from algorithms.d3qn_agent import D3QNAgent
from core.traffic_env import TrafficEnvironment
from experiments.comprehensive_training import ComprehensiveTrainer
from evaluation.performance_comparison import PerformanceComparator
from utils.production_logger import create_production_logger

__all__ = [
    'D3QNAgent',
    'TrafficEnvironment', 
    'ComprehensiveTrainer',
    'PerformanceComparator',
    'create_production_logger'
]
