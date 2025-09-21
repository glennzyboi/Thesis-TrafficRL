# 🧪 D3QN Traffic Control System - User Testing Guide

## 🎯 Overview
This guide provides comprehensive testing commands for evaluating the trained D3QN model performance on specific test data, including data split verification and performance analysis.

## 📊 Data Split Verification

### Check Available Data Splits
```bash
# Verify train/validation/test split implementation
python -c "
from train_d3qn import load_scenarios_index
print('=== DATA SPLIT VERIFICATION ===')
for split in ['train', 'validation', 'test']:
    scenarios = load_scenarios_index(split=split)
    print(f'{split.upper()}: {len(scenarios)} scenarios')
    if scenarios:
        print(f'  Range: {scenarios[0]['name']} to {scenarios[-1]['name']}')
    print()
"
```

### View Detailed Split Information
```bash
# Get detailed information about data splits
python -c "
from train_d3qn import load_scenarios_index
import json

splits = {}
for split_name in ['train', 'validation', 'test']:
    scenarios = load_scenarios_index(split=split_name)
    splits[split_name] = {
        'count': len(scenarios),
        'scenarios': [s['name'] for s in scenarios] if scenarios else []
    }

print(json.dumps(splits, indent=2))
"
```

## 🏆 Model Performance Testing

### 1. Test Trained Model on Specific Test Data

#### Single Episode Test (Quick)
```bash
# Test the best trained model on one test scenario with GUI
python performance_comparison.py --episodes 1 --gui --test_split
```

#### Comprehensive Test (Multiple Episodes)
```bash
# Test on all test scenarios (no GUI for speed)
python performance_comparison.py --episodes 10 --test_split
```

#### Test with Specific Scenario
```bash
# Test on validation split for intermediate evaluation
python performance_comparison.py --episodes 5 --validation_split --gui
```

### 2. Test Specific Model Checkpoint

#### Test Best Model
```bash
# Test the best saved model
python -c "
from performance_comparison import run_model_test
from pathlib import Path

# Find the latest comprehensive results
result_dirs = list(Path('comprehensive_results').glob('*'))
if result_dirs:
    latest_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_dir / 'models' / 'best_model.keras'
    
    print(f'Testing model: {model_path}')
    results = run_model_test(str(model_path), episodes=3, use_gui=True)
    print('Test completed!')
else:
    print('No trained models found!')
"
```

#### Test Specific Checkpoint
```bash
# Test a specific checkpoint (replace with actual checkpoint)
python performance_comparison.py --model_path "comprehensive_results/extended_training_fixed_throughput/models/checkpoint_ep100.keras" --episodes 3 --gui
```

### 3. Advanced Performance Analysis

#### Generate Performance Report for Test Data
```bash
# Create comprehensive performance analysis on test data
python -c "
from performance_comparison import ModelTester
from train_d3qn import load_scenarios_index

# Load test scenarios
test_scenarios = load_scenarios_index(split='test')
print(f'Running analysis on {len(test_scenarios)} test scenarios...')

# Run comprehensive test
tester = ModelTester()
results = tester.run_comprehensive_test(
    test_scenarios=test_scenarios,
    episodes_per_scenario=2,
    save_detailed_results=True
)

print('Test Results Summary:')
print(f'D3QN Average Reward: {results[\"d3qn_avg_reward\"]:.2f}')
print(f'Fixed-Time Average Reward: {results[\"baseline_avg_reward\"]:.2f}')
print(f'Improvement: {results[\"improvement_percent\"]:.1f}%')
"
```

#### Compare Against Multiple Baselines
```bash
# Run full comparison with statistical analysis
python performance_comparison.py --episodes 20 --test_split --statistical_analysis --export_results
```

## 🎮 Interactive Testing Commands

### 1. Visual Testing with GUI

#### Watch Model in Action (Recommended)
```bash
# Watch the D3QN agent control traffic lights in real-time
python -c "
from traffic_env import TrafficEnvironment
from d3qn_agent import D3QNAgent
from train_d3qn import load_scenarios_index, CONFIG
import numpy as np

# Load test scenario
test_bundles = load_scenarios_index(split='test')
if test_bundles:
    bundle = test_bundles[0]  # Use first test scenario
    
    # Initialize environment with GUI
    env = TrafficEnvironment(
        net_file=CONFIG['NET_FILE'],
        rou_file=bundle['consolidated_file'],
        use_gui=True,  # Enable GUI
        num_seconds=600,  # 10 minutes
        warmup_time=60,
        step_length=1.0,
        min_phase_time=8,
        max_phase_time=90
    )
    
    # Load trained agent
    agent = D3QNAgent(state_size=159, action_size=11, sequence_length=10)
    agent.load('models/best_d3qn_model.keras')
    
    print('🎮 INTERACTIVE TEST - Watch the agent control traffic!')
    print(f'📊 Test scenario: {bundle[\"name\"]}')
    print('⏸️  Press SPACE in SUMO GUI to pause/resume')
    print('🔍 Click vehicles to see their info')
    print('⏱️  Simulation will run for 10 minutes')
    
    # Run episode
    state = env.reset()
    total_reward = 0
    step = 0
    
    try:
        while not env._is_done():
            action = agent.act(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            state = next_state
            step += 1
            
            if step % 60 == 0:  # Print every minute
                print(f'⏱️  Step {step}: Reward={reward:.2f}, Total={total_reward:.2f}')
                print(f'   Vehicles: {info.get(\"vehicles\", 0)}, Completed: {info.get(\"completed_trips\", 0)}')
    
    except KeyboardInterrupt:
        print('🛑 Test stopped by user')
    
    finally:
        env.close()
        print(f'✅ Test completed! Total reward: {total_reward:.2f}')
else:
    print('❌ No test scenarios available!')
"
```

#### Compare D3QN vs Fixed-Time Side-by-Side
```bash
# This requires manual setup, but provides visual comparison
# Run D3QN test first, then fixed-time on same scenario

# Step 1: Run D3QN
python performance_comparison.py --episodes 1 --gui --test_split --d3qn_only

# Step 2: Run Fixed-Time (run this in another terminal after first completes)
python performance_comparison.py --episodes 1 --gui --test_split --baseline_only
```

### 2. Quick Performance Checks

#### Check Model Loading
```bash
# Verify model can be loaded successfully
python -c "
from d3qn_agent import D3QNAgent
import os

model_files = [
    'models/best_d3qn_model.keras',
    'comprehensive_results/final_validation/models/best_model.keras'
]

for model_path in model_files:
    if os.path.exists(model_path):
        try:
            agent = D3QNAgent(state_size=159, action_size=11, sequence_length=10)
            agent.load(model_path)
            print(f'✅ Successfully loaded: {model_path}')
        except Exception as e:
            print(f'❌ Failed to load {model_path}: {e}')
    else:
        print(f'⚠️  File not found: {model_path}')
"
```

#### Validate Environment Setup
```bash
# Test environment initialization
python -c "
from traffic_env import TrafficEnvironment
from train_d3qn import CONFIG, load_scenarios_index

test_bundles = load_scenarios_index(split='test')
if test_bundles:
    bundle = test_bundles[0]
    
    try:
        env = TrafficEnvironment(
            net_file=CONFIG['NET_FILE'],
            rou_file=bundle['consolidated_file'],
            use_gui=False,
            num_seconds=60,
            warmup_time=10
        )
        
        state = env.reset()
        print(f'✅ Environment initialized successfully')
        print(f'   State shape: {len(state)}')
        print(f'   Traffic scenario: {bundle[\"name\"]}')
        
        # Test one step
        action = 0  # Hold current phase
        next_state, reward, done, info = env.step(action)
        print(f'   Step test passed: reward={reward:.2f}')
        
        env.close()
        print('✅ Environment test completed successfully')
        
    except Exception as e:
        print(f'❌ Environment test failed: {e}')
else:
    print('❌ No test bundles available')
"
```

## 📈 Results Analysis Commands

### 1. Analyze Latest Training Results
```bash
# Analyze results from the most recent training
python results_analysis.py

# Or analyze specific experiment
python -c "
from results_analysis import ResultsAnalyzer

# Replace with your experiment name
analyzer = ResultsAnalyzer('comprehensive_results/extended_training_fixed_throughput')
results = analyzer.generate_comprehensive_report()
print('Analysis complete!')
"
```

### 2. Export Results for External Analysis
```bash
# Export training data to CSV for analysis in Excel/R/Python
python -c "
import json
import pandas as pd
from pathlib import Path

# Find latest results
result_dirs = list(Path('comprehensive_results').glob('*'))
if result_dirs:
    latest_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
    results_file = latest_dir / 'complete_results.json'
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Export training episodes
        df = pd.DataFrame(data['training_results'])
        export_file = latest_dir / 'training_data_export.csv'
        df.to_csv(export_file, index=False)
        
        print(f'✅ Training data exported to: {export_file}')
        print(f'   Episodes: {len(df)}')
        print(f'   Columns: {list(df.columns)}')
    else:
        print('❌ No results file found')
else:
    print('❌ No result directories found')
"
```

## 🎛️ Custom Testing Scenarios

### 1. Test on Specific Traffic Conditions

#### High Traffic Test
```bash
# Test model performance during high traffic periods
python -c "
from performance_comparison import run_targeted_test

# This would test on scenarios known to have high traffic
results = run_targeted_test(
    traffic_condition='high',
    episodes=5,
    use_gui=True
)
print('High traffic test completed!')
"
```

#### Rush Hour Simulation
```bash
# Test during simulated rush hour conditions
python performance_comparison.py --episodes 3 --rush_hour --gui
```

### 2. Stress Testing

#### Long Duration Test
```bash
# Test model stability over extended periods
python -c "
from traffic_env import TrafficEnvironment
from d3qn_agent import D3QNAgent
from train_d3qn import CONFIG

# Extended test - 30 minutes simulation
env = TrafficEnvironment(
    net_file=CONFIG['NET_FILE'],
    rou_file='data/routes/consolidated/bundle_20250708_cycle_1.rou.xml',
    use_gui=True,
    num_seconds=1800,  # 30 minutes
    warmup_time=120
)

agent = D3QNAgent(state_size=159, action_size=11, sequence_length=10)
agent.load('models/best_d3qn_model.keras')

print('🏋️ STRESS TEST - 30 minute simulation')
print('This will test model stability over extended periods')

state = env.reset()
total_reward = 0

try:
    while not env._is_done():
        action = agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

    print(f'✅ Stress test completed! Total reward: {total_reward:.2f}')
    
except Exception as e:
    print(f'❌ Stress test failed: {e}')
finally:
    env.close()
"
```

## 🚨 Troubleshooting Commands

### Fix Common Issues
```bash
# If SUMO GUI doesn't appear
export DISPLAY=:0  # On Linux
# On Windows, make sure SUMO is in PATH

# If models not found
ls -la models/
ls -la comprehensive_results/*/models/

# If environment fails to initialize
python -c "
import os
print('SUMO_HOME:', os.environ.get('SUMO_HOME', 'Not set'))
print('Current directory:', os.getcwd())
"

# Clear any stuck SUMO processes (Windows)
taskkill /f /im sumo-gui.exe
taskkill /f /im sumo.exe
```

## 📋 Summary Commands for Quick Testing

```bash
# 🔥 QUICK START TESTING COMMANDS 🔥

# 1. Verify everything is working
python -c "from train_d3qn import load_scenarios_index; print('Test scenarios:', len(load_scenarios_index('test')))"

# 2. Quick visual test (5 minutes)
python performance_comparison.py --episodes 1 --gui --test_split

# 3. Comprehensive performance test (no GUI, faster)
python performance_comparison.py --episodes 10 --test_split

# 4. Generate analysis plots
python results_analysis.py

# 5. Check latest training progress
python -c "
from pathlib import Path
result_dirs = list(Path('comprehensive_results').glob('*'))
if result_dirs:
    latest = max(result_dirs, key=lambda x: x.stat().st_mtime)
    print(f'Latest training: {latest.name}')
else:
    print('No training results found')
"
```

## 📊 Expected Results

When testing is successful, you should see:

- **D3QN Performance**: 40-60% improvement in waiting time over fixed-time
- **Vehicle Throughput**: Balanced with passenger optimization
- **Visual Behavior**: Smooth, realistic traffic light changes
- **Convergence**: Clear learning progression in reward curves
- **Stability**: Consistent performance across different test scenarios

## 🎓 Research Validation

The testing framework ensures:
- ✅ **No data leakage** (temporal train/validation/test split)
- ✅ **Reproducible results** (fixed random seeds)
- ✅ **Statistical significance** (multiple episodes per scenario)
- ✅ **Real-world applicability** (realistic timing constraints)
- ✅ **Defense-ready validation** (comprehensive baseline comparison)

---

**Happy Testing! 🚀**

If you encounter any issues, check the troubleshooting section or review the model training logs for debugging information.
