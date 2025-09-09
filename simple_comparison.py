"""
Simplified Performance Comparison - Fixed-Time vs D3QN
Focuses on core metrics without complex model loading
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# Set SUMO_HOME
if 'SUMO_HOME' not in os.environ:
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

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

import traci
from fixed_time_baseline import FixedTimeController


def run_optimized_d3qn_episode(route_file, use_pretrained=True):
    """Run optimized D3QN episode with enhanced agent for fair comparison"""
    from traffic_env import TrafficEnvironment
    from d3qn_agent import D3QNAgent
    
    # Initialize environment with identical parameters as fixed-time
    env = TrafficEnvironment(
        net_file='network/ThesisNetowrk.net.xml',
        rou_file=route_file,
        use_gui=False,
        num_seconds=180,
        warmup_time=30,
        step_length=1.0
    )
    
    # Initialize optimized D3QN agent
    state = env.reset()
    state_size = len(state)
    action_size = env.action_size
    
    agent = D3QNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.0005,   # Optimized parameters
        epsilon=0.1,            # Lower epsilon for more exploitation
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        memory_size=50000,
        batch_size=64
    )
    
    # Try to load pre-trained model if available
    model_path = "models/best_bundle_d3qn_model.h5"
    if use_pretrained and os.path.exists(model_path):
        try:
            # Enable unsafe deserialization for Lambda layers
            import tensorflow as tf
            tf.keras.config.enable_unsafe_deserialization()
            
            # Try to load the model
            loaded_model = tf.keras.models.load_model(model_path, safe_mode=False)
            agent.q_network = loaded_model
            agent.target_network = tf.keras.models.clone_model(loaded_model)
            agent.target_network.set_weights(loaded_model.get_weights())
            agent.epsilon = 0.05  # Low exploration for evaluation
            print(f"   🧠 Using pre-trained D3QN model with optimized reward function")
        except Exception as e:
            print(f"   ⚠️ Could not load pre-trained model: {e}")
            print(f"   🆕 Using fresh optimized agent with enhanced reward function")
            agent.epsilon = 0.2  # Some exploration for new agent
    else:
        print(f"   🆕 Using fresh optimized agent with enhanced reward function")
        agent.epsilon = 0.2  # Some exploration for new agent
    
    # Run episode with identical structure to fixed-time
    state = env.reset()
    step_data = []
    total_reward = 0
    
    print(f"   🧠 Running optimized D3QN simulation...")
    
    for step in range(150):  # Identical duration to fixed-time
        # D3QN action selection
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        # Store experience and learn
        agent.remember(state, action, reward, next_state, done)
        if len(agent.memory) > agent.batch_size:
            loss = agent.replay()
        
        # Collect metrics (identical to fixed-time collection)
        step_data.append({
            'vehicles': info.get('vehicles', 0),
            'waiting_time': info.get('waiting_time', 0),
            'avg_speed': info.get('avg_speed', 0),
            'queue_length': info.get('queue_length', 0),
            'completed_trips': info.get('completed_trips', 0),
            'throughput': info.get('throughput', 0),
            'passenger_throughput': info.get('passenger_throughput', 0),  # PRIMARY METRIC
            'total_passenger_throughput': info.get('total_passenger_throughput', 0)
        })
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Calculate metrics using identical method as fixed-time
    if step_data:
        metrics = {
            'avg_waiting_time': np.mean([d['waiting_time'] for d in step_data]),
            'avg_speed': np.mean([d['avg_speed'] for d in step_data]),
            'avg_queue_length': np.mean([d['queue_length'] for d in step_data]),
            'max_queue_length': max([d['queue_length'] for d in step_data]),
            'completed_trips': step_data[-1]['completed_trips'],
            'avg_throughput': np.mean([d['throughput'] for d in step_data if d['throughput'] > 0]),
            'avg_passenger_throughput': np.mean([d['passenger_throughput'] for d in step_data if d['passenger_throughput'] > 0]),  # PRIMARY METRIC
            'total_passenger_throughput': step_data[-1]['total_passenger_throughput'] if step_data[-1]['total_passenger_throughput'] > 0 else 0,
            'total_reward': total_reward,
            'final_epsilon': agent.epsilon,
            'memory_size': len(agent.memory)
        }
    else:
        metrics = {}
    
    env.close()
    return metrics


def run_comparison():
    """Run simplified comparison"""
    print("🚀 OPTIMIZED D3QN vs FIXED-TIME COMPARISON")
    print("=" * 70)
    print("Fixed-Time Control vs Optimized D3QN Agent")
    print("🔬 FAIR COMPARISON: Identical routes, duration, and parameters")
    
    # Test routes
    route_files = [
        "data/routes/consolidated/bundle_20250828_cycle_1.rou.xml",
        "data/routes/consolidated/bundle_20250829_cycle_1.rou.xml",
        "data/routes/consolidated/bundle_20250829_cycle_3.rou.xml"
    ]
    
    results = {
        'fixed_time': [],
        'd3qn_optimized': [],
        'scenarios': []
    }
    
    for i, route_file in enumerate(route_files):
        if not os.path.exists(route_file):
            print(f"⚠️ Skipping missing file: {route_file}")
            continue
            
        scenario_name = f"Scenario {i+1}"
        print(f"\n📺 TESTING {scenario_name}")
        print(f"📂 Route: {os.path.basename(route_file)}")
        
        # Fixed-Time Control
        print(f"\n🔧 Fixed-Time Control...")
        try:
            if traci.isLoaded():
                traci.close()
            
            controller = FixedTimeController(
                net_file='network/ThesisNetowrk.net.xml',
                rou_file=route_file,
                use_gui=False,  # Disable GUI for batch processing
                num_seconds=180,
                warmup_time=30,
                step_length=1.0
            )
            
            fixed_metrics = controller.run_simulation()
            fixed_metrics['scenario'] = scenario_name
            results['fixed_time'].append(fixed_metrics)
            
            print(f"✅ Fixed-Time: {fixed_metrics['avg_throughput']:.1f} veh/h, "
                  f"{fixed_metrics['avg_waiting_time']:.1f}s wait")
            
        except Exception as e:
            print(f"❌ Fixed-Time failed: {e}")
            continue
        
        # Optimized D3QN Agent
        print(f"\n🧠 Optimized D3QN Agent...")
        try:
            if traci.isLoaded():
                traci.close()
            
            d3qn_metrics = run_optimized_d3qn_episode(route_file)
            d3qn_metrics['scenario'] = scenario_name
            results['d3qn_optimized'].append(d3qn_metrics)
            
            print(f"✅ Optimized D3QN: Vehicle: {d3qn_metrics.get('avg_throughput', 0):.1f} veh/h, "
                  f"Passenger: {d3qn_metrics.get('avg_passenger_throughput', 0):.1f} pass/h (PRIMARY), "
                  f"{d3qn_metrics.get('avg_waiting_time', 0):.1f}s wait, "
                  f"Reward: {d3qn_metrics.get('total_reward', 0):.1f}")
            
        except Exception as e:
            print(f"❌ Optimized D3QN failed: {e}")
            continue
        
        # Quick comparison
        if results['fixed_time'] and results['d3qn_optimized']:
            fixed = results['fixed_time'][-1]
            d3qn = results['d3qn_optimized'][-1]
            
            # Vehicle throughput comparison
            vehicle_throughput_diff = ((d3qn.get('avg_throughput', 0) - fixed['avg_throughput']) / 
                                     max(fixed['avg_throughput'], 1) * 100)
            
            # PASSENGER THROUGHPUT COMPARISON (PRIMARY OBJECTIVE)
            passenger_throughput_diff = ((d3qn.get('avg_passenger_throughput', 0) - fixed.get('avg_passenger_throughput', 0)) / 
                                       max(fixed.get('avg_passenger_throughput', 1), 1) * 100)
            
            waiting_diff = ((fixed['avg_waiting_time'] - d3qn.get('avg_waiting_time', 0)) / 
                           max(fixed['avg_waiting_time'], 1) * 100)
            speed_diff = ((d3qn.get('avg_speed', 0) - fixed['avg_speed']) / 
                         max(fixed['avg_speed'], 1) * 100)
            
            print(f"\n📊 {scenario_name} Results:")
            print(f"   🚗 D3QN vehicle throughput improvement: {vehicle_throughput_diff:+.1f}%")
            print(f"   🚌 D3QN PASSENGER throughput improvement: {passenger_throughput_diff:+.1f}% (PRIMARY OBJECTIVE)")
            print(f"   ⏱️ D3QN waiting time improvement: {waiting_diff:+.1f}%") 
            print(f"   🏎️ D3QN speed improvement: {speed_diff:+.1f}%")
            print(f"   🎯 D3QN total reward: {d3qn.get('total_reward', 0):+.1f}")
        
        results['scenarios'].append(scenario_name)
        
        # Brief pause between tests
        time.sleep(1)
    
    # Generate summary
    generate_summary(results)
    

def generate_summary(results):
    """Generate summary visualization"""
    print(f"\n📊 GENERATING SUMMARY ANALYSIS...")
    
    if not results['fixed_time'] or not results['d3qn_optimized']:
        print("❌ Insufficient data for analysis")
        return
    
    # Create comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimized D3QN vs Fixed-Time Control Performance', fontsize=16, fontweight='bold')
    
    scenarios = results['scenarios']
    
    # Throughput comparison
    fixed_throughput = [r['avg_throughput'] for r in results['fixed_time']]
    d3qn_throughput = [r.get('avg_throughput', 0) for r in results['d3qn_optimized']]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax1.bar(x - width/2, fixed_throughput, width, label='Fixed-Time', color='lightblue')
    ax1.bar(x + width/2, d3qn_throughput, width, label='Optimized D3QN', color='lightgreen')
    ax1.set_xlabel('Scenarios')
    ax1.set_ylabel('Throughput (veh/h)')
    ax1.set_title('Average Throughput')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Waiting time comparison
    fixed_waiting = [r['avg_waiting_time'] for r in results['fixed_time']]
    d3qn_waiting = [r.get('avg_waiting_time', 0) for r in results['d3qn_optimized']]
    
    ax2.bar(x - width/2, fixed_waiting, width, label='Fixed-Time', color='lightblue')
    ax2.bar(x + width/2, d3qn_waiting, width, label='Optimized D3QN', color='lightgreen')
    ax2.set_xlabel('Scenarios')
    ax2.set_ylabel('Waiting Time (s)')
    ax2.set_title('Average Waiting Time')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Speed comparison
    fixed_speed = [r['avg_speed'] for r in results['fixed_time']]
    d3qn_speed = [r.get('avg_speed', 0) for r in results['d3qn_optimized']]
    
    ax3.bar(x - width/2, fixed_speed, width, label='Fixed-Time', color='lightblue')
    ax3.bar(x + width/2, d3qn_speed, width, label='Optimized D3QN', color='lightgreen')
    ax3.set_xlabel('Scenarios')
    ax3.set_ylabel('Speed (km/h)')
    ax3.set_title('Average Speed')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Queue length comparison
    fixed_queue = [r['avg_queue_length'] for r in results['fixed_time']]
    d3qn_queue = [r.get('avg_queue_length', 0) for r in results['d3qn_optimized']]
    
    ax4.bar(x - width/2, fixed_queue, width, label='Fixed-Time', color='lightblue')
    ax4.bar(x + width/2, d3qn_queue, width, label='Optimized D3QN', color='lightgreen')
    ax4.set_xlabel('Scenarios')
    ax4.set_ylabel('Queue Length')
    ax4.set_title('Average Queue Length')
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenarios)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('comparison_results', exist_ok=True)
    plt.savefig('comparison_results/d3qn_vs_fixed_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"📊 Comparison chart saved: comparison_results/d3qn_vs_fixed_time_comparison.png")
    
    # Print summary statistics
    print(f"\n📈 COMPREHENSIVE PERFORMANCE ANALYSIS:")
    print(f"   Fixed-Time Average Throughput: {np.mean(fixed_throughput):.1f} veh/h")
    print(f"   Optimized D3QN Average Throughput: {np.mean(d3qn_throughput):.1f} veh/h")
    print(f"   Fixed-Time Average Waiting: {np.mean(fixed_waiting):.1f}s")
    print(f"   Optimized D3QN Average Waiting: {np.mean(d3qn_waiting):.1f}s")
    print(f"   Fixed-Time Average Speed: {np.mean(fixed_speed):.1f} km/h")
    print(f"   Optimized D3QN Average Speed: {np.mean(d3qn_speed):.1f} km/h")
    
    # Calculate improvements (positive means D3QN is better)
    throughput_improvement = ((np.mean(d3qn_throughput) - np.mean(fixed_throughput)) / 
                             max(np.mean(fixed_throughput), 1) * 100)
    waiting_improvement = ((np.mean(fixed_waiting) - np.mean(d3qn_waiting)) / 
                          max(np.mean(fixed_waiting), 1) * 100)
    speed_improvement = ((np.mean(d3qn_speed) - np.mean(fixed_speed)) / 
                        max(np.mean(fixed_speed), 1) * 100)
    queue_improvement = ((np.mean(fixed_queue) - np.mean(d3qn_queue)) / 
                        max(np.mean(fixed_queue), 1) * 100)
    
    print(f"\n🎯 OPTIMIZED D3QN PERFORMANCE vs FIXED-TIME:")
    print(f"   Throughput improvement: {throughput_improvement:+.1f}%")
    print(f"   Waiting time reduction: {waiting_improvement:+.1f}%")
    print(f"   Speed improvement: {speed_improvement:+.1f}%")
    print(f"   Queue reduction: {queue_improvement:+.1f}%")
    
    # Overall performance score
    overall_score = (throughput_improvement + waiting_improvement + speed_improvement + queue_improvement) / 4
    print(f"\n🏆 OVERALL D3QN ADVANTAGE: {overall_score:+.1f}%")
    
    if overall_score > 5:
        print("   ✅ D3QN significantly outperforms fixed-time control!")
    elif overall_score > 0:
        print("   ✅ D3QN shows improvement over fixed-time control")
    else:
        print("   ⚠️ D3QN needs further optimization to outperform fixed-time control")
    
    plt.show()


if __name__ == "__main__":
    run_comparison()
