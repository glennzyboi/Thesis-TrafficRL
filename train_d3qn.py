"""
D3QN Traffic Signal Control Training Script
Supports both single-agent and multi-agent (MARL) training modes
Trains LSTM-enhanced D3QN agents to control traffic signals in SUMO
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import pandas as pd
import random
from collections import defaultdict

from d3qn_agent import D3QNAgent
from traffic_env import TrafficEnvironment

# Configuration
CONFIG = {
    # Files
    'NET_FILE': 'network/ThesisNetowrk.net.xml',
    'ROU_FILE': 'data/routes/20250828_cycle_1/ECOLAND_20250828_cycle1.rou.xml',
    
    # Environment settings
    'USE_GUI': False,  # Disable GUI for stability during training
    'EPISODE_DURATION': 300,  # 5 minutes per episode for better learning
    'WARMUP_TIME': 30,   # 30 seconds warmup for quicker start
    'STEP_LENGTH': 1.0,  # 1 second per step
    
    # Training parameters
    'EPISODES': 100,  # Increased episodes for better learning
    'TARGET_UPDATE_FREQ': 10,  # Update target network every N episodes
    'SAVE_FREQ': 10,  # Save model every N episodes
    
    # Optimized Agent parameters (based on SUMO+RL research)
    'LEARNING_RATE': 0.0005,   # Reduced for more stable learning
    'EPSILON': 1.0,             # Start with full exploration
    'EPSILON_MIN': 0.05,        # Higher minimum for continued exploration
    'EPSILON_DECAY': 0.9995,    # Slower decay for longer exploration
    'MEMORY_SIZE': 50000,       # Larger memory for better experience diversity
    'BATCH_SIZE': 64,           # Larger batch for more stable gradients
    
    # Directories
    'MODEL_DIR': 'models',
    'LOGS_DIR': 'logs',
    'PLOTS_DIR': 'plots',
    
    # LSTM parameters
    'SEQUENCE_LENGTH': 10,      # LSTM temporal memory length
    
    # MARL parameters
    'MARL_MODE': False,         # Enable multi-agent training
    'COORDINATION_WEIGHT': 0.1, # Weight for coordination rewards
}

def create_directories():
    """Create necessary directories"""
    for dir_name in [CONFIG['MODEL_DIR'], CONFIG['LOGS_DIR'], CONFIG['PLOTS_DIR']]:
        os.makedirs(dir_name, exist_ok=True)

def load_scenarios_index():
    """Load available bundles from scenarios_index.csv"""
    scenarios_file = os.path.join("data", "processed", "scenarios_index.csv")
    
    if not os.path.exists(scenarios_file):
        print(f"❌ Scenarios index not found: {scenarios_file}")
        return []
    
    df = pd.read_csv(scenarios_file)
    bundles = []
    
    for _, row in df.iterrows():
        day = row['Day']
        cycle = row['CycleNum']
        intersections = row['Intersections'].split(',')
        
        # Check if consolidated route file exists
        consolidated_file = f"data/routes/consolidated/bundle_{day}_cycle_{cycle}.rou.xml"
        
        if os.path.exists(consolidated_file):
            bundles.append({
                'day': day,
                'cycle': cycle,
                'intersections': [i.strip() for i in intersections],
                'consolidated_file': consolidated_file,
                'name': f"Day {day}, Cycle {cycle}"
            })
        else:
            print(f"⚠️ Missing consolidated route file: {consolidated_file}")
    
    return bundles

def select_random_bundle(bundles):
    """Randomly select a bundle for training"""
    if not bundles:
        return None, None
    
    bundle = random.choice(bundles)
    return bundle, bundle['consolidated_file']

def plot_training_progress(episode_rewards, episode_lengths, losses, save_path):
    """Plot and save training progress"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Episode lengths
    ax2.plot(episode_lengths)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    
    # Training losses
    if losses:
        ax3.plot(losses)
        ax3.set_title('Training Loss')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
    
    # Moving average of rewards
    if len(episode_rewards) >= 10:
        moving_avg = []
        window = 10
        for i in range(len(episode_rewards)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(episode_rewards[start:i+1]))
        ax4.plot(moving_avg)
        ax4.set_title(f'Moving Average Rewards (window={window})')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Reward')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_agent():
    """Main training function - supports both single-agent and MARL modes"""
    if CONFIG['MARL_MODE']:
        print("🤝 MARL D3QN TRAFFIC SIGNAL CONTROL TRAINING")
        print("🚦 Training separate agents for each intersection")
        return train_marl_agents()
    else:
        print("🚀 SINGLE-AGENT D3QN TRAFFIC SIGNAL CONTROL TRAINING")
        return train_single_agent()

def train_single_agent():
    """Single-agent training function"""
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Load available bundles for varied training
    bundles = load_scenarios_index()
    
    if not bundles:
        print("❌ No training bundles available! Using default route...")
        route_file = CONFIG['ROU_FILE']
    else:
        print(f"✅ Found {len(bundles)} valid traffic bundles for training")
        # Use first bundle initially
        route_file = bundles[0]['consolidated_file']
    
    # Initialize environment with realistic traffic signal constraints
    print("🏗️ Initializing environment...")
    env = TrafficEnvironment(
        net_file=CONFIG['NET_FILE'],
        rou_file=route_file,
        use_gui=CONFIG['USE_GUI'],
        num_seconds=CONFIG['EPISODE_DURATION'],
        warmup_time=CONFIG['WARMUP_TIME'],
        step_length=CONFIG['STEP_LENGTH'],
        min_phase_time=8,   # 8 seconds minimum (based on ITE standards & RL research)
        max_phase_time=90   # 90 seconds maximum (optimized for urban arterials)
    )
    
    try:
        # Get initial state to determine state size
        initial_state = env.reset()
        state_size = len(initial_state)
        action_size = env.action_size
        
        print(f"🧠 State size: {state_size}, Action size: {action_size}")
        
        # Initialize LSTM-enhanced agent
        agent = D3QNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=CONFIG['LEARNING_RATE'],
            epsilon=CONFIG['EPSILON'],
            epsilon_min=CONFIG['EPSILON_MIN'],
            epsilon_decay=CONFIG['EPSILON_DECAY'],
            memory_size=CONFIG['MEMORY_SIZE'],
            batch_size=CONFIG['BATCH_SIZE'],
            sequence_length=CONFIG['SEQUENCE_LENGTH']
        )
        
        # Training tracking
        episode_rewards = []
        episode_lengths = []
        losses = []
        best_reward = float('-inf')
        
        print(f"\n🎯 Starting training for {CONFIG['EPISODES']} episodes...")
        print("   Press Ctrl+C to stop training early\n")
        
        start_time = time.time()
        
        for episode in range(CONFIG['EPISODES']):
            print(f"📺 Episode {episode + 1}/{CONFIG['EPISODES']}")
            
            # Reset environment and agent state history
            state = env.reset()
            agent.reset_state_history()  # Clear LSTM state history for new episode
            episode_reward = 0
            episode_steps = 0
            
            while True:
                # Agent selects action
                action = agent.act(state, training=True)
                
                # Environment step
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Update state and tracking
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Train agent
                if len(agent.memory) > agent.batch_size:
                    loss = agent.replay()
                    if loss is not None:
                        losses.append(loss)
                
                # Render environment (show progress)
                if episode_steps % 100 == 0:
                    env.render()
                    print(f"     Step {episode_steps}: Reward={reward:.2f}, "
                          f"Vehicles={info['vehicles']}, Epsilon={agent.epsilon:.3f}")
                
                if done:
                    break
            
            # Episode completed
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            
            # Update target network
            if episode % CONFIG['TARGET_UPDATE_FREQ'] == 0:
                agent.update_target_model()
                print(f"     🎯 Target network updated")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                model_path = os.path.join(CONFIG['MODEL_DIR'], 'best_d3qn_model.h5')
                agent.save(model_path)
                print(f"     💾 New best model saved! Reward: {best_reward:.2f}")
            
            # Periodic saves
            if (episode + 1) % CONFIG['SAVE_FREQ'] == 0:
                model_path = os.path.join(CONFIG['MODEL_DIR'], f'd3qn_model_ep{episode + 1}.h5')
                agent.save(model_path)
                
                # Save training progress plot
                plot_path = os.path.join(CONFIG['PLOTS_DIR'], f'training_progress_ep{episode + 1}.png')
                plot_training_progress(episode_rewards, episode_lengths, losses, plot_path)
                
                print(f"     📊 Progress saved: Model and plots updated")
            
            # Episode summary
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            
            print(f"     ✅ Episode completed:")
            print(f"        Reward: {episode_reward:.2f} | Steps: {episode_steps}")
            print(f"        Avg Reward (last 10): {avg_reward:.2f}")
            print(f"        Epsilon: {agent.epsilon:.3f} | Time: {elapsed_time/60:.1f}m")
            print(f"        Memory size: {len(agent.memory)}")
            print()
        
        print("🎉 Training completed!")
        
        # Final saves
        final_model_path = os.path.join(CONFIG['MODEL_DIR'], 'final_d3qn_model.h5')
        agent.save(final_model_path)
        
        final_plot_path = os.path.join(CONFIG['PLOTS_DIR'], 'final_training_progress.png')
        plot_training_progress(episode_rewards, episode_lengths, losses, final_plot_path)
        
        # Training summary
        total_time = time.time() - start_time
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"   Total episodes: {len(episode_rewards)}")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best reward: {best_reward:.2f}")
        print(f"   Average reward: {np.mean(episode_rewards):.2f}")
        print(f"   Final epsilon: {agent.epsilon:.3f}")
        print(f"   Models saved in: {CONFIG['MODEL_DIR']}")
        print(f"   Plots saved in: {CONFIG['PLOTS_DIR']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        
        # Save current progress
        model_path = os.path.join(CONFIG['MODEL_DIR'], 'interrupted_d3qn_model.h5')
        agent.save(model_path)
        
        if episode_rewards:
            plot_path = os.path.join(CONFIG['PLOTS_DIR'], 'interrupted_training_progress.png')
            plot_training_progress(episode_rewards, episode_lengths, losses, plot_path)
        
        print(f"     💾 Progress saved to {CONFIG['MODEL_DIR']}")
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        raise
    
    finally:
        # Clean up
        env.close()
        print("🧹 Environment closed")

def train_marl_agents():
    """Multi-agent training function"""
    from d3qn_agent import MARLAgentManager
    
    print("=" * 60)
    print("🧠 LSTM-enhanced agents with coordination mechanisms")
    
    # Create directories
    create_directories()
    
    # Load available bundles
    bundles = load_scenarios_index()
    
    if not bundles:
        print("❌ No training bundles available!")
        return
    
    print(f"✅ Found {len(bundles)} valid traffic bundles:")
    for i, bundle in enumerate(bundles, 1):
        intersections = ', '.join(bundle['intersections'])
        print(f"   {i}. {bundle['name']} ({intersections})")
    
    # Initialize environment with first bundle and realistic constraints
    initial_bundle, initial_route_file = select_random_bundle(bundles)
    env = TrafficEnvironment(
        net_file=CONFIG['NET_FILE'],
        rou_file=initial_route_file,
        use_gui=CONFIG['USE_GUI'],
        num_seconds=CONFIG['EPISODE_DURATION'],
        warmup_time=CONFIG['WARMUP_TIME'],
        step_length=CONFIG['STEP_LENGTH'],
        min_phase_time=8,   # 8 seconds minimum (based on ITE standards & RL research)
        max_phase_time=90   # 90 seconds maximum (optimized for urban arterials)
    )
    
    try:
        # Initialize environment and get agent configs
        env.reset()
        initial_states = env.get_marl_states()
        agent_configs = env.get_agent_configs()
        
        print(f"🤖 Agents detected: {len(agent_configs)}")
        for agent_id, config in agent_configs.items():
            print(f"   {agent_id}: State={config['state_size']}, Actions={config['action_size']}")
        
        # Initialize MARL system
        marl_system = MARLAgentManager(
            agent_configs=agent_configs,
            coordination_weight=CONFIG['COORDINATION_WEIGHT']
        )
        
        # Training tracking
        episode_rewards = defaultdict(list)
        episode_metrics = []
        
        print(f"\n🎯 Starting MARL training for {CONFIG['EPISODES']} episodes...")
        print(f"   Training across {len(bundles)} different traffic scenarios")
        print(f"   Expected scenario distribution: ~{CONFIG['EPISODES']/len(bundles):.1f} episodes per scenario")
        print("   This prevents overfitting to specific traffic patterns")
        
        start_time = time.time()
        
        for episode in range(CONFIG['EPISODES']):
            print(f"\n🤝 Episode {episode + 1}/{CONFIG['EPISODES']}")
            
            # Select random bundle for this episode
            bundle, route_file = select_random_bundle(bundles)
            if not bundle:
                continue
            
            print(f"   🎲 Selected: {bundle['name']}")
            
            # Close and reinitialize environment with new route and constraints
            env.close()
            env = TrafficEnvironment(
                net_file=CONFIG['NET_FILE'],
                rou_file=route_file,
                use_gui=CONFIG['USE_GUI'],
                num_seconds=CONFIG['EPISODE_DURATION'],
                warmup_time=CONFIG['WARMUP_TIME'],
                step_length=CONFIG['STEP_LENGTH'],
                min_phase_time=10,  # 10 seconds minimum (safety standard)
                max_phase_time=120  # 120 seconds maximum (efficiency standard)
            )
            
            # Reset environment and agents
            env.reset()
            states = env.get_marl_states()
            marl_system.reset_episode()
            
            # Episode tracking
            episode_total_rewards = defaultdict(float)
            episode_steps = 0
            
            # Main episode loop
            while True:
                # All agents select actions
                actions = marl_system.act(states, training=True)
                
                # Environment step with all agent actions
                next_states, rewards, done, info = env.step_marl(actions)
                
                # Add coordination reward
                coordination_reward = marl_system.calculate_coordination_reward(rewards)
                for agent_id in rewards:
                    rewards[agent_id] += coordination_reward
                
                # Store experiences for all agents
                dones = {agent_id: done for agent_id in actions.keys()}
                marl_system.remember(states, actions, rewards, next_states, dones)
                
                # Update tracking
                for agent_id, reward in rewards.items():
                    episode_total_rewards[agent_id] += reward
                
                episode_steps += 1
                
                # Train agents periodically
                if episode_steps % 10 == 0:
                    losses = marl_system.replay()
                    if losses and episode_steps % 50 == 0:  # Reduce output frequency
                        avg_loss = np.mean(list(losses.values()))
                        print(f"     🧠 Step {episode_steps}: Avg Loss={avg_loss:.4f}")
                
                # Progress updates
                if episode_steps % 30 == 0:
                    total_vehicles = info.get('vehicles', 0)
                    completed_trips = info.get('completed_trips', 0)
                    passenger_throughput = info.get('passenger_throughput', 0)
                    print(f"     📊 Step {episode_steps}: Vehicles={total_vehicles}, "
                          f"Completed={completed_trips}, Passengers={passenger_throughput:.1f}")
                
                # Update states
                states = next_states
                
                if done:
                    break
            
            # Episode summary
            avg_reward = np.mean(list(episode_total_rewards.values())) if episode_total_rewards else 0
            total_vehicles = info.get('vehicles', 0)
            completed_trips = info.get('completed_trips', 0)
            passenger_throughput = info.get('passenger_throughput', 0)
            
            print(f"   ✅ Episode {episode + 1} completed in {episode_steps} steps")
            print(f"   📈 Avg reward: {avg_reward:.2f}, Vehicles: {total_vehicles}, "
                  f"Completed: {completed_trips}, Passengers: {passenger_throughput:.1f}")
            
            # Record rewards
            for agent_id, reward in episode_total_rewards.items():
                episode_rewards[agent_id].append(reward)
            
            # Store episode metrics
            episode_metrics.append({
                'episode': episode + 1,
                'avg_reward': avg_reward,
                'vehicles': total_vehicles,
                'completed_trips': completed_trips,
                'passenger_throughput': passenger_throughput,
                'scenario': bundle['name']
            })
            
            # Update target networks and save models periodically
            if (episode + 1) % CONFIG['TARGET_UPDATE_FREQ'] == 0:
                marl_system.update_target_networks()
            
            if (episode + 1) % CONFIG['SAVE_FREQ'] == 0:
                marl_model_dir = os.path.join(CONFIG['MODEL_DIR'], 'marl')
                marl_system.save_models(marl_model_dir)
                print(f"     💾 Models saved at episode {episode + 1}")
                
                # Show training progress
                if episode_rewards:
                    recent_rewards = {aid: np.mean(rewards[-5:]) for aid, rewards in episode_rewards.items() if rewards}
                    print(f"     📊 Recent performance: {recent_rewards}")
        
        # Final training summary
        training_time = time.time() - start_time
        print(f"\n🎉 MARL Training completed in {training_time:.1f} seconds!")
        print(f"=" * 60)
        print(f"📊 Final Performance Summary:")
        print(f"   Total episodes: {len(episode_metrics)}")
        print(f"   Training scenarios used: {len(set(m['scenario'] for m in episode_metrics))}")
        print(f"   Average episode length: {np.mean([episode_steps]):.1f} steps")
        
        for agent_id in marl_system.agent_ids:
            if agent_id in episode_rewards and episode_rewards[agent_id]:
                agent_rewards = episode_rewards[agent_id]
                print(f"   Agent {agent_id}:")
                print(f"     Final reward (last 5): {np.mean(agent_rewards[-5:]):.2f}")
                print(f"     Best reward: {max(agent_rewards):.2f}")
                print(f"     Learning progress: {'+' if agent_rewards[-1] > agent_rewards[0] else '-'}")
                print(f"     Final epsilon: {marl_system.agents[agent_id].epsilon:.3f}")
        
        # Save final models
        final_marl_dir = os.path.join(CONFIG['MODEL_DIR'], 'marl_final')
        marl_system.save_models(final_marl_dir)
        
        # Save training metrics
        metrics_df = pd.DataFrame(episode_metrics)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(CONFIG['LOGS_DIR'], f'marl_training_metrics_{timestamp}.csv')
        metrics_df.to_csv(metrics_file, index=False)
        print(f"📄 Training metrics saved to {metrics_file}")
        
        # Performance analysis
        print(f"\n📈 Performance Analysis:")
        print(f"   Scenario diversity: {len(set(m['scenario'] for m in episode_metrics))}/{len(bundles)} scenarios used")
        print(f"   Average passenger throughput: {np.mean([m['passenger_throughput'] for m in episode_metrics]):.1f}")
        print(f"   Average completion rate: {np.mean([m['completed_trips'] for m in episode_metrics]):.1f} trips/episode")
        
        # Check for overfitting indicators
        scenario_counts = {}
        for metric in episode_metrics:
            scenario = metric['scenario']
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        print(f"   Episode distribution per scenario:")
        for scenario, count in scenario_counts.items():
            print(f"     {scenario}: {count} episodes ({count/len(episode_metrics)*100:.1f}%)")
        
        max_scenario_pct = max(scenario_counts.values()) / len(episode_metrics)
        if max_scenario_pct > 0.4:
            print(f"   ⚠️ Warning: One scenario used {max_scenario_pct*100:.1f}% of episodes - consider more scenarios")
        else:
            print(f"   ✅ Good scenario distribution - low overfitting risk")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        # Save progress
        if 'marl_system' in locals():
            interrupt_dir = os.path.join(CONFIG['MODEL_DIR'], 'marl_interrupted')
            marl_system.save_models(interrupt_dir)
            print(f"💾 Models saved to {interrupt_dir}")
            
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("🔄 Environment closed")

def test_agent(model_path, episodes=5):
    """Test a trained agent"""
    print(f"🧪 Testing trained agent: {model_path}")
    
    # Initialize environment (with GUI for testing)
    env = TrafficEnvironment(
        net_file=CONFIG['NET_FILE'],
        rou_file=CONFIG['ROU_FILE'],
        use_gui=True,  # Always use GUI for testing
        num_seconds=CONFIG['EPISODE_DURATION'],
        warmup_time=CONFIG['WARMUP_TIME']
    )
    
    try:
        # Initialize agent
        initial_state = env.reset()
        agent = D3QNAgent(
            state_size=len(initial_state),
            action_size=env.action_size
        )
        
        # Load trained model
        agent.load(model_path)
        agent.epsilon = 0.0  # No exploration during testing
        
        test_rewards = []
        
        for episode in range(episodes):
            print(f"\n🎮 Test Episode {episode + 1}/{episodes}")
            
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            while True:
                # Agent selects action (no exploration)
                action = agent.act(state, training=False)
                
                # Environment step
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                if episode_steps % 50 == 0:
                    print(f"     Step {episode_steps}: Reward={reward:.2f}, Vehicles={info['vehicles']}")
                
                if done:
                    break
            
            test_rewards.append(episode_reward)
            print(f"     ✅ Episode reward: {episode_reward:.2f}, Steps: {episode_steps}")
        
        print(f"\n📊 TEST SUMMARY:")
        print(f"   Average reward: {np.mean(test_rewards):.2f}")
        print(f"   Best reward: {np.max(test_rewards):.2f}")
        print(f"   Worst reward: {np.min(test_rewards):.2f}")
        
    finally:
        env.close()

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='D3QN Traffic Signal Control')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                       help='Mode: train or test')
    parser.add_argument('--model', type=str, default='models/best_d3qn_model.h5',
                       help='Model path for testing')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes (overrides config)')
    parser.add_argument('--no-gui', action='store_true',
                       help='Disable SUMO GUI')
    parser.add_argument('--marl', action='store_true',
                       help='Enable multi-agent training')
    parser.add_argument('--gui', action='store_true',
                       help='Enable GUI (overrides no-gui)')
    
    args = parser.parse_args()
    
    # Override config based on arguments
    if args.episodes:
        CONFIG['EPISODES'] = args.episodes
    if args.no_gui:
        CONFIG['USE_GUI'] = False
    if args.gui:
        CONFIG['USE_GUI'] = True
    if args.marl:
        CONFIG['MARL_MODE'] = True
    
    if args.mode == 'train':
        train_agent()
    elif args.mode == 'test':
        test_agent(args.model)

if __name__ == '__main__':
    main()
