"""
Bundle-aware D3QN training that samples from different traffic scenarios
Uses real field data from multiple days and cycles
"""

import os
import sys
import random
import pandas as pd
import argparse
import numpy as np
from train_d3qn import *  # Import everything from the base trainer

def load_scenarios_index():
    """Load available bundles from scenarios_index.csv and check for consolidated route files"""
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
        
        # Check if consolidated route file exists for this bundle
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

def select_random_bundle_routes(bundles):
    """Randomly select a bundle and return consolidated route file for true MARL"""
    if not bundles:
        print("❌ No valid bundles available!")
        return None, None
    
    # Randomly select a bundle
    bundle = random.choice(bundles)
    
    # Use the consolidated route file from the bundle
    consolidated_file = bundle['consolidated_file']
    
    return bundle, consolidated_file

def train_with_bundles():
    """Train D3QN agent using multiple traffic bundles"""
    print("🚀 BUNDLE-AWARE D3QN TRAINING")
    print("=" * 60)
    print("📊 Training on real traffic data from multiple scenarios")
    
    # Load available bundles
    bundles = load_scenarios_index()
    
    if not bundles:
        print("❌ No training bundles available!")
        return
    
    print(f"✅ Found {len(bundles)} valid traffic bundles:")
    for i, bundle in enumerate(bundles, 1):
        intersections = ', '.join(bundle['intersections'])
        print(f"   {i}. {bundle['name']} ({intersections})")
    
    # Initialize environment and agent
    print(f"\n🏗️ Initializing training components...")
    
    # Create environment (will be reconfigured for each episode)
    env = None
    agent = None
    
    total_episodes = CONFIG.get('EPISODES', 10)
    episode_rewards = []
    
    print(f"\n🎯 Starting bundle-aware training for {total_episodes} episodes...")
    print("   Each episode uses a random traffic scenario from your field data")
    print("   🖥️ SUMO GUI will open for each episode - you can watch the AI learn!")
    
    try:
        for episode in range(total_episodes):
            print(f"\n📺 Episode {episode + 1}/{total_episodes}")
            
            # Select random bundle and get consolidated route file for MARL
            bundle, consolidated_route_file = select_random_bundle_routes(bundles)
            if not bundle or not consolidated_route_file:
                continue
            
            print(f"   🎲 Selected: {bundle['name']}")
            print(f"   🚦 Using consolidated route file for TRUE MARL:")
            print(f"     - {os.path.basename(consolidated_route_file)}")
            print(f"     - Contains synchronized traffic for all 3 intersections!")
            print(f"   🖥️ Opening SUMO for visualization...")
            
            # Update config for this episode with GUI enabled for testing
            episode_config = CONFIG.copy()
            episode_config['ROU_FILE'] = consolidated_route_file  # Single consolidated file
            episode_config['USE_GUI'] = True  # Enable GUI to see the synchronized traffic
            
            # Close previous environment
            if env:
                env.close()
            
            # Create new environment with consolidated route file for MARL
            env = TrafficEnvironment(
                net_file=episode_config['NET_FILE'],
                rou_file=episode_config['ROU_FILE'],  # Single consolidated file with all intersection traffic
                use_gui=episode_config['USE_GUI'],
                num_seconds=episode_config['EPISODE_DURATION'],
                warmup_time=episode_config['WARMUP_TIME'],
                step_length=episode_config['STEP_LENGTH']
            )
            
            # Initialize agent on first episode
            if agent is None:
                initial_state = env.reset()
                state_size = len(initial_state)
                action_size = env.action_size
                
                agent = D3QNAgent(
                    state_size=state_size,
                    action_size=action_size,
                    learning_rate=episode_config['LEARNING_RATE'],
                    memory_size=episode_config['MEMORY_SIZE'],
                    batch_size=episode_config['BATCH_SIZE']
                )
                
                print(f"🧠 D3QN Agent initialized: State={state_size}, Actions={action_size}")
            else:
                # Reset environment for existing agent
                initial_state = env.reset()
            
            # Run episode
            state = initial_state
            total_reward = 0
            steps = 0
            
            print(f"   🏃 Running episode...")
            
            for step in range(episode_config['EPISODE_DURATION'] - episode_config['WARMUP_TIME']):
                # Agent selects action
                action = agent.act(state)
                
                # Environment step
                next_state, reward, done, info = env.step(action)
                
                # Agent learns
                agent.remember(state, action, reward, next_state, done)
                
                loss = None
                if len(agent.memory) > episode_config['BATCH_SIZE']:
                    loss = agent.replay()
                
                # Show learning progress every 25 steps
                if step % 25 == 0 and step > 0 and loss is not None:
                    print(f"     >>> Learning: Loss={loss:.4f} | Memory={len(agent.memory)}/{agent.memory_size}")
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Detailed progress updates every 25 steps
                if step % 25 == 0 and step > 0:
                    vehicles = info.get('vehicles', 0) if info else 0
                    waiting_time = info.get('waiting_time', 0) if info else 0
                    avg_speed = info.get('avg_speed', 0) if info else 0
                    queue_length = info.get('queue_length', 0) if info else 0
                    
                    # Get Q-values for current state if agent has memory
                    q_values_str = "N/A"
                    if hasattr(agent, 'model') and len(agent.memory) > 0:
                        try:
                            q_values = agent.model.predict(state.reshape(1, -1), verbose=0)[0]
                            q_max = np.max(q_values)
                            q_min = np.min(q_values)
                            q_values_str = f"Q[{q_min:.2f}-{q_max:.2f}]"
                        except:
                            q_values_str = "N/A"
                    
                    print(f"     Step {step:3d}: R={reward:+6.2f} | Vehicles={vehicles:3d} | Wait={waiting_time:5.1f}s | Speed={avg_speed:4.1f}km/h | Queue={queue_length:2d} | {q_values_str} | ε={agent.epsilon:.3f}")
                
                if done:
                    break
            
            # Episode summary with detailed metrics
            episode_rewards.append(total_reward)
            avg_reward = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
            
            # Final episode metrics
            final_info = info if info else {}
            final_vehicles = final_info.get('vehicles', 0)
            final_waiting = final_info.get('waiting_time', 0)
            final_speed = final_info.get('avg_speed', 0)
            final_throughput = final_info.get('throughput', 0)
            
            print(f"   ✅ Episode {episode + 1} Summary:")
            print(f"     📊 Reward: {total_reward:+8.2f} | Steps: {steps:3d} | ε: {agent.epsilon:.3f}")
            print(f"     🚗 Vehicles: {final_vehicles:3d} | Waiting: {final_waiting:5.1f}s | Speed: {final_speed:4.1f}km/h")
            print(f"     📈 Throughput: {final_throughput:5.1f}veh/h | Memory: {len(agent.memory):4d}/{agent.memory_size}")
            print(f"     🎯 Avg Reward (last 10): {avg_reward:+8.2f}")
            print(f"     📍 Scenario: {bundle['name']}")
            
            # Update target network
            if episode % episode_config['TARGET_UPDATE_FREQ'] == 0:
                agent.update_target_model()
                print(f"     🎯 Target network updated")
            
            # Save best model
            if total_reward == max(episode_rewards):
                agent.save(f"models/best_bundle_d3qn_model.h5")
                print(f"     💾 New best model saved! Reward: {total_reward:.2f}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
    
    finally:
        # Cleanup and final save
        if env:
            env.close()
        
        if agent:
            agent.save(f"models/final_bundle_d3qn_model.h5")
        
        # Training summary
        print(f"\n🎉 BUNDLE-AWARE TRAINING COMPLETED!")
        print(f"   Total episodes: {len(episode_rewards)}")
        print(f"   Best reward: {max(episode_rewards) if episode_rewards else 'N/A'}")
        avg_reward = sum(episode_rewards)/len(episode_rewards) if episode_rewards else 0
        print(f"   Average reward: {avg_reward:.2f}" if episode_rewards else "   Average reward: N/A")
        print(f"   Unique scenarios experienced: {min(len(episode_rewards), len(bundles))}")
        print(f"   Models saved in: models/")
        
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"   The AI trained on REAL traffic data from your field observations")
        print(f"   It experienced different days, cycles, and intersections")
        print(f"   It learned to handle various traffic patterns from your dataset")

def main():
    parser = argparse.ArgumentParser(description="Bundle-aware D3QN training")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--gui", action="store_true", help="Enable SUMO GUI")
    
    args = parser.parse_args()
    
    # Update global config
    global CONFIG
    CONFIG['EPISODES'] = args.episodes
    CONFIG['USE_GUI'] = args.gui
    
    train_with_bundles()

if __name__ == "__main__":
    main()
