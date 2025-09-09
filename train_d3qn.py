"""
D3QN Traffic Signal Control Training Script
Trains a Dueling Double Deep Q-Network agent to control traffic signals in SUMO
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from d3qn_agent import D3QNAgent
from traffic_env import TrafficEnvironment

# Configuration
CONFIG = {
    # Files
    'NET_FILE': 'network/ThesisNetowrk.net.xml',
    'ROU_FILE': 'data/routes/test_training_routes.rou.xml',
    
    # Environment settings
    'USE_GUI': True,  # Show SUMO GUI for visualization
    'EPISODE_DURATION': 1800,  # 30 minutes per episode
    'WARMUP_TIME': 300,  # 5 minutes warmup
    'STEP_LENGTH': 1.0,  # 1 second per step
    
    # Training parameters
    'EPISODES': 50,  # Number of training episodes
    'TARGET_UPDATE_FREQ': 10,  # Update target network every N episodes
    'SAVE_FREQ': 10,  # Save model every N episodes
    
    # Agent parameters
    'LEARNING_RATE': 0.001,
    'EPSILON': 1.0,
    'EPSILON_MIN': 0.01,
    'EPSILON_DECAY': 0.995,
    'MEMORY_SIZE': 10000,
    'BATCH_SIZE': 32,
    
    # Directories
    'MODEL_DIR': 'models',
    'LOGS_DIR': 'logs',
    'PLOTS_DIR': 'plots'
}

def create_directories():
    """Create necessary directories"""
    for dir_name in [CONFIG['MODEL_DIR'], CONFIG['LOGS_DIR'], CONFIG['PLOTS_DIR']]:
        os.makedirs(dir_name, exist_ok=True)

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
    """Main training function"""
    print("🚀 D3QN TRAFFIC SIGNAL CONTROL TRAINING")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Initialize environment
    print("🏗️ Initializing environment...")
    env = TrafficEnvironment(
        net_file=CONFIG['NET_FILE'],
        rou_file=CONFIG['ROU_FILE'],
        use_gui=CONFIG['USE_GUI'],
        num_seconds=CONFIG['EPISODE_DURATION'],
        warmup_time=CONFIG['WARMUP_TIME'],
        step_length=CONFIG['STEP_LENGTH']
    )
    
    try:
        # Get initial state to determine state size
        initial_state = env.reset()
        state_size = len(initial_state)
        action_size = env.action_size
        
        print(f"🧠 State size: {state_size}, Action size: {action_size}")
        
        # Initialize agent
        agent = D3QNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=CONFIG['LEARNING_RATE'],
            epsilon=CONFIG['EPSILON'],
            epsilon_min=CONFIG['EPSILON_MIN'],
            epsilon_decay=CONFIG['EPSILON_DECAY'],
            memory_size=CONFIG['MEMORY_SIZE'],
            batch_size=CONFIG['BATCH_SIZE']
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
            
            # Reset environment
            state = env.reset()
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
    
    args = parser.parse_args()
    
    # Override config based on arguments
    if args.episodes:
        CONFIG['EPISODES'] = args.episodes
    if args.no_gui:
        CONFIG['USE_GUI'] = False
    
    if args.mode == 'train':
        train_agent()
    elif args.mode == 'test':
        test_agent(args.model)

if __name__ == '__main__':
    main()
