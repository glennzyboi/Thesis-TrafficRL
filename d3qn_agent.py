"""
Dueling Double Deep Q-Network (D3QN) Agent for Traffic Signal Control
Implements the D3QN algorithm with experience replay and target networks
"""

import numpy as np
import tensorflow as tf
import random
from collections import deque
import os

class D3QNAgent:
    """
    Dueling Double Deep Q-Network Agent with LSTM for traffic signal control
    Includes temporal memory for learning traffic patterns over time
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.9990,
                 memory_size=100000, batch_size=128, sequence_length=15):
        """
        Initialize the D3QN agent with LSTM for temporal learning
        
        Args:
            state_size: Dimension of the state space
            action_size: Number of possible actions
            learning_rate: Learning rate for the neural network
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            memory_size: Size of the replay buffer
            batch_size: Batch size for training
            sequence_length: Number of timesteps to remember for LSTM
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = 0.98  # Higher discount factor for long-term rewards
        self.tau = 0.001   # Soft update parameter for target network
        
        # LSTM-specific parameters
        self.sequence_length = sequence_length
        self.state_history = deque(maxlen=sequence_length)  # Store recent states for LSTM
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        
        # Initialize target network with same weights
        self.update_target_model()
        
        print(f"Optimized D3QN Agent with LSTM initialized:")
        print(f"  State size: {state_size}")
        print(f"  Action size: {action_size}")
        print(f"  Learning rate: {learning_rate} (reduced for stability)")
        print(f"  Epsilon decay: {epsilon_decay} (slower for longer exploration)")
        print(f"  Memory size: {memory_size} (increased for diversity)")
        print(f"  Batch size: {batch_size} (increased for stability)")
        print(f"  Gamma: {self.gamma} (optimized for traffic control)")
        print(f"  LSTM sequence length: {sequence_length} (temporal memory)")
    
    def _build_model(self):
        """
        Build the Dueling DQN model with LSTM for temporal learning
        The model includes LSTM layers for traffic pattern recognition
        """
        # Input layer for sequences: (batch_size, sequence_length, state_size)
        inputs = tf.keras.Input(shape=(self.sequence_length, self.state_size))
        
        # LSTM layers for temporal pattern learning
        lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm2 = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2)(lstm1)
        
        # Shared dense layers after LSTM
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm2)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
        
        # Value stream
        value_stream = tf.keras.layers.Dense(32, activation='relu')(dense2)
        value = tf.keras.layers.Dense(1, activation='linear', name='value')(value_stream)
        
        # Advantage stream  
        advantage_stream = tf.keras.layers.Dense(32, activation='relu')(dense2)
        advantage = tf.keras.layers.Dense(self.action_size, activation='linear', name='advantage')(advantage_stream)
        
        # Combine value and advantage to get Q-values using Dueling DQN formula
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # Use a custom layer to avoid Lambda serialization issues
        
        class DuelingLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(DuelingLayer, self).__init__(**kwargs)
            
            def call(self, inputs):
                value, advantage = inputs
                mean_advantage = tf.reduce_mean(advantage, axis=1, keepdims=True)
                advantage_normalized = advantage - mean_advantage
                return value + advantage_normalized
        
        # Apply dueling combination
        q_values = DuelingLayer(name='dueling_combination')([value, advantage])
        
        model = tf.keras.Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mean_squared_error')
        
        return model
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer and update state history for LSTM"""
        # Add current state to history for LSTM
        self.state_history.append(state)
        
        # Create sequences for LSTM input
        current_sequence = self._get_state_sequence()
        next_sequence = self._get_next_state_sequence(next_state)
        
        self.memory.append((current_sequence, action, reward, next_sequence, done))
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy with LSTM sequence input
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            action: Selected action
        """
        # Add current state to history
        self.state_history.append(state)
        
        if training and np.random.random() <= self.epsilon:
            # Random action (exploration)
            return random.randrange(self.action_size)
        
        # Get state sequence for LSTM input
        state_sequence = self._get_state_sequence()
        state_sequence = np.array(state_sequence).reshape(1, self.sequence_length, self.state_size)
        
        # Get Q-values from the network
        q_values = self.q_network.predict(state_sequence, verbose=0)
        
        # Return action with highest Q-value
        return np.argmax(q_values[0])
    
    def replay(self):
        """
        Train the model on a batch of experiences from replay buffer
        Uses Double DQN approach: main network selects actions, target network evaluates them
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Extract sequences (already prepared for LSTM)
        states = np.array([e[0] for e in batch])  # Shape: (batch_size, sequence_length, state_size)
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])  # Shape: (batch_size, sequence_length, state_size)
        dones = np.array([e[4] for e in batch])
        
        # Reshape data for LSTM input: (batch_size, sequence_length, state_size)
        states = states.reshape(self.batch_size, self.sequence_length, self.state_size)
        next_states = next_states.reshape(self.batch_size, self.sequence_length, self.state_size)
        
        # Get current Q-values with LSTM input
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Double DQN: Use main network to select actions for next states
        next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
        
        # Use target network to evaluate the selected actions
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Calculate target Q-values
        targets = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                # Double DQN target: Q_target(s', argmax_a Q_main(s', a))
                targets[i][actions[i]] = rewards[i] + self.gamma * next_q_values[i][next_actions[i]]
        
        # Train the model
        history = self.q_network.fit(states, targets, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0]
    
    def save(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Use newer Keras format for better compatibility
        if filepath.endswith('.h5'):
            filepath = filepath.replace('.h5', '.keras')
        self.q_network.save(filepath)
        print(f"LSTM D3QN Model saved to {filepath}")
    
    def load(self, filepath):
        """Load a trained model"""
        # Try both .keras and .h5 formats
        if not os.path.exists(filepath):
            if filepath.endswith('.h5'):
                filepath = filepath.replace('.h5', '.keras')
            elif filepath.endswith('.keras'):
                filepath = filepath.replace('.keras', '.h5')
        
        if os.path.exists(filepath):
            try:
                # Enable unsafe deserialization for loading models with Lambda layers
                tf.keras.config.enable_unsafe_deserialization()
                self.q_network = tf.keras.models.load_model(filepath, safe_mode=False)
                self.update_target_model()
                # Reset state history when loading a model
                self.reset_state_history()
                print(f"LSTM D3QN Model loaded from {filepath}")
            except Exception as e:
                print(f"Error loading model from {filepath}: {e}")
                print("Continuing with randomly initialized model...")
        else:
            print(f"Model file {filepath} not found")
    
    def _get_state_sequence(self):
        """Get current state sequence for LSTM input"""
        # Pad with zeros if we don't have enough history yet
        sequence = list(self.state_history)
        while len(sequence) < self.sequence_length:
            sequence.insert(0, np.zeros(self.state_size, dtype=np.float32))
        
        # Take only the most recent sequence_length states and ensure consistent shape
        recent_sequence = sequence[-self.sequence_length:]
        
        # Ensure all states have the correct shape and type
        normalized_sequence = []
        for state in recent_sequence:
            if isinstance(state, np.ndarray):
                normalized_sequence.append(state.astype(np.float32))
            else:
                normalized_sequence.append(np.array(state, dtype=np.float32))
        
        return normalized_sequence
    
    def _get_next_state_sequence(self, next_state):
        """Get next state sequence for LSTM training"""
        # Create sequence with next_state appended
        sequence = list(self.state_history)[1:]  # Remove oldest state
        sequence.append(next_state)  # Add new state
        
        # Pad if necessary
        while len(sequence) < self.sequence_length:
            sequence.insert(0, np.zeros(self.state_size, dtype=np.float32))
        
        # Normalize sequence
        recent_sequence = sequence[-self.sequence_length:]
        normalized_sequence = []
        for state in recent_sequence:
            if isinstance(state, np.ndarray):
                normalized_sequence.append(state.astype(np.float32))
            else:
                normalized_sequence.append(np.array(state, dtype=np.float32))
        
        return normalized_sequence
    
    def reset_state_history(self):
        """Reset state history at the beginning of each episode"""
        self.state_history.clear()
    
    def get_model_summary(self):
        """Get a summary of the model architecture"""
        return self.q_network.summary()


class MARLAgentManager:
    """
    Simple Multi-Agent Manager for coordinating D3QN agents
    Handles multiple agents without complex coordination mechanisms
    """
    
    def __init__(self, agent_configs, coordination_weight=0.1):
        """
        Initialize MARL manager
        
        Args:
            agent_configs: List of tuples (agent_id, state_size, action_size)
            coordination_weight: Weight for coordination rewards
        """
        self.agents = {}
        self.agent_ids = []
        self.coordination_weight = coordination_weight
        
        # Initialize individual agents
        for agent_id, config in agent_configs.items():
            self.agents[agent_id] = D3QNAgent(
                state_size=config['state_size'],
                action_size=config['action_size'],
                learning_rate=0.0005,
                sequence_length=10,
                memory_size=50000,
                batch_size=64
            )
            self.agent_ids.append(agent_id)
        
        print(f"🤝 MARL Manager initialized with {len(self.agents)} agents")
        for agent_id in self.agent_ids:
            print(f"   {agent_id}: Ready")
    
    def reset_episode(self):
        """Reset all agents for new episode"""
        for agent in self.agents.values():
            agent.reset_state_history()
        print(f"🔄 All {len(self.agents)} agents reset for new episode")
    
    def act(self, states, training=True):
        """Get actions from all agents"""
        actions = {}
        for agent_id, state in states.items():
            if agent_id in self.agents:
                actions[agent_id] = self.agents[agent_id].act(state, training=training)
        return actions
    
    def remember(self, states, actions, rewards, next_states, dones):
        """Store experiences for all agents"""
        for agent_id in self.agent_ids:
            if agent_id in states:
                self.agents[agent_id].remember(
                    states[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_states[agent_id],
                    dones.get(agent_id, False)
                )
    
    def replay(self):
        """Train all agents"""
        losses = {}
        for agent_id, agent in self.agents.items():
            loss = agent.replay()
            if loss is not None:
                losses[agent_id] = loss
        return losses
    
    def update_target_networks(self):
        """Update target networks for all agents"""
        for agent in self.agents.values():
            agent.update_target_model()
        print("🎯 All agent target networks updated")
    
    def save_models(self, directory):
        """Save all agent models"""
        os.makedirs(directory, exist_ok=True)
        for agent_id, agent in self.agents.items():
            model_path = os.path.join(directory, f"marl_agent_{agent_id}.keras")
            agent.save(model_path)
        print(f"💾 All {len(self.agents)} agent models saved to {directory}")
    
    def calculate_coordination_reward(self, individual_rewards):
        """Calculate simple coordination reward based on performance balance"""
        if len(individual_rewards) <= 1:
            return 0.0
        
        # Reward balanced performance across agents
        reward_values = list(individual_rewards.values())
        performance_std = np.std(reward_values)
        coordination_bonus = self.coordination_weight * (1.0 / (1.0 + performance_std))
        
        return coordination_bonus
