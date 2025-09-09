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
    Dueling Double Deep Q-Network Agent for traffic signal control
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32):
        """
        Initialize the D3QN agent
        
        Args:
            state_size: Dimension of the state space
            action_size: Number of possible actions
            learning_rate: Learning rate for the neural network
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            memory_size: Size of the replay buffer
            batch_size: Batch size for training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        
        # Initialize target network with same weights
        self.update_target_model()
        
        print(f"D3QN Agent initialized:")
        print(f"  State size: {state_size}")
        print(f"  Action size: {action_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Memory size: {memory_size}")
    
    def _build_model(self):
        """
        Build the Dueling DQN model
        The model splits into value and advantage streams
        """
        # Input layer
        inputs = tf.keras.Input(shape=(self.state_size,))
        
        # Shared layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
        
        # Value stream
        value_stream = tf.keras.layers.Dense(64, activation='relu')(dense2)
        value = tf.keras.layers.Dense(1, activation='linear')(value_stream)
        
        # Advantage stream  
        advantage_stream = tf.keras.layers.Dense(64, activation='relu')(dense2)
        advantage = tf.keras.layers.Dense(self.action_size, activation='linear')(advantage_stream)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        mean_advantage = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
        advantage_normalized = tf.keras.layers.Subtract()([advantage, mean_advantage])
        q_values = tf.keras.layers.Add()([value, advantage_normalized])
        
        model = tf.keras.Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        
        return model
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            action: Selected action
        """
        if training and np.random.random() <= self.epsilon:
            # Random action (exploration)
            return random.randrange(self.action_size)
        
        # Get Q-values from the network
        state = np.array(state).reshape(1, -1)
        q_values = self.q_network.predict(state, verbose=0)
        
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
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Get current Q-values
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
                targets[i][actions[i]] = rewards[i] + 0.95 * next_q_values[i][next_actions[i]]
        
        # Train the model
        history = self.q_network.fit(states, targets, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0]
    
    def save(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.q_network.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.q_network = tf.keras.models.load_model(filepath)
            self.update_target_model()
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")
    
    def get_model_summary(self):
        """Get a summary of the model architecture"""
        return self.q_network.summary()
