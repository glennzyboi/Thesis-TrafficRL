"""
Dueling Double Deep Q-Network (D3QN) Agent for Traffic Signal Control - No LSTM Version
Implements the D3QN algorithm with experience replay and target networks without LSTM
"""

import numpy as np
import tensorflow as tf
import random
from collections import deque
import os

class D3QNAgentNoLSTM:
    """
    Dueling Double Deep Q-Network Agent without LSTM for traffic signal control
    Simplified version for comparison with LSTM-based agent
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.0005,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995,
                 memory_size=75000, batch_size=128):
        """
        Initialize the D3QN agent without LSTM
        
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
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = 0.95
        self.tau = 0.005
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        
        # Initialize target network with same weights
        self.update_target_model()
        
        print(f"D3QN Agent without LSTM (Comparison Version):")
        print(f"  State size: {state_size}")
        print(f"  Action size: {action_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epsilon decay: {epsilon_decay}")
        print(f"  Memory size: {memory_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Architecture: Dense layers only (no temporal memory)")
    
    def _build_model(self):
        """
        Build the Dueling DQN model without LSTM
        Uses the EXACT same architecture as LSTM version but with dense layers instead of LSTM
        """
        # Input layer for current state only (instead of sequence)
        inputs = tf.keras.Input(shape=(self.state_size,))
        
        # Dense layers matching LSTM version parameter count
        # LSTM version: LSTM(128) -> LSTM(64) -> Dense(128) -> Dense(64) = 221,927 params
        # Non-LSTM version needs similar parameter count
        # Strategy: Use larger dense layers to match LSTM complexity
        
        # First layer (replacing LSTM(128) with larger dense layer)
        dense1 = tf.keras.layers.Dense(512, activation='relu', 
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        
        # Second layer (replacing LSTM(64) with larger dense layer)
        dense2 = tf.keras.layers.Dense(256, activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.3)(dense2)
        
        # Third layer (matching LSTM version's Dense(128))
        dense3 = tf.keras.layers.Dense(128, activation='relu', 
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(dropout2)
        dropout3 = tf.keras.layers.Dropout(0.3)(dense3)
        
        # Fourth layer (matching LSTM version's Dense(64))
        dense4 = tf.keras.layers.Dense(64, activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(dropout3)
        
        # Value stream with regularization (exact match to LSTM version)
        value_stream = tf.keras.layers.Dense(32, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.001))(dense4)
        value_dropout = tf.keras.layers.Dropout(0.2)(value_stream)
        value = tf.keras.layers.Dense(1, activation='linear', name='value')(value_dropout)
        
        # Advantage stream with regularization (exact match to LSTM version)
        advantage_stream = tf.keras.layers.Dense(32, activation='relu',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.001))(dense4)
        advantage_dropout = tf.keras.layers.Dropout(0.2)(advantage_stream)
        advantage = tf.keras.layers.Dense(self.action_size, activation='linear', name='advantage')(advantage_dropout)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # Using the exact same approach as LSTM version
        
        # Reshape advantage to (batch, actions, 1) for pooling
        advantage_reshaped = tf.keras.layers.Reshape((self.action_size, 1))(advantage)
        
        # Compute mean advantage across actions
        advantage_mean = tf.keras.layers.GlobalAveragePooling1D()(advantage_reshaped)
        
        # Reshape back and repeat to match advantage shape
        advantage_mean = tf.keras.layers.RepeatVector(self.action_size)(advantage_mean)
        advantage_mean = tf.keras.layers.Flatten()(advantage_mean)
        
        # Normalize advantage by subtracting mean
        advantage_normalized = tf.keras.layers.Subtract(name='advantage_normalized')([advantage, advantage_mean])
        
        # Combine value and normalized advantage
        q_values = tf.keras.layers.Add(name='q_values')([value, advantage_normalized])
        
        model = tf.keras.Model(inputs=inputs, outputs=q_values)
        
        # Compile with exact same settings as LSTM version
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=5.0)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model
    
    def update_target_model(self):
        """Copy weights from main model to target model (matching LSTM version)"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer (matching LSTM version)"""
        # Ensure state has correct shape and type
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        else:
            state = np.array(state, dtype=np.float32)
        
        if isinstance(next_state, np.ndarray):
            next_state = next_state.astype(np.float32)
        else:
            next_state = np.array(next_state, dtype=np.float32)
        
        # Ensure state is 1D for non-LSTM
        if state.ndim > 1:
            state = state.flatten()
        if next_state.ndim > 1:
            next_state = next_state.flatten()
        
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # For non-LSTM, we just use the current state
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size):
        """
        Train the model on a batch of experiences (matching LSTM version)
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, batch_size)
        
        # Extract experiences (matching LSTM version structure)
        states = np.array([e[0] for e in batch])  # Shape: (batch_size, state_size)
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])  # Shape: (batch_size, state_size)
        dones = np.array([e[4] for e in batch])
        
        # Get current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Double DQN: Use main network to select actions for next states
        next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
        
        # Use target network to evaluate the selected actions
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Calculate target Q-values
        targets = current_q_values.copy()
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                # Double DQN target: Q_target(s', argmax_a Q_main(s', a))
                targets[i][actions[i]] = rewards[i] + self.gamma * next_q_values[i][next_actions[i]]
        
        # Train the model
        history = self.q_network.fit(states, targets, epochs=1, verbose=0)
        
        # Soft update target network each training step for stability (matching LSTM version)
        main_weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        new_weights = []
        for tw, mw in zip(target_weights, main_weights):
            new_weights.append(self.tau * mw + (1.0 - self.tau) * tw)
        self.target_network.set_weights(new_weights)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0]
    
    
    def save(self, filepath):
        """
        Save the model to file (matching LSTM version interface)
        """
        self.q_network.save(filepath)
        print(f"Model saved to {filepath}")
    
    def save_model(self, filepath):
        """
        Save the model to file
        """
        self.q_network.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the model from file
        """
        if os.path.exists(filepath):
            self.q_network = tf.keras.models.load_model(filepath)
            self.target_network = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")
    
    def get_model_summary(self):
        """
        Get model architecture summary
        """
        return self.q_network.summary()
    
    def reset_state_history(self):
        """
        Reset state history (compatibility method for non-LSTM agent)
        """
        # No-op for non-LSTM agent
        pass
    
    def load(self, filepath):
        """
        Load the model from file (matching LSTM version interface)
        """
        if os.path.exists(filepath):
            self.q_network = tf.keras.models.load_model(filepath)
            self.target_network = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")