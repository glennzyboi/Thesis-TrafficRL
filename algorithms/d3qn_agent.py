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
    CORRECTED: LSTM now predicts traffic first, then uses prediction for action selection
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.0005,  # OPTIMIZED: Balanced learning rate for stability
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995,
                 memory_size=75000, batch_size=128, sequence_length=10):  # OPTIMIZED: Better parameters for convergence
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
        self.gamma = 0.95  # OPTIMIZED: Balanced discount factor for immediate and long-term rewards
        self.tau = 0.005   # OPTIMIZED: Faster target network updates for better learning
        
        # LSTM-specific parameters
        self.sequence_length = sequence_length
        self.state_history = deque(maxlen=sequence_length)  # Store recent states for LSTM
        
        # Neural networks
        self.lstm_layers = self._build_lstm_layers()
        self.traffic_predictor = self._build_traffic_predictor()
        self.q_network = self._build_q_network_with_prediction()
        self.target_network = self._build_q_network_with_prediction()
        
        # Separate optimizers for different components
        self.traffic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate * 2)  # Higher LR for prediction
        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Initialize target network with same weights
        self.update_target_model()
        
        # Prediction metrics tracking
        self.prediction_history = []
        self.accuracy_history = []
        
        print(f"Simplified D3QN Agent with Core State Representation:")
        print(f"  State size: {state_size} (4 core metrics per lane + 4 global context)")
        print(f"  Action size: {action_size} (6 actions: 2 phases × 3 traffic lights)")
        print(f"  Learning rate: {learning_rate} (optimized for faster convergence)")
        print(f"  Epsilon decay: {epsilon_decay} (balanced exploration)")
        print(f"  Memory size: {memory_size} (diverse experience replay)")
        print(f"  Batch size: {batch_size} (stable learning)")
        print(f"  Gamma: {self.gamma} (long-term reward focus)")
        print(f"  LSTM sequence length: {sequence_length} (temporal pattern learning for date-based traffic)")
    
    def _build_lstm_layers(self):
        """Build LSTM layers for temporal pattern learning"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
            tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)
        ])
        return model
    
    def _build_traffic_predictor(self):
        """Build traffic prediction head (PRIMARY OUTPUT)"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary: heavy (1) or light (0) traffic
        ])
        return model
    
    def _build_q_network_with_prediction(self):
        """Build Q-network that uses traffic prediction for action selection"""
        # Input: state sequence
        state_input = tf.keras.Input(shape=(self.sequence_length, self.state_size), name='state_sequence')
        
        # Get LSTM output
        lstm_output = self.lstm_layers(state_input)
        
        # Get traffic prediction
        traffic_prediction = self.traffic_predictor(lstm_output)
        
        # Combine LSTM output with traffic prediction for Q-value estimation
        combined_features = tf.keras.layers.Concatenate()([lstm_output, traffic_prediction])
        
        # Dueling DQN architecture
        # Value stream
        value_stream = tf.keras.layers.Dense(64, activation='relu')(combined_features)
        value_stream = tf.keras.layers.Dropout(0.2)(value_stream)
        value = tf.keras.layers.Dense(1, activation='linear', name='value')(value_stream)
        
        # Advantage stream
        advantage_stream = tf.keras.layers.Dense(64, activation='relu')(combined_features)
        advantage_stream = tf.keras.layers.Dropout(0.2)(advantage_stream)
        advantage = tf.keras.layers.Dense(self.action_size, activation='linear', name='advantage')(advantage_stream)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        advantage_mean = tf.keras.layers.GlobalAveragePooling1D()(
            tf.keras.layers.Reshape((self.action_size, 1))(advantage)
        )
        advantage_mean = tf.keras.layers.RepeatVector(self.action_size)(advantage_mean)
        advantage_mean = tf.keras.layers.Flatten()(advantage_mean)
        
        advantage_normalized = tf.keras.layers.Subtract(name='advantage_normalized')([advantage, advantage_mean])
        q_values = tf.keras.layers.Add(name='q_values')([value, advantage_normalized])
        
        model = tf.keras.Model(inputs=state_input, outputs=q_values)
        return model
    
    def predict_traffic(self, state_sequence):
        """
        Predict heavy/light traffic (PRIMARY FUNCTION)
        
        Args:
            state_sequence: Sequence of states for LSTM
            
        Returns:
            traffic_prediction: 0 (light) or 1 (heavy) traffic
        """
        if len(state_sequence.shape) == 2:
            state_sequence = np.expand_dims(state_sequence, 0)
        
        # Get LSTM output
        lstm_output = self.lstm_layers(state_sequence, training=False)
        
        # Get traffic prediction
        traffic_prediction = self.traffic_predictor(lstm_output, training=False)
        
        return traffic_prediction.numpy()[0][0]  # Return scalar probability
    
    def is_heavy_traffic(self, traffic_metrics):
        """
        Determine if traffic is heavy based on metrics
        
        Args:
            traffic_metrics: Dictionary of traffic metrics
            
        Returns:
            bool: True if heavy traffic, False if light traffic
        """
        if not traffic_metrics:
            return False
        
        # Multiple criteria for heavy traffic
        queue_length = traffic_metrics.get('queue_length', 0)
        waiting_time = traffic_metrics.get('waiting_time', 0)
        vehicle_density = traffic_metrics.get('vehicle_density', 0)
        congestion_level = traffic_metrics.get('congestion_level', 0)
        
        # Heavy traffic if ANY of these conditions are met
        heavy_conditions = [
            queue_length > 100,           # Long queues
            waiting_time > 15,            # High waiting times
            vehicle_density > 0.8,        # High vehicle density
            congestion_level > 0.7        # High congestion
        ]
        
        return any(heavy_conditions)
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done, traffic_metrics=None):
        """Store experience in replay buffer and update state history for LSTM"""
        # Ensure state has correct shape and type
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        else:
            state = np.array(state, dtype=np.float32)
        
        if isinstance(next_state, np.ndarray):
            next_state = next_state.astype(np.float32)
        else:
            next_state = np.array(next_state, dtype=np.float32)
        
        # Add current state to history for LSTM (ensure it's 1D)
        if state.ndim > 1:
            state = state.flatten()
        self.state_history.append(state)
        
        # Create sequences for LSTM input
        current_sequence = self._get_state_sequence()
        next_sequence = self._get_next_state_sequence(next_state)
        
        self.memory.append((current_sequence, action, reward, next_sequence, done, traffic_metrics))
    
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
        
        # Get Q-values from the network (which now uses traffic prediction internally)
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
        # Each memory entry contains: (state_sequence, action, reward, next_state_sequence, done)
        # where state_sequence and next_state_sequence are lists of states
        
        states = np.array([e[0] for e in batch])  # Shape: (batch_size, sequence_length, state_size)
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])  # Shape: (batch_size, sequence_length, state_size)
        dones = np.array([e[4] for e in batch])
        
        # Data is already in correct shape for LSTM input
        # No reshaping needed since sequences are already prepared
        
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
        # Soft update target network each training step for stability
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
    
    def train_traffic_predictor(self, batch):
        """
        Train traffic prediction head (PRIMARY TRAINING)
        
        Args:
            batch: Batch of experiences
            
        Returns:
            prediction_loss: Loss from traffic prediction training
        """
        if len(batch) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch_sample = random.sample(batch, self.batch_size)
        
        # Prepare data
        states = []
        traffic_labels = []
        
        for state_sequence, action, reward, next_state_sequence, done, traffic_metrics in batch_sample:
            if traffic_metrics is not None:
                states.append(state_sequence)
                
                # Create traffic label
                is_heavy = self.is_heavy_traffic(traffic_metrics)
                traffic_labels.append(1 if is_heavy else 0)
        
        if len(states) == 0:
            return 0.0
        
        states = np.array(states)
        traffic_labels = np.array(traffic_labels, dtype=np.float32)
        
        # Train traffic predictor
        with tf.GradientTape() as tape:
            # Get LSTM output
            lstm_output = self.lstm_layers(states, training=True)
            
            # Get traffic prediction
            predictions = self.traffic_predictor(lstm_output, training=True)
            
            # Calculate loss
            prediction_loss = tf.keras.losses.binary_crossentropy(traffic_labels, predictions)
        
        # Update traffic predictor weights
        traffic_vars = self.traffic_predictor.trainable_variables
        gradients = tape.gradient(prediction_loss, traffic_vars)
        self.traffic_optimizer.apply_gradients(zip(gradients, traffic_vars))
        
        # Calculate accuracy
        predicted_labels = (predictions > 0.5).numpy().astype(int)
        accuracy = np.mean(predicted_labels == traffic_labels)
        
        # Store metrics
        self.prediction_history.append({
            'loss': float(prediction_loss),
            'accuracy': float(accuracy),
            'predictions': predicted_labels.tolist(),
            'actual': traffic_labels.astype(int).tolist()
        })
        self.accuracy_history.append(float(accuracy))
        
        return float(prediction_loss)
    
    def train_q_network_with_prediction(self, batch):
        """
        Train Q-network with traffic prediction (SECONDARY TRAINING)
        
        Args:
            batch: Batch of experiences
            
        Returns:
            q_loss: Loss from Q-network training
        """
        if len(batch) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch_sample = random.sample(batch, self.batch_size)
        
        # Prepare data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state_sequence, action, reward, next_state_sequence, done, traffic_metrics in batch_sample:
            states.append(state_sequence)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state_sequence)
            dones.append(done)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Get current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Get target Q-values
        next_q_values = self.target_network.predict(next_states, verbose=0)
        target_q_values = rewards + (self.gamma * np.max(next_q_values, axis=1) * (1 - dones))
        
        # Create target Q-values
        target_q = current_q_values.copy()
        target_q[np.arange(self.batch_size), actions] = target_q_values
        
        # Train Q-network
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            q_loss = tf.keras.losses.Huber(delta=1.0)(target_q, q_values)
        
        # Update Q-network weights
        q_vars = self.q_network.trainable_variables
        gradients = tape.gradient(q_loss, q_vars)
        self.q_optimizer.apply_gradients(zip(gradients, q_vars))
        
        return float(q_loss)
    
    def train_both(self, batch):
        """
        Train both traffic predictor and Q-network
        
        Args:
            batch: Batch of experiences
            
        Returns:
            dict: Training metrics
        """
        # Train traffic predictor (PRIMARY)
        prediction_loss = self.train_traffic_predictor(batch)
        
        # Train Q-network (SECONDARY)
        q_loss = self.train_q_network_with_prediction(batch)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {
            'prediction_loss': prediction_loss,
            'q_loss': q_loss,
            'epsilon': self.epsilon,
            'prediction_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0
        }
    
    def get_prediction_metrics(self):
        """Get traffic prediction performance metrics"""
        if not self.prediction_history:
            return {}
        
        recent_history = self.prediction_history[-10:]  # Last 10 predictions
        
        return {
            'avg_accuracy': np.mean([h['accuracy'] for h in recent_history]),
            'avg_loss': np.mean([h['loss'] for h in recent_history]),
            'total_predictions': len(self.prediction_history),
            'recent_accuracy_trend': self.accuracy_history[-10:] if len(self.accuracy_history) >= 10 else self.accuracy_history
        }
    
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
        for i, state in enumerate(recent_sequence):
            if isinstance(state, np.ndarray):
                # Ensure state is 1D and has correct size
                if state.ndim > 1:
                    state = state.flatten()
                if len(state) != self.state_size:
                    # Pad or truncate to correct size
                    if len(state) < self.state_size:
                        state = np.pad(state, (0, self.state_size - len(state)), 'constant')
                    else:
                        state = state[:self.state_size]
                normalized_sequence.append(state.astype(np.float32))
            else:
                state = np.array(state, dtype=np.float32)
                if state.ndim > 1:
                    state = state.flatten()
                if len(state) != self.state_size:
                    if len(state) < self.state_size:
                        state = np.pad(state, (0, self.state_size - len(state)), 'constant')
                    else:
                        state = state[:self.state_size]
                normalized_sequence.append(state.astype(np.float32))
            
        
        return np.array(normalized_sequence, dtype=np.float32)
    
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
                # Ensure state is 1D and has correct size
                if state.ndim > 1:
                    state = state.flatten()
                if len(state) != self.state_size:
                    # Pad or truncate to correct size
                    if len(state) < self.state_size:
                        state = np.pad(state, (0, self.state_size - len(state)), 'constant')
                    else:
                        state = state[:self.state_size]
                normalized_sequence.append(state.astype(np.float32))
            else:
                state = np.array(state, dtype=np.float32)
                if state.ndim > 1:
                    state = state.flatten()
                if len(state) != self.state_size:
                    if len(state) < self.state_size:
                        state = np.pad(state, (0, self.state_size - len(state)), 'constant')
                    else:
                        state = state[:self.state_size]
                normalized_sequence.append(state.astype(np.float32))
        
        return np.array(normalized_sequence, dtype=np.float32)
    
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
                learning_rate=0.0001,  # FIXED: Reduced for stability
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
