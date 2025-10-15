"""
CORRECTED Dueling Double Deep Q-Network (D3QN) Agent for Traffic Signal Control
Implements LSTM with PRIMARY traffic prediction and SECONDARY action selection
This is the correct implementation for the thesis methodology
"""

import numpy as np
import tensorflow as tf
import random
from collections import deque
import os

class D3QNAgentCorrected:
    """
    CORRECTED D3QN Agent with LSTM for traffic prediction and prediction-informed action selection
    
    ARCHITECTURE:
    1. LSTM processes state sequences
    2. LSTM predicts heavy/light traffic (PRIMARY)
    3. Traffic prediction + state used for Q-value estimation (SECONDARY)
    4. Actions selected based on Q-values informed by traffic prediction
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.0005,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995,
                 memory_size=75000, batch_size=128, sequence_length=10):
        """
        Initialize the CORRECTED D3QN agent with traffic prediction as primary function
        
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
        self.gamma = 0.95
        self.tau = 0.005
        
        # LSTM-specific parameters
        self.sequence_length = sequence_length
        self.state_history = deque(maxlen=sequence_length)
        
        # CORRECTED ARCHITECTURE: Traffic prediction is PRIMARY
        self.lstm_layers = self._build_lstm_layers()
        self.traffic_predictor = self._build_traffic_predictor()
        self.q_network = self._build_q_network_with_prediction()
        self.target_network = self._build_q_network_with_prediction()
        
        # Separate optimizers for different components
        self.traffic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate * 2)  # Higher LR for prediction
        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Initialize target network
        self.update_target_model()
        
        # Prediction metrics tracking
        self.prediction_history = []
        self.accuracy_history = []
        
        print(f"CORRECTED D3QN Agent with Traffic Prediction as PRIMARY Function:")
        print(f"  State size: {state_size}")
        print(f"  Action size: {action_size}")
        print(f"  LSTM sequence length: {sequence_length}")
        print(f"  Architecture: LSTM → Traffic Prediction → Q-Value Estimation")
        print(f"  Traffic prediction is PRIMARY, Q-value estimation is SECONDARY")
    
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
    
    def act(self, state):
        """
        Select action based on traffic prediction and Q-values
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
        """
        # Add state to history
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        else:
            state = np.array(state, dtype=np.float32)
        
        self.state_history.append(state)
        
        # Need sequence for LSTM
        if len(self.state_history) < self.sequence_length:
            return random.choice(range(self.action_size))
        
        # Convert history to sequence
        state_sequence = np.array(list(self.state_history))
        state_sequence = np.expand_dims(state_sequence, 0)
        
        # Get traffic prediction
        traffic_prediction = self.predict_traffic(state_sequence)
        
        # Get Q-values (which already use traffic prediction internally)
        q_values = self.q_network(state_sequence, training=False)
        
        # Epsilon-greedy action selection
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done, traffic_metrics=None):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            traffic_metrics: Traffic metrics for prediction training
        """
        # Ensure states have correct shape and type
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        else:
            state = np.array(state, dtype=np.float32)
        
        if isinstance(next_state, np.ndarray):
            next_state = next_state.astype(np.float32)
        else:
            next_state = np.array(next_state, dtype=np.float32)
        
        # Store experience
        self.memory.append((state, action, reward, next_state, done, traffic_metrics))
    
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
        
        for state, action, reward, next_state, done, traffic_metrics in batch_sample:
            if traffic_metrics is not None:
                # Add state to history for sequence
                if len(self.state_history) >= self.sequence_length:
                    state_sequence = np.array(list(self.state_history))
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
    
    def train_q_network(self, batch):
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
        
        for state, action, reward, next_state, done, traffic_metrics in batch_sample:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Get current Q-values
        current_q_values = self.q_network(states, training=True)
        
        # Get target Q-values
        next_q_values = self.target_network(next_states, training=False)
        target_q_values = rewards + (self.gamma * np.max(next_q_values, axis=1) * (1 - dones))
        
        # Create target Q-values
        target_q = current_q_values.numpy()
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
    
    def train(self, batch):
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
        q_loss = self.train_q_network(batch)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {
            'prediction_loss': prediction_loss,
            'q_loss': q_loss,
            'epsilon': self.epsilon,
            'prediction_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0
        }
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_network.set_weights(self.q_network.get_weights())
    
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
        """Save the model"""
        # Save LSTM layers
        self.lstm_layers.save(f"{filepath}_lstm.keras")
        
        # Save traffic predictor
        self.traffic_predictor.save(f"{filepath}_traffic_predictor.keras")
        
        # Save Q-network
        self.q_network.save(f"{filepath}_q_network.keras")
        
        # Save target network
        self.target_network.save(f"{filepath}_target_network.keras")
        
        # Save other parameters
        params = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'gamma': self.gamma,
            'tau': self.tau
        }
        
        import json
        with open(f"{filepath}_params.json", 'w') as f:
            json.dump(params, f)
    
    def load(self, filepath):
        """Load the model"""
        # Load LSTM layers
        self.lstm_layers = tf.keras.models.load_model(f"{filepath}_lstm.keras")
        
        # Load traffic predictor
        self.traffic_predictor = tf.keras.models.load_model(f"{filepath}_traffic_predictor.keras")
        
        # Load Q-network
        self.q_network = tf.keras.models.load_model(f"{filepath}_q_network.keras")
        
        # Load target network
        self.target_network = tf.keras.models.load_model(f"{filepath}_target_network.keras")
        
        # Load parameters
        import json
        with open(f"{filepath}_params.json", 'r') as f:
            params = json.load(f)
        
        # Update parameters
        for key, value in params.items():
            setattr(self, key, value)
