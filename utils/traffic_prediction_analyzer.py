"""
Traffic Prediction Accuracy Analyzer for LSTM
Measures LSTM's ability to predict heavy vs light traffic during training
Author: D3QN Thesis Project
Date: October 13, 2025
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class TrafficPredictionAnalyzer:
    """
    Analyzes LSTM's traffic prediction accuracy during training.
    Implements realistic expectations (60-70% accuracy) with limited data.
    """
    
    def __init__(self, heavy_traffic_thresholds: Dict[str, float] = None):
        """
        Initialize traffic prediction analyzer.
        
        Args:
            heavy_traffic_thresholds: Criteria for determining heavy traffic
        """
        self.thresholds = heavy_traffic_thresholds or {
            'queue_length': 100,      # vehicles
            'waiting_time': 15,       # seconds
            'vehicle_density': 0.8,   # ratio
            'congestion_level': 0.7   # ratio
        }
        
        # Prediction history for analysis
        self.prediction_history = []
        self.accuracy_history = []
        
    def is_heavy_traffic(self, metrics: Dict[str, Any]) -> bool:
        """
        Determine if current traffic conditions are heavy.
        
        Args:
            metrics: Current traffic metrics dictionary
        
        Returns:
            True if traffic is heavy, False otherwise
        """
        # Extract metrics with defaults
        queue_length = metrics.get('queue_length', 0)
        waiting_time = metrics.get('waiting_time', 0)
        vehicle_density = metrics.get('vehicle_density', 0)
        congestion_level = metrics.get('congestion_level', 0)
        
        # Heavy traffic if ANY condition is met (OR logic)
        heavy_conditions = [
            queue_length > self.thresholds['queue_length'],
            waiting_time > self.thresholds['waiting_time'],
            vehicle_density > self.thresholds['vehicle_density'],
            congestion_level > self.thresholds['congestion_level']
        ]
        
        return any(heavy_conditions)
    
    def calculate_prediction_metrics(self, predictions: np.ndarray, 
                                   actual_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive prediction metrics.
        
        Args:
            predictions: Binary predictions (0 or 1)
            actual_labels: Actual binary labels (0 or 1)
        
        Returns:
            Dictionary of prediction metrics
        """
        # Convert to binary if needed
        if predictions.dtype != np.int32:
            predictions = (predictions > 0.5).astype(np.int32)
        if actual_labels.dtype != np.int32:
            actual_labels = actual_labels.astype(np.int32)
        
        # Basic metrics
        accuracy = accuracy_score(actual_labels, predictions)
        precision = precision_score(actual_labels, predictions, zero_division=0)
        recall = recall_score(actual_labels, predictions, zero_division=0)
        f1 = f1_score(actual_labels, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(actual_labels, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Specific metrics for traffic prediction
        heavy_traffic_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        light_traffic_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        heavy_traffic_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        light_traffic_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'heavy_traffic_recall': float(heavy_traffic_recall),
            'light_traffic_precision': float(light_traffic_precision),
            'heavy_traffic_precision': float(heavy_traffic_precision),
            'light_traffic_recall': float(light_traffic_recall),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_predictions': len(predictions),
            'heavy_traffic_actual': int(np.sum(actual_labels)),
            'heavy_traffic_predicted': int(np.sum(predictions))
        }
    
    def analyze_episode_predictions(self, episode_number: int, 
                                  state_sequences: List[np.ndarray],
                                  actual_metrics: List[Dict[str, Any]],
                                  lstm_predictions: np.ndarray) -> Dict[str, Any]:
        """
        Analyze LSTM predictions for a single episode.
        
        Args:
            episode_number: Episode number
            state_sequences: List of state sequences for LSTM
            actual_metrics: List of actual traffic metrics
            lstm_predictions: LSTM predictions (0-1 probabilities)
        
        Returns:
            Episode prediction analysis
        """
        # Convert LSTM predictions to binary
        binary_predictions = (lstm_predictions > 0.5).astype(np.int32)
        
        # Calculate actual labels
        actual_labels = np.array([
            1 if self.is_heavy_traffic(metrics) else 0 
            for metrics in actual_metrics
        ])
        
        # Calculate metrics
        metrics = self.calculate_prediction_metrics(binary_predictions, actual_labels)
        
        # Add episode context
        episode_analysis = {
            'episode_number': episode_number,
            'timestamp': np.datetime64('now'),
            **metrics
        }
        
        # Store for history
        self.prediction_history.append(episode_analysis)
        self.accuracy_history.append(metrics['accuracy'])
        
        return episode_analysis
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get overall training progress for traffic prediction.
        
        Returns:
            Training progress summary
        """
        if not self.accuracy_history:
            return {
                'total_episodes': 0,
                'average_accuracy': 0,
                'best_accuracy': 0,
                'worst_accuracy': 0,
                'accuracy_trend': 'no_data',
                'prediction_quality': 'insufficient_data'
            }
        
        accuracies = np.array(self.accuracy_history)
        
        # Calculate trends
        if len(accuracies) >= 10:
            early_avg = np.mean(accuracies[:5])
            late_avg = np.mean(accuracies[-5:])
            trend = 'improving' if late_avg > early_avg else 'declining'
        else:
            trend = 'insufficient_data'
        
        # Determine prediction quality
        avg_accuracy = np.mean(accuracies)
        if avg_accuracy >= 0.70:
            quality = 'excellent'
        elif avg_accuracy >= 0.65:
            quality = 'good'
        elif avg_accuracy >= 0.60:
            quality = 'acceptable'
        elif avg_accuracy >= 0.55:
            quality = 'poor'
        else:
            quality = 'very_poor'
        
        return {
            'total_episodes': len(accuracies),
            'average_accuracy': float(avg_accuracy),
            'best_accuracy': float(np.max(accuracies)),
            'worst_accuracy': float(np.min(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'accuracy_trend': trend,
            'prediction_quality': quality,
            'target_achieved': avg_accuracy >= 0.60,  # Realistic target
            'excellent_threshold': avg_accuracy >= 0.70
        }
    
    def get_confusion_matrix_summary(self) -> Dict[str, Any]:
        """
        Get confusion matrix summary across all episodes.
        
        Returns:
            Confusion matrix summary
        """
        if not self.prediction_history:
            return {'error': 'no_prediction_data'}
        
        # Aggregate confusion matrix data
        total_tp = sum(ep['true_positives'] for ep in self.prediction_history)
        total_fp = sum(ep['false_positives'] for ep in self.prediction_history)
        total_tn = sum(ep['true_negatives'] for ep in self.prediction_history)
        total_fn = sum(ep['false_negatives'] for ep in self.prediction_history)
        
        # Calculate overall metrics
        total_predictions = total_tp + total_fp + total_tn + total_fn
        overall_accuracy = (total_tp + total_tn) / total_predictions if total_predictions > 0 else 0
        
        return {
            'total_predictions': int(total_predictions),
            'true_positives': int(total_tp),
            'false_positives': int(total_fp),
            'true_negatives': int(total_tn),
            'false_negatives': int(total_fn),
            'overall_accuracy': float(overall_accuracy),
            'heavy_traffic_recall': float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0,
            'light_traffic_precision': float(total_tn / (total_tn + total_fn)) if (total_tn + total_fn) > 0 else 0
        }
    
    def generate_academic_summary(self) -> Dict[str, Any]:
        """
        Generate academic summary for thesis documentation.
        
        Returns:
            Academic summary suitable for thesis
        """
        progress = self.get_training_progress()
        confusion = self.get_confusion_matrix_summary()
        
        # Academic positioning
        if progress['average_accuracy'] >= 0.70:
            academic_position = "LSTM achieved excellent traffic prediction accuracy, demonstrating strong temporal pattern learning capability."
        elif progress['average_accuracy'] >= 0.65:
            academic_position = "LSTM achieved good traffic prediction accuracy, showing effective temporal pattern learning with limited data."
        elif progress['average_accuracy'] >= 0.60:
            academic_position = "LSTM achieved acceptable traffic prediction accuracy, demonstrating basic temporal pattern learning capability."
        else:
            academic_position = "LSTM showed limited traffic prediction accuracy due to data constraints, but still contributed to temporal learning in Q-value estimation."
        
        return {
            'prediction_accuracy': progress['average_accuracy'],
            'academic_positioning': academic_position,
            'data_limitations_acknowledged': progress['average_accuracy'] < 0.80,
            'temporal_learning_demonstrated': progress['average_accuracy'] > 0.55,
            'realistic_expectations': {
                'target_accuracy': 0.60,
                'achieved_accuracy': progress['average_accuracy'],
                'data_requirement': 'More extensive training data needed for 80% accuracy',
                'proof_of_concept': 'Demonstrates LSTM potential for traffic signal control'
            },
            'confusion_matrix_summary': confusion,
            'training_progress': progress
        }


def create_traffic_prediction_head(input_dim: int) -> tf.keras.Model:
    """
    Create traffic prediction head for LSTM.
    
    Args:
        input_dim: Input dimension from LSTM output
    
    Returns:
        Keras model for traffic prediction
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_prediction_head(prediction_head: tf.keras.Model, 
                         lstm_outputs: np.ndarray, 
                         labels: np.ndarray,
                         epochs: int = 5) -> Dict[str, float]:
    """
    Train traffic prediction head.
    
    Args:
        prediction_head: Keras model for prediction
        lstm_outputs: LSTM output features
        labels: Binary labels (0 or 1)
        epochs: Training epochs
    
    Returns:
        Training history
    """
    history = prediction_head.fit(
        lstm_outputs, labels,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    return {
        'final_loss': float(history.history['loss'][-1]),
        'final_accuracy': float(history.history['accuracy'][-1]),
        'val_loss': float(history.history['val_loss'][-1]),
        'val_accuracy': float(history.history['val_accuracy'][-1])
    }
