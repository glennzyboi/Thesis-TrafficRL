"""
Simple Traffic Prediction Dashboard for LSTM Performance Monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

class TrafficPredictionDashboard:
    """Simple dashboard for monitoring LSTM traffic prediction performance"""
    
    def __init__(self, output_dir: str = "prediction_dashboard"):
        self.output_dir = output_dir
        self.prediction_data = []
        self.accuracy_history = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
    
    def log_prediction(self, episode: int, predictions: list, actual_labels: list):
        """Log prediction results for an episode"""
        # Convert predictions to binary
        binary_predictions = (np.array(predictions) > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = np.mean(binary_predictions == actual_labels)
        
        # Confusion matrix components
        tp = np.sum((binary_predictions == 1) & (actual_labels == 1))
        fp = np.sum((binary_predictions == 1) & (actual_labels == 0))
        tn = np.sum((binary_predictions == 0) & (actual_labels == 0))
        fn = np.sum((binary_predictions == 0) & (actual_labels == 1))
        
        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store data
        prediction_data = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        self.prediction_data.append(prediction_data)
        self.accuracy_history.append(accuracy)
        
        # Save data
        self._save_data()
        
        return prediction_data
    
    def _save_data(self):
        """Save prediction data to JSON files"""
        # Save all prediction data
        with open(f"{self.output_dir}/data/prediction_data.json", 'w') as f:
            json.dump(self.prediction_data, f, indent=2)
        
        # Save accuracy history
        with open(f"{self.output_dir}/data/accuracy_history.json", 'w') as f:
            json.dump(self.accuracy_history, f, indent=2)
    
    def plot_accuracy_trend(self, window_size: int = 10):
        """Plot accuracy trend over episodes"""
        if len(self.accuracy_history) < 2:
            return
        
        # Calculate moving average
        episodes = list(range(len(self.accuracy_history)))
        moving_avg = []
        
        for i in range(len(self.accuracy_history)):
            start_idx = max(0, i - window_size + 1)
            window_data = self.accuracy_history[start_idx:i+1]
            moving_avg.append(np.mean(window_data))
        
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, self.accuracy_history, alpha=0.3, color='blue', label='Raw Accuracy')
        plt.plot(episodes, moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size})')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random (50%)')
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Target (80%)')
        
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.title('LSTM Traffic Prediction Accuracy Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{self.output_dir}/plots/accuracy_trend.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if not self.prediction_data:
            return "No prediction data available"
        
        # Calculate overall metrics
        avg_accuracy = np.mean([d['accuracy'] for d in self.prediction_data])
        avg_precision = np.mean([d['precision'] for d in self.prediction_data])
        avg_recall = np.mean([d['recall'] for d in self.prediction_data])
        avg_f1_score = np.mean([d['f1_score'] for d in self.prediction_data])
        
        # Calculate overall confusion matrix
        total_tp = sum(d['true_positives'] for d in self.prediction_data)
        total_fp = sum(d['false_positives'] for d in self.prediction_data)
        total_tn = sum(d['true_negatives'] for d in self.prediction_data)
        total_fn = sum(d['false_negatives'] for d in self.prediction_data)
        
        # Recent performance (last 10 episodes)
        recent_data = self.prediction_data[-10:] if len(self.prediction_data) >= 10 else self.prediction_data
        recent_accuracy = np.mean([d['accuracy'] for d in recent_data])
        
        report = f"""
# LSTM Traffic Prediction Performance Report

## Overall Performance
- **Average Accuracy**: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)
- **Average Precision**: {avg_precision:.3f}
- **Average Recall**: {avg_recall:.3f}
- **Average F1-Score**: {avg_f1_score:.3f}

## Confusion Matrix (Overall)
- **True Positives**: {total_tp:,} (Heavy traffic correctly predicted)
- **False Positives**: {total_fp:,} (Light traffic incorrectly predicted as heavy)
- **True Negatives**: {total_tn:,} (Light traffic correctly predicted)
- **False Negatives**: {total_fn:,} (Heavy traffic incorrectly predicted as light)

## Recent Performance (Last {len(recent_data)} Episodes)
- **Recent Accuracy**: {recent_accuracy:.3f} ({recent_accuracy*100:.1f}%)

## Target Achievement
- **Target Accuracy**: 80%
- **Current Accuracy**: {avg_accuracy*100:.1f}%
- **Achievement**: {'✅ ACHIEVED' if avg_accuracy >= 0.8 else '❌ NOT ACHIEVED'}
"""
        
        # Save report
        with open(f"{self.output_dir}/prediction_performance_report.md", 'w') as f:
            f.write(report)
        
        return report
    
    def create_dashboard(self):
        """Create complete dashboard with all plots"""
        print("Creating LSTM Traffic Prediction Dashboard...")
        
        # Generate plots
        self.plot_accuracy_trend()
        
        # Generate summary report
        report = self.generate_summary_report()
        
        print(f"Dashboard created in: {self.output_dir}")
        print(f"Plots saved in: {self.output_dir}/plots")
        print(f"Data saved in: {self.output_dir}/data")
        print(f"Report saved as: {self.output_dir}/prediction_performance_report.md")
        
        return report
