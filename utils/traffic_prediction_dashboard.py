"""
Traffic Prediction Dashboard for LSTM Performance Monitoring
Tracks and visualizes LSTM traffic prediction accuracy and performance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import json
import os
from datetime import datetime

class TrafficPredictionDashboard:
    """
    Dashboard for monitoring LSTM traffic prediction performance
    Tracks accuracy, confusion matrix, and prediction trends
    """
    
    def __init__(self, output_dir: str = "prediction_dashboard"):
        self.output_dir = output_dir
        self.prediction_data = []
        self.accuracy_history = []
        self.confusion_matrices = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def log_prediction(self, episode: int, predictions: List[float], 
                      actual_labels: List[int], traffic_metrics: List[Dict]):
        """
        Log prediction results for an episode
        
        Args:
            episode: Episode number
            predictions: LSTM predictions (0-1 probabilities)
            actual_labels: Actual traffic labels (0 or 1)
            traffic_metrics: Traffic metrics for context
        """
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
            'false_negatives': int(fn),
            'predictions': predictions,
            'actual_labels': actual_labels,
            'traffic_metrics': traffic_metrics
        }
        
        self.prediction_data.append(prediction_data)
        self.accuracy_history.append(accuracy)
        
        # Store confusion matrix
        confusion_matrix = {
            'episode': episode,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
        self.confusion_matrices.append(confusion_matrix)
        
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
        
        # Save confusion matrices
        with open(f"{self.output_dir}/data/confusion_matrices.json", 'w') as f:
            json.dump(self.confusion_matrices, f, indent=2)
    
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
    
    def plot_confusion_matrix(self, episode: int = None):
        """Plot confusion matrix for specific episode or overall"""
        if episode is not None:
            # Find specific episode
            episode_data = next((d for d in self.prediction_data if d['episode'] == episode), None)
            if episode_data is None:
                return
            
            tp = episode_data['true_positives']
            fp = episode_data['false_positives']
            tn = episode_data['true_negatives']
            fn = episode_data['false_negatives']
            title = f'Confusion Matrix - Episode {episode}'
        else:
            # Overall confusion matrix
            tp = sum(d['true_positives'] for d in self.prediction_data)
            fp = sum(d['false_positives'] for d in self.prediction_data)
            tn = sum(d['true_negatives'] for d in self.prediction_data)
            fn = sum(d['false_negatives'] for d in self.prediction_data)
            title = 'Overall Confusion Matrix'
        
        # Create confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Light', 'Predicted Heavy'],
                   yticklabels=['Actual Light', 'Actual Heavy'])
        
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Add metrics text
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}'
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/confusion_matrix_{episode or 'overall'}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_distribution(self, episode: int = None):
        """Plot distribution of prediction probabilities"""
        if episode is not None:
            # Specific episode
            episode_data = next((d for d in self.prediction_data if d['episode'] == episode), None)
            if episode_data is None:
                return
            
            predictions = episode_data['predictions']
            actual_labels = episode_data['actual_labels']
            title = f'Prediction Distribution - Episode {episode}'
        else:
            # All episodes
            predictions = []
            actual_labels = []
            for d in self.prediction_data:
                predictions.extend(d['predictions'])
                actual_labels.extend(d['actual_labels'])
            title = 'Overall Prediction Distribution'
        
        if not predictions:
            return
        
        predictions = np.array(predictions)
        actual_labels = np.array(actual_labels)
        
        # Separate predictions by actual label
        light_traffic_preds = predictions[actual_labels == 0]
        heavy_traffic_preds = predictions[actual_labels == 1]
        
        plt.figure(figsize=(12, 6))
        
        # Plot histograms
        plt.hist(light_traffic_preds, bins=20, alpha=0.7, label='Light Traffic', color='blue', density=True)
        plt.hist(heavy_traffic_preds, bins=20, alpha=0.7, label='Heavy Traffic', color='red', density=True)
        
        # Add threshold line
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold (0.5)')
        
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{self.output_dir}/plots/prediction_distribution_{episode or 'overall'}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_comparison(self):
        """Plot comparison of different metrics over time"""
        if len(self.prediction_data) < 2:
            return
        
        episodes = [d['episode'] for d in self.prediction_data]
        accuracy = [d['accuracy'] for d in self.prediction_data]
        precision = [d['precision'] for d in self.prediction_data]
        recall = [d['recall'] for d in self.prediction_data]
        f1_score = [d['f1_score'] for d in self.prediction_data]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(episodes, accuracy, label='Accuracy', color='blue')
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Target (80%)')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(episodes, precision, label='Precision', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Precision')
        plt.title('Precision Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(episodes, recall, label='Recall', color='green')
        plt.xlabel('Episode')
        plt.ylabel('Recall')
        plt.title('Recall Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(episodes, f1_score, label='F1-Score', color='purple')
        plt.xlabel('Episode')
        plt.ylabel('F1-Score')
        plt.title('F1-Score Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if not self.prediction_data:
            return "No prediction data available"
        
        # Calculate overall metrics
        total_predictions = sum(len(d['predictions']) for d in self.prediction_data)
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
        
        # Best and worst episodes
        best_episode = max(self.prediction_data, key=lambda x: x['accuracy'])
        worst_episode = min(self.prediction_data, key=lambda x: x['accuracy'])
        
        report = f"""
# LSTM Traffic Prediction Performance Report

## Overall Performance
- **Total Predictions**: {total_predictions:,}
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

## Best and Worst Episodes
- **Best Episode**: {best_episode['episode']} (Accuracy: {best_episode['accuracy']:.3f})
- **Worst Episode**: {worst_episode['episode']} (Accuracy: {worst_episode['accuracy']:.3f})

## Target Achievement
- **Target Accuracy**: 80%
- **Current Accuracy**: {avg_accuracy*100:.1f}%
- **Achievement**: {'✅ ACHIEVED' if avg_accuracy >= 0.8 else '❌ NOT ACHIEVED'}

## Recommendations
"""
        
        if avg_accuracy < 0.6:
            report += "- **Critical**: Accuracy below 60%. Consider increasing training data or adjusting model architecture.\n"
        elif avg_accuracy < 0.8:
            report += "- **Improvement Needed**: Accuracy below target 80%. Consider fine-tuning hyperparameters or increasing training episodes.\n"
        else:
            report += "- **Excellent**: Accuracy meets or exceeds target. Model is performing well.\n"
        
        if avg_precision < 0.7:
            report += "- **Precision Issue**: Low precision indicates many false positives. Consider adjusting threshold or improving heavy traffic detection.\n"
        
        if avg_recall < 0.7:
            report += "- **Recall Issue**: Low recall indicates many false negatives. Consider improving heavy traffic detection sensitivity.\n"
        
        # Save report
        with open(f"{self.output_dir}/prediction_performance_report.md", 'w') as f:
            f.write(report)
        
        return report
    
    def create_dashboard(self):
        """Create complete dashboard with all plots"""
        print("Creating LSTM Traffic Prediction Dashboard...")
        
        # Generate all plots
        self.plot_accuracy_trend()
        self.plot_confusion_matrix()  # Overall
        self.plot_prediction_distribution()  # Overall
        self.plot_metrics_comparison()
        
        # Generate summary report
        report = self.generate_summary_report()
        
        print(f"Dashboard created in: {self.output_dir}")
        print(f"Plots saved in: {self.output_dir}/plots")
        print(f"Data saved in: {self.output_dir}/data")
        print(f"Report saved as: {self.output_dir}/prediction_performance_report.md")
        
        return report
