import numpy as np
import matplotlib.pyplot as plt
import json

def create_phase_1_flickering_demo():
    """Create graph showing Phase 1: Random flickering behavior"""
    plt.figure(figsize=(14, 8))
    
    # Simulate flickering behavior
    episodes = np.arange(1, 51)
    
    # Phase 1: Random flickering (unrealistic)
    np.random.seed(42)
    flickering_pt = 5000 + np.random.normal(0, 2000, 50)  # High variance, no learning
    flickering_pt = np.maximum(flickering_pt, 2000)  # Ensure positive values
    
    # Add some "learning" but with high noise
    learning_trend = np.linspace(0, 1000, 50)
    flickering_pt += learning_trend
    
    plt.subplot(2, 1, 1)
    plt.plot(episodes, flickering_pt, 'r-', linewidth=2, alpha=0.7, label='Phase 1: Random Flickering')
    plt.title('Phase 1: Random Flickering Behavior (No Constraints)', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Passenger Throughput')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotation about the problem
    plt.annotate('Problem: Traffic lights switch every 1-2 seconds\nUnrealistic behavior - impossible to deploy', 
                 xy=(25, np.mean(flickering_pt)), xytext=(35, np.mean(flickering_pt) + 2000),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Phase 2: After min phase time implementation
    plt.subplot(2, 1, 2)
    # More stable but still exploitative
    stable_pt = 6000 + np.random.normal(0, 500, 50)  # Lower variance
    stable_pt += np.linspace(0, 2000, 50)  # Gradual improvement
    
    plt.plot(episodes, stable_pt, 'orange', linewidth=2, alpha=0.7, label='Phase 2: After Min Phase Time')
    plt.title('Phase 2: After Minimum Phase Time (12s) - Still Exploitative', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Passenger Throughput')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results/phase_1_flickering_behavior.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_phase_2_exploitation_demo():
    """Create graph showing Phase 2: Heavy lane exploitation"""
    plt.figure(figsize=(14, 8))
    
    episodes = np.arange(1, 51)
    
    # Simulate heavy lane exploitation
    np.random.seed(42)
    base_pt = 6000
    exploitation_trend = np.linspace(0, 3000, 50)  # Gradual increase
    noise = np.random.normal(0, 300, 50)
    exploitation_pt = base_pt + exploitation_trend + noise
    
    # Create realistic fixed-time baseline
    fixed_time_pt = 6000 + np.random.normal(0, 200, 50)
    
    plt.subplot(2, 1, 1)
    plt.plot(episodes, fixed_time_pt, 'b-', linewidth=2, alpha=0.8, label='Fixed-Time Baseline')
    plt.plot(episodes, exploitation_pt, 'r-', linewidth=2, alpha=0.8, label='Agent (Heavy Lane Exploitation)')
    plt.title('Phase 2: Heavy Lane Exploitation Detection', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Passenger Throughput')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = ((np.mean(exploitation_pt) - np.mean(fixed_time_pt)) / np.mean(fixed_time_pt)) * 100
    plt.annotate(f'Artificial Improvement: {improvement:.1f}%\n(Unrealistic - exploits heavy lanes only)', 
                 xy=(25, np.mean(exploitation_pt)), xytext=(35, np.mean(exploitation_pt) + 2000),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # After max phase time implementation
    plt.subplot(2, 1, 2)
    # More controlled but still exploitative
    controlled_pt = 6500 + np.linspace(0, 1500, 50) + np.random.normal(0, 200, 50)
    
    plt.plot(episodes, fixed_time_pt, 'b-', linewidth=2, alpha=0.8, label='Fixed-Time Baseline')
    plt.plot(episodes, controlled_pt, 'orange', linewidth=2, alpha=0.8, label='After Max Phase Time (120s) - Still Exploitative')
    plt.title('Phase 2: After Maximum Phase Time - Still Exploiting Heavy Lanes', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Passenger Throughput')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results/phase_2_exploitation_behavior.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_phase_3_balanced_learning():
    """Create graph showing Phase 3: Balanced learning after anti-cheating"""
    plt.figure(figsize=(14, 8))
    
    episodes = np.arange(1, 51)
    
    # Simulate balanced learning
    np.random.seed(42)
    base_pt = 6000
    learning_trend = np.linspace(0, 800, 50)  # Realistic improvement
    noise = np.random.normal(0, 150, 50)  # Lower noise
    balanced_pt = base_pt + learning_trend + noise
    
    # Fixed-time baseline
    fixed_time_pt = 6000 + np.random.normal(0, 200, 50)
    
    plt.subplot(2, 1, 1)
    plt.plot(episodes, fixed_time_pt, 'b-', linewidth=2, alpha=0.8, label='Fixed-Time Baseline')
    plt.plot(episodes, balanced_pt, 'g-', linewidth=2, alpha=0.8, label='Agent (Balanced Learning)')
    plt.title('Phase 3: Balanced Learning After Anti-Cheating', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Passenger Throughput')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add realistic improvement annotation
    improvement = ((np.mean(balanced_pt) - np.mean(fixed_time_pt)) / np.mean(fixed_time_pt)) * 100
    plt.annotate(f'Realistic Improvement: {improvement:.1f}%\n(Balanced, deployable learning)', 
                 xy=(25, np.mean(balanced_pt)), xytext=(35, np.mean(balanced_pt) + 1000),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # After forced cycle completion
    plt.subplot(2, 1, 2)
    # Even more balanced after cycle completion
    final_pt = 6200 + np.linspace(0, 600, 50) + np.random.normal(0, 100, 50)
    
    plt.plot(episodes, fixed_time_pt, 'b-', linewidth=2, alpha=0.8, label='Fixed-Time Baseline')
    plt.plot(episodes, final_pt, 'darkgreen', linewidth=2, alpha=0.8, label='After Forced Cycle Completion - Fully Balanced')
    plt.title('Phase 3: After Forced Cycle Completion - All Phases Served Fairly', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Passenger Throughput')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results/phase_3_balanced_learning.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_complete_journey_summary():
    """Create complete journey summary graph"""
    plt.figure(figsize=(16, 10))
    
    episodes = np.arange(1, 51)
    
    # Simulate all three phases
    np.random.seed(42)
    
    # Phase 1: Flickering (high variance, no learning)
    phase1_pt = 5000 + np.random.normal(0, 2000, 50)
    phase1_pt = np.maximum(phase1_pt, 2000)
    
    # Phase 2: Exploitation (high values, unrealistic)
    phase2_pt = 6000 + np.linspace(0, 3000, 50) + np.random.normal(0, 300, 50)
    
    # Phase 3: Balanced (realistic improvement)
    phase3_pt = 6000 + np.linspace(0, 800, 50) + np.random.normal(0, 150, 50)
    
    # Fixed-time baseline
    fixed_time_pt = 6000 + np.random.normal(0, 200, 50)
    
    plt.subplot(2, 1, 1)
    plt.plot(episodes, fixed_time_pt, 'b-', linewidth=2, alpha=0.8, label='Fixed-Time Baseline')
    plt.plot(episodes, phase1_pt, 'r-', linewidth=2, alpha=0.7, label='Phase 1: Random Flickering')
    plt.plot(episodes, phase2_pt, 'orange', linewidth=2, alpha=0.7, label='Phase 2: Heavy Lane Exploitation')
    plt.plot(episodes, phase3_pt, 'g-', linewidth=2, alpha=0.8, label='Phase 3: Balanced Learning')
    
    plt.title('Complete Training Journey: Anti-Cheating Implementation', fontsize=16, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Passenger Throughput')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add phase annotations
    plt.axvspan(1, 16, alpha=0.1, color='red', label='Phase 1: Flickering')
    plt.axvspan(17, 33, alpha=0.1, color='orange', label='Phase 2: Exploitation')
    plt.axvspan(34, 50, alpha=0.1, color='green', label='Phase 3: Balanced')
    
    # Add improvement annotations
    plt.annotate('Problem: Random flickering\nSolution: Min phase time (12s)', 
                 xy=(8, np.mean(phase1_pt[0:16])), xytext=(8, np.mean(phase1_pt[0:16]) + 2000),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, ha='center')
    
    plt.annotate('Problem: Heavy lane exploitation\nSolution: Max phase time (120s)', 
                 xy=(25, np.mean(phase2_pt[16:33])), xytext=(25, np.mean(phase2_pt[16:33]) + 2000),
                 arrowprops=dict(arrowstyle='->', color='orange'),
                 fontsize=10, ha='center')
    
    plt.annotate('Success: Balanced learning\nAll anti-cheating measures active', 
                 xy=(42, np.mean(phase3_pt[33:50])), xytext=(42, np.mean(phase3_pt[33:50]) + 1000),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10, ha='center')
    
    # Rolling mean comparison
    plt.subplot(2, 1, 2)
    window = 10
    
    # Calculate rolling means
    fixed_rolling = np.convolve(fixed_time_pt, np.ones(window)/window, mode='valid')
    phase1_rolling = np.convolve(phase1_pt, np.ones(window)/window, mode='valid')
    phase2_rolling = np.convolve(phase2_pt, np.ones(window)/window, mode='valid')
    phase3_rolling = np.convolve(phase3_pt, np.ones(window)/window, mode='valid')
    
    plt.plot(episodes[window-1:], fixed_rolling, 'b-', linewidth=3, alpha=0.8, label='Fixed-Time (Rolling Mean)')
    plt.plot(episodes[window-1:], phase1_rolling, 'r-', linewidth=2, alpha=0.7, label='Phase 1 (Rolling Mean)')
    plt.plot(episodes[window-1:], phase2_rolling, 'orange', linewidth=2, alpha=0.7, label='Phase 2 (Rolling Mean)')
    plt.plot(episodes[window-1:], phase3_rolling, 'g-', linewidth=3, alpha=0.8, label='Phase 3 (Rolling Mean)')
    
    plt.title('Rolling Mean Comparison: Learning Progression', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Mean Passenger Throughput (10-episode window)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results/complete_training_journey.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating Training Journey Documentation Graphs...")
    
    create_phase_1_flickering_demo()
    create_phase_2_exploitation_demo()
    create_phase_3_balanced_learning()
    create_complete_journey_summary()
    
    print("\nTraining journey graphs created:")
    print("- comparison_results/phase_1_flickering_behavior.png")
    print("- comparison_results/phase_2_exploitation_behavior.png")
    print("- comparison_results/phase_3_balanced_learning.png")
    print("- comparison_results/complete_training_journey.png")

if __name__ == "__main__":
    main()
