import os
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(root, 'Chapter 4', 'figures')
    ensure_dir(out_dir)

    # Colors per phase
    colors = {
        'P1': '#4C78A8',
        'P2': '#F28E2B',
        'P3': '#59A14F',
        'P4': '#E15759',
    }

    total_time = 300
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))

    # Helper function to draw timeline
    def draw_timeline(ax, phases_data, title):
        y_pos = 0
        y_spacing = 2.0
        
        for phase_name, spans, color in phases_data:
            if len(spans) > 0:
                # Phase was served
                for (t0, t1) in spans:
                    ax.barh(y_pos, width=t1 - t0, left=t0, height=1.0, color=color, 
                           edgecolor='black', alpha=0.8, linewidth=2)
                    # Add duration label on bar
                    duration = t1 - t0
                    if duration > 15:  # Only label if bar is large enough
                        ax.text(t0 + duration/2, y_pos, f'{int(duration)}s', 
                               ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            else:
                # Phase was skipped
                ax.barh(y_pos, width=total_time * 0.15, left=0, height=1.0, 
                       color='lightcoral', edgecolor='red', linestyle='--', 
                       alpha=0.4, hatch='///')
                ax.text(total_time * 0.5, y_pos, 'SKIPPED', ha='center', va='center', 
                       fontsize=9, fontweight='bold', color='red')
            
            ax.text(-25, y_pos, phase_name, va='center', ha='right', 
                   fontsize=9, fontweight='bold')
            y_pos += y_spacing
        
        ax.set_ylim(-0.8, y_pos - y_spacing + 1)
        ax.set_xlim(-100, total_time + 20)
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, axis='x', alpha=0.3, linestyle=':')
        ax.axvline(x=total_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
        # Add explanation text
        explanation = "Rule: ALL vehicle phases must be served at least once within 300s"
        ax.text(total_time/2, ax.get_ylim()[1] * 0.98, explanation, ha='center', 
               va='top', fontsize=10, color='darkblue', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Ecoland (4 vehicle phases)
    ecoland_before = [
        ('Phase 1', [(0, 140), (180, 300)], colors['P1']),
        ('Phase 2', [(140, 180)], colors['P2']),
        ('Phase 3', [], colors['P3']),  # SKIPPED
        ('Phase 4', [], colors['P4']),  # SKIPPED
    ]
    ecoland_after = [
        ('Phase 1', [(0, 90), (240, 270)], colors['P1']),
        ('Phase 2', [(90, 130)], colors['P2']),
        ('Phase 3', [(130, 180)], colors['P3']),
        ('Phase 4', [(180, 240)], colors['P4']),
    ]
    
    draw_timeline(axes[0, 0], ecoland_before, 'BEFORE: Ecoland (4 vehicle phases, skipping 3 & 4)')
    draw_timeline(axes[0, 1], ecoland_after, 'AFTER: Ecoland (All 4 phases served; 12–120s each)')

    # JohnPaul (5 vehicle phases)
    john_before = [
        ('Phase 1', [(0, 120), (240, 300)], colors['P1']),
        ('Phase 2', [(120, 160)], colors['P2']),
        ('Phase 3', [], colors['P3']),  # SKIPPED
        ('Phase 4', [], colors['P4']),  # SKIPPED
        ('Phase 5', [], '#B07AA1'),     # SKIPPED
    ]
    john_after = [
        ('Phase 1', [(0, 70), (250, 270)], colors['P1']),
        ('Phase 2', [(70, 110)], colors['P2']),
        ('Phase 3', [(110, 150)], colors['P3']),
        ('Phase 4', [(150, 190)], colors['P4']),
        ('Phase 5', [(190, 240)], '#B07AA1'),
    ]
    
    draw_timeline(axes[1, 0], john_before, 'BEFORE: JohnPaul (5 vehicle phases, skipping 3–5)')
    draw_timeline(axes[1, 1], john_after, 'AFTER: JohnPaul (All 5 phases served; 12–120s each)')

    # Sandawa (3 vehicle phases)
    sandawa_before = [
        ('Phase 1', [(0, 220), (280, 300)], colors['P1']),
        ('Phase 2', [(220, 280)], colors['P2']),
        ('Phase 3', [], colors['P3']),  # SKIPPED
    ]
    sandawa_after = [
        ('Phase 1', [(0, 130), (240, 270)], colors['P1']),
        ('Phase 2', [(130, 180)], colors['P2']),
        ('Phase 3', [(180, 240)], colors['P3']),
    ]
    
    draw_timeline(axes[2, 0], sandawa_before, 'BEFORE: Sandawa (3 vehicle phases, skipping 3)')
    draw_timeline(axes[2, 1], sandawa_after, 'AFTER: Sandawa (All 3 phases served; 12–120s each)')

    plt.suptitle('Forced Cycle Completion (Vehicle Phases Only): All approaches must be served; agent chooses durations within 12–120s', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = os.path.join(out_dir, 'forced_cycle_completion_diagram.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved:', out_path)


if __name__ == '__main__':
    main()
