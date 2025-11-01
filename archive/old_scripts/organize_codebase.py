"""
Temporary script to organize codebase for thesis defense.
Moves files to appropriate directories without breaking functionality.
"""
import os
import shutil
from pathlib import Path

# Define root directory
ROOT = Path(__file__).parent

# Files to move to archive/analysis_documents
ANALYSIS_FILES = [
    "350EP_TRAINING_COMPLETE_ANALYSIS.md",
    "350EP_TRAINING_LAUNCHED.md",
    "5EP_TEST_RESULTS_ANALYSIS.md",
    "ACADEMIC_DEFENSIBILITY_AUDIT.md",
    "BUS_NUMBERS_FIXED.md",
    "COMPLETE_FIX_CHECKLIST_SUMMARY.md",
    "COMPREHENSIVE_ACTION_PLAN.md",
    "COMPREHENSIVE_ANALYSIS_AND_FIXES_SUMMARY.md",
    "COMPREHENSIVE_LOGGER_TEST_RESULTS.md",
    "COMPREHENSIVE_LOGGER_VERIFIED.md",
    "COMPREHENSIVE_STABILITY_TEST_PLAN.md",
    "COMPREHENSIVE_VALIDATION_ANALYSIS.md",
    "COMPREHENSIVE_VALIDATION_STATUS.md",
    "CONSERVATIVE_FIXES_APPLIED.md",
    "CRITICAL_50EP_TEST_ANALYSIS.md",
    "CRITICAL_ACADEMIC_INTEGRITY_ANALYSIS.md",
    "CRITICAL_ANALYSIS_RL_DATA_LEAKAGE_DEBATE.md",
    "CRITICAL_ISSUES_IDENTIFIED.md",
    "CRITICAL_TRAINING_ANALYSIS_HONEST_ASSESSMENT.md",
    "CURRENT_STATUS_AND_NEXT_STEPS.md",
    "DASHBOARD_DATA_AVAILABILITY.md",
    "DASHBOARD_DATA_MAPPING_COMPLETE.md",
    "DASHBOARD_DATA_VERIFICATION_REPORT.md",
    "DASHBOARD_METRICS_OPTIMIZATION.md",
    "DASHBOARD_POPULATION_GUIDE.md",
    "DASHBOARD_READY_FINAL_STATUS.md",
    "DATA_AUTHENTICITY_VERIFICATION_REPORT.md",
    "DATA_MAPPING_AUDIT.md",
    "DATABASE_SCHEMA_ANALYSIS.md",
    "EVIDENCE_BASED_FIXES_APPLIED.md",
    "EVIDENCE_BASED_TRAIN_TEST_METHODOLOGY.md",
    "EXTENDED_30EP_STABILIZATION_ANALYSIS.md",
    "FINAL_100EP_TRAINING_ANALYSIS.md",
    "FINAL_ACTION_PLAN_WITH_DATA_LEAKAGE_CONFIRMED.md",
    "FINAL_DATABASE_POPULATION_READY.md",
    "HONEST_COMPREHENSIVE_TRAINING_READINESS_ANALYSIS.md",
    "HONEST_FIXES_SUMMARY.md",
    "HONEST_TRAINING_STATUS.md",
    "LANE_THROUGHPUT_TRACKING_ANALYSIS.md",
    "LANE_TRACKING_TEST_RESULTS_ANALYSIS.md",
    "LOGGER_DATABASE_VERIFICATION.md",
    "LOSS_INSTABILITY_CRITICAL_ANALYSIS.md",
    "LOSS_STABILIZATION_FIXES.md",
    "LSTM_METRIC_PRIORITY_ANALYSIS.md",
    "MISSING_DATA_ANALYSIS.md",
    "PHASE_2_STATUS.md",
    "PIPELINE_VERIFIED_READY_FOR_50EP.md",
    "PRE_TRAINING_CRITICAL_ASSESSMENT.md",
    "PT_METRICS_FINAL_FIX_SUMMARY.md",
    "PT_METRICS_FIX_AND_HYPERPARAMETER_TUNING.md",
    "PT_METRICS_FIX_COMPLETE.md",
    "PT_METRICS_STILL_ZERO_ANALYSIS.md",
    "READY_FOR_350EP_TRAINING.md",
    "STABILITY_TEST_50EP_COMPREHENSIVE_ANALYSIS.md",
    "STABILITY_TEST_50EP_LAUNCHED.md",
    "TECHNICAL_AUDIT_CRITICAL_FINDINGS.md",
    "TRAINING_INTERRUPTED_ACTUAL_ANALYSIS.md",
    "TRAINING_RESULTS_ANALYSIS.md",
    "VALIDATION_ANALYSIS_COMPLETE.md",
    "VALIDATION_RESULTS_SUMMARY.md",
    "VEHICLE_BREAKDOWN_100_PERCENT_REAL.md",
    "VEHICLE_BREAKDOWN_CORRECTED.md",
    "VEHICLE_TYPE_FIX_ANALYSIS.md",
    "VERIFICATION_TEST_COMPREHENSIVE_ANALYSIS.md",
]

# Visualization scripts (already in scripts/)
VISUALIZATION_SCRIPTS = [
    "scripts/generate_chapter4_comparison_figures.py",
    "scripts/generate_lstm_validation_figures.py",
    "scripts/generate_forced_cycle_completion_diagram.py",
    "scripts/generate_training_phase_graphs.py",
    "scripts/create_realistic_cheating_demo.py",
    "scripts/create_cheating_demonstration.py",
    "scripts/cap_training_plots_300.py",
]

# Data processing scripts (already in scripts/ or root)
DATA_PROCESSING_SCRIPTS = [
    "scripts/consolidate_bundle_routes.py",
    "scripts/consolidate_bundle_routes_utf8.py",
    "scripts/generate_balanced_routes.py",
    "scripts/prepare_dashboard_data.py",
]

# Utility/temporary scripts to archive
UTILITY_SCRIPTS = [
    "add_final_50_records.py",
    "add_missing_50_records.py",
    "analyze_logs_for_database_mapping.py",
    "analyze_queue_lengths.py",
    "analyze_raw_validation_data.py",
    "analyze_real_training_data.py",
    "analyze_validation_results.py",
    "analyze_vehicle_breakdown.py",
    "check_vehicle_types.py",
    "check_waiting_times.py",
    "clean_database_and_repopulate.py",
    "clean_database_fresh_start.py",
    "cleanup_old_results.py",
    "compile_hybrid_training_data.py",
    "create_validation_summary_csv.py",
    "database_schema_summary.py",
    "debug_lane_metrics.py",
    "estimate_realistic_traffic_metrics.py",
    "explain_lane_metrics.py",
    "extract_intersection_throughput.py",
    "final_database_population_with_estimates.py",
    "final_database_summary.py",
    "fix_all_database_issues.py",
    "fix_final_lane_metrics.py",
    "fix_remaining_lane_metrics.py",
    "fix_training_passenger_throughput.py",
    "fix_validation_passenger_throughput.py",
    "generate_accurate_enhanced_data.py",
    "generate_corrected_validation_stats.py",
    "generate_realistic_enhanced_data.py",
    "investigate_database_issues.py",
    "merge_enhanced_logging.py",
    "populate_database_corrected.py",
    "populate_database_final_complete.py",
    "populate_database_real_data.py",
    "populate_database_with_real_analysis.py",
    "populate_dashboard_final.py",
    "review_hybrid_data.py",
    "show_csv_sample.py",
    "show_raw_validation_data.py",
    "show_raw_values.py",
    "test_enhanced_logging.py",
    "test_validation_equality.py",
    "use_only_real_log_data.py",
    "verify_corrected_data.py",
    "verify_dashboard_data_completeness.py",
    "verify_database_samples.py",
    "verify_fixed_database.py",
    "verify_real_log_data.py",
]

def move_file_safe(src, dst_dir):
    """Safely move a file if it exists."""
    src_path = ROOT / src
    dst_dir_path = ROOT / dst_dir
    
    if not src_path.exists():
        return False, f"File not found: {src}"
    
    if not dst_dir_path.exists():
        dst_dir_path.mkdir(parents=True, exist_ok=True)
    
    dst_path = dst_dir_path / src_path.name
    
    try:
        shutil.move(str(src_path), str(dst_path))
        return True, f"Moved: {src} -> {dst_path}"
    except Exception as e:
        return False, f"Error moving {src}: {e}"

def main():
    print("=" * 80)
    print("ORGANIZING CODEBASE FOR THESIS DEFENSE")
    print("=" * 80)
    print()
    
    # Create directories
    (ROOT / "archive" / "analysis_documents").mkdir(parents=True, exist_ok=True)
    (ROOT / "archive" / "old_scripts").mkdir(parents=True, exist_ok=True)
    (ROOT / "scripts" / "visualization").mkdir(parents=True, exist_ok=True)
    (ROOT / "scripts" / "data_processing").mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    errors = []
    
    # Move analysis documents
    print("Moving analysis documents...")
    for f in ANALYSIS_FILES:
        success, msg = move_file_safe(f, "archive/analysis_documents")
        if success:
            moved_count += 1
            print(f"  [OK] {msg}")
        else:
            errors.append(msg)
    
    # Move visualization scripts
    print("\nMoving visualization scripts...")
    for f in VISUALIZATION_SCRIPTS:
        src_path = ROOT / f
        if src_path.exists():
            dst_path = ROOT / "scripts" / "visualization" / src_path.name
            try:
                shutil.move(str(src_path), str(dst_path))
                moved_count += 1
                print(f"  [OK] Moved: {f} -> scripts/visualization/{src_path.name}")
            except Exception as e:
                errors.append(f"Error moving {f}: {e}")
    
    # Move data processing scripts
    print("\nMoving data processing scripts...")
    for f in DATA_PROCESSING_SCRIPTS:
        src_path = ROOT / f
        if src_path.exists():
            dst_path = ROOT / "scripts" / "data_processing" / src_path.name
            try:
                shutil.move(str(src_path), str(dst_path))
                moved_count += 1
                print(f"  [OK] Moved: {f} -> scripts/data_processing/{src_path.name}")
            except Exception as e:
                errors.append(f"Error moving {f}: {e}")
    
    # Move utility scripts
    print("\nMoving utility scripts...")
    for f in UTILITY_SCRIPTS:
        success, msg = move_file_safe(f, "archive/old_scripts")
        if success:
            moved_count += 1
            print(f"  [OK] {msg}")
        else:
            errors.append(msg)
    
    # Move compare_training_stability to visualization
    stability_script = ROOT / "scripts" / "compare_training_stability.py"
    if stability_script.exists():
        dst_path = ROOT / "scripts" / "visualization" / "compare_training_stability.py"
        try:
            shutil.move(str(stability_script), str(dst_path))
            moved_count += 1
            print(f"  [OK] Moved: scripts/compare_training_stability.py -> scripts/visualization/")
        except Exception as e:
            errors.append(f"Error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"ORGANIZATION COMPLETE")
    print("=" * 80)
    print(f"Files moved: {moved_count}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    print("\nNext steps:")
    print("  1. Update imports in affected scripts")
    print("  2. Update README.md with new structure")
    print("  3. Test main entry points")

if __name__ == "__main__":
    main()

