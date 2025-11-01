"""
Centralized Training Configuration for D3QN Traffic Control System
This file serves as the single source of truth for all training parameters.
"""

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # ========================================================================
    # CORE TRAINING PARAMETERS (OPTIMIZED BASED ON TEST RUN ANALYSIS)
    # ========================================================================
    "episodes": 350,                    # Final training: 350 episodes (converged at 300)
    "learning_rate": 0.0005,           # As used in final training (Chapter 4, Section 4.2.6)
    "epsilon": 1.0,                    # Initial exploration rate
    "epsilon_decay": 0.9995,           # As used in final training (Chapter 4, Section 4.2.6)
    "epsilon_min": 0.01,               # Minimum exploration rate
    "memory_size": 75000,              # As used in final training (Chapter 4, Section 4.2.6)
    "batch_size": 64,                  # As used in final training (Chapter 4, Section 4.2.6)
    "gamma": 0.95,                     # As used in final training (Chapter 4, Section 4.2.6)
    
    # ========================================================================
    # NETWORK ARCHITECTURE
    # ========================================================================
    "lstm_sequence_length": 10,        # LSTM temporal memory window
    "hidden_units_lstm1": 128,         # First LSTM layer units
    "hidden_units_lstm2": 64,          # Second LSTM layer units
    "hidden_units_dense1": 128,        # First dense layer units
    "hidden_units_dense2": 64,         # Second dense layer units
    "dropout_rate": 0.3,               # Dropout rate for regularization
    "recurrent_dropout_rate": 0.2,     # Recurrent dropout for LSTM
    "l2_regularization": 0.001,        # L2 regularization strength
    
    # ========================================================================
    # TRAINING SCHEDULE - OPTIMIZED
    # ========================================================================
    "validation_freq": 15,             # More frequent validation (from analysis)
    "save_freq": 25,                   # More frequent checkpoints
    "log_freq": 1,                     # Logging every N episodes
    "early_stopping_patience": 15,     # CRITICAL FIX: Reduced patience to prevent divergence
    "early_stopping_min_delta": 3.0,   # Reduced threshold for sensitivity
    
    # ========================================================================
    # HYBRID TRAINING (OFFLINE/ONLINE) - OPTIMIZED
    # ========================================================================
    "training_mode": "hybrid",         # "offline", "online", or "hybrid"
    "offline_ratio": 0.75,             # 75% offline for better stability (from analysis)
    "online_memory_size": 20000,       # Increased for better online adaptation
    "online_batch_size": 128,          # Match main batch size for consistency
    "online_epsilon_decay": 0.9998,    # Gradual exploration decay for online
    "online_lr_reduction": 0.7,        # More aggressive LR reduction (from analysis)
    
    # ========================================================================
    # TRAFFIC ENVIRONMENT CONSTRAINTS
    # ========================================================================
    "min_phase_time": 12,              # Minimum phase duration (ITE compliance)
    "max_phase_time": 120,             # Maximum phase duration (efficiency threshold)
    "yellow_time": 3,                  # Yellow phase duration (safety standard)
    "all_red_time": 2,                 # All-red clearance time
    "warmup_time": 30,                 # Simulation warmup period
    "step_length": 1.0,                # Simulation step size (seconds)
    "num_seconds": 300,                # Episode duration (5 minutes)
    "episode_duration": 300,           # Episode duration (alias for num_seconds)
    
    # ========================================================================
    # FAIR CYCLING CONSTRAINTS
    # ========================================================================
    "max_steps_per_cycle": 200,        # Force cycle completion threshold
    "min_cycle_duration": 60,          # Minimum complete cycle duration
    "max_cycle_duration": 180,         # Maximum cycle before forced completion
    "fairness_threshold": 0.25,        # Minimum time allocation per direction
    
    # ========================================================================
    # DATA PROCESSING
    # ========================================================================
    "train_split": 0.7,                # Training data percentage
    "val_split": 0.2,                  # Validation data percentage
    "test_split": 0.1,                 # Test data percentage
    "temporal_ordering": True,         # Preserve temporal order in splits
    "shuffle_episodes": False,         # Don't shuffle to maintain temporal consistency
    
    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================
    "min_episodes_for_stats": 25,      # Minimum episodes for statistical validity
    "confidence_level": 0.95,          # Confidence level for intervals
    "statistical_power": 0.8,          # Target statistical power
    "effect_size_threshold": 0.5,      # Minimum meaningful effect size (Cohen's d)
    "multiple_comparison_correction": "bonferroni",  # Multiple comparison method
    
    # ========================================================================
    # PERFORMANCE MONITORING
    # ========================================================================
    "reward_stability_window": 5,      # Window for reward stability monitoring
    "reward_degradation_threshold": 0.95,  # Degradation detection threshold (95%)
    "memory_usage_monitoring": True,    # Monitor memory usage
    "gpu_monitoring": True,            # Monitor GPU usage if available
    
    # ========================================================================
    # OUTPUT AND LOGGING
    # ========================================================================
    "output_base_dir": "comprehensive_results",
    "production_logs_dir": "production_logs",
    "log_level": "INFO",               # Logging level
    "save_training_plots": True,       # Generate training progress plots
    "save_dashboard_plots": True,      # Generate dashboard visualizations
    "plot_dpi": 300,                   # Plot resolution for publication quality
    "save_individual_episodes": True,  # Save detailed episode results
    
    # ========================================================================
    # REPRODUCIBILITY
    # ========================================================================
    "random_seed": 42,                 # Random seed for reproducibility
    "numpy_seed": 42,                  # NumPy random seed
    "tensorflow_seed": 42,             # TensorFlow random seed
    "sumo_seed_base": 12345,           # Base seed for SUMO (incremented per episode)
    
    # ========================================================================
    # HARDWARE OPTIMIZATION
    # ========================================================================
    "use_gpu": True,                   # Enable GPU acceleration if available
    "gpu_memory_growth": True,         # Allow GPU memory growth
    "cpu_threads": -1,                 # Use all available CPU threads (-1 = auto)
    "parallel_evaluation": True,       # Run baseline comparisons in parallel
}

# ============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# ============================================================================

DEVELOPMENT_CONFIG = {
    **TRAINING_CONFIG,
    "episodes": 10,                    # Reduced for quick testing
    "validation_freq": 2,              # More frequent validation
    "save_freq": 5,                    # More frequent saves
    "log_level": "DEBUG",              # Verbose logging
    "use_gpu": False,                  # CPU-only for debugging
}

TESTING_CONFIG = {
    **TRAINING_CONFIG,
    "episodes": 5,                     # Minimal episodes for tests
    "memory_size": 1000,               # Reduced memory for speed
    "batch_size": 16,                  # Smaller batches
    "validation_freq": 1,              # Validate every episode
    "save_freq": 2,                    # Frequent saves for testing
    "log_level": "WARNING",            # Minimal logging
}

PRODUCTION_CONFIG = {
    **TRAINING_CONFIG,
    "episodes": 500,                   # Extended training
    "validation_freq": 25,             # Less frequent validation
    "save_freq": 100,                  # Less frequent saves
    "log_level": "INFO",               # Standard logging
    "early_stopping_patience": 25,     # More patience for production
}

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config(config):
    """Validate configuration parameters for consistency and correctness"""
    errors = []
    
    # Check required parameters
    required_params = [
        'episodes', 'learning_rate', 'memory_size', 'batch_size',
        'min_phase_time', 'max_phase_time', 'warmup_time'
    ]
    
    for param in required_params:
        if param not in config:
            errors.append(f"Missing required parameter: {param}")
    
    # Check parameter ranges
    if config.get('learning_rate', 0) <= 0 or config.get('learning_rate', 0) > 1:
        errors.append("learning_rate must be between 0 and 1")
    
    if config.get('epsilon_decay', 0) <= 0 or config.get('epsilon_decay', 0) > 1:
        errors.append("epsilon_decay must be between 0 and 1")
    
    if config.get('min_phase_time', 0) >= config.get('max_phase_time', 0):
        errors.append("min_phase_time must be less than max_phase_time")
    
    if config.get('episodes', 0) <= 0:
        errors.append("episodes must be positive")
    
    # Check split ratios
    splits = [config.get('train_split', 0), config.get('val_split', 0), config.get('test_split', 0)]
    if abs(sum(splits) - 1.0) > 0.001:
        errors.append("train_split + val_split + test_split must equal 1.0")
    
    return errors

def get_config(environment="production"):
    """Get configuration for specified environment"""
    config_map = {
        "development": DEVELOPMENT_CONFIG,
        "testing": TESTING_CONFIG,
        "production": PRODUCTION_CONFIG,
        "default": TRAINING_CONFIG
    }
    
    config = config_map.get(environment, TRAINING_CONFIG)
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return config

# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

def print_config_summary(config):
    """Print a summary of the current configuration"""
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"Episodes: {config['episodes']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Memory Size: {config['memory_size']:,}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Training Mode: {config.get('training_mode', 'standard')}")
    if config.get('training_mode') == 'hybrid':
        offline_eps = int(config['episodes'] * config['offline_ratio'])
        online_eps = config['episodes'] - offline_eps
        print(f"  - Offline Episodes: {offline_eps} ({config['offline_ratio']:.0%})")
        print(f"  - Online Episodes: {online_eps} ({1-config['offline_ratio']:.0%})")
    print(f"Traffic Constraints: {config['min_phase_time']}-{config['max_phase_time']}s")
    print(f"Validation Frequency: Every {config['validation_freq']} episodes")
    print(f"Statistical Power Target: {config['statistical_power']}")
    print("=" * 50)

def save_config_to_file(config, filepath):
    """Save configuration to JSON file for reproducibility"""
    import json
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)
    
    print(f"Configuration saved to: {filepath}")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Get production configuration
    config = get_config("production")
    print_config_summary(config)
    
    # Save configuration for reproducibility
    save_config_to_file(config, "config/last_used_config.json")