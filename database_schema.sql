-- =====================================================
-- D3QN Traffic Control System - Supabase Database Schema
-- =====================================================
-- This schema supports real-time logging, analysis, and deployment
-- Compatible with Supabase PostgreSQL with real-time subscriptions

-- =====================================================
-- 1. EXPERIMENTS TABLE
-- =====================================================
CREATE TABLE experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    config JSONB NOT NULL, -- Store hyperparameters, settings
    status VARCHAR(50) DEFAULT 'running', -- 'running', 'completed', 'failed', 'paused'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    total_episodes INTEGER DEFAULT 0,
    best_reward DECIMAL(10,4),
    convergence_episode INTEGER,
    metadata JSONB -- Additional experiment info
);

-- Index for quick experiment lookups
CREATE INDEX idx_experiments_name ON experiments(experiment_name);
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_created ON experiments(created_at);

-- =====================================================
-- 2. TRAINING EPISODES TABLE
-- =====================================================
CREATE TABLE training_episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    episode_number INTEGER NOT NULL,
    scenario_info JSONB NOT NULL, -- day, cycle, scenario details
    
    -- Core RL Metrics
    total_reward DECIMAL(10,4) NOT NULL,
    steps_completed INTEGER NOT NULL,
    epsilon_value DECIMAL(8,6),
    avg_loss DECIMAL(12,8),
    memory_size INTEGER,
    
    -- Traffic Performance Metrics
    total_vehicles INTEGER,
    completed_trips INTEGER,
    passenger_throughput DECIMAL(10,2),
    avg_waiting_time DECIMAL(8,3),
    avg_speed DECIMAL(8,3),
    avg_queue_length DECIMAL(8,3),
    max_queue_length INTEGER,
    travel_time_index DECIMAL(8,4),
    
    -- Public Transport Metrics
    buses_processed INTEGER DEFAULT 0,
    jeepneys_processed INTEGER DEFAULT 0,
    pt_passenger_throughput DECIMAL(10,2) DEFAULT 0,
    pt_avg_waiting DECIMAL(8,3) DEFAULT 0,
    pt_service_efficiency DECIMAL(6,4) DEFAULT 0,
    
    -- Reward Components (for analysis)
    reward_components JSONB, -- breakdown of reward calculation
    
    -- Timing
    episode_duration_minutes DECIMAL(8,4),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(experiment_id, episode_number)
);

-- Indexes for performance
CREATE INDEX idx_episodes_experiment ON training_episodes(experiment_id);
CREATE INDEX idx_episodes_number ON training_episodes(episode_number);
CREATE INDEX idx_episodes_timestamp ON training_episodes(timestamp);
CREATE INDEX idx_episodes_reward ON training_episodes(total_reward);

-- =====================================================
-- 3. TRAINING STEPS TABLE (Interval-based logging)
-- =====================================================
CREATE TABLE training_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    episode_number INTEGER NOT NULL,
    step_number INTEGER NOT NULL,
    
    -- Real-time Traffic State
    active_vehicles INTEGER,
    queue_lengths JSONB, -- per intersection/lane
    waiting_times JSONB, -- per intersection
    speeds JSONB, -- per intersection
    
    -- RL State/Action
    state_vector JSONB, -- compressed state representation
    action_taken INTEGER,
    immediate_reward DECIMAL(8,4),
    
    -- Intersection-specific (for MARL)
    intersection_metrics JSONB, -- per-intersection performance
    
    -- Timing
    simulation_time INTEGER, -- simulation seconds
    real_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(experiment_id, episode_number, step_number)
);

-- Partitioning for performance (steps generate lots of data)
-- CREATE INDEX idx_steps_experiment_episode ON training_steps(experiment_id, episode_number);
-- Consider partitioning by experiment_id for large deployments

-- =====================================================
-- 4. MODEL CHECKPOINTS TABLE
-- =====================================================
CREATE TABLE model_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    episode_number INTEGER NOT NULL,
    checkpoint_type VARCHAR(50) NOT NULL, -- 'best', 'regular', 'final'
    
    -- Model Performance at Checkpoint
    validation_reward DECIMAL(10,4),
    validation_metrics JSONB,
    
    -- Model Storage (could be file path or blob reference)
    model_path TEXT, -- Path to stored model file
    model_size_mb DECIMAL(8,2),
    model_hash VARCHAR(64), -- For integrity verification
    
    -- Metadata
    training_time_minutes DECIMAL(10,4),
    hyperparameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Performance comparison
    vs_baseline_improvement JSONB -- improvement over fixed-time
);

CREATE INDEX idx_checkpoints_experiment ON model_checkpoints(experiment_id);
CREATE INDEX idx_checkpoints_type ON model_checkpoints(checkpoint_type);

-- =====================================================
-- 5. VALIDATION RESULTS TABLE
-- =====================================================
CREATE TABLE validation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    validation_episode INTEGER NOT NULL,
    checkpoint_id UUID REFERENCES model_checkpoints(id),
    
    -- Test Scenario
    test_scenario JSONB NOT NULL,
    
    -- Performance Metrics
    total_reward DECIMAL(10,4),
    traffic_metrics JSONB, -- comprehensive traffic performance
    comparison_metrics JSONB, -- vs baseline comparison
    
    -- Statistical Analysis
    confidence_interval JSONB,
    statistical_significance BOOLEAN,
    
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- 6. BASELINE COMPARISONS TABLE
-- =====================================================
CREATE TABLE baseline_comparisons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    baseline_type VARCHAR(100) NOT NULL, -- 'fixed_time', 'adaptive', etc.
    
    -- Comparison Scenarios
    test_scenarios JSONB NOT NULL,
    
    -- Performance Results
    d3qn_metrics JSONB NOT NULL,
    baseline_metrics JSONB NOT NULL,
    improvement_percentages JSONB NOT NULL,
    
    -- Statistical Analysis
    statistical_tests JSONB, -- t-tests, effect sizes, etc.
    significance_results JSONB,
    
    -- Metadata
    comparison_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT
);

-- =====================================================
-- 7. REAL-TIME MONITORING VIEW
-- =====================================================
-- View for real-time dashboard monitoring
CREATE VIEW real_time_training AS
SELECT 
    e.experiment_name,
    e.status,
    e.total_episodes,
    e.best_reward,
    te.episode_number,
    te.total_reward as current_reward,
    te.passenger_throughput,
    te.avg_waiting_time,
    te.completed_trips,
    te.timestamp,
    -- Latest step info
    ts.active_vehicles,
    ts.simulation_time
FROM experiments e
LEFT JOIN training_episodes te ON e.id = te.experiment_id
LEFT JOIN training_steps ts ON te.experiment_id = ts.experiment_id 
    AND te.episode_number = ts.episode_number
WHERE te.episode_number = e.total_episodes -- Latest episode
ORDER BY te.timestamp DESC;

-- =====================================================
-- 8. PERFORMANCE ANALYTICS VIEW
-- =====================================================
CREATE VIEW performance_analytics AS
SELECT 
    e.experiment_name,
    COUNT(te.id) as episodes_completed,
    AVG(te.total_reward) as avg_reward,
    MAX(te.total_reward) as best_reward,
    AVG(te.passenger_throughput) as avg_passenger_throughput,
    AVG(te.avg_waiting_time) as avg_waiting_time,
    AVG(te.completed_trips) as avg_completed_trips,
    STDDEV(te.total_reward) as reward_std,
    -- Learning progress
    (SELECT total_reward FROM training_episodes WHERE experiment_id = e.id ORDER BY episode_number DESC LIMIT 1) as latest_reward,
    (SELECT total_reward FROM training_episodes WHERE experiment_id = e.id ORDER BY episode_number ASC LIMIT 1) as first_reward
FROM experiments e
LEFT JOIN training_episodes te ON e.id = te.experiment_id
GROUP BY e.id, e.experiment_name;

-- =====================================================
-- 9. TRIGGERS AND FUNCTIONS
-- =====================================================

-- Function to update experiment statistics
CREATE OR REPLACE FUNCTION update_experiment_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE experiments 
    SET 
        total_episodes = (
            SELECT MAX(episode_number) 
            FROM training_episodes 
            WHERE experiment_id = NEW.experiment_id
        ),
        best_reward = (
            SELECT MAX(total_reward) 
            FROM training_episodes 
            WHERE experiment_id = NEW.experiment_id
        )
    WHERE id = NEW.experiment_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update experiment stats
CREATE TRIGGER update_experiment_stats_trigger
    AFTER INSERT ON training_episodes
    FOR EACH ROW
    EXECUTE FUNCTION update_experiment_stats();

-- =====================================================
-- 10. SAMPLE DATA INSERTION PROCEDURE
-- =====================================================

-- Function for inserting training episode (used by Python logger)
CREATE OR REPLACE FUNCTION insert_training_episode(
    p_experiment_name VARCHAR(255),
    p_episode_data JSONB
) RETURNS UUID AS $$
DECLARE
    exp_id UUID;
    episode_id UUID;
BEGIN
    -- Get or create experiment
    SELECT id INTO exp_id FROM experiments WHERE experiment_name = p_experiment_name;
    
    IF exp_id IS NULL THEN
        INSERT INTO experiments (experiment_name, config, status)
        VALUES (p_experiment_name, p_episode_data->'config', 'running')
        RETURNING id INTO exp_id;
    END IF;
    
    -- Insert episode
    INSERT INTO training_episodes (
        experiment_id,
        episode_number,
        scenario_info,
        total_reward,
        steps_completed,
        epsilon_value,
        avg_loss,
        memory_size,
        total_vehicles,
        completed_trips,
        passenger_throughput,
        avg_waiting_time,
        avg_speed,
        avg_queue_length,
        max_queue_length,
        travel_time_index,
        buses_processed,
        jeepneys_processed,
        pt_passenger_throughput,
        pt_avg_waiting,
        pt_service_efficiency,
        reward_components,
        episode_duration_minutes
    ) VALUES (
        exp_id,
        (p_episode_data->>'episode_number')::INTEGER,
        p_episode_data->'scenario_info',
        (p_episode_data->>'total_reward')::DECIMAL,
        (p_episode_data->>'steps_completed')::INTEGER,
        (p_episode_data->>'epsilon_value')::DECIMAL,
        (p_episode_data->>'avg_loss')::DECIMAL,
        (p_episode_data->>'memory_size')::INTEGER,
        (p_episode_data->>'total_vehicles')::INTEGER,
        (p_episode_data->>'completed_trips')::INTEGER,
        (p_episode_data->>'passenger_throughput')::DECIMAL,
        (p_episode_data->>'avg_waiting_time')::DECIMAL,
        (p_episode_data->>'avg_speed')::DECIMAL,
        (p_episode_data->>'avg_queue_length')::DECIMAL,
        (p_episode_data->>'max_queue_length')::INTEGER,
        (p_episode_data->>'travel_time_index')::DECIMAL,
        COALESCE((p_episode_data->>'buses_processed')::INTEGER, 0),
        COALESCE((p_episode_data->>'jeepneys_processed')::INTEGER, 0),
        COALESCE((p_episode_data->>'pt_passenger_throughput')::DECIMAL, 0),
        COALESCE((p_episode_data->>'pt_avg_waiting')::DECIMAL, 0),
        COALESCE((p_episode_data->>'pt_service_efficiency')::DECIMAL, 0),
        p_episode_data->'reward_components',
        (p_episode_data->>'episode_duration_minutes')::DECIMAL
    ) RETURNING id INTO episode_id;
    
    RETURN episode_id;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 11. INDEXES FOR ANALYTICS
-- =====================================================

-- GIN indexes for JSONB queries
CREATE INDEX idx_episodes_scenario_gin ON training_episodes USING GIN (scenario_info);
CREATE INDEX idx_episodes_rewards_gin ON training_episodes USING GIN (reward_components);
CREATE INDEX idx_steps_metrics_gin ON training_steps USING GIN (intersection_metrics);

-- Composite indexes for common queries
CREATE INDEX idx_episodes_exp_perf ON training_episodes(experiment_id, total_reward DESC, episode_number);
CREATE INDEX idx_episodes_time_series ON training_episodes(experiment_id, episode_number, timestamp);

-- =====================================================
-- 12. SECURITY & PERMISSIONS
-- =====================================================

-- Create roles for different access levels
-- (Uncomment when setting up production)

-- CREATE ROLE d3qn_training_writer;
-- CREATE ROLE d3qn_analyst_reader;
-- CREATE ROLE d3qn_admin;

-- Grant permissions
-- GRANT INSERT, UPDATE ON experiments, training_episodes, training_steps TO d3qn_training_writer;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO d3qn_analyst_reader;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO d3qn_admin;

-- =====================================================
-- 13. MAINTENANCE
-- =====================================================

-- Function to clean up old step data (for production maintenance)
CREATE OR REPLACE FUNCTION cleanup_old_steps(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM training_steps 
    WHERE real_timestamp < (NOW() - INTERVAL '1 day' * days_to_keep);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE experiments IS 'Master table for training experiments with metadata and configuration';
COMMENT ON TABLE training_episodes IS 'Episode-level training results and performance metrics';
COMMENT ON TABLE training_steps IS 'Step-level training data for detailed analysis (interval-based)';
COMMENT ON TABLE model_checkpoints IS 'Model snapshots and their performance metrics';
COMMENT ON TABLE validation_results IS 'Validation test results against held-out data';
COMMENT ON TABLE baseline_comparisons IS 'Performance comparisons against baseline methods';

-- =====================================================
-- SCHEMA COMPLETE
-- =====================================================
