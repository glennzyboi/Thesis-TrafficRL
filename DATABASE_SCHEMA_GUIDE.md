# 🗄️ Database Schema Design Guide for D3QN Traffic Control System

## 📋 Overview

This guide describes the database schema requirements for storing D3QN training logs and supporting the practical output phase. It focuses on **what data needs to be stored**, **how logs flow from training to database**, and **what features will be needed for real-world deployment**.

## 🎯 Core Requirements for Practical Output

### 1. **Real-time Monitoring Dashboard**
- **Live training progress** tracking across multiple experiments
- **Performance metrics visualization** (reward curves, traffic metrics)
- **Alert system** for training issues or performance degradation
- **Comparison views** between different training runs

### 2. **Research Validation & Analysis**
- **Statistical analysis** capabilities for thesis defense
- **Baseline comparison** storage and retrieval
- **Reproducibility tracking** with complete parameter logs
- **Publication-ready data export** (CSV, JSON formats)

### 3. **Production Deployment Support**
- **Model versioning** and checkpoint management
- **Performance benchmarking** against established baselines
- **Hyperparameter optimization** history and analysis
- **A/B testing** framework for different model versions

---

## 📊 Data Flow Architecture

### Training to Database Pipeline

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   D3QN Training     │    │   Production        │    │   Database          │
│                     │    │   Logger            │    │                     │
│ • Episode Results   │────▶ • JSON Buffering   │────▶ • Structured Storage │
│ • Step-by-step Data │    │ • Local Backup     │    │ • Real-time Access   │
│ • Model Checkpoints │    │ • Error Handling   │    │ • Analytics Support  │
│ • Performance Metrics│   │ • Batch Sync       │    │ • Reporting Engine   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### Key Data Collection Points

1. **Episode Level** (Every episode completion)
   - Complete performance metrics
   - Reward component breakdown
   - Traffic statistics
   - Public transport metrics

2. **Step Level** (Configurable intervals, e.g., every 10 steps)
   - Real-time state information
   - Action decisions
   - Immediate rewards
   - Traffic light status

3. **Checkpoint Level** (Model saves)
   - Model performance validation
   - Hyperparameter snapshots
   - Comparison with previous versions

---

## 🏗️ Database Schema Structure

### Core Tables Overview

#### 1. **EXPERIMENTS** - Master Control Table
**Purpose**: Track and manage different training experiments

**Essential Fields**:
- **Experiment Identity**: Unique name, description, creation timestamp
- **Configuration**: Complete hyperparameter set (learning rate, epsilon, batch size, etc.)
- **Status Tracking**: Running, completed, failed, paused
- **Performance Summary**: Best reward achieved, convergence episode, total duration
- **Research Context**: Which baseline being compared, dataset used

**Why This Matters for Practical Output**:
- Allows comparing different hyperparameter combinations
- Tracks which experiments are production-ready
- Enables reproducible research for thesis defense

#### 2. **TRAINING_EPISODES** - Episode Performance Data
**Purpose**: Store complete results for each training episode

**Essential Fields**:

**RL-Specific Metrics**:
- `episode_number`, `total_reward`, `steps_completed`
- `epsilon_value` (exploration rate), `avg_loss`, `memory_size`

**Traffic Performance Metrics**:
- `total_vehicles`, `completed_trips`, `passenger_throughput`
- `avg_waiting_time`, `avg_speed`, `avg_queue_length`, `max_queue_length`
- `travel_time_index` (efficiency measure)

**Public Transport Metrics** (Your Novel Contribution):
- `buses_processed`, `jeepneys_processed`
- `pt_passenger_throughput`, `pt_avg_waiting`, `pt_service_efficiency`

**Reward Components** (For Analysis):
- `waiting_penalty`, `queue_penalty`, `speed_reward`
- `passenger_throughput_reward`, `vehicle_throughput_bonus`
- `public_transport_bonus` (your enhanced feature)

**Scenario Context**:
- `scenario_day`, `scenario_cycle`, `intersections_included`
- `episode_duration_minutes`

**Why This Matters**:
- Enables detailed learning curve analysis
- Supports statistical significance testing
- Tracks public transport performance (key thesis contribution)

#### 3. **TRAINING_STEPS** - Real-time State Monitoring
**Purpose**: Capture detailed step-by-step behavior (interval-based to manage volume)

**Essential Fields**:
- **Traffic State**: `active_vehicles`, `queue_lengths_per_intersection`, `waiting_times`
- **RL Decision**: `action_taken`, `immediate_reward`, `state_representation`
- **Multi-Agent Data**: `intersection_specific_metrics`, `coordination_scores`
- **Timing**: `simulation_time`, `real_timestamp`

**Data Structure Example**:
```json
{
  "step_number": 120,
  "queue_lengths": {
    "ECOLAND": {"north": 3, "south": 7, "east": 2, "west": 5},
    "JOHNPAUL": {"north": 4, "south": 6, "east": 3, "west": 2},
    "SANDAWA": {"north": 2, "south": 8, "east": 4, "west": 3}
  },
  "action_taken": 3,
  "immediate_reward": 2.34
}
```

**Why This Matters**:
- Enables fine-grained behavior analysis
- Supports debugging of agent decisions
- Allows visualization of traffic flow patterns

#### 4. **MODEL_CHECKPOINTS** - Model Version Management
**Purpose**: Track model saves and their performance characteristics

**Essential Fields**:
- **Model Identity**: `checkpoint_episode`, `model_file_path`, `model_size`
- **Performance**: `validation_reward`, `baseline_comparison_results`
- **Configuration**: `hyperparameters_snapshot`, `training_time_elapsed`
- **Validation Results**: `test_episode_performance`, `statistical_significance`

**Why This Matters**:
- Supports model rollback if performance degrades
- Enables A/B testing between model versions
- Critical for production deployment decisions

#### 5. **BASELINE_COMPARISONS** - Performance Benchmarking
**Purpose**: Store comprehensive comparisons against fixed-time and other baselines

**Essential Fields**:
- **Test Scenarios**: Which traffic scenarios were used for comparison
- **D3QN Results**: Complete performance metrics from your trained model
- **Baseline Results**: Performance from fixed-time, adaptive, other methods
- **Statistical Analysis**: t-test results, effect sizes, confidence intervals
- **Improvement Metrics**: Percentage improvements per metric

**Why This Matters**:
- Essential for thesis defense (proving your method works)
- Supports publication claims with statistical backing
- Enables comparison with future research

#### 6. **VALIDATION_RESULTS** - Test Set Performance
**Purpose**: Track performance on held-out test data (critical for avoiding overfitting claims)

**Essential Fields**:
- **Test Scenario Details**: Which specific test scenarios were used
- **Performance Metrics**: Complete traffic and RL performance data
- **Cross-Validation**: If using k-fold or temporal validation
- **Generalization Assessment**: How well model performs on unseen data

---

## 🔄 Log Data Flow Process

### 1. **During Training** (Real-time)

```python
# Every Episode Completion
episode_data = {
    "experiment_id": "thesis_final_run",
    "episode_number": 45,
    "total_reward": 164.67,
    "passenger_throughput": 6512.73,
    "buses_processed": 45,
    "jeepneys_processed": 67,
    "reward_components": {
        "waiting_penalty": -15.2,
        "passenger_throughput_reward": 130.2,
        "public_transport_bonus": 8.9
    },
    "scenario_info": {
        "day": "20250708",
        "cycle": 1,
        "intersections": ["ECOLAND", "JOHNPAUL", "SANDAWA"]
    }
}
# → Insert into TRAINING_EPISODES table
```

```python
# Every 10 Steps (Configurable)
step_data = {
    "experiment_id": "thesis_final_run",
    "episode_number": 45,
    "step_number": 120,
    "active_vehicles": 185,
    "queue_lengths": {...},  # Per intersection
    "action_taken": 3,
    "immediate_reward": 2.34
}
# → Insert into TRAINING_STEPS table
```

### 2. **Model Checkpoints** (Every 20 episodes)

```python
checkpoint_data = {
    "experiment_id": "thesis_final_run",
    "episode_number": 40,
    "model_path": "models/checkpoint_ep40.keras",
    "validation_performance": {
        "avg_reward": 156.3,
        "passenger_throughput": 6234.5,
        "vs_baseline_improvement": 45.2
    }
}
# → Insert into MODEL_CHECKPOINTS table
```

### 3. **End of Training** (Comprehensive analysis)

```python
final_comparison = {
    "experiment_id": "thesis_final_run",
    "baseline_type": "fixed_time",
    "test_scenarios": [...],
    "d3qn_performance": {...},
    "baseline_performance": {...},
    "statistical_results": {
        "waiting_time_improvement": 51.1,
        "p_value": 0.001,
        "effect_size": 1.23
    }
}
# → Insert into BASELINE_COMPARISONS table
```

---

## 🎯 Practical Output Features Enabled

### 1. **Real-time Training Dashboard**

**Live Metrics Display**:
```sql
-- Example query for dashboard
SELECT episode_number, total_reward, passenger_throughput, 
       buses_processed, jeepneys_processed
FROM training_episodes 
WHERE experiment_id = 'current_training'
ORDER BY episode_number DESC LIMIT 50;
```

**Performance Trends**:
- Reward progression over episodes
- Learning stability (reward variance)
- Public transport metrics evolution
- Traffic efficiency improvements

### 2. **Research Validation Queries**

**Statistical Significance Testing**:
```sql
-- Compare D3QN vs baseline performance
SELECT 
    AVG(d3qn_avg_waiting) as d3qn_waiting,
    AVG(baseline_avg_waiting) as baseline_waiting,
    (AVG(baseline_avg_waiting) - AVG(d3qn_avg_waiting)) / AVG(baseline_avg_waiting) * 100 as improvement_percent
FROM baseline_comparisons 
WHERE experiment_id = 'thesis_final_run';
```

**Convergence Analysis**:
```sql
-- Moving average for convergence detection
SELECT episode_number,
       AVG(total_reward) OVER (ORDER BY episode_number ROWS 9 PRECEDING) as moving_avg_reward
FROM training_episodes 
WHERE experiment_id = 'thesis_final_run';
```

### 3. **Production Deployment Queries**

**Best Model Selection**:
```sql
-- Find best performing checkpoint
SELECT model_path, validation_reward, episode_number
FROM model_checkpoints 
WHERE experiment_id = 'production_candidate'
ORDER BY validation_reward DESC LIMIT 1;
```

**Performance Monitoring**:
```sql
-- Track public transport efficiency over time
SELECT DATE(timestamp) as training_date,
       AVG(pt_service_efficiency) as daily_pt_efficiency,
       AVG(passenger_throughput) as daily_passenger_throughput
FROM training_episodes 
GROUP BY DATE(timestamp)
ORDER BY training_date;
```

---

## 📈 Data Volume Considerations

### Expected Data Volumes

**For 200-episode training run**:
- **TRAINING_EPISODES**: 200 rows (~50KB total)
- **TRAINING_STEPS**: ~6,000 rows (200 episodes × 300 steps ÷ 10 interval) (~2MB)
- **MODEL_CHECKPOINTS**: ~10 rows (~5KB)

**For production deployment** (multiple experiments):
- Scale by number of experiments
- Consider data archival strategy for old experiments
- Implement data compression for step-level data

### Performance Optimization Recommendations

1. **Index Strategy**:
   - Index on `experiment_id` + `episode_number`
   - Index on `timestamp` for time-series queries
   - Index on `total_reward` for performance ranking

2. **Data Partitioning**:
   - Partition by experiment_id for large deployments
   - Consider time-based partitioning for historical data

3. **Archival Strategy**:
   - Move old step-level data to archive tables
   - Keep episode-level data for long-term analysis

---

## 🔍 Analytics Capabilities Enabled

### 1. **Training Analysis**
- **Learning curve visualization**: Reward progression, convergence detection
- **Hyperparameter sensitivity**: Compare experiments with different parameters
- **Stability analysis**: Reward variance, performance consistency

### 2. **Traffic Performance Analysis**
- **Public transport optimization**: Track PT-specific improvements over time
- **Rush hour performance**: Analyze performance under different traffic conditions
- **Intersection-specific analysis**: MARL coordination effectiveness

### 3. **Research Validation**
- **Statistical significance**: Automated p-value calculations
- **Effect size analysis**: Practical significance of improvements
- **Baseline comparisons**: Multiple baseline method comparisons

### 4. **Production Readiness**
- **Model selection**: Automated best model identification
- **Performance degradation detection**: Alert if performance drops
- **A/B testing support**: Compare different model versions

---

## 🚀 Implementation Recommendations

### 1. **Database Choice**
- **PostgreSQL**: Excellent JSON support, advanced analytics
- **MongoDB**: Document-based, natural fit for JSON logs
- **InfluxDB**: Time-series optimized for real-time monitoring

### 2. **Data Ingestion Strategy**
- **Batch processing**: Buffer logs locally, sync periodically
- **Real-time streaming**: Direct database writes (for live monitoring)
- **Hybrid approach**: Critical data real-time, detailed data batched

### 3. **Monitoring & Alerting**
- **Performance thresholds**: Alert if reward drops significantly
- **Training progress**: Notify on convergence or training completion
- **Error detection**: Alert on training failures or data issues

---

## 📊 Expected Practical Output Benefits

### For Research & Thesis Defense
- **Comprehensive evidence** of model superiority with statistical backing
- **Detailed analysis** of public transport optimization (novel contribution)
- **Reproducible results** with complete parameter and performance tracking
- **Publication-ready data** with automated export capabilities

### For Real-world Deployment
- **Model versioning** and rollback capabilities
- **Performance monitoring** in production environments
- **Continuous improvement** through ongoing training analysis
- **Scalable architecture** for multiple intersection deployments

### For Future Research
- **Baseline establishment** for future traffic RL research
- **Hyperparameter optimization** guidance for similar problems
- **Public transport integration** patterns for other cities
- **MARL coordination** insights for larger traffic networks

---

This schema design supports your complete research pipeline from training through thesis defense to real-world deployment, with special emphasis on your novel public transport optimization contribution.

