# 🗄️ Database Schema & Deployment Guide

## 📋 Overview

This guide covers the complete database schema design for the D3QN Traffic Control System, including local-to-cloud data flow, Supabase integration, and production deployment.

## 🏗️ Database Schema Architecture

### Core Tables Structure

```mermaid
graph TD
    A[experiments] --> B[training_episodes]
    A --> C[training_steps]
    A --> D[model_checkpoints]
    A --> E[validation_results]
    A --> F[baseline_comparisons]
    
    B --> G[Episode Metrics]
    C --> H[Real-time State]
    D --> I[Model Storage]
    E --> J[Test Results]
    F --> K[Performance Analysis]
```

### 1. **experiments** - Master Control Table
- **Purpose**: Track training experiments with metadata
- **Key Fields**: 
  - `experiment_name` (unique identifier)
  - `config` (JSONB - hyperparameters)
  - `status` (running/completed/failed)
  - `best_reward`, `convergence_episode`

### 2. **training_episodes** - Episode Performance
- **Purpose**: Store complete episode results and metrics
- **Key Fields**:
  - Core RL metrics (reward, loss, epsilon)
  - Traffic performance (throughput, waiting time, speed)
  - Public transport metrics (buses, jeepneys processed)
  - Reward component breakdown

### 3. **training_steps** - Real-time Monitoring
- **Purpose**: Interval-based step logging for detailed analysis
- **Key Fields**:
  - Traffic state (vehicles, queues, speeds)
  - RL state/action pairs
  - Intersection-specific metrics (MARL)

### 4. **model_checkpoints** - Model Versioning
- **Purpose**: Track model saves and performance
- **Key Fields**:
  - Model file path and metadata
  - Validation performance
  - Hyperparameter snapshot

### 5. **validation_results** - Test Performance
- **Purpose**: Store validation and test results
- **Key Fields**:
  - Test scenario details
  - Performance vs baseline
  - Statistical analysis results

### 6. **baseline_comparisons** - Benchmark Analysis
- **Purpose**: Comprehensive baseline comparison storage
- **Key Fields**:
  - Multiple baseline types
  - Statistical significance tests
  - Improvement percentages

## 🔄 Data Flow Architecture

### Local Training → Database Pipeline

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   D3QN Training │    │  Production      │    │   Supabase      │
│                 │────▶   Logger         │────▶   PostgreSQL    │
│   • Episodes    │    │  • JSON Buffer   │    │   • Real-time   │
│   • Steps       │    │  • File Backup   │    │   • Analytics   │
│   • Metrics     │    │  • DB Sync       │    │   • Monitoring  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Local Files    │    │  Batch Sync      │    │  Real-time      │
│  • Backup       │    │  • Error Retry   │    │  Dashboard      │
│  • Offline      │    │  • Validation    │    │  • Live Plots   │
│  • Recovery     │    │  • Compression   │    │  • Alerts       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Synchronization Modes

#### 1. **Real-time Sync** (Production)
```python
# During training - immediate database writes
logger = ProductionLoggerWithDB(
    experiment_name="production_run",
    enable_db=True,
    log_interval=10  # Sync every 10 steps
)

# Episode completion triggers immediate sync
await logger.log_episode(episode_data)
```

#### 2. **Batch Sync** (Development/Backup)
```python
# Sync local logs to database after training
db_manager = DatabaseManager()
stats = db_manager.sync_local_logs_to_db("production_logs")
print(f"Synced {stats['episodes']} episodes")
```

#### 3. **Offline Mode** (No Database)
```python
# Local files only - no database dependency
logger = ProductionLogger(
    experiment_name="offline_test",
    log_interval=5
)
# Automatically falls back to file-only logging
```

## 🚀 Supabase Deployment Setup

### 1. **Database Setup**
```sql
-- Execute database_schema.sql in Supabase SQL editor
-- This creates all tables, indexes, functions, and triggers

-- Enable Row Level Security (RLS) for production
ALTER TABLE experiments ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_episodes ENABLE ROW LEVEL SECURITY;
-- ... repeat for other tables
```

### 2. **Environment Configuration**
```bash
# Set environment variables for production
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_HOST="db.your-project.supabase.co"
export SUPABASE_PORT="5432"
export SUPABASE_DB="postgres"
export SUPABASE_USER="postgres"
export SUPABASE_PASSWORD="your-secure-password"
```

### 3. **Database Configuration File** (Development)
```json
{
  "host": "db.your-project.supabase.co",
  "port": 5432,
  "database": "postgres",
  "user": "postgres",
  "password": "your-password",
  "sslmode": "require"
}
```

### 4. **Python Integration**
```python
# Install required packages
pip install asyncpg psycopg2-binary

# Initialize database logging
from database_integration import ProductionLoggerWithDB

logger = ProductionLoggerWithDB(
    experiment_name="thesis_final_run",
    enable_db=True
)

await logger.initialize(training_config)
```

## 📊 Data Schema Details

### Episode Data Structure
```json
{
  "episode_number": 1,
  "scenario_info": {
    "day": "20250708",
    "cycle": 1,
    "intersections": ["ECOLAND", "JOHNPAUL", "SANDAWA"]
  },
  "total_reward": 164.67,
  "steps_completed": 300,
  "epsilon_value": 0.885,
  "avg_loss": 0.007,
  "memory_size": 300,
  "total_vehicles": 219,
  "completed_trips": 398,
  "passenger_throughput": 6512.73,
  "avg_waiting_time": 17.18,
  "avg_speed": 18.51,
  "avg_queue_length": 90.03,
  "max_queue_length": 135,
  "travel_time_index": 2.16,
  "buses_processed": 45,
  "jeepneys_processed": 67,
  "pt_passenger_throughput": 890.5,
  "pt_avg_waiting": 12.3,
  "pt_service_efficiency": 0.87,
  "reward_components": {
    "waiting_penalty": -15.2,
    "queue_penalty": -8.5,
    "speed_reward": 12.1,
    "passenger_throughput_reward": 130.2,
    "vehicle_throughput_bonus": 15.8,
    "public_transport_bonus": 8.9
  },
  "episode_duration_minutes": 1.13
}
```

### Step Data Structure (Interval-based)
```json
{
  "episode_number": 1,
  "step_number": 60,
  "active_vehicles": 45,
  "queue_lengths": {
    "ECOLAND": {"north": 3, "south": 7, "east": 2, "west": 5},
    "JOHNPAUL": {"north": 4, "south": 6, "east": 3, "west": 2},
    "SANDAWA": {"north": 2, "south": 8, "east": 4, "west": 3}
  },
  "waiting_times": {
    "ECOLAND": 15.2,
    "JOHNPAUL": 12.8,
    "SANDAWA": 18.5
  },
  "speeds": {
    "ECOLAND": 18.5,
    "JOHNPAUL": 16.2,
    "SANDAWA": 14.8
  },
  "state_vector": [0.15, 0.23, 0.45, "..."],
  "action_taken": 3,
  "immediate_reward": 2.34,
  "intersection_metrics": {
    "coordination_score": 0.85,
    "efficiency_index": 0.78
  },
  "simulation_time": 360
}
```

## 🔍 Analytics and Monitoring

### Real-time Dashboard Queries
```sql
-- Current training status
SELECT * FROM real_time_training 
WHERE experiment_name = 'thesis_final_run';

-- Performance trends
SELECT episode_number, total_reward, passenger_throughput 
FROM training_episodes 
WHERE experiment_id = (
  SELECT id FROM experiments 
  WHERE experiment_name = 'thesis_final_run'
)
ORDER BY episode_number;

-- Public transport performance
SELECT 
  AVG(buses_processed) as avg_buses,
  AVG(jeepneys_processed) as avg_jeepneys,
  AVG(pt_passenger_throughput) as avg_pt_throughput
FROM training_episodes 
WHERE experiment_id = (SELECT id FROM experiments WHERE experiment_name = 'thesis_final_run');
```

### Performance Analytics Views
```sql
-- Training progress analysis
SELECT * FROM performance_analytics 
WHERE experiment_name = 'thesis_final_run';

-- Convergence detection
SELECT episode_number, 
       AVG(total_reward) OVER (ORDER BY episode_number ROWS 9 PRECEDING) as moving_avg
FROM training_episodes 
WHERE experiment_id = (SELECT id FROM experiments WHERE experiment_name = 'thesis_final_run')
ORDER BY episode_number;
```

## 🔐 Security Considerations

### 1. **Row Level Security**
```sql
-- Ensure users only see their experiments
CREATE POLICY experiment_access ON experiments
FOR ALL USING (created_by = auth.uid());
```

### 2. **API Key Management**
```python
# Use environment variables, never hardcode
import os
DB_PASSWORD = os.getenv('SUPABASE_PASSWORD')
if not DB_PASSWORD:
    raise ValueError("Database password not configured")
```

### 3. **Connection Pooling**
```python
# Limit connections to prevent DoS
pool = await asyncpg.create_pool(
    **config,
    min_size=2,
    max_size=10,
    command_timeout=30
)
```

## 📈 Performance Optimization

### 1. **Indexing Strategy**
```sql
-- Critical indexes for query performance
CREATE INDEX idx_episodes_exp_time ON training_episodes(experiment_id, timestamp);
CREATE INDEX idx_steps_interval ON training_steps(experiment_id, episode_number, step_number);
CREATE INDEX idx_episodes_reward ON training_episodes(total_reward DESC);
```

### 2. **Data Partitioning** (Large Scale)
```sql
-- Partition by experiment for large deployments
CREATE TABLE training_steps_partitioned (
    LIKE training_steps INCLUDING ALL
) PARTITION BY HASH (experiment_id);
```

### 3. **Compression and Archival**
```sql
-- Archive old training data
CREATE OR REPLACE FUNCTION archive_old_experiments(days_old INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
BEGIN
    -- Move old data to archive tables
    INSERT INTO training_episodes_archive 
    SELECT * FROM training_episodes 
    WHERE timestamp < (NOW() - INTERVAL '1 day' * days_old);
    
    DELETE FROM training_episodes 
    WHERE timestamp < (NOW() - INTERVAL '1 day' * days_old);
    
    RETURN 1;
END;
$$ LANGUAGE plpgsql;
```

## 🚀 Production Deployment Checklist

### Pre-deployment
- [ ] Database schema deployed to Supabase
- [ ] Environment variables configured
- [ ] Database connection tested
- [ ] Row level security enabled
- [ ] Backup strategy implemented

### During Training
- [ ] Real-time monitoring dashboard active
- [ ] Error alerting configured
- [ ] Local file backup enabled
- [ ] Resource monitoring active

### Post-training
- [ ] Data validation completed
- [ ] Results exported for analysis
- [ ] Models backed up to cloud storage
- [ ] Performance benchmarks documented

## 🔧 Troubleshooting

### Common Issues
```python
# Connection timeout
asyncio.get_event_loop().run_until_complete(
    asyncio.wait_for(db_operation(), timeout=30)
)

# Memory management for large datasets
# Use cursor-based pagination
cursor = conn.cursor()
cursor.execute("SELECT * FROM training_episodes WHERE ...")
while True:
    rows = cursor.fetchmany(1000)
    if not rows:
        break
    process_batch(rows)
```

### Error Recovery
```python
# Automatic retry with exponential backoff
import asyncio
from functools import wraps

async def retry_on_failure(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

## 📊 Data Export and Analysis

### Export Training Data
```python
# Export for external analysis
from database_integration import DatabaseManager

db = DatabaseManager()
db.export_experiment_data("thesis_final_run", "exports/")
```

### Integration with Analysis Tools
```python
# Direct integration with pandas/matplotlib
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host/db")
df = pd.read_sql("""
    SELECT episode_number, total_reward, passenger_throughput 
    FROM training_episodes 
    WHERE experiment_id = (
        SELECT id FROM experiments 
        WHERE experiment_name = 'thesis_final_run'
    )
""", engine)

# Now use df for plotting, analysis, etc.
```

---

This comprehensive database system ensures:
- ✅ **Real-time monitoring** during training
- ✅ **Robust data storage** with backup strategies  
- ✅ **Scalable architecture** for large experiments
- ✅ **Research reproducibility** with complete audit trails
- ✅ **Production deployment** ready for real-world use

The system bridges local development with cloud deployment, providing a seamless path from research to production.
