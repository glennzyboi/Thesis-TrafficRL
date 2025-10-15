# Real-Time Dashboard Integration Guide

**Date**: October 13, 2025  
**Purpose**: Complete guide for integrating real-time database logging with training  
**Status**: READY TO INTEGRATE

---

## SYSTEM ARCHITECTURE

```
Training Script (comprehensive_training.py)
    ↓
Real-Time Logger (utils/realtime_dashboard_logger.py)
    ↓
SQLite Database (dashboard_data/training.db)
    ↓
API Server (scripts/dashboard_api_server.py)
    ↓
Frontend Dashboard (https://traffic-compare-17.vercel.app)
```

---

## STEP 1: INTEGRATE LOGGER INTO TRAINING SCRIPT

Add these lines to `experiments/comprehensive_training.py`:

### Import the Logger

```python
# Add this import at the top with other imports
from utils.realtime_dashboard_logger import RealtimeDashboardLogger
```

### Initialize in __init__

```python
class ComprehensiveTrainer:
    def __init__(self, experiment_name: str = None):
        # ... existing code ...
        
        # Initialize production logger
        self.logger = create_production_logger(self.experiment_name)
        
        # ADD THIS: Initialize dashboard logger
        self.dashboard_logger = RealtimeDashboardLogger(
            experiment_name=self.experiment_name,
            db_path='dashboard_data/training.db'
        )
```

### Log Episode Data After Each Episode

Find the section where episode completes (around line 250-300) and add:

```python
def train_episode(self, agent, env, episode, bundle, phase='offline'):
    # ... existing training code ...
    
    # Calculate episode metrics
    episode_metrics = {
        'completed_trips': env.metrics.get('completed_trips', 0),
        'waiting_time': env.metrics.get('waiting_time', 0),
        'avg_speed': env.metrics.get('avg_speed', 0),
        'queue_length': env.metrics.get('queue_length', 0),
        'max_queue_length': env.metrics.get('max_queue_length', 0),
        'total_reward': episode_reward,
        'loss': episode_loss / max(steps, 1),
        'epsilon': agent.epsilon,
        'scenario': bundle.get('scenario_name', f'Episode {episode}'),
        'steps': steps,
        'duration_seconds': 300,
        
        # Reward components (if available)
        'throughput_reward': env.metrics.get('throughput_reward', 0),
        'waiting_penalty': env.metrics.get('waiting_penalty', 0),
        'speed_reward': env.metrics.get('speed_reward', 0),
        'queue_penalty': env.metrics.get('queue_penalty', 0),
        'emergency_penalty': env.metrics.get('emergency_penalty', 0)
    }
    
    # ADD THIS: Log to dashboard database in real-time
    self.dashboard_logger.log_episode(
        episode_number=episode,
        metrics=episode_metrics,
        vehicle_breakdown=None,  # Will be estimated automatically
        phase=phase
    )
    
    # ... rest of existing code ...
```

### Mark Training Complete

At the end of training (in the `run()` method):

```python
def run(self, agent_type='lstm', episodes=None, offline_episodes=None):
    # ... existing training code ...
    
    # Training complete
    print("Training completed successfully")
    
    # ADD THIS: Mark experiment as complete in database
    self.dashboard_logger.mark_experiment_complete(total_episodes=episodes)
```

---

## STEP 2: LOG EVALUATION RESULTS

In `evaluation/performance_comparison.py`, add logging for evaluation:

### Import

```python
from utils.realtime_dashboard_logger import RealtimeDashboardLogger
```

### Log Results

```python
class PerformanceComparator:
    def run_comparison(self, ...):
        # ... existing code ...
        
        # After evaluation completes, log results
        dashboard_logger = RealtimeDashboardLogger(
            experiment_name=experiment_name,
            db_path='dashboard_data/training.db'
        )
        
        # Log D3QN results
        dashboard_logger.log_evaluation_results(
            agent_type='d3qn',
            episodes=d3qn_results
        )
        
        # Log Fixed-Time results
        dashboard_logger.log_evaluation_results(
            agent_type='fixed_time',
            episodes=fixed_time_results
        )
        
        # Calculate and log summary statistics
        d3qn_avg = {
            'vehicles': np.mean([ep['completed_trips'] for ep in d3qn_results]),
            'passengers': np.mean([ep['passengers_completed'] for ep in d3qn_results]),
            'waiting': np.mean([ep['avg_waiting_time'] for ep in d3qn_results]),
            'speed': np.mean([ep['avg_speed'] for ep in d3qn_results]),
            'queue': np.mean([ep['avg_queue_length'] for ep in d3qn_results])
        }
        
        fixed_avg = {
            'vehicles': np.mean([ep['completed_trips'] for ep in fixed_time_results]),
            'passengers': np.mean([ep['passengers_completed'] for ep in fixed_time_results]),
            'waiting': np.mean([ep['avg_waiting_time'] for ep in fixed_time_results]),
            'speed': np.mean([ep['avg_speed'] for ep in fixed_time_results]),
            'queue': np.mean([ep['avg_queue_length'] for ep in fixed_time_results])
        }
        
        improvements = {
            'throughput': ((d3qn_avg['vehicles'] - fixed_avg['vehicles']) / fixed_avg['vehicles'] * 100),
            'passengers': ((d3qn_avg['passengers'] - fixed_avg['passengers']) / fixed_avg['passengers'] * 100),
            'waiting_time': -((d3qn_avg['waiting'] - fixed_avg['waiting']) / fixed_avg['waiting'] * 100),
            'speed': ((d3qn_avg['speed'] - fixed_avg['speed']) / fixed_avg['speed'] * 100),
            'queue_length': -((d3qn_avg['queue'] - fixed_avg['queue']) / fixed_avg['queue'] * 100)
        }
        
        stats = {
            'p_value': self.statistical_results.get('p_value', 0),
            'cohens_d': self.statistical_results.get('cohens_d', 0)
        }
        
        dashboard_logger.update_summary_statistics(
            d3qn_avg=d3qn_avg,
            fixed_avg=fixed_avg,
            improvements=improvements,
            stats=stats
        )
```

---

## STEP 3: START API SERVER

The API server provides endpoints for the frontend to fetch data.

### Install Dependencies

```bash
pip install flask flask-cors
```

### Start Server

```bash
python scripts/dashboard_api_server.py
```

This will start the API server on http://localhost:5000

### Available Endpoints

```
GET /api/experiments
GET /api/experiments/<name>/status
GET /api/experiments/<name>/episodes
GET /api/experiments/<name>/episodes/<number>
GET /api/experiments/<name>/progress
GET /api/experiments/<name>/vehicles
GET /api/experiments/<name>/summary
GET /api/experiments/<name>/evaluation
GET /api/experiments/<name>/latest
GET /health
```

---

## STEP 4: FRONTEND INTEGRATION

Update your frontend to fetch from the API:

### Example: Fetch Training Progress

```javascript
// Fetch training progress for charts
async function fetchTrainingProgress(experimentName) {
    const response = await fetch(`http://localhost:5000/api/experiments/${experimentName}/progress`);
    const data = await response.json();
    
    if (data.success) {
        const progress = data.progress;
        
        // Use in charts
        updateVehiclesChart(progress.episodes, progress.vehicles);
        updatePassengersChart(progress.episodes, progress.passengers);
        updateRewardsChart(progress.episodes, progress.rewards);
    }
}
```

### Example: Real-Time Latest Episode

```javascript
// Poll for latest episode every 5 seconds
setInterval(async () => {
    const response = await fetch(`http://localhost:5000/api/experiments/final_defense_training_350ep/latest`);
    const data = await response.json();
    
    if (data.success) {
        const latest = data.latest;
        
        // Update UI
        updateCurrentEpisode(latest.episode_number);
        updateCurrentMetrics({
            vehicles: latest.vehicles_completed,
            passengers: latest.passengers_completed,
            waiting: latest.avg_waiting_time,
            speed: latest.avg_speed
        });
    }
}, 5000);
```

### Example: Fetch Summary Statistics

```javascript
// Fetch summary for comparison view
async function fetchSummary(experimentName) {
    const response = await fetch(`http://localhost:5000/api/experiments/${experimentName}/summary`);
    const data = await response.json();
    
    if (data.success) {
        const summary = data.summary;
        
        // Update comparison table
        updateComparisonTable({
            d3qn: {
                vehicles: summary.d3qn_avg_vehicles,
                passengers: summary.d3qn_avg_passengers,
                waiting: summary.d3qn_avg_waiting,
                speed: summary.d3qn_avg_speed
            },
            fixedTime: {
                vehicles: summary.fixed_avg_vehicles,
                passengers: summary.fixed_avg_passengers,
                waiting: summary.fixed_avg_waiting,
                speed: summary.fixed_avg_speed
            },
            improvements: {
                throughput: summary.throughput_improvement,
                passengers: summary.passengers_improvement,
                waiting: summary.waiting_improvement,
                speed: summary.speed_improvement
            }
        });
    }
}
```

---

## DATABASE SCHEMA

The SQLite database has these tables:

### experiments
- id, name, created_at, updated_at, status, total_episodes, completed_episodes, config

### episodes
- id, experiment_name, episode_number, timestamp, phase, scenario_name
- duration_seconds, steps
- vehicles_completed, passengers_completed
- avg_waiting_time, avg_speed, avg_queue_length, max_queue_length
- total_reward, avg_loss, epsilon
- throughput_reward, waiting_penalty, speed_reward, queue_penalty, emergency_penalty

### vehicle_breakdown
- id, experiment_name, episode_number
- cars, jeepneys, buses, motorcycles, trucks, tricycles, other
- cars_passengers, jeepneys_passengers, buses_passengers, etc.

### evaluation_results
- id, experiment_name, agent_type, episode_number
- vehicles_completed, passengers_completed
- avg_waiting_time, avg_speed, avg_queue_length, max_queue_length, total_reward

### summary_statistics
- id, experiment_name, updated_at
- d3qn_avg_* (vehicles, passengers, waiting, speed, queue)
- fixed_avg_* (vehicles, passengers, waiting, speed, queue)
- *_improvement (throughput, passengers, waiting, speed, queue)
- p_value, cohens_d

---

## DATA THAT FRONTEND RECEIVES

### Per-Episode Data (Raw Values)

```json
{
    "episode_number": 25,
    "timestamp": "2025-10-13T10:30:00",
    "phase": "online",
    "scenario_name": "Day 20250812, Cycle 1",
    "duration_seconds": 300,
    "steps": 60,
    
    "vehicles_completed": 485,
    "passengers_completed": 728,
    
    "vehicle_breakdown": {
        "cars": 170,
        "jeepneys": 121,
        "buses": 39,
        "motorcycles": 97,
        "trucks": 34,
        "tricycles": 24
    },
    
    "passenger_breakdown": {
        "cars": 221,
        "jeepneys": 1694,
        "buses": 1365,
        "motorcycles": 136,
        "trucks": 51,
        "tricycles": 60
    },
    
    "avg_waiting_time": 7.33,
    "avg_speed": 14.9,
    "avg_queue_length": 92,
    "max_queue_length": 135,
    
    "total_reward": -209.19,
    "avg_loss": 0.0646,
    "epsilon": 0.01,
    
    "throughput_reward": 485.0,
    "waiting_penalty": -146.6,
    "speed_reward": 149.0,
    "queue_penalty": -92.0,
    "emergency_penalty": 0.0
}
```

### Training Progress (For Charts)

```json
{
    "episodes": [1, 2, 3, ..., 300],
    "vehicles": [485, 492, 478, ...],
    "passengers": [728, 738, 717, ...],
    "rewards": [-209, -195, -188, ...],
    "losses": [0.065, 0.062, 0.059, ...],
    "waiting_times": [7.33, 7.12, 6.95, ...],
    "speeds": [14.9, 15.1, 15.3, ...],
    "queue_lengths": [92, 89, 87, ...],
    "phases": ["offline", "offline", ..., "online", ...]
}
```

### Summary Statistics (For Comparison)

```json
{
    "d3qn_avg_vehicles": 483.2,
    "d3qn_avg_passengers": 725.8,
    "d3qn_avg_waiting": 7.33,
    "d3qn_avg_speed": 14.9,
    "d3qn_avg_queue": 92.0,
    
    "fixed_avg_vehicles": 423.6,
    "fixed_avg_passengers": 636.4,
    "fixed_avg_waiting": 12.45,
    "fixed_avg_speed": 11.2,
    "fixed_avg_queue": 135.0,
    
    "throughput_improvement": 14.1,
    "passengers_improvement": 14.1,
    "waiting_improvement": -41.1,
    "speed_improvement": 33.0,
    "queue_improvement": -31.9,
    
    "p_value": 0.0023,
    "cohens_d": 0.89
}
```

### Vehicle Breakdown (Aggregated)

```json
{
    "vehicles": {
        "cars": 51000,
        "jeepneys": 36300,
        "buses": 11700,
        "motorcycles": 29100,
        "trucks": 10200,
        "tricycles": 7300
    },
    "passengers": {
        "cars": 66300,
        "jeepneys": 508200,
        "buses": 409500,
        "motorcycles": 40740,
        "trucks": 15300,
        "tricycles": 18250
    },
    "public_transport": {
        "vehicles": 48000,
        "passengers": 917700,
        "percentage": 62.8
    }
}
```

---

## COMPLETE WORKFLOW

### During Training

1. Training script runs episode
2. After episode completes, calls `dashboard_logger.log_episode()`
3. Data is immediately written to SQLite database
4. Frontend polls API every 5 seconds
5. API reads from database and returns latest data
6. Frontend updates charts and displays in real-time

### After Evaluation

1. Evaluation script completes
2. Calls `dashboard_logger.log_evaluation_results()`
3. Calls `dashboard_logger.update_summary_statistics()`
4. Frontend fetches summary and displays comparison

---

## TESTING THE INTEGRATION

### Test 1: Start Training

```bash
python experiments/comprehensive_training.py --agent_type lstm --episodes 5 --experiment_name test_dashboard
```

### Test 2: Check Database

```bash
sqlite3 dashboard_data/training.db "SELECT * FROM episodes ORDER BY episode_number DESC LIMIT 5;"
```

### Test 3: Start API Server

```bash
python scripts/dashboard_api_server.py
```

### Test 4: Test API Endpoint

```bash
curl http://localhost:5000/api/experiments/test_dashboard/episodes
```

### Test 5: Frontend Fetch

```javascript
fetch('http://localhost:5000/api/experiments/test_dashboard/progress')
    .then(res => res.json())
    .then(data => console.log(data));
```

---

## SUMMARY

What we built:

1. **RealtimeDashboardLogger**: Writes to SQLite database during training
2. **Dashboard API Server**: Provides REST endpoints for frontend
3. **Database Schema**: Structured storage for all training/evaluation data
4. **Integration Points**: Clear hooks in training and evaluation scripts

What the frontend gets:

- Raw episode values (vehicles, passengers per episode)
- Vehicle type breakdown (cars, jeepneys, buses, etc.)
- Training progression (for charts)
- Summary statistics (D3QN vs Fixed-Time)
- Real-time updates (latest episode every 5 seconds)

All data is clean, structured, and ready for the frontend to visualize. No emojis, no formatting - just raw data that the FE can use.



