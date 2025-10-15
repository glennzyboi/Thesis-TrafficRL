# Dashboard Real-Time Logging - Quick Start Guide

**Date**: October 13, 2025  
**Purpose**: Quick reference for setting up real-time dashboard logging  
**Time to Setup**: 10 minutes

---

## WHAT THIS DOES

During training, data is automatically logged to a SQLite database. Your frontend fetches from an API server that reads this database, providing real-time updates of training progress.

```
Training → Database → API Server → Frontend Dashboard
```

---

## QUICK SETUP (3 Steps)

### Step 1: Install Dependencies (1 minute)

```bash
pip install flask flask-cors
```

### Step 2: Add Logger to Training Script (5 minutes)

Edit `experiments/comprehensive_training.py`:

**Add import:**
```python
from utils.realtime_dashboard_logger import RealtimeDashboardLogger
```

**Initialize in __init__:**
```python
def __init__(self, experiment_name: str = None):
    # ... existing code ...
    
    # Add this
    self.dashboard_logger = RealtimeDashboardLogger(
        experiment_name=self.experiment_name,
        db_path='dashboard_data/training.db'
    )
```

**Log after each episode (find where episode completes):**
```python
# After episode completes, add this:
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
    'duration_seconds': 300
}

self.dashboard_logger.log_episode(
    episode_number=episode,
    metrics=episode_metrics,
    phase=phase  # 'offline' or 'online'
)
```

**Mark complete at end of training:**
```python
# At end of run() method:
self.dashboard_logger.mark_experiment_complete(total_episodes=episodes)
```

### Step 3: Start API Server (1 minute)

```bash
python scripts/dashboard_api_server.py
```

Server runs on http://localhost:5000

---

## FRONTEND INTEGRATION

### Fetch Training Progress

```javascript
// Get all episodes for charts
fetch('http://localhost:5000/api/experiments/final_defense_training_350ep/progress')
    .then(res => res.json())
    .then(data => {
        const { episodes, vehicles, passengers, rewards } = data.progress;
        // Update charts with this data
    });
```

### Fetch Latest Episode (Real-Time)

```javascript
// Poll every 5 seconds for latest
setInterval(() => {
    fetch('http://localhost:5000/api/experiments/final_defense_training_350ep/latest')
        .then(res => res.json())
        .then(data => {
            const latest = data.latest;
            // Update current episode display
            updateDisplay({
                episode: latest.episode_number,
                vehicles: latest.vehicles_completed,
                passengers: latest.passengers_completed,
                waiting: latest.avg_waiting_time,
                speed: latest.avg_speed
            });
        });
}, 5000);
```

### Fetch Summary Statistics

```javascript
// Get D3QN vs Fixed-Time comparison
fetch('http://localhost:5000/api/experiments/final_defense_training_350ep/summary')
    .then(res => res.json())
    .then(data => {
        const summary = data.summary;
        // Update comparison table
        updateComparison({
            d3qn_vehicles: summary.d3qn_avg_vehicles,
            fixed_vehicles: summary.fixed_avg_vehicles,
            improvement: summary.throughput_improvement
        });
    });
```

### Fetch Vehicle Breakdown

```javascript
// Get vehicle and passenger breakdown
fetch('http://localhost:5000/api/experiments/final_defense_training_350ep/vehicles')
    .then(res => res.json())
    .then(data => {
        const breakdown = data.breakdown;
        // Update pie charts
        updateVehiclePieChart(breakdown.vehicles);
        updatePassengerPieChart(breakdown.passengers);
        
        // Show public transport percentage
        showPublicTransportStat(breakdown.public_transport.percentage);
    });
```

---

## API ENDPOINTS

All endpoints return JSON with `{success: true/false, ...data}`

### Experiments
- `GET /api/experiments` - List all experiments
- `GET /api/experiments/<name>/status` - Experiment status

### Episodes
- `GET /api/experiments/<name>/episodes` - All episodes
- `GET /api/experiments/<name>/episodes/<number>` - Specific episode
- `GET /api/experiments/<name>/latest` - Latest episode (for real-time)

### Charts & Stats
- `GET /api/experiments/<name>/progress` - Training progress data for charts
- `GET /api/experiments/<name>/vehicles` - Vehicle/passenger breakdown
- `GET /api/experiments/<name>/summary` - Summary statistics
- `GET /api/experiments/<name>/evaluation` - Evaluation results

### Health
- `GET /health` - Check if server is running

---

## DATA FORMAT

### Episode Data (What Frontend Receives)

```json
{
    "episode_number": 25,
    "timestamp": "2025-10-13T10:30:00",
    "phase": "online",
    "vehicles_completed": 485,
    "passengers_completed": 728,
    "avg_waiting_time": 7.33,
    "avg_speed": 14.9,
    "avg_queue_length": 92,
    "max_queue_length": 135,
    "total_reward": -209.19,
    "avg_loss": 0.0646,
    "epsilon": 0.01
}
```

### Progress Data (For Charts)

```json
{
    "episodes": [1, 2, 3, ..., 300],
    "vehicles": [485, 492, 478, ...],
    "passengers": [728, 738, 717, ...],
    "rewards": [-209, -195, -188, ...],
    "waiting_times": [7.33, 7.12, ...],
    "speeds": [14.9, 15.1, ...]
}
```

### Summary (For Comparison)

```json
{
    "d3qn_avg_vehicles": 483.2,
    "fixed_avg_vehicles": 423.6,
    "throughput_improvement": 14.1,
    "p_value": 0.0023
}
```

---

## TESTING

### Test 1: Run Short Training

```bash
python experiments/comprehensive_training.py \
    --agent_type lstm \
    --episodes 5 \
    --experiment_name test_dashboard
```

### Test 2: Check Database

```bash
sqlite3 dashboard_data/training.db "SELECT episode_number, vehicles_completed FROM episodes;"
```

### Test 3: Test API

```bash
curl http://localhost:5000/api/experiments/test_dashboard/episodes
```

---

## WHAT THE FRONTEND SHOULD DO

### On Dashboard Load
1. Fetch experiment list: `/api/experiments`
2. Select experiment to display
3. Fetch progress data: `/api/experiments/<name>/progress`
4. Render charts with vehicles, passengers, rewards over episodes

### During Training (Real-Time)
1. Poll `/api/experiments/<name>/latest` every 5 seconds
2. Update "Current Episode" section
3. Update progress charts with new data point

### After Training Complete
1. Fetch summary: `/api/experiments/<name>/summary`
2. Display D3QN vs Fixed-Time comparison
3. Show improvement percentages
4. Display statistical significance

### Additional Views
1. Fetch vehicles breakdown: `/api/experiments/<name>/vehicles`
2. Display pie charts
3. Highlight public transport percentage

---

## TROUBLESHOOTING

### Database not found
- Check that training created `dashboard_data/training.db`
- Logger creates it automatically on first run

### API server not responding
- Make sure it's running: `python scripts/dashboard_api_server.py`
- Check port 5000 is not in use

### No data in database
- Verify logger is integrated in training script
- Check that `log_episode()` is called after each episode
- Look for errors in training logs

### CORS errors
- API server has CORS enabled by default
- If issues persist, update CORS config in `dashboard_api_server.py`

---

## PRODUCTION DEPLOYMENT

### For Production API Server

Use a production WSGI server instead of Flask dev server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 scripts.dashboard_api_server:app
```

### Database Location

For production, use absolute path:

```python
db_path='/var/data/training.db'
```

### API URL in Frontend

Update frontend to point to production server:

```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

fetch(`${API_URL}/api/experiments/${name}/progress`)
```

---

## SUMMARY

You now have:

1. **Real-time logger** that writes to SQLite during training
2. **API server** that provides REST endpoints for frontend
3. **Clean data format** - raw values, no formatting, ready for FE to use
4. **Vehicle breakdown** - cars, jeepneys, buses with passenger counts
5. **Live updates** - frontend can poll for latest episode
6. **Summary stats** - D3QN vs Fixed-Time comparison after evaluation

The frontend handles all visualization and formatting. The backend just provides clean, structured data.



