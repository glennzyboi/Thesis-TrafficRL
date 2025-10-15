# API Response Examples for Frontend Integration

**Date**: October 13, 2025  
**Purpose**: Exact JSON responses frontend will receive from API

---

## BASE URL

```
http://localhost:5000
```

For production, update to your deployed API URL.

---

## 1. LIST ALL EXPERIMENTS

**Endpoint**: `GET /api/experiments`

**Response**:
```json
{
    "success": true,
    "experiments": [
        {
            "name": "final_defense_training_350ep",
            "created_at": "2025-10-11 10:00:00",
            "updated_at": "2025-10-11 20:30:00",
            "status": "completed",
            "total_episodes": 300,
            "completed_episodes": 300
        },
        {
            "name": "test_dashboard",
            "created_at": "2025-10-13 14:20:00",
            "updated_at": "2025-10-13 14:45:00",
            "status": "running",
            "total_episodes": 350,
            "completed_episodes": 45
        }
    ]
}
```

---

## 2. EXPERIMENT STATUS

**Endpoint**: `GET /api/experiments/final_defense_training_350ep/status`

**Response**:
```json
{
    "success": true,
    "status": {
        "name": "final_defense_training_350ep",
        "created_at": "2025-10-11 10:00:00",
        "updated_at": "2025-10-11 20:30:00",
        "status": "completed",
        "total_episodes": 300,
        "completed_episodes": 300
    }
}
```

---

## 3. ALL EPISODES

**Endpoint**: `GET /api/experiments/final_defense_training_350ep/episodes`

**Response**:
```json
{
    "success": true,
    "total": 300,
    "episodes": [
        {
            "id": 1,
            "experiment_name": "final_defense_training_350ep",
            "episode_number": 1,
            "timestamp": "2025-10-11 10:05:23",
            "phase": "offline",
            "scenario_name": "Day 20250812, Cycle 1",
            "duration_seconds": 300.0,
            "steps": 60,
            "vehicles_completed": 485,
            "passengers_completed": 728,
            "avg_waiting_time": 7.33,
            "avg_speed": 14.9,
            "avg_queue_length": 92.0,
            "max_queue_length": 135,
            "total_reward": -209.19,
            "avg_loss": 0.0646,
            "epsilon": 0.01,
            "throughput_reward": 485.0,
            "waiting_penalty": -146.6,
            "speed_reward": 149.0,
            "queue_penalty": -92.0,
            "emergency_penalty": 0.0
        },
        {
            "id": 2,
            "episode_number": 2,
            "vehicles_completed": 492,
            "passengers_completed": 738,
            ...
        }
        // ... 298 more episodes
    ]
}
```

---

## 4. SPECIFIC EPISODE DETAIL

**Endpoint**: `GET /api/experiments/final_defense_training_350ep/episodes/25`

**Response**:
```json
{
    "success": true,
    "episode": {
        "episode_number": 25,
        "timestamp": "2025-10-11 12:15:45",
        "phase": "offline",
        "scenario_name": "Day 20250814, Cycle 2",
        "duration_seconds": 300.0,
        "steps": 60,
        "vehicles_completed": 501,
        "passengers_completed": 752,
        "avg_waiting_time": 6.95,
        "avg_speed": 15.3,
        "avg_queue_length": 87.0,
        "max_queue_length": 128,
        "total_reward": -188.45,
        "avg_loss": 0.0592,
        "epsilon": 0.008,
        "vehicle_breakdown": {
            "cars": 175,
            "jeepneys": 125,
            "buses": 40,
            "motorcycles": 100,
            "trucks": 35,
            "tricycles": 26
        },
        "passenger_breakdown": {
            "cars": 228,
            "jeepneys": 1750,
            "buses": 1400,
            "motorcycles": 140,
            "trucks": 53,
            "tricycles": 65
        }
    }
}
```

---

## 5. TRAINING PROGRESS (For Charts)

**Endpoint**: `GET /api/experiments/final_defense_training_350ep/progress`

**Response**:
```json
{
    "success": true,
    "progress": {
        "episodes": [1, 2, 3, 4, 5, ..., 300],
        "vehicles": [485, 492, 478, 501, 489, ..., 495],
        "passengers": [728, 738, 717, 752, 734, ..., 743],
        "rewards": [-209.19, -195.23, -188.45, -175.67, ..., -165.34],
        "losses": [0.0646, 0.0612, 0.0592, 0.0578, ..., 0.0445],
        "waiting_times": [7.33, 7.12, 6.95, 6.78, ..., 6.21],
        "speeds": [14.9, 15.1, 15.3, 15.5, ..., 16.2],
        "queue_lengths": [92, 89, 87, 84, ..., 78],
        "phases": ["offline", "offline", ..., "online", "online"]
    }
}
```

**Frontend Use**:
```javascript
const { episodes, vehicles, passengers } = data.progress;

// Plot vehicles over episodes
updateVehiclesChart(episodes, vehicles);

// Plot passengers over episodes
updatePassengersChart(episodes, passengers);

// Plot rewards over episodes
updateRewardsChart(episodes, rewards);
```

---

## 6. VEHICLE BREAKDOWN (Aggregated)

**Endpoint**: `GET /api/experiments/final_defense_training_350ep/vehicles`

**Response**:
```json
{
    "success": true,
    "breakdown": {
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
        "vehicle_percentages": {
            "cars": 34.9,
            "jeepneys": 24.8,
            "buses": 8.0,
            "motorcycles": 19.9,
            "trucks": 7.0,
            "tricycles": 5.0
        },
        "passenger_percentages": {
            "cars": 6.2,
            "jeepneys": 47.7,
            "buses": 38.4,
            "motorcycles": 3.8,
            "trucks": 1.4,
            "tricycles": 1.7
        },
        "totals": {
            "vehicles": 145600,
            "passengers": 1058290
        },
        "public_transport": {
            "vehicles": 48000,
            "passengers": 917700,
            "percentage": 86.7
        }
    }
}
```

**Frontend Use**:
```javascript
const { vehicles, passengers, public_transport } = data.breakdown;

// Pie chart of vehicles
renderPieChart('vehicles', vehicles);

// Pie chart of passengers (shows dominance of jeepneys/buses)
renderPieChart('passengers', passengers);

// Highlight public transport
showStat('Public Transport Passengers', `${public_transport.percentage}%`);
```

---

## 7. SUMMARY STATISTICS (D3QN vs Fixed-Time)

**Endpoint**: `GET /api/experiments/final_defense_training_350ep/summary`

**Response**:
```json
{
    "success": true,
    "summary": {
        "id": 1,
        "experiment_name": "final_defense_training_350ep",
        "updated_at": "2025-10-11 21:00:00",
        
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
}
```

**Frontend Use**:
```javascript
const summary = data.summary;

// Comparison table
renderComparisonTable({
    metric: 'Vehicle Throughput',
    d3qn: summary.d3qn_avg_vehicles,
    fixedTime: summary.fixed_avg_vehicles,
    improvement: summary.throughput_improvement,
    unit: 'vehicles/episode'
});

// Show improvement badges
if (summary.throughput_improvement > 10) {
    showBadge('Excellent', 'green');
} else if (summary.throughput_improvement > 5) {
    showBadge('Good', 'yellow');
}

// Statistical significance
if (summary.p_value < 0.05) {
    showSignificance('Statistically Significant', summary.p_value);
}
```

---

## 8. EVALUATION RESULTS

**Endpoint**: `GET /api/experiments/final_defense_training_350ep/evaluation`

**Response**:
```json
{
    "success": true,
    "d3qn": [
        {
            "id": 1,
            "experiment_name": "final_defense_training_350ep",
            "agent_type": "d3qn",
            "episode_number": 1,
            "vehicles_completed": 485,
            "passengers_completed": 728,
            "avg_waiting_time": 7.33,
            "avg_speed": 14.9,
            "avg_queue_length": 92.0,
            "max_queue_length": 135,
            "total_reward": -209.19
        },
        // ... 24 more D3QN evaluation episodes
    ],
    "fixed_time": [
        {
            "id": 26,
            "agent_type": "fixed_time",
            "episode_number": 1,
            "vehicles_completed": 425,
            "passengers_completed": 638,
            "avg_waiting_time": 12.45,
            "avg_speed": 11.2,
            "avg_queue_length": 135.0,
            "max_queue_length": 180,
            "total_reward": 0.0
        },
        // ... 24 more Fixed-Time evaluation episodes
    ]
}
```

---

## 9. LATEST EPISODE (Real-Time)

**Endpoint**: `GET /api/experiments/final_defense_training_350ep/latest`

**Response**:
```json
{
    "success": true,
    "latest": {
        "episode_number": 45,
        "timestamp": "2025-10-13 14:45:12",
        "phase": "online",
        "scenario_name": "Day 20250820, Cycle 3",
        "vehicles_completed": 489,
        "passengers_completed": 734,
        "avg_waiting_time": 6.78,
        "avg_speed": 15.5,
        "avg_queue_length": 84.0,
        "max_queue_length": 125,
        "total_reward": -175.67,
        "avg_loss": 0.0578,
        "epsilon": 0.005,
        "vehicle_breakdown": {
            "cars": 171,
            "jeepneys": 122,
            "buses": 39,
            "motorcycles": 98,
            "trucks": 34,
            "tricycles": 25
        }
    }
}
```

**Frontend Use (Poll Every 5 Seconds)**:
```javascript
setInterval(async () => {
    const response = await fetch(`${API_URL}/api/experiments/final_defense_training_350ep/latest`);
    const data = await response.json();
    
    if (data.success) {
        const latest = data.latest;
        
        // Update current episode display
        updateElement('#current-episode', latest.episode_number);
        updateElement('#current-vehicles', latest.vehicles_completed);
        updateElement('#current-passengers', latest.passengers_completed);
        updateElement('#current-reward', latest.total_reward.toFixed(2));
        
        // Update progress bar
        const progress = (latest.episode_number / 350) * 100;
        updateProgressBar(progress);
    }
}, 5000);
```

---

## 10. HEALTH CHECK

**Endpoint**: `GET /health`

**Response**:
```json
{
    "success": true,
    "status": "healthy",
    "database": true
}
```

---

## ERROR RESPONSES

### Experiment Not Found

**Response** (404):
```json
{
    "success": false,
    "error": "Experiment not found"
}
```

### Episode Not Found

**Response** (404):
```json
{
    "success": false,
    "error": "Episode not found"
}
```

### Server Error

**Response** (500):
```json
{
    "success": false,
    "error": "Database connection failed"
}
```

---

## FRONTEND INTEGRATION EXAMPLE

### Complete Dashboard Component

```javascript
class TrainingDashboard {
    constructor(experimentName) {
        this.experimentName = experimentName;
        this.apiUrl = 'http://localhost:5000';
    }
    
    async init() {
        // Load initial data
        await this.loadProgress();
        await this.loadVehicleBreakdown();
        await this.loadSummary();
        
        // Start real-time updates
        this.startRealTimeUpdates();
    }
    
    async loadProgress() {
        const response = await fetch(`${this.apiUrl}/api/experiments/${this.experimentName}/progress`);
        const data = await response.json();
        
        if (data.success) {
            this.renderProgressCharts(data.progress);
        }
    }
    
    async loadVehicleBreakdown() {
        const response = await fetch(`${this.apiUrl}/api/experiments/${this.experimentName}/vehicles`);
        const data = await response.json();
        
        if (data.success) {
            this.renderVehicleBreakdown(data.breakdown);
        }
    }
    
    async loadSummary() {
        const response = await fetch(`${this.apiUrl}/api/experiments/${this.experimentName}/summary`);
        const data = await response.json();
        
        if (data.success) {
            this.renderSummary(data.summary);
        }
    }
    
    startRealTimeUpdates() {
        setInterval(async () => {
            const response = await fetch(`${this.apiUrl}/api/experiments/${this.experimentName}/latest`);
            const data = await response.json();
            
            if (data.success) {
                this.updateCurrentEpisode(data.latest);
            }
        }, 5000);
    }
    
    renderProgressCharts(progress) {
        // Plot vehicles over episodes
        new Chart('vehicles-chart', {
            type: 'line',
            data: {
                labels: progress.episodes,
                datasets: [{
                    label: 'Vehicles Completed',
                    data: progress.vehicles
                }]
            }
        });
        
        // Plot passengers over episodes
        new Chart('passengers-chart', {
            type: 'line',
            data: {
                labels: progress.episodes,
                datasets: [{
                    label: 'Passengers Completed',
                    data: progress.passengers
                }]
            }
        });
    }
    
    renderVehicleBreakdown(breakdown) {
        // Vehicles pie chart
        new Chart('vehicles-pie', {
            type: 'pie',
            data: {
                labels: Object.keys(breakdown.vehicles),
                datasets: [{
                    data: Object.values(breakdown.vehicles)
                }]
            }
        });
        
        // Passengers pie chart (highlights public transport)
        new Chart('passengers-pie', {
            type: 'pie',
            data: {
                labels: Object.keys(breakdown.passengers),
                datasets: [{
                    data: Object.values(breakdown.passengers)
                }]
            }
        });
        
        // Show public transport stat
        document.getElementById('public-transport-percentage').textContent = 
            `${breakdown.public_transport.percentage.toFixed(1)}%`;
    }
    
    renderSummary(summary) {
        // Create comparison table
        const table = `
            <tr>
                <td>Vehicle Throughput</td>
                <td>${summary.d3qn_avg_vehicles.toFixed(1)}</td>
                <td>${summary.fixed_avg_vehicles.toFixed(1)}</td>
                <td class="${summary.throughput_improvement > 0 ? 'positive' : 'negative'}">
                    ${summary.throughput_improvement > 0 ? '+' : ''}${summary.throughput_improvement.toFixed(1)}%
                </td>
            </tr>
            <tr>
                <td>Passenger Throughput</td>
                <td>${summary.d3qn_avg_passengers.toFixed(1)}</td>
                <td>${summary.fixed_avg_passengers.toFixed(1)}</td>
                <td class="positive">
                    +${summary.passengers_improvement.toFixed(1)}%
                </td>
            </tr>
        `;
        
        document.getElementById('comparison-table').innerHTML = table;
        
        // Show statistical significance
        if (summary.p_value < 0.05) {
            document.getElementById('significance-badge').innerHTML = 
                `<span class="badge-success">Statistically Significant (p=${summary.p_value.toFixed(4)})</span>`;
        }
    }
    
    updateCurrentEpisode(latest) {
        document.getElementById('current-episode').textContent = latest.episode_number;
        document.getElementById('current-vehicles').textContent = latest.vehicles_completed;
        document.getElementById('current-passengers').textContent = latest.passengers_completed;
        document.getElementById('current-reward').textContent = latest.total_reward.toFixed(2);
    }
}

// Initialize dashboard
const dashboard = new TrainingDashboard('final_defense_training_350ep');
dashboard.init();
```

---

## SUMMARY

All API responses are clean JSON with:
- Raw values (vehicles/passengers as counts, NOT hourly rates)
- No formatting (no emojis, no text decorations)
- Structured data ready for frontend visualization
- Real-time updates via polling

Frontend handles all display logic, colors, icons, and formatting.



