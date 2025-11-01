# Dashboard Metrics Optimization - Raw Values vs Hourly Rates

**Date**: October 11, 2025  
**Purpose**: Optimize metrics for dashboard visualization with raw episode values  
**Status**: ✅ **RECOMMENDATION - USE RAW VALUES FOR DASHBOARD**

---

## 🎯 **Current vs Recommended Metrics**

### **Current System (Hourly Rates)**
```python
# Current calculation
vehicle_throughput = completed_trips / simulation_duration_hours  # veh/h
passenger_throughput = passenger_count / simulation_duration_hours  # pass/h
```

**Problems for Dashboard**:
- ❌ **Hard to interpret**: "6,473 veh/h" doesn't show episode progress
- ❌ **Not episode-specific**: Hourly rate assumes 1-hour simulation
- ❌ **Inconsistent**: Different episodes have different durations
- ❌ **Dashboard unfriendly**: Hard to visualize trends

### **Recommended System (Raw Episode Values)**
```python
# Recommended for dashboard
vehicles_completed = completed_trips  # Raw count per episode
passengers_completed = passenger_count  # Raw count per episode
episode_duration = current_step * step_length  # Actual duration in seconds
```

**Benefits for Dashboard**:
- ✅ **Easy to interpret**: "485 vehicles completed in Episode 25"
- ✅ **Episode-specific**: Shows actual performance per episode
- ✅ **Consistent**: Same metric across all episodes
- ✅ **Dashboard friendly**: Easy to plot trends, compare episodes

---

## 📊 **Dashboard Metrics Structure**

### **Per-Episode Raw Metrics**

```python
episode_metrics = {
    # Episode Info
    'episode_number': 25,
    'scenario_name': 'Day 20250812, Cycle 1',
    'episode_duration_seconds': 300,  # Actual duration
    'episode_type': 'online',  # offline/online/validation
    
    # Raw Vehicle Metrics
    'vehicles_completed': 485,  # Raw count
    'vehicles_started': 355,    # Vehicles that entered
    'vehicles_active': 180,     # Vehicles still in network
    'completion_rate': 0.73,    # vehicles_completed / vehicles_started
    
    # Raw Passenger Metrics
    'passengers_completed': 728,  # Raw count (vehicles × capacity)
    'passengers_by_type': {
        'cars': 120,      # 120 cars × 1.3 passengers = 156 passengers
        'jeepneys': 45,   # 45 jeepneys × 14 passengers = 630 passengers
        'buses': 8,       # 8 buses × 35 passengers = 280 passengers
        'motorcycles': 12 # 12 motorcycles × 1.4 passengers = 17 passengers
    },
    
    # Performance Metrics
    'avg_waiting_time': 7.33,    # seconds
    'avg_speed': 14.9,           # km/h
    'avg_queue_length': 92,      # vehicles
    'max_queue_length': 135,     # vehicles
    
    # Training Metrics
    'total_reward': -209.19,
    'loss': 0.0646,
    'epsilon': 0.01,
    
    # Timestamps
    'start_time': '2025-10-11 10:30:00',
    'end_time': '2025-10-11 10:35:00',
    'duration_minutes': 5.0
}
```

### **Aggregated Dashboard Metrics**

```python
dashboard_summary = {
    # Current Episode
    'current_episode': 25,
    'current_vehicles': 485,
    'current_passengers': 728,
    
    # Running Totals
    'total_episodes': 300,
    'total_vehicles': 145000,      # Sum of all episodes
    'total_passengers': 218000,    # Sum of all episodes
    
    # Averages
    'avg_vehicles_per_episode': 483,
    'avg_passengers_per_episode': 727,
    'avg_episode_duration': 5.2,   # minutes
    
    # Trends (Last 10 episodes)
    'recent_vehicles': [485, 492, 478, 501, 489, 495, 487, 493, 481, 485],
    'recent_passengers': [728, 738, 717, 752, 734, 743, 731, 740, 722, 728],
    
    # Performance
    'best_episode': 23,  # Episode with most vehicles
    'best_vehicles': 501,
    'best_passengers': 752,
    
    'worst_episode': 7,  # Episode with least vehicles
    'worst_vehicles': 445,
    'worst_passengers': 668
}
```

---

## 🔧 **Implementation Changes Needed**

### **1. Modify Episode Logging**

```python
# In experiments/comprehensive_training.py
def log_episode_metrics(self, episode, scenario, metrics):
    """Log episode metrics with raw values for dashboard"""
    
    # Calculate raw metrics
    vehicles_completed = metrics.get('completed_trips', 0)
    passengers_completed = self.calculate_passenger_throughput(vehicles_completed)
    
    # Vehicle type breakdown
    vehicle_breakdown = self.get_vehicle_type_breakdown()
    
    episode_data = {
        'episode_number': episode,
        'scenario_name': scenario,
        'episode_duration_seconds': 300,  # Fixed duration
        
        # Raw counts
        'vehicles_completed': vehicles_completed,
        'passengers_completed': passengers_completed,
        
        # Vehicle type breakdown
        'vehicle_breakdown': vehicle_breakdown,
        
        # Performance metrics
        'avg_waiting_time': metrics.get('waiting_time', 0),
        'avg_speed': metrics.get('avg_speed', 0),
        'avg_queue_length': metrics.get('queue_length', 0),
        'max_queue_length': metrics.get('max_queue_length', 0),
        
        # Training metrics
        'total_reward': metrics.get('total_reward', 0),
        'loss': metrics.get('loss', 0),
        'epsilon': self.agent.epsilon,
        
        # Timestamps
        'timestamp': datetime.now().isoformat(),
        'duration_minutes': 5.0
    }
    
    return episode_data
```

### **2. Vehicle Type Breakdown Function**

```python
def get_vehicle_type_breakdown(self):
    """Get breakdown of vehicle types in current episode"""
    
    breakdown = {
        'cars': 0,
        'jeepneys': 0,
        'buses': 0,
        'motorcycles': 0,
        'trucks': 0,
        'tricycles': 0,
        'other': 0
    }
    
    # Count vehicles by type
    for veh_id in traci.simulation.getArrivedIDList():
        veh_type = traci.vehicle.getTypeID(veh_id)
        veh_id_lower = veh_id.lower()
        
        if 'car' in veh_id_lower:
            breakdown['cars'] += 1
        elif 'jeepney' in veh_id_lower or 'jeep' in veh_id_lower:
            breakdown['jeepneys'] += 1
        elif 'bus' in veh_id_lower:
            breakdown['buses'] += 1
        elif 'motor' in veh_id_lower:
            breakdown['motorcycles'] += 1
        elif 'truck' in veh_id_lower:
            breakdown['trucks'] += 1
        elif 'trike' in veh_id_lower:
            breakdown['tricycles'] += 1
        else:
            breakdown['other'] += 1
    
    return breakdown
```

### **3. Dashboard API Endpoints**

```python
# Dashboard API endpoints
@app.get("/api/episodes/{episode_id}")
async def get_episode_details(episode_id: int):
    """Get detailed episode metrics"""
    return {
        'episode_number': episode_id,
        'vehicles_completed': 485,
        'passengers_completed': 728,
        'vehicle_breakdown': {
            'cars': 120,
            'jeepneys': 45,
            'buses': 8,
            'motorcycles': 12
        },
        'performance_metrics': {...},
        'training_metrics': {...}
    }

@app.get("/api/dashboard/summary")
async def get_dashboard_summary():
    """Get dashboard summary with trends"""
    return {
        'current_episode': 25,
        'total_episodes': 300,
        'total_vehicles': 145000,
        'total_passengers': 218000,
        'recent_trends': [...],
        'performance_stats': {...}
    }

@app.get("/api/vehicles/breakdown")
async def get_vehicle_breakdown():
    """Get vehicle type breakdown across all episodes"""
    return {
        'total_breakdown': {
            'cars': 45000,
            'jeepneys': 35000,
            'buses': 8000,
            'motorcycles': 15000
        },
        'passenger_breakdown': {
            'cars': 58500,      # 45000 × 1.3
            'jeepneys': 490000, # 35000 × 14
            'buses': 280000,    # 8000 × 35
            'motorcycles': 21000 # 15000 × 1.4
        }
    }
```

---

## 📈 **Dashboard Visualization Benefits**

### **1. Episode Progress Charts**
```javascript
// Vehicles completed per episode
const vehicleChart = {
  x: [1, 2, 3, 4, 5, ...],  // Episode numbers
  y: [485, 492, 478, 501, 489, ...],  // Vehicles completed
  type: 'scatter',
  mode: 'lines+markers'
};

// Passengers completed per episode
const passengerChart = {
  x: [1, 2, 3, 4, 5, ...],  // Episode numbers
  y: [728, 738, 717, 752, 734, ...],  // Passengers completed
  type: 'scatter',
  mode: 'lines+markers'
};
```

### **2. Vehicle Type Distribution**
```javascript
// Pie chart of vehicle types
const vehicleTypeChart = {
  labels: ['Cars', 'Jeepneys', 'Buses', 'Motorcycles', 'Trucks'],
  values: [45000, 35000, 8000, 15000, 2000],
  type: 'pie'
};

// Bar chart of passenger contribution
const passengerContributionChart = {
  x: ['Cars', 'Jeepneys', 'Buses', 'Motorcycles'],
  y: [58500, 490000, 280000, 21000],  // Passenger counts
  type: 'bar'
};
```

### **3. Real-time Episode Monitoring**
```javascript
// Live episode progress
const currentEpisode = {
  episode: 25,
  vehicles: 485,
  passengers: 728,
  progress: 0.73,  // completion rate
  eta: '2 minutes remaining'
};
```

---

## 🎯 **Answer to Your Questions**

### **Q1: "Does throughput have to be an hour always?"**

**Answer**: **NO!** For dashboard purposes, raw episode values are much better.

**Current**: `6,473 veh/h` (confusing, assumes 1-hour simulation)  
**Better**: `485 vehicles completed in Episode 25` (clear, episode-specific)

### **Q2: "Can we get raw values per episode?"**

**Answer**: **YES!** This is actually the recommended approach.

**Raw Values Per Episode**:
- ✅ **Vehicles completed**: 485 (actual count)
- ✅ **Passengers completed**: 728 (actual count)
- ✅ **Episode duration**: 300 seconds (actual duration)
- ✅ **Vehicle breakdown**: Cars: 120, Jeepneys: 45, etc.

---

## 📝 **Implementation Plan**

### **Phase 1: Modify Logging (Today)**
1. Update episode logging to capture raw values
2. Add vehicle type breakdown
3. Include episode duration and timestamps

### **Phase 2: Dashboard API (When team is ready)**
1. Create API endpoints for raw episode data
2. Add aggregation functions for trends
3. Include vehicle type breakdown endpoints

### **Phase 3: Frontend Integration (When team is ready)**
1. Display raw episode values
2. Show vehicle type breakdown
3. Create trend visualizations

---

## ✅ **Benefits of Raw Values**

1. **Clearer interpretation**: "485 vehicles" vs "6,473 veh/h"
2. **Episode-specific**: Shows actual performance per episode
3. **Dashboard friendly**: Easy to plot and compare
4. **Consistent**: Same metric across all episodes
5. **Detailed breakdown**: Vehicle types, passenger counts
6. **Real-time monitoring**: Live episode progress

---

**Status**: ✅ **RECOMMENDED - USE RAW VALUES FOR DASHBOARD**  
**Implementation**: **Modify logging to capture raw episode values**  
**Benefit**: **Much clearer and more useful for dashboard visualization**




