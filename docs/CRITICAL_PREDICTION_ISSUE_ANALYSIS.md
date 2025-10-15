# CRITICAL: LSTM Prediction Issue Analysis & Fix

**Date**: October 15, 2025  
**Issue**: LSTM constantly predicts light traffic (accuracy=1.0 but TP=FP=TN=FN=0)  
**Root Cause**: Traffic classification based on wrong criteria  
**Status**: 🔴 **CRITICAL FIX REQUIRED**

---

## 🚨 **THE PROBLEM**

### **Observed Behavior**
```json
{
  "accuracy": 1.0,
  "precision": 0.0,
  "recall": 0.0,
  "f1_score": 0.0,
  "true_positives": 0,
  "false_positives": 0,
  "true_negatives": 0,
  "false_negatives": 0
}
```

**Analysis**:
- **All predictions**: Light traffic (0)
- **All actual labels**: Light traffic (0)
- **Accuracy 1.0**: Meaningless - always predicting same class
- **No learning**: LSTM cannot learn temporal patterns

### **Root Cause**

**Current Implementation** (WRONG):
```python
def is_heavy_traffic(self, traffic_metrics):
    heavy_conditions = [
        queue_length > 100,           # Never true
        waiting_time > 15,            # Never true
        vehicle_density > 0.8,        # Never true
        congestion_level > 0.7        # Never true
    ]
    return any(heavy_conditions)  # Always False
```

**Why It's Wrong**:
1. **Queue length**: In 300-second episodes, queues rarely exceed 100
2. **Waiting time**: Average waiting time is ~7-10s, never > 15s
3. **Vehicle density**: Not properly tracked
4. **Congestion level**: Not defined in metrics

**Result**: Every episode is classified as light traffic → No temporal pattern to learn

---

## ✅ **THE CORRECT SOLUTION**

### **User's Insight (CORRECT)**

**Traffic should be classified by WEEKDAY patterns**:
- **Heavy Traffic**: Monday, Tuesday, Friday (workdays with heavy commute)
- **Light Traffic**: Wednesday, Thursday, Saturday, Sunday (lighter traffic days)

**Bundle Format**: `bundle_YYYYMMDD_cycle_N.rou.xml`
- Example: `bundle_20250812_cycle_1.rou.xml` → Date: 2025-08-12 (Tuesday)

### **Correct Implementation**

```python
def is_heavy_traffic_from_date(self, date_string):
    """
    Determine if traffic is heavy based on day of week
    
    Heavy Traffic Days: Monday (0), Tuesday (1), Friday (4)
    Light Traffic Days: Wednesday (2), Thursday (3), Saturday (5), Sunday (6)
    
    Args:
        date_string: Date in format YYYYMMDD (e.g., "20250812")
        
    Returns:
        bool: True if heavy traffic day, False if light traffic day
    """
    from datetime import datetime
    
    # Parse date
    date = datetime.strptime(date_string, "%Y%m%d")
    
    # Get day of week (0=Monday, 6=Sunday)
    day_of_week = date.weekday()
    
    # Heavy traffic: Monday (0), Tuesday (1), Friday (4)
    heavy_traffic_days = [0, 1, 4]
    
    return day_of_week in heavy_traffic_days
```

---

## 🔧 **REQUIRED CHANGES**

### **Change 1: Update `is_heavy_traffic()` Method**

**Location**: `algorithms/d3qn_agent.py` (lines 156-183)

**Current** (WRONG):
```python
def is_heavy_traffic(self, traffic_metrics):
    if not traffic_metrics:
        return False
    
    queue_length = traffic_metrics.get('queue_length', 0)
    waiting_time = traffic_metrics.get('waiting_time', 0)
    vehicle_density = traffic_metrics.get('vehicle_density', 0)
    congestion_level = traffic_metrics.get('congestion_level', 0)
    
    heavy_conditions = [
        queue_length > 100,
        waiting_time > 15,
        vehicle_density > 0.8,
        congestion_level > 0.7
    ]
    
    return any(heavy_conditions)
```

**New** (CORRECT):
```python
def is_heavy_traffic(self, scenario_info):
    """
    Determine if traffic is heavy based on day of week from scenario date
    
    Heavy Traffic Days: Monday, Tuesday, Friday (peak commute days)
    Light Traffic Days: Wednesday, Thursday, Saturday, Sunday
    
    Args:
        scenario_info: Dictionary with 'day' key (date in YYYYMMDD format)
        
    Returns:
        bool: True if heavy traffic day, False if light traffic day
    """
    from datetime import datetime
    
    if not scenario_info or 'day' not in scenario_info:
        return False
    
    try:
        # Parse date from scenario (format: YYYYMMDD or integer like 20250812)
        date_str = str(scenario_info['day'])
        date = datetime.strptime(date_str, "%Y%m%d")
        
        # Get day of week (0=Monday, 6=Sunday)
        day_of_week = date.weekday()
        
        # Heavy traffic: Monday (0), Tuesday (1), Friday (4)
        heavy_traffic_days = [0, 1, 4]
        
        return day_of_week in heavy_traffic_days
    
    except (ValueError, KeyError):
        # If parsing fails, default to light traffic
        return False
```

### **Change 2: Update `remember()` Signature**

**Location**: `algorithms/d3qn_agent.py` (line 189)

**Current**:
```python
def remember(self, state, action, reward, next_state, done, traffic_metrics=None):
```

**New**:
```python
def remember(self, state, action, reward, next_state, done, scenario_info=None):
```

### **Change 3: Update Memory Tuple Structure**

**Location**: `algorithms/d3qn_agent.py` (line 211)

**Current**:
```python
self.memory.append((current_sequence, action, reward, next_sequence, done, traffic_metrics))
```

**New**:
```python
self.memory.append((current_sequence, action, reward, next_sequence, done, scenario_info))
```

### **Change 4: Update Training Methods**

**Location**: `algorithms/d3qn_agent.py` (lines 320-326)

**Current**:
```python
for state_sequence, action, reward, next_state_sequence, done, traffic_metrics in batch_sample:
    if traffic_metrics is not None:
        states.append(state_sequence)
        
        # Create traffic label
        is_heavy = self.is_heavy_traffic(traffic_metrics)
        traffic_labels.append(1 if is_heavy else 0)
```

**New**:
```python
for state_sequence, action, reward, next_state_sequence, done, scenario_info in batch_sample:
    if scenario_info is not None:
        states.append(state_sequence)
        
        # Create traffic label from scenario date
        is_heavy = self.is_heavy_traffic(scenario_info)
        traffic_labels.append(1 if is_heavy else 0)
```

### **Change 5: Update Training Script**

**Location**: `experiments/comprehensive_training.py` (lines 527-541)

**Current**:
```python
# Store experience with traffic metrics
traffic_metrics = {
    'queue_length': env.metrics.get('queue_length', 0),
    'waiting_time': env.metrics.get('waiting_time', 0),
    'vehicle_density': env.metrics.get('vehicle_density', 0),
    'congestion_level': env.metrics.get('congestion_level', 0)
}
agent.remember(state, action, reward, next_state, done, traffic_metrics)
```

**New**:
```python
# Store experience with scenario info (for date-based traffic classification)
# scenario_info is already available in this scope from episode setup
agent.remember(state, action, reward, next_state, done, scenario_info)
```

### **Change 6: Update Prediction Monitoring**

**Location**: `experiments/comprehensive_training.py` (lines 513-521)

**Current**:
```python
# Determine actual traffic label
traffic_metrics = {
    'queue_length': env.metrics.get('queue_length', 0),
    'waiting_time': env.metrics.get('waiting_time', 0),
    'vehicle_density': env.metrics.get('vehicle_density', 0),
    'congestion_level': env.metrics.get('congestion_level', 0)
}
is_heavy_traffic = agent.is_heavy_traffic(traffic_metrics)
episode_actual_labels.append(1 if is_heavy_traffic else 0)
```

**New**:
```python
# Determine actual traffic label from scenario date
is_heavy_traffic = agent.is_heavy_traffic(scenario_info)
episode_actual_labels.append(1 if is_heavy_traffic else 0)
```

---

## 📊 **EXPECTED RESULTS AFTER FIX**

### **Before Fix**
```json
{
  "accuracy": 1.0,
  "precision": 0.0,
  "recall": 0.0,
  "true_positives": 0,
  "false_positives": 0,
  "true_negatives": 0,
  "false_negatives": 0
}
```

### **After Fix** (Expected)
```json
{
  "accuracy": 0.60-0.75,
  "precision": 0.65-0.80,
  "recall": 0.55-0.75,
  "true_positives": 30-50,
  "false_positives": 10-20,
  "true_negatives": 25-40,
  "false_negatives": 5-15
}
```

**Explanation**:
- **Mixed predictions**: Some heavy, some light (realistic)
- **Learning visible**: LSTM learns weekday patterns
- **Accuracy 60-75%**: Realistic for limited data
- **Confusion matrix filled**: Actual learning happening

---

## 🗑️ **CLEANUP OLD RESULTS**

### **Directories to Remove**

```bash
# Remove old/incomplete training results
comprehensive_results/lstm_progressive_test_50ep/
comprehensive_results/final_defense_training_350ep/  # Keep this - it's your best +14% result!

# Remove non-LSTM comparison results (if not needed)
non_lstm_rebalanced_rewards/
lstm_rebalanced_rewards/
```

### **Keep These**
- `comprehensive_results/final_defense_training_350ep/` - **KEEP** (your +14% throughput result)
- `comprehensive_results/comprehensive_training/` - **KEEP** (current test run)

---

## ✅ **IMPLEMENTATION PRIORITY**

1. **CRITICAL**: Fix `is_heavy_traffic()` method (5 minutes)
2. **CRITICAL**: Update training script to pass `scenario_info` (5 minutes)
3. **CRITICAL**: Update training methods to use `scenario_info` (5 minutes)
4. **RECOMMENDED**: Cleanup old results (2 minutes)
5. **TEST**: Run 10-episode test to verify fix (50 minutes)

---

## 🎯 **BOTTOM LINE**

**Current State**: 
- ❌ LSTM predicting constant value (no learning)
- ❌ Wrong classification criteria (metrics-based)
- ❌ Accuracy 1.0 but meaningless

**After Fix**:
- ✅ LSTM learns weekday patterns (temporal)
- ✅ Correct classification (date-based)
- ✅ Realistic accuracy (60-75%)
- ✅ Academic defensibility (concrete learning)

**User's insight was correct** - dates should determine traffic patterns, not instantaneous metrics!
