# Training Stabilization Action Plan

**Generated:** October 10, 2025  
**Status:** CRITICAL - Immediate action required  
**Goal:** Stabilize training while maintaining throughput improvement  

---

## The Problem

**DISCOVERY:** Comparing 50-episode (aggressive) vs 100-episode (conservative) training:

| Metric | Conservative (100ep) | Aggressive (50ep) | Change |
|--------|---------------------|-------------------|---------|
| **Loss Trend** | +15% increase | **+209% increase** | **14× worse** ❌ |
| **Throughput** | -32% degradation | **+6.3% improvement** | **+38pp better** ✅ |
| **Avg Reward** | -274.20 | -399.73 | -46% worse ❌ |

**THE PARADOX:** We achieved throughput goal BUT destroyed training stability!

---

## Root Cause: Aggressive Reward Rebalancing

**What Changed:**
```python
# Conservative (100-episode):
throughput_focus = 57% (45% throughput + 12% bonus)
waiting_weight = 28%
→ Result: Stable loss (+15%), bad throughput (-32%)

# Aggressive (50-episode):
throughput_focus = 75% (55% throughput + 20% bonus)
waiting_weight = 15%
→ Result: Unstable loss (+209%), GOOD throughput (+6.3%)
```

**Why Loss Exploded:**
1. **Reward Volatility:** 75% weight on volatile throughput metric
2. **Lost Balance:** Reduced stable waiting time weight by 50%
3. **Q-Value Divergence:** Network can't predict highly variable rewards
4. **Compounding Effect:** Instability worsens over episodes

---

## Solution: Moderate Rebalancing

### Proposed Reward Weights

**"Goldilocks" Approach - Not too conservative, not too aggressive:**

```python
reward = (
    waiting_reward * 0.22 +      # 22% (was 28% conservative, 15% aggressive)
    throughput_reward * 0.50 +   # 50% (was 45% conservative, 55% aggressive)
    speed_reward * 0.12 +        # 12% (was 15% conservative, 10% aggressive)
    queue_reward * 0.08 +        # 8% (was 10% conservative, 5% aggressive)
    pressure_term * 0.05 +       # 5% (maintained across both)
    throughput_bonus * 0.15      # 15% (was 12% conservative, 20% aggressive)
)
# Total throughput focus: 65% (50% + 15%)
# Between conservative 57% and aggressive 75%
```

**Expected Outcome:**
- **Throughput:** 0% to -15% degradation (vs baseline)
  - Better than conservative -32%
  - May not reach aggressive +6.3%, but acceptable
- **Loss:** +50-100% increase over training
  - Better than aggressive +209%
  - Worse than conservative +15%, but acceptable
- **Balance:** Maintains multi-objective learning

---

## Additional Stabilization Measures

### 1. Learning Rate Reduction
```python
# Current: 0.0005
# Proposed: 0.0003 (-40%)
# Why: Slower Q-value updates = more stable convergence
```

### 2. Tighter Gradient Clipping
```python
# Current: clipnorm=5.0
# Proposed: clipnorm=1.0
# Why: Prevents extreme gradient spikes
```

### 3. More Frequent Target Updates
```python
# Current: Every 20 episodes
# Proposed: Every 10 episodes
# Why: Faster Q-value stabilization
```

### 4. Smaller Huber Loss Delta
```python
# Current: delta=1.0
# Proposed: delta=0.5
# Why: Less sensitivity to reward outliers
```

---

## Implementation Steps

### Step 1: Update Reward Function (10 minutes)

<function_calls>
<invoke name="search_replace">
<parameter name="file_path">core/traffic_env.py




