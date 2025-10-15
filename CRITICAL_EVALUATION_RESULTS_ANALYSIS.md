# 🚨 CRITICAL: Evaluation Results Analysis

**Date:** October 10, 2025  
**Status:** ⚠️ **MAJOR DISCREPANCY DISCOVERED**  
**Issue:** D3QN agent underperforming in evaluation despite training success  

---

## Executive Summary

**PROBLEM:** The D3QN agent that showed +5.9% throughput improvement during training is showing **-31.6% degradation** in evaluation!

### Training Results (188 episodes):
- Average throughput: **5,834 veh/h**
- vs Fixed-Time baseline (5,507 veh/h): **+5.9% improvement** ✅

### Evaluation Results (25 episodes):
- D3QN throughput: **3,886 veh/h** 
- Fixed-Time throughput: **5,677 veh/h**
- Performance: **-31.6% degradation** ❌

**Discrepancy:** 1,948 veh/h difference between training and evaluation!

---

## Detailed Analysis

### 1. Throughput Comparison

| Metric | Training (188ep) | Evaluation (25ep) | Difference |
|--------|------------------|-------------------|------------|
| **D3QN Throughput** | 5,834 veh/h | 3,886 veh/h | **-1,948 veh/h** |
| **Completed Trips** | 486.2 trips/ep | ~290 trips/ep | **-196 trips/ep** |
| **vs Baseline** | +5.9% | -31.6% | **-37.5 percentage points** |

### 2. Other Metrics (from Evaluation)

**What D3QN IMPROVED:**
- ✅ Waiting Time: -34.4% (10.84s → 7.11s) - **EXCELLENT**
- ✅ Speed: +7.6% (14.46 → 15.56 km/h) - **GOOD**
- ✅ Completed Trips: +14.9% (426 → 489) - **WAIT, WHAT?**
- ✅ Max Queue: -19.2% (163 → 132) - **GOOD**

**What D3QN DEGRADED:**
- ❌ Throughput: -31.6% (5,677 → 3,886 veh/h) - **TERRIBLE**

### 3. The Paradox

**CRITICAL OBSERVATION:**
- Completed trips: **+14.9% improvement** (426 → 489)
- Throughput: **-31.6% degradation** (5,677 → 3,886)

**This doesn't make sense!** Throughput = Completed Trips / Time

Let me recalculate:
- Fixed-Time: 426 trips / 0.075 hours = 5,680 veh/h ✅
- D3QN: 489 trips / 0.075 hours = 6,520 veh/h ✅✅✅

**WAIT!** If D3QN completed MORE trips (489 vs 426), throughput should be HIGHER, not LOWER!

---

## Root Cause Investigation

### Hypothesis 1: Metric Calculation Error ⭐ **MOST LIKELY**

The evaluation script might be calculating throughput incorrectly for D3QN.

**Evidence:**
- Completed trips show +14.9% improvement
- But throughput shows -31.6% degradation
- **Mathematically impossible** unless time duration is different

**Check needed:**
1. Does D3QN simulation run for full 300 seconds?
2. Is throughput calculated the same way for both?
3. Is there a bug in the D3QN metrics collection?

### Hypothesis 2: Different Scenario Sets

Training used 46 training scenarios repeatedly, evaluation used 25 test scenarios.

**Evidence:**
- Fixed-Time evaluation: 5,677 veh/h
- Fixed-Time baseline (from training data): 5,507 veh/h
- **Difference: +170 veh/h** (test scenarios might be easier)

**BUT:** This doesn't explain why D3QN gets 3,886 when completing MORE trips!

### Hypothesis 3: Simulation Duration Mismatch

D3QN might be ending simulations early or running them differently.

**Evidence needed:**
- Check actual simulation time for D3QN episodes
- Verify warmup time is excluded from calculations
- Check if episodes are terminating early

### Hypothesis 4: Throughput vs Completed Trips Definitions

There might be a difference between:
- `completed_trips`: Vehicles that finished their route
- `throughput`: Vehicles processed per hour (might include active vehicles?)

---

## Immediate Investigation Steps

### Step 1: Check D3QN Metric Calculation

**File:** `evaluation/performance_comparison.py`, `_run_d3qn_episode` method

Need to verify:
```python
# How is throughput calculated for D3QN?
# Is it using the same formula as Fixed-Time?
# Are simulation durations the same?
```

### Step 2: Review Training Metric Calculation

**File:** `experiments/comprehensive_training.py`

During training, where does `final_completed_trips` come from?
- Is it from `env.get_final_metrics()`?
- Is it actually hourly throughput or just trip count?

### Step 3: Check Traffic Environment Metrics

**File:** `core/traffic_env.py`

How does the environment calculate final metrics?

---

## Possible Explanations

### Scenario A: Training Used Wrong Metric

**Hypothesis:** Training logged "throughput" as trips/episode, not veh/h

**Evidence:**
- Training: 486 trips/episode × 12 = 5,832 veh/h
- Evaluation D3QN: 489 trips/episode / 0.075h = 6,520 veh/h
- But evaluation reports: 3,886 veh/h ← **MISMATCH!**

**Conclusion:** Can't be this, numbers still don't match.

### Scenario B: Evaluation Has a Bug

**Hypothesis:** Evaluation is miscalculating D3QN throughput

**Evidence:**
- More completed trips (+14.9%)
- Lower throughput (-31.6%)
- **Logically impossible** unless calculation is wrong

**Action:** Review `evaluation/performance_comparison.py` line by line

### Scenario C: D3QN Is Optimizing Wrong Metric

**Hypothesis:** D3QN learned to maximize cumulative reward, not throughput

**Evidence:**
- Excellent waiting time improvement (-34%)
- Good speed improvement (+7.6%)
- More completed trips (+14.9%)
- But... lower throughput? **DOESN'T COMPUTE**

---

## Statistical Significance (from Evaluation)

Despite the terrible throughput result, the statistical analysis shows:

```
AVG_THROUGHPUT:
  p-value: 0.000000
  Effect size (Cohen's d): -7.972 (large)
  95% CI: [-1856.860, -1725.545]
  Significant: Yes
```

The degradation is **statistically significant** with a **very large effect size**.

**BUT:** If completed trips are higher, how can throughput be lower?

---

## Next Actions

### Priority 1: Debug Metric Calculation ⭐ **URGENT**

1. Add debug logging to evaluation script
2. Print both completed trips AND throughput for each episode
3. Verify calculation formula matches training
4. Check simulation duration is actually 300 seconds

### Priority 2: Manual Verification

Run a single episode manually and track:
- Start time
- End time
- Completed trips
- Calculated throughput
- Compare with reported metrics

### Priority 3: Review Code

Search for all instances of:
- `avg_throughput` calculation
- `completed_trips` usage
- Simulation duration handling

---

## Hypothesis: Evaluation Bug Confirmed

Looking at the data more carefully:

**From performance_report.txt:**
```
Completed Trips:
  Fixed-Time: 425.80
  D3QN:       489.20
  Improvement: +14.9%
```

**Expected throughput calculation:**
```
Fixed-Time: 425.80 / 0.075h = 5,677 veh/h ✅ (matches report)
D3QN: 489.20 / 0.075h = 6,523 veh/h ✅✅✅

But report says: 3,886 veh/h ❌❌❌
```

**CONCLUSION:** There's a **BUG** in how D3QN throughput is being calculated or reported!

---

## Impact on Thesis

**If bug is confirmed and fixed:**
- D3QN actually achieved **+15% throughput improvement!** (6,523 vs 5,677)
- ALL other metrics also improved
- **Thesis goals exceeded dramatically!**

**If not a bug:**
- Need to understand why more completed trips = lower throughput
- Might indicate fundamental problem with approach
- Would need major revisions

---

## Status

🔴 **CRITICAL ISSUE IDENTIFIED**  
⏸️ **EVALUATION RESULTS ON HOLD**  
🔍 **INVESTIGATION REQUIRED**  

**Next Step:** Debug evaluation script to find throughput calculation error

---

*Analysis completed: October 10, 2025 @ 21:55*









