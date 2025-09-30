# Performance Analysis Takeaways

## (a) 2 Nodes with Autoscaling

**Baseline Performance:**
- 1 node: 47.0 min
- "Perfect split" estimate for 2 nodes: 47.0 ÷ 2 = 23.5 min
- **Actual time: 10.2 min**

### Analysis

**Performance vs Perfect Split:**
- Beat the "perfect split" guess by: 23.5 − 10.2 = **13.3 min faster**
- How much better: 13.3 ÷ 23.5 = **56.5% better** than the simple half-time estimate

**Performance vs Single Node:**
- Times faster than 1 node: 47.0 ÷ 10.2 = **4.59×**
- Total time saved: 47.0 − 10.2 = 36.8 min
- Percentage reduction: 36.8 ÷ 47.0 = **78.2% less time**

### 📊 Key Takeaway
> Instead of the expected ~23.5 min on 2 nodes, you finished in 10.2 min—that's ~2.3× faster than even a perfect 2-way split and ~4.6× faster than 1 node (saving 78% of the time).

## (b) 2 Nodes without Autoscaling

**Baseline Performance:**
- 1 node: 47.0 min
- "Perfect split" estimate for 2 nodes: 47.0 ÷ 2 = 23.5 min
- **Actual time: 15.1 min**

### Analysis

**Performance vs Perfect Split:**
- Beat the "perfect split" guess by: 23.5 − 15.1 = **8.4 min faster**
- How much better: 8.4 ÷ 23.5 = **35.5%** better than the simple half-time estimate

**Performance vs Single Node:**
- Times faster than 1 node: 47.0 ÷ 15.1 = **3.10×**
- Total time saved: 47.0 − 15.1 = 31.9 min
- Percentage reduction: 31.9 ÷ 47.0 = **67.8% less time**

### 📊 Key Takeaway
> Instead of the expected ~23.5 min on 2 nodes, you finished in 15.1 min—that's ~36% better than a perfect 2-way split and ~3.1× faster than 1 node (saving 68% of the time).

## (c) Autoscaling vs. No Autoscaling Comparison

**Both configurations use 2 nodes:**
- No autoscaling: 15.1 min
- **Autoscaling: 10.2 min**

### Analysis

**Direct Comparison:**
- Minutes faster: 15.1 − 10.2 = **4.9 min**
- Percent faster: 4.9 ÷ 15.1 = **32.5%**
- Times faster: 15.1 ÷ 10.2 = **1.48×**

**Context vs Perfect Split (23.5 min):**
- Autoscaling beats it by **56.5%**
- No autoscaling beats it by **35.5%**
- Difference: ~**21.0 percentage points**

### 📊 Key Takeaway
> With the same 2 nodes, autoscaling finishes ~1.48× faster than no-autoscaling (~32% less time), saving 4.9 minutes on this workload.

---

## Summary

| Configuration | Time (min) | vs 1 Node | vs Perfect Split |
|---------------|------------|-----------|------------------|
| 1 Node (baseline) | 47.0 | - | - |
| 2 Nodes (no autoscaling) | 15.1 | 3.10× faster | 35.5% better |
| 2 Nodes (with autoscaling) | 10.2 | 4.59× faster | 56.5% better |
