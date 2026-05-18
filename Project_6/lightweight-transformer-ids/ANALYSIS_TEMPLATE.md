# Ablation Study Analysis Template

This template should be completed for each ablation study to document findings and reasoning.

---

## Study Metadata

- **Ablation Name**: [e.g., architecture_ablation]
- **Timestamp**: [Auto-generated]
- **Researcher**: [Your name]
- **Date Completed**: [YYYY-MM-DD]
- **Data Used**: [CIC-IoT dataset, 50-50 split, etc.]

---

## Executive Summary

**Research Question**: 
[What specific hypothesis are we testing?]

**Key Finding**: 
[Most important result in 2-3 sentences]

**Recommendation**: 
[What should we do based on these results?]

---

## Detailed Analysis

### 1. Before Experiments

#### Hypothesis
[What did we expect to happen and why?]

#### Rationale
[Scientific justification for the ablation]

#### Expected Outcomes
[Specific predictions about results]

### 2. Experimental Results

#### Summary Statistics
| Experiment | Key Metric 1 | Key Metric 2 | Key Metric 3 |
|------------|-------------|-------------|-------------|
| exp_1 | value | value | value |
| exp_2 | value | value | value |
| exp_3 | value | value | value |

#### Detailed Observations
- Observation 1: [What happened?]
- Observation 2: [What changed?]
- Observation 3: [What was unexpected?]

### 3. Analysis

#### Key Findings
1. **Finding 1**: [Description with supporting metrics]
   - Evidence: [Which experiments support this?]
   - Confidence: [High/Medium/Low] - Why?

2. **Finding 2**: [Description with supporting metrics]
   - Evidence: [Which experiments support this?]
   - Confidence: [High/Medium/Low] - Why?

3. **Finding 3**: [Description with supporting metrics]
   - Evidence: [Which experiments support this?]
   - Confidence: [High/Medium/Low] - Why?

#### Tradeoff Analysis

**Accuracy vs. Efficiency**
```
[Describe the accuracy-efficiency tradeoff observed]
- At metric=value: accuracy=X%, latency=Yms
- At metric=value: accuracy=X%, latency=Yms
- Optimal point: [Where is the sweet spot?]
```

**Performance by Class**
```
[Analyze precision/recall for benign vs. attack classes]
- Benign class: precision=X%, recall=X%
- Attack class: precision=X%, recall=X%
- Implications: [What does this mean?]
```

### 4. Surprising Results

[Document any unexpected findings and hypothesize why]

Example:
- **Unexpected**: We expected k=32 to improve accuracy, but it didn't.
- **Why**: [Possible explanations - overfitting, saturation, data characteristics, etc.]

### 5. Statistical Significance

[If multiple runs were averaged, document confidence intervals and significance tests]

```
Example:
- k=8: 96.1 ± 0.3% (n=3 runs)
- k=16: 96.3 ± 0.2% (n=3 runs)
- Difference: Not significant (t-test p=0.23)
```

---

## Comparisons with Prior Work

[Compare findings with previous ablations or literature]

- **vs. Ablation X**: We expected similar results but found...
- **vs. Literature**: Our findings align/conflict with...
- **Implications**: What does this mean for our understanding?

---

## Recommendations

### For Next Steps
1. **Short-term**: [Immediate actions based on findings]
2. **Follow-up Ablation**: [What should we test next?]
3. **Configuration Update**: [Change base config based on findings]

### For Deployment
- **Recommended Configuration**: [Best settings for this component]
- **Deployment Constraint 1**: [e.g., max latency = 10ms]
  - **Solution**: [Use these settings]
- **Deployment Constraint 2**: [e.g., max model size = 5MB]
  - **Solution**: [Use these settings]

### For Paper/Report
[Key findings worth highlighting]

---

## Limitations

[Be honest about limitations that could affect conclusions]

- Limitation 1: [How could this bias results?]
- Limitation 2: [What wasn't tested?]
- Limitation 3: [What assumptions were made?]

---

## Code Changes Required

[Any code changes needed based on findings]

```python
# Example: Update base configuration
old_config = ModelConfig(k=16, depth=4)
new_config = ModelConfig(k=8, depth=6)  # Based on our findings
```

---

## Visualization Summary

[Key plots/tables to understand this ablation]

### Plot 1: [Title]
[Description of what the plot shows and key takeaways]

### Plot 2: [Title]
[Description of what the plot shows and key takeaways]

---

## Appendix: Raw Data

[Reference to detailed results]

- **CSV Results**: `results.csv` in experiment directory
- **JSON Results**: `results.json` in experiment directory
- **Logs**: Check `logs/<ablation_name>/<timestamp>/ablation.log` for detailed training logs

---

## Sign-Off

- **Analyzed by**: [Name]
- **Date**: [YYYY-MM-DD]
- **Approved by**: [Name if applicable]
- **Ready for Integration**: Yes / No
- **Comments**: [Any final notes]

---

## Example: Completed Linformer K Sweep Analysis

**Ablation Name**: architecture_ablation
**Timestamp**: 2026-05-03_14-32-10

**Executive Summary**:
The k (projection dimension) sweep revealed that k=8 provides the optimal accuracy-efficiency tradeoff for CIC-IoT classification. Larger k values show diminishing returns in accuracy improvement while significantly increasing latency.

**Key Findings**:
1. Accuracy plateaus beyond k=16: going from k=8 (96.1%) to k=32 (96.4%) gains only 0.3% accuracy
2. Latency scales roughly linearly with k: from 2.8ms (k=8) to 12.3ms (k=64)
3. k=4 is too restrictive: forces model to compress 78 features to 4 dimensions, causing 2% accuracy loss

**Recommendation**: Use k=8 as default for CIC-IoT. This provides 96.1% accuracy with 2.8ms latency, suitable for real-time IDS.

**Updated Configuration**:
```yaml
model:
  k: 8  # Changed from 16 (previous default)
  depth: 4
  heads: 4
  dim: 64
```

---
