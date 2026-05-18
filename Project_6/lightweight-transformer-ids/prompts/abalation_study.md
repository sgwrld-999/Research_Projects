### ROLE

You are a senior ML systems engineer working on an IDS project.

Your task is to **implement a complete ablation study by extending the existing repository**, strictly following its current structure and conventions.

---

### OBJECTIVE

Implement a **fully automated, reproducible ablation study pipeline** that:

* Reuses existing models, pipelines, and scripts
* Adds configurable experimentation
* Logs results systematically
* Stores outputs in a **clean, timestamped structure**

---

## 📂 REPOSITORY CONSTRAINT (VERY IMPORTANT)

You MUST follow the existing structure:

```
configs/
data/
logs/
models/
pipeline/
prompts/
results/
src/
tests/
```

### Rules:

* DO NOT create random folders
* DO NOT duplicate scripts unnecessarily
* Place everything in **logically correct directories**

---

## 📊 DATASET REQUIREMENT

Use the CICIoT2023 dataset.

* Use **existing loaders and preprocessing**
* Use **same dataset splits**
* Only modify preprocessing when explicitly required (ablation cases)

---

## 🧠 EXECUTION STRATEGY

---

### STEP 1: Analyze Existing Codebase

* Understand:

  * `models/` → architectures
  * `pipeline/` → training + evaluation
  * `configs/` → hyperparameters
  * `main_ciciot.py` → entry point
  * `tests/` → evaluation scripts

**Task:**
Design ablation without breaking this structure.

---

### STEP 2: Add Ablation Entry Script

Create a new script:

```
src/run_ablation.py
```

This script should:

* Load base configs
* Override parameters per experiment
* Call existing training pipeline
* Handle logging + saving

---

### STEP 3: Config-Driven Ablation

Extend `configs/`:

```
configs/ablation/
    architecture.yaml
    design.yaml
    efficiency.yaml
```

Each config defines:

* Variable to change
* Values to sweep

---

### STEP 4: Architectural Ablation

Support via config overrides:

* Transformer vs Linformer
* Linformer k = 8,16,32,64
* Depth N = 2,4,6,8
* Embedding d = 32,64,128
* Heads h = 2,4,8

Use:

* Existing model constructors
* No duplication

---

### STEP 5: Design Ablation

Add flags in configs:

* Positional encoding on/off
* Embedding type
* Pooling strategy
* Loss function
* Correlation pruning
* Log transform

Modify only relevant modules.

---

### STEP 6: Efficiency Benchmarking

Create:

```
pipeline/benchmark.py
```

Compute:

* Params
* FLOPs
* Latency
* Peak RAM

Run across:

* MLP, CNN, LSTM, Transformer, Linformer

---

## 📁 LOGGING + RESULT STORAGE (CRITICAL)

---

### 🔹 LOGS STRUCTURE

Store logs inside:

```
logs/<experiment_name>/<timestamp>/
```

Example:

```
logs/linformer_k_sweep/2026-05-03_14-32-10/
    train.log
    metrics.json
    config.yaml
```

---

### 🔹 RESULTS STRUCTURE

Store results inside:

```
results/ciciot_binary_classification/<experiment_name>/<timestamp>/
```

Example:

```
results/ciciot_binary_classification/linformer_k_sweep/2026-05-03_14-32-10/
    results.csv
    summary.json
```

---

### 🔹 TIMESTAMP FORMAT

Use:

```
YYYY-MM-DD_HH-MM-SS
```

Generate programmatically for every run.

---

### 🔹 WHAT TO SAVE

For each experiment:

* Config used
* Metrics:

  * Accuracy, Precision, Recall, F1
* Efficiency stats
* Model checkpoints (optional)

---

## ⚙️ EXECUTION FLOW

Example usage:

```bash
python src/run_ablation.py --config configs/ablation/architecture.yaml
```

---

## ⚠️ STRICT RULES

* DO NOT modify:

  * Existing training pipeline
  * Dataset splits
* DO NOT mix multiple variables
* DO NOT overwrite logs/results
* ALWAYS use timestamped directories
* ALWAYS save config snapshot

---

## 🧠 REASONING REQUIREMENT

For EACH experiment (log this as well):

### Before run:

* What is changing?
* Why is it important?

### After run:

* What changed in results?
* Why did it happen?
* What tradeoff is observed?

Store this in:

```
analysis.txt
```

inside each experiment folder.

---

## FINAL GOAL

Deliver:

* Clean ablation pipeline integrated into repo
* Fully reproducible experiments
* Proper experiment tracking (industry + research level)
* Clear justification of all design choices

---
