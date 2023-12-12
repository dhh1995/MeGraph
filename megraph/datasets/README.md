# Dataset

This folder includes implementations of datasets and evaluators. 

The `manager.py` manages all datasets implemented or imported in this folder. check [`datasets`](datasets) directory see which datasets are available and registered via `add_dataset()` function.

The `datasets` directory contains datasets from GNN benchmark, OGB benchmark, LRGB benchmark, and our Graph Theory Benchmark. 

The `utils` directory contains utilities for generating datasets from the Graph Theory Benchmark. 

The `evaluator.py` file includes evaluation criteria, such as accuracy, L1, L2, F1, and AP (Average Precision).
