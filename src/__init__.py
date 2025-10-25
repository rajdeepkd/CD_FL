"""
Cross-domain federated learning framework.

This package simulates heterogeneous clients (each backed by its own CSV)
and runs different FL strategies (FedAvg, FedProx, PersonalizedHead).

Key modules:
- src.data: loading and preprocessing per-client data
- src.models: simple MLP classifiers
- src.fl: client/server logic and strategies
- src.utils: logging, config, metrics, serialization utilities
- src.experiments: experiment runner
"""