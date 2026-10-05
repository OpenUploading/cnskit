# Scope and roadmap

## Implemented in 0.2

- Verified MaleCNS graph ingestion and explicit induced subgraphs.
- Regression/classification with episode masks and held-out model selection.
- Readout, adapter and bounded dynamics adaptation with a fixed sparse graph.
- Explicit-state, batched inference; portable graph-bound checkpoints.
- Local train/predict CLI; reproducible synthetic and real-graph validation examples.

## Next investigations

- Matched recurrent baselines, graph rewiring/edge ablations and multiple-seed evaluations on substantive user tasks.
- CUDA memory/latency characterization, sparse execution alternatives and full-graph training profiles.
- Richer synaptic dynamics, typed observation schemas and task-specific sensory encoders, evaluated separately from anatomical evidence.
- Optimizer resume, streaming datasets and distributed execution as measured needs emerge.

These are research and engineering directions, not supported features or promised performance. Reward-only RL, learned topology, spiking dynamics, general intelligence and population-level biological fidelity are not established by this release.
