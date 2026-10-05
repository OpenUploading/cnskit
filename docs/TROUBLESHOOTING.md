# Troubleshooting

| Symptom | Action |
|---|---|
| `torch` import fails | Install the `[train]` extra and a PyTorch wheel compatible with your platform. |
| Unknown body ID | Inspect the traced-body selection. Never replace excluded IDs with positional indices. |
| Graph integrity failure | Re-prepare from the original source files; do not edit the manifest to bypass verification. |
| Training/validation overlap | Split by independent episode/session and assign stable distinct IDs. |
| Memory pressure | Begin with a selected subgraph, smaller batches and shorter TBPTT windows. Sparse edges do not eliminate activation memory. |
| Existing output | Choose a new run directory; exports intentionally refuse overwrite. |
| Divergent checkpoint input | Preserve ordered input channels and fitted normalization. Reset state between unrelated episodes. |
| Poor generalization | Compare against matched recurrent baselines and examine your splits, labels, task encoding and graph selection. |

The SDK runs locally without Vercel, an account or a hosted job scheduler. GPU speed and biological accuracy are not guaranteed.
