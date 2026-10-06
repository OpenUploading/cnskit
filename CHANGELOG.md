# Changelog

## Unreleased

- Add a complete real-MaleCNS task walkthrough with user-supplied regression datasets, three-seed zero-edge ablations and verified stateful inference.
- Replace illustrative README neuron bindings with runnable commands, explicit outputs and measured limitations.

## 0.2.2

- Add optional patience-based early stopping while restoring the absolute best validation epoch.
- Support unlabeled target placeholders under boolean masks (NaN for regression, -1 for classification).
- Add `cnskit evaluate` for held-out datasets and expose learning symbols to IDEs.
- Extend regression coverage for masked supervision, stopping criteria and CLI evaluation.

## 0.2.1

- Stream checkpoint hashing and retain only trainable tensors during best-epoch selection.
- Publish Linux Python 3.11 / 3.13 CI and document custom PyTorch training loops.

## 0.2.0

- Public source release under Apache-2.0 in OpenUploading.
- New `cnskit` graph, episode, supervised training and explicit-state inference APIs.
- Three fixed-graph adaptation modes; named channels and portable verified checkpoints.
- Local `cnskit train` / `predict` commands and English guides.
- Numerical, gradient, held-out training and real-MaleCNS validation examples.
- Retained `cns_tinker` compatibility namespace; excluded unrelated website/media assets.
