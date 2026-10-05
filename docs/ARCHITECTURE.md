# Architecture

![CNSKit training and inference](assets/cnskit-overview.svg)

1. **Import.** Verify graph artifacts and provenance; retain exact neuron IDs. Matrix rows receive from columns. Incoming-count normalization is an authored numerical choice.
2. **Bind.** A learned or fixed projection maps named task channels to an explicit set of input neurons. Another explicit set provides the readout. No automatic biological mapping is implied.
3. **Execute.** A cached sparse COO operator performs recurrent multiplication. Per-neuron gain and leak determine bounded rate dynamics. Batch and time axes remain explicit.
4. **Fit.** The local trainer resets episode state, uses masked supervision and truncated backpropagation, clips gradients, and restores the checkpoint with lowest held-out validation loss. Training-only normalization is exported.
5. **Deploy.** An inference bundle binds parameters to a graph fingerprint and ordered channels. Streaming callers supply their own per-episode state; there is no shared hidden session.

`readout` fits the output projection; `adapters` also fits the input projection; `dynamics` additionally adapts sigmoid-bounded cell-wise gain/leak. All modes preserve the anatomical edge weights. No dense neuron-by-neuron matrix is constructed by the learning implementation.

The export contains `manifest.json`, `graph.npz`, `body_ids.npy` and `parameters.npz`. Arrays are read without pickle. Hashes detect corruption, not replacement of an entire unsigned bundle. The export is an inference artifact, not a resumable optimizer snapshot.

The older `cns_tinker` namespace provides scenario/replay helpers, a synthetic local runtime, explicit-current MaleCNS sessions and ridge fitting. Its cloud/configuration scaffolds are not a managed training service.
