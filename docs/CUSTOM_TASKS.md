# Custom tasks

[Documentation](README.md) / Custom tasks

CNSKit does not turn a task description into a trained policy. The application supplies three things: observations, an input mapping, and a readout that has meaning in its environment.

## Path A: prototype with existing synthetic adapters

Run `python examples/local_task.py`. It alternates left/right target features and prints action transitions. The scenario chooses an existing encoder and decoder; the synthetic `Session` supplies the recurrent state. Replace the example's observation dictionary with your environment's measurements, then consume `action.action` and `action.values` in your application.

Feature names must match the selected adapter. Unknown features are currently ignored, not automatically bound. One call advances a fixed model timestep (`1000 / observation_hz` milliseconds); it does not wait for wall-clock time. Sessions are stateful and not thread-safe. Use one per episode and reset between trials.

## Path B: use the real anatomical graph

Prepare the graph using [MALECNS.md](MALECNS.md), then run:

```sh
python examples/real_graph_task.py --graph /path/to/traced-graph
```

This example selects annotated LC4 input cells and DNp01 readout cells, applies a bounded pulse, and prints states. It does **not** decode a validated escape behavior. Change the selected IDs and your encoder only with an explicit interpretation of their annotations.

The application loop is:

```text
application observation
  -> your encoder: {body_id: current_in_arbitrary_units}
  -> MaleCNSSession.step(currents)
  -> MaleCNSSession.read(readout_ids)
  -> your decoder/controller
  -> application action
```

Readout values are activity, not action labels. A threshold or fitted decoder is an application decision. Evaluate success on held-out episodes before describing it as a learned task capability.

## Training and extension boundaries

Use [RidgeReadout](READOUT_TRAINING.md) to fit a local continuous readout and save its JSON checkpoint. There is no autograd integration, connectome weight-update API or end-to-end task trainer. You own the train/validation/test split and task evaluation. Editing adaptation settings or a task specification does not train anything.

Dynamic adapter registration is not implemented. To add a built-in synthetic adapter, extend the existing factories in `cns_tinker/sensory/adapters.py` or `cns_tinker/actions/adapters.py`, add a recipe and behavior-focused tests, and document the input/output units. For real-graph tasks, application-owned encoder/readout functions can wrap the public session API without modifying the runtime.
