# Contributing

Start with the [documentation](docs/README.md). Install Python 3.11+ and run:

```sh
python -m pip install -e ".[train,malecns,server,server-test,dev]"
python -m pytest -q
python examples/train_sequence.py --out runs/check
python -m build
```

Use a focused branch and include a runnable example for public API changes. Test state ownership, split isolation, sparse/dense numerical agreement and checkpoint replay. Small artificial graph fixtures keep CI independent of dataset downloads.

Document ordered channels, units, graph selection, normalization and trainable parameters. Benchmark claims need reproducible scripts, environment details, matched baselines and held-out splits. Do not commit credentials, personal data, raw connectomes or media without redistribution rights. Contributions are under Apache-2.0.
