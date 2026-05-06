# Examples

These files are lightweight fixtures for inspecting the framework interfaces.
They are not biological results.

- `tiny_manifest.csv` shows the expected manifest schema.
- `example_experiment.yaml` shows the intended config-driven experiment shape.
- `fake_run_outputs/metrics_summary.csv` shows the compact result table format.

Try the dry-run CLI after installing the package in editable mode:

```bash
evo-amr train --config examples/example_experiment.yaml
evo-amr report --config examples/example_experiment.yaml
```
