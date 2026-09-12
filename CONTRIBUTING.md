# Contributing

This is a research preview. Improvements should make behavior testable and claims
more precise. Install Foresight and this package, then run:

```bash
python -m unittest discover -v
python -m capability_gap demo --output runs/contribution-check
git diff --check
```

For controller changes, add regression tests for observable outcomes rather than
mirroring implementation details. Keep task success, tool success and model
confidence separate. Do not promote predictions or model-generated labels into
ground truth. Do not commit private traces, model credentials or training data.

Document any new provider's request limits and any executor's verification boundary.
Label scripted examples clearly and report failed adaptation cases alongside gains.
Contributions are provided under Apache-2.0.
