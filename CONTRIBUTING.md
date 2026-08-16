# Contributing

Thank you for your interest in this project.

## Before submitting a change

Please:

1. Create a focused branch.
2. Add or update tests.
3. Run the benchmark with a fixed seed.
4. Document changes to assumptions or metrics.
5. Run the test suite.

```bash
python -m pytest
```

## Scientific changes

Changes to the following require documentation:

- Array geometry.
- Speed of sound.
- Delay sign convention.
- Angular coordinate convention.
- SNR generation.
- Benchmark seed.
- Error metric.
- Search bounds.