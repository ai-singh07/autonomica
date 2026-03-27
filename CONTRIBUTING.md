# Contributing to Autonomica

Thank you for helping make AI agent governance better. All contributions are welcome — bug fixes, new features, integrations, and documentation improvements.

## Dev environment

```bash
git clone https://github.com/ai-singh07/autonomica
cd autonomica
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+

## Running tests

```bash
pytest                              # full suite (~436 tests, < 2 s)
pytest tests/test_scorer.py -v     # single module
pytest -k "test_tool_overrides"    # single test by name
```

The suite is fast and has no external dependencies — no network calls, no real database writes (SQLite in-memory or temp files).

## Making changes

1. **Open an issue first** for anything non-trivial so we can align before you invest time writing code.
2. Fork the repo and create a branch: `git checkout -b feat/my-feature`.
3. Write tests. PRs without tests for new behaviour will not be merged.
4. Run the full suite and confirm it passes: `pytest`.
5. Open a PR against `main` with a clear description of what changed and why.

## PR guidelines

- Keep PRs focused — one logical change per PR.
- Match the existing code style (no formatter config yet; just follow the surrounding code).
- Update docstrings and README if your change affects the public API.
- Do not bump the version in `pyproject.toml` — maintainers handle releases.

## Good first issues

New to the codebase? Start here:

- [`good first issue`](https://github.com/ai-singh07/autonomica/issues?q=label%3A%22good+first+issue%22) — approachable tasks with clear scope
- [`help wanted`](https://github.com/ai-singh07/autonomica/issues?q=label%3A%22help+wanted%22) — larger items where outside help is welcome

## Questions

Open a [GitHub Discussion](https://github.com/ai-singh07/autonomica/discussions) or file an issue — we're responsive.
