# Contributing

Thanks for helping improve the ReAct Planning Agent. This guide covers local setup, the
quality gates CI enforces, the spec-driven (OpenSpec) workflow, branching, commit
conventions and how CI/CD is wired.

## Development setup

Requires Python 3.10+ (CI tests 3.10, 3.11, 3.12 and 3.13).

```bash
git clone https://github.com/imranmali/react-planning-agent.git
cd react-planning-agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
react-agent "What is 12 * 7?"      # prints 84
```

The core agent has no runtime dependencies and runs fully offline. The optional hosted
model backend is installed with `pip install -e ".[anthropic]"`.

## Quality checks

Run these before pushing; CI runs exactly the same commands:

```bash
ruff check .                 # lint
ruff format --check .        # formatting (use `ruff format .` to fix)
mypy                         # strict type check of src/ and tests/ (configured in pyproject.toml)
pytest --cov --cov-report=term-missing   # tests; fails below 90% coverage
openspec validate --all --strict         # spec and change validation
```

Tests must be deterministic and must not use the network.

## OpenSpec workflow

Behavior changes are spec-driven with [OpenSpec](https://github.com/Fission-AI/OpenSpec).
Install the CLI with `npm install -g @fission-ai/openspec@1.14.3` (the version CI uses).

1. **Propose** - create a change and write its artifacts:

   ```bash
   openspec new change add-my-feature
   # edit openspec/changes/add-my-feature/{proposal.md,design.md,tasks.md,specs/**/spec.md}
   openspec validate add-my-feature --strict
   ```

   Open a PR with the proposal so it can be reviewed before implementation.

2. **Apply** - implement the tasks in `tasks.md`, checking each one off (`- [x]`) as it is
   verified. Keep `openspec validate --all --strict` green.

3. **Archive** - once the change is implemented and merged, fold its spec deltas into
   `openspec/specs/` and move it to the archive:

   ```bash
   openspec archive add-my-feature --yes
   ```

Pure chores (CI, tooling, docs) do not need a change proposal.

## Branching model

- `main` is protected and always releasable. Never push to it directly.
- Work on short-lived branches cut from `main`:
  - `feature/<topic>` for new functionality (e.g. `feature/loop-detection`)
  - `fix/<topic>` for bug fixes (e.g. `fix/parser-fenced-json`)
- Open a pull request (the [template](.github/pull_request_template.md) has a checklist);
  merge only after CI is green and the PR is reviewed.
- Releases are cut by pushing a `vX.Y.Z` tag on `main` whose version matches
  `project.version` in `pyproject.toml`.

## Commit convention

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
`<type>(<optional scope>): <summary>`, with a body explaining what and why.

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`.

```text
feat(guards): stop runs that alternate between two actions
fix(parser): accept lowercase "final answer" labels
docs: document exit codes in the README
test(critic): cover bounded revision limit
ci: add Python 3.13 to the test matrix
feat(cli)!: rename --max-steps to --max-iterations

BREAKING CHANGE: --max-steps has been removed.
```

Record user-visible changes under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).

## CI/CD

| Workflow | Trigger | What it does |
| --- | --- | --- |
| [`ci.yml`](.github/workflows/ci.yml) | push to `main`, pull requests, manual | ruff lint + format check, mypy, pytest on 3.10-3.13 with the 90% coverage gate, `openspec validate --all --strict`, then builds the sdist/wheel and the Docker image and smoke tests both |
| [`deploy.yml`](.github/workflows/deploy.yml) | push to `main` (non-docs changes), manual | builds and pushes `ghcr.io/imranmali/react-planning-agent` tagged `sha-<short-sha>` and `latest`, then smoke tests the pushed image |
| [`release.yml`](.github/workflows/release.yml) | push of a `v*` tag | checks the tag matches `project.version`, builds the sdist/wheel and publishes a GitHub Release with generated notes |

Dependabot ([`dependabot.yml`](.github/dependabot.yml)) proposes weekly updates for
GitHub Actions and Python dependencies.

### Releasing

1. Bump `project.version` in `pyproject.toml` and move the `[Unreleased]` entries in
   `CHANGELOG.md` under the new version; merge via PR.
2. Tag the merge commit on `main` and push the tag:

   ```bash
   git tag -a v0.2.0 -m "v0.2.0"
   git push origin v0.2.0
   ```

### Container image

```bash
docker build -t react-planning-agent .
docker run --rm react-planning-agent "What is 12 * 7?"
docker run --rm ghcr.io/imranmali/react-planning-agent:latest --help
```

## License

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).
