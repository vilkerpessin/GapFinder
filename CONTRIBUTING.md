# Contributing to GapFinder

Thank you for your interest in contributing! GapFinder is an early-stage open-source project, and we welcome all contributions.

## Ways to Contribute

- **Bug reports**: Found something broken? Open an issue
- **Code**: Fix bugs or add features via pull requests
- **Documentation**: Improve docs, add examples, fix typos

## Getting Started

1. Fork the repository
2. Clone your fork and set up locally (see [Setup Local](README.md#setup-local) in the README)
3. Create a feature branch: `git checkout -b feat/your-feature`

## Pull Requests

We use the [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow) as our main workflow.

- **Small fixes** (typos, simple bugs): Submit a PR directly
- **Larger changes** (new features, refactors): Open an issue first to discuss

Before submitting:

1. Run tests: `pytest tests/ -v`
2. Follow [Conventional Commits](https://www.conventionalcommits.org/)

Expect feedback within a few days.

## Commit Messages

We use the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type: description
```

Types:
- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation only
- `test`: adding or updating tests
- `refactor`: code change without behavior change

Example:
```
feat: add CSV export option
```

## Testing

```bash
pytest tests/ -v

pytest tests/ --cov=app --cov-report=html
```

## Writing Bug Reports

A good bug report includes:

- A quick summary
- Steps to reproduce (be specific!)
- What you expected would happen
- What actually happens
- Notes (why you think this might be happening, or things you tried)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
