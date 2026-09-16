# Releasing

There is no release step. Bump the version and add a changelog section in your
PR; merging it to `main` publishes to PyPI and creates the GitHub Release.

## 1. In your PR — bump and write the changelog

```bash
make bump         # patch — bug fix, e.g. 0.2.4 → 0.2.5
make bump-minor   # new tool, new optional argument, e.g. 0.2.4 → 0.3.0
make bump-major   # tool removed/renamed, output shape changed, e.g. 0.2.4 → 1.0.0
```

Each updates `pyproject.toml` and `uv.lock` and nothing else — no commit, no
tag. Commit both with the rest of your work.

Then add a `## [X.Y.Z] - YYYY-MM-DD` section at the **top** of `CHANGELOG.md`.
It becomes the GitHub release notes verbatim, so write it for readers: a short
paragraph on what the release is about, then `### Added` / `### Changed` /
`### Fixed` bullets. Don't add an install block — the release appends one.

A PR that doesn't touch `langchain_zerogpu/` (docs, tests, CI) doesn't need a
bump and ships nothing. If another release PR merges first with the same
version, rebase and bump again.

## 2. PR CI checks it

The **Version bump + CHANGELOG** job fails the PR if `langchain_zerogpu/`
changed (or the version changed) and either:

- the `pyproject.toml` version isn't higher than `main`'s, or
- the top `CHANGELOG.md` section isn't `## [<that version>]`.

## 3. Merge — CI publishes

1. **CI** runs on `main` — lint, mypy, unit tests on Python 3.10–3.13.
2. If CI is green, **Release** checks the merged commit. If the version isn't
   released yet, it builds the wheel + sdist, pushes the tag `vX.Y.Z`,
   publishes to PyPI (Trusted Publishing, `pypi` environment), and creates the
   GitHub Release `langchain-zerogpu X.Y.Z` with the changelog section as notes
   and the built files attached.
3. If the version is already released, the run goes green and does nothing.

Step 3 is every ordinary merge. A skipped release is not a failure — check the
run's summary for the reason if you expected a publish.

### When it declines to release

| Summary line | Fix |
|---|---|
| `vX.Y.Z is already released` | Nothing — this is every ordinary merge. |
| `pyproject.toml is X but uv.lock says Y` | `uv lock`, commit `uv.lock`, push. |
| `the top CHANGELOG.md section is '## [Y]'` | Add the `## [X]` section at the top, push. |
| `X, which is older than the latest release Y` | Bump above the latest release. |

If CI fails, Release never starts.

### If something goes wrong mid-release

- **Failed before the tag** (build, `twine check`, empty changelog section):
  nothing was published. Fix it and push; the same bump releases.
- **Failed after the tag** (PyPI publish or the GitHub Release): re-run the
  failed Release run, or **Actions → Release → Run workflow**. It reuses the
  tag, skips files already on PyPI, and creates the missing release.

A new version can take a minute or two to show up in `pip install` after the
run goes green. That's PyPI propagation, not a failed publish.

## One-time setup (already done)

- PyPI trusted publisher: owner `zerogpu`, repo `langchain-zerogpu`, workflow
  `release.yml`, environment `pypi`. Renaming the workflow file or the
  environment breaks publishing until the publisher is updated.
- GitHub environment `pypi` exists under *Settings → Environments*. Adding
  required reviewers there turns every release into a one-click approval.
