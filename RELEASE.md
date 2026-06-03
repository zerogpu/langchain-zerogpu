# Releasing `langchain-zerogpu`

Releases are automated: pushing a `v<version>` tag builds the package, publishes
it to PyPI via **Trusted Publishing (OIDC)**, and creates a GitHub Release.
This project follows [Semantic Versioning](https://semver.org/) and keeps a
[Keep a Changelog](https://keepachangelog.com/)-style `CHANGELOG.md`.

## One-time setup

1. **Create the PyPI project's trusted publisher** (no API token needed):
   - Go to <https://pypi.org/manage/account/publishing/> (or, once the project
     exists, its *Publishing* settings).
   - Add a **GitHub Actions** publisher:
     - Owner: `zerogpu`
     - Repository: `langchain-zerogpu`
     - Workflow: `release.yml`
     - Environment: `pypi`
2. **Create the `pypi` environment** in the GitHub repo
   (*Settings → Environments → New environment → `pypi`*). Optionally add
   required reviewers to gate publishes.

> First-ever publish: if PyPI cannot resolve the trusted publisher because the
> project does not exist yet, do a single manual `0.1.0` upload (see
> [Manual fallback](#manual-fallback)), then rely on the workflow thereafter.

## Cutting a release

1. Make sure `main` is green in CI.
2. Bump the version in **`pyproject.toml`** (`project.version`).
3. Move the relevant notes in **`CHANGELOG.md`** from `[Unreleased]` into a new
   `## [X.Y.Z] - YYYY-MM-DD` section and update the link references at the
   bottom. The release workflow fails if no `## [X.Y.Z]` section exists.

   **This section becomes the GitHub release notes verbatim** (matching the
   style of [zerogpu-router releases](https://github.com/zerogpu/zerogpu-router/releases)),
   so write it for readers:
   - open with a short narrative paragraph saying what the release is about,
   - follow with the `### Added` / `### Changed` / `### Fixed` subsections with
     concrete, file-level bullets.

   The workflow appends an `### Install` block automatically — don't add one.
4. Commit the bump:
   ```bash
   git add pyproject.toml CHANGELOG.md uv.lock
   git commit -m "Release vX.Y.Z"
   git push origin main
   ```
5. Tag and push — this triggers the `Release` workflow:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```
6. Watch the **Release** workflow in the Actions tab. It will:
   - verify the tag matches `pyproject.toml` and that `CHANGELOG.md` has a
     matching section,
   - build the wheel + sdist and run `twine check`,
   - publish to PyPI through the `pypi` environment (OIDC),
   - create a GitHub Release titled `langchain-zerogpu X.Y.Z`, with the
     `## [X.Y.Z]` CHANGELOG section as the notes, an `### Install` block, and
     the built artifacts attached.
7. Verify the install:
   ```bash
   pip install --upgrade langchain-zerogpu
   python -c "import langchain_zerogpu; print(langchain_zerogpu.__version__)"
   ```

## Manual fallback

If you must publish without the workflow (e.g. the very first upload):

```bash
uv build
uvx twine check dist/*
uvx twine upload dist/*          # needs a PyPI API token in ~/.pypirc or $TWINE_*
git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z
```

## Post-release

- Confirm the new version renders on <https://pypi.org/project/langchain-zerogpu/>.
- Add a fresh `## [Unreleased]` section back to the top of `CHANGELOG.md` for the
  next cycle.
