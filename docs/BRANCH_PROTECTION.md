# GitHub Repository Rules — Setup Checklist

The workflow in [GIT_WORKFLOW.md](GIT_WORKFLOW.md) is enforced by three things:

1. **`.github/workflows/ci.yml`** — runs lint + tests + branch-name validation on every PR.
2. **`.github/CODEOWNERS`** — auto-routes review requests by module path.
3. **A repository ruleset** on `main` — encodes "no direct push, ≥1 review, squash-merge only, status checks must pass, linear history". The ruleset is **not** auto-applied from the repo — an admin must import it once via Settings UI or the GitHub API.

This document is the checklist for that one-time admin setup.

## A. Repository ruleset — `main` branch protection

The canonical JSON lives at [.github/rulesets/main-protection.json](../.github/rulesets/main-protection.json). It encodes:

| Rule | Setting |
|---|---|
| Block direct deletion | `deletion` |
| Block force pushes | `non_fast_forward` |
| Require linear history | `required_linear_history` |
| Require PR before merging | `pull_request` |
| Required approving reviews | **1** |
| Dismiss stale reviews on new push | **yes** |
| Require CODEOWNERS approval | **yes** |
| Require review-thread resolution | **yes** |
| Allowed merge methods | **squash** only |
| Required status checks (strict) | `lint + tests`, `branch name follows convention` |

### Option 1 — Import via Settings UI (easiest)

1. Open *Settings → Rules → Rulesets → New branch ruleset → Import a ruleset*.
2. Upload [.github/rulesets/main-protection.json](../.github/rulesets/main-protection.json).
3. Confirm `Enforcement status: Active` and `Targets: refs/heads/main`.
4. Save.

### Option 2 — Import via REST API

Requires a token with `Administration: write` repo permission:

```bash
gh api \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  /repos/chenbenlu/omr-jianpu/rulesets \
  --input .github/rulesets/main-protection.json
```

To list / update / delete:

```bash
gh api /repos/chenbenlu/omr-jianpu/rulesets                  # list
gh api -X PUT  /repos/chenbenlu/omr-jianpu/rulesets/<id> --input <file>
gh api -X DELETE /repos/chenbenlu/omr-jianpu/rulesets/<id>
```

## B. Repository-level settings (not part of the ruleset)

These live under *Settings → General* and *Settings → Pull Requests* and must be set by hand once:

- **Default branch**: `main`.
- **Allow merge commits**: ☐ off.
- **Allow squash merging**: ☑ on. *Default commit message: "Pull request title and description"*.
- **Allow rebase merging**: ☐ off.
- **Always suggest updating pull request branches**: ☑ on.
- **Automatically delete head branches**: ☑ on. (Branch is auto-deleted on merge, per GIT_WORKFLOW.md §2.)

## C. CODEOWNERS

[.github/CODEOWNERS](../.github/CODEOWNERS) maps directories to handles. Until Members B / C / D add their GitHub handles, the placeholder `@chenbenlu` keeps PRs from being blocked. Each member should update their own line in a `chore/<owner>-claim-codeowner` PR when they onboard.

## D. CI status checks

The status-check names referenced by the ruleset (`lint + tests` and `branch name follows convention`) come from job names in [.github/workflows/ci.yml](../.github/workflows/ci.yml). If a job is renamed in `ci.yml`, **update the JSON ruleset in the same PR** or merges to `main` will start failing because the required check no longer exists.

## E. Verification

After importing:

1. Open a throwaway PR (e.g. add a trailing newline to a docs file) on a branch like `chore/A-test-branch-protection`. Confirm:
   - Direct push to `main` from your terminal is rejected.
   - The PR cannot be merged until CI passes and at least one approving review is recorded.
   - "Merge" and "Rebase and merge" buttons are disabled; only "Squash and merge" is offered.
2. Close the throwaway PR without merging; delete the branch.
