# One-time repository setup

## 1. Push the project

```bash
git init
git add .
git commit -m "feat: end-to-end MLOps pipeline for Cats vs Dogs"
git branch -M main
git remote add origin https://github.com/<your-user>/cats-vs-dogs-mlops.git
git push -u origin main
```

The first push to `main` runs the whole pipeline, including the deployment
stage. No secrets need to be configured: the workflow authenticates to GitHub
Container Registry with the built-in `GITHUB_TOKEN`.

## 2. Allow the workflow to publish packages

*Settings → Actions → General → Workflow permissions* → **Read and write
permissions**.

The published image appears under the repository's *Packages*. It is private by
default, which is fine — the deploy job pulls it with the same token. To make it
public: *Packages → cats-vs-dogs-mlops → Package settings → Change visibility*.

## 3. Enable auto-merge (for the auto-approval flow)

*Settings → General → Pull Requests* → tick **Allow auto-merge**.

## 4. Protect `main` so the tests actually gate merges

*Settings → Branches → Add branch ruleset* (or classic branch protection) for
`main`:

* **Require status checks to pass before merging** → add `Lint & unit tests`,
  `Train & track model` and `Build & publish image`.
* **Require a pull request before merging** → 1 approval (supplied
  automatically by `pr-auto-merge.yml`).

With those on, the flow is: branch → PR → CI runs → bot approves → auto-merge
once green → deploy + smoke test on `main`. No human clicks anywhere.

## 5. Demonstrating the flow for the recording

```bash
git checkout -b feature/demo-change
# make a visible change, e.g. bump APP_VERSION in k8s/base/deployment.yaml
git commit -am "feat: demo change"
git push -u origin feature/demo-change
gh pr create --fill        # or open the PR in the browser
```

Then watch the Actions tab: CI runs, the bot approves, auto-merge lands the PR,
and the deploy job rolls the new image out and smoke-tests it.
