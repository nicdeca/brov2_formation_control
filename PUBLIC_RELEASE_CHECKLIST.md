# Public release checklist

Before changing the GitHub repository visibility to public:

- [ ] Run `make public-check`.
- [ ] Remove `.venv-ros` / `.venv_ros` from Git history, not only from the current tree.
- [ ] Verify `git ls-files` contains no virtual environments, caches, generated bags, logs, or outputs.
- [ ] Verify documentation contains no user-specific filesystem paths.
- [ ] Confirm the repository license. The ROS package metadata currently declares MIT, but the repository snapshot does not contain a top-level `LICENSE` file.
- [ ] Add the paper/preprint URL to the README when it becomes available.
- [ ] Optionally add `CITATION.cff` once the final author list and bibliographic data are fixed.

Recommended history cleanup for the currently tracked ROS virtual environment:

```bash
# Make a backup / clone first. git-filter-repo rewrites commit IDs.
git filter-repo --path .venv-ros --invert-paths
```

`git filter-repo` may remove the `origin` remote as a safety measure. Re-add it
if necessary and force-push the rewritten branches/tags only after checking the
result. If collaborators already use the private repository, coordinate the
history rewrite with them.
