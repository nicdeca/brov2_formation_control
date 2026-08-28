# README media

This directory contains visual material embedded in the repository README.

Generate the main formation-control animation with:

```bash
bash scripts/generate_readme_animation.sh
```

The generated repository-relative file is:

```text
docs/media/formation_animation.gif
```

This README animation is separate from experiment-run animations, which are
stored under each run's `mission/plots/` directory by the experiment plotting
pipeline.
