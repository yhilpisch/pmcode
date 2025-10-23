<img src="https://theaiengineer.dev/tae_logo_gw_flat.png" width=35% align=right>
<br><br>

# Python & Mathematics for Data Science and Machine Learning — Code Companion

**&copy; Dr. Yves J. Hilpisch | The Python Quants GmbH**  
**AI-powered by GPT-5**

Welcome to the code companion repository for the book “Python & Mathematics for Data Science and Machine Learning”. This repository contains the executable code assets that accompany the book. It is designed as a practical counterpart: book manuscript + this code repo = your complete learning resources.

What’s included here are the runnable Python scripts and Jupyter notebooks referenced throughout the chapters.

## Contents

- `notebooks/` — Chapter notebooks (`chNN_*.ipynb`) used in the book.
- `code/chapters/` — Python scripts exported from the notebooks for headless/CLI execution.
- `code/figures/` — Standalone figure-generation scripts used to produce the figures in the book.
- `requirements.txt` — Minimal dependency set to run the notebooks and scripts.

Note: Many figure scripts save their outputs (SVG/PNG) to a local `figures/` directory when executed from the repository root. Use the commands below from the repo root for consistent behavior.

## Getting Started

1) Create and activate a local virtual environment in the repository (Python 3.11+ recommended):

```
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\\Scripts\\Activate.ps1

# optional but recommended
python -m pip install -U pip
```

2) Install dependencies into the active environment:

```
pip install -r requirements.txt
```

3) Run a notebook of interest (interactive):

```
jupyter lab notebooks/ch02_python_essentials.ipynb
```

4) Run a chapter script (headless):

```
python code/chapters/ch02_python_essentials.py
```

5) Generate a figure used in the book:

```
python code/figures/ch04_projection.py
```

If you execute scripts in batch or on a server, you can enforce a headless Matplotlib backend via the environment variable `MPLBACKEND=Agg`.

## Reproducibility

- Scripts and notebooks prefer deterministic random number generation when relevant (fixed seeds or reproducible generators).
- If results slightly differ across platforms or versions, verify your library versions match those in `requirements.txt`.

## Tips

- Execute from the repository root so relative paths used by figure scripts resolve correctly.
- If your environment lacks a GUI, set `MPLBACKEND=Agg` to avoid display issues when generating plots.

## Acknowledgments

This companion repository is maintained for readers of the book. The learning experience is optimized when used alongside the book manuscript.

**&copy; Dr. Yves J. Hilpisch | The Python Quants GmbH**  
**AI-powered by GPT-5**

## Disclaimer

This repository is provided "as is" for educational purposes. No warranties, guarantees, or representations of any kind are made regarding correctness, completeness, fitness for a particular purpose, or non‑infringement. Code and examples are intended for illustration and learning; they may omit production concerns (error handling, security, performance, robustness). Use at your own risk. Dependencies evolve over time; version changes can affect behavior or break examples.

<img src="https://theaiengineer.dev/tae_logo_gw_flat.png" width=35% align=right>
