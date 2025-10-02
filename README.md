<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width=35% align=right>

# Python & Mathematics for Data Science and Machine Learning — Code Companion

# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

Welcome to the code companion repository for the book “Python & Mathematics for Data Science and Machine Learning”. This repository contains the executable code assets that accompany the book. It is designed as a practical counterpart: book PDF + this code repo = your complete learning resources.

What’s included here are the runnable Python scripts and Jupyter notebooks referenced throughout the chapters. The book sources (AsciiDoc, build tooling, etc.) are not part of this repository to keep it focused and lightweight for readers.

## Contents

- `notebooks/` — Chapter notebooks (`chNN_*.ipynb`) used in the book.
- `code/chapters/` — Python scripts exported from the notebooks for headless/CLI execution.
- `code/figures/` — Standalone figure-generation scripts used to produce the figures in the book.
- `requirements.txt` — Minimal dependency set to run the notebooks and scripts.

Note: Many figure scripts save their outputs (SVG/PNG) to a local `figures/` directory when executed from the repository root. Use the commands below from the repo root for consistent behavior.

## Getting Started

1) Create and activate a Python environment (Python 3.11+ recommended).

2) Install dependencies:

   pip install -r requirements.txt

3) Run a notebook of interest (interactive):

   jupyter lab notebooks/ch02_python_essentials.ipynb

4) Run a chapter script (headless):

   python code/chapters/ch02_python_essentials.py

5) Generate a figure used in the book:

   python code/figures/ch04_projection.py

If you execute scripts in batch or on a server, you can enforce a headless Matplotlib backend via the environment variable `MPLBACKEND=Agg`.

## Reproducibility

- Scripts and notebooks prefer deterministic random number generation when relevant (fixed seeds or reproducible generators).
- If results slightly differ across platforms or versions, verify your library versions match those in `requirements.txt`.

## Tips

- Execute from the repository root so relative paths used by figure scripts resolve correctly.
- If your environment lacks a GUI, set `MPLBACKEND=Agg` to avoid display issues when generating plots.

## Acknowledgments

This companion repository is maintained for readers of the book. The learning experience is optimized when used alongside the book PDF.

# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width=35% align=right>

