<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width=35% align=right>

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

The materials in this repository are provided strictly for illustrative and educational purposes and are supplied on an “as is” and “as available” basis. To the maximum extent permitted by applicable law:

- No warranties or guarantees of any kind are made, whether express or implied (including but not limited to warranties of accuracy, completeness, non‑infringement, merchantability, or fitness for a particular purpose).
- The authors and contributors shall not be liable for any direct, indirect, incidental, special, consequential, exemplary, or punitive damages arising out of or relating to the use of, reliance on, or inability to use these materials, even if advised of the possibility of such damages.
- The code and examples are not professional advice (legal, financial, engineering, or otherwise). You are solely responsible for your use of the materials, including verifying correctness, ensuring appropriate safeguards (security, privacy, safety), and complying with all applicable laws and regulations.
- External links, datasets, and third‑party libraries are subject to their own licenses and terms; availability and behavior may change without notice.
- Product names, logos, and brands are property of their respective owners; use here is for identification purposes only and does not imply endorsement.

Software dependencies evolve over time; version changes can affect behavior or break examples. Pin versions where reproducibility is critical and run tests appropriate for your use case.

<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width=35% align=right>
