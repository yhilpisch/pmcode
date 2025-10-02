# Validation Tools Guide

# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

This folder contains two validators to help you run and verify all code assets non‑interactively.

## 1) Validate Chapter Scripts

Script: `tools/check_chapter_scripts.py`

What it does:
- Discovers `code/chapters/ch*.py` and runs each script headlessly.
- Forces a non-GUI Matplotlib backend and intercepts `plt.show()` to auto‑save figures to a folder, then closes them.
- Prints detailed, per‑script diagnostics with timings.

Usage examples:

  python tools/check_chapter_scripts.py

  PMDS_FILTER=ch07 python tools/check_chapter_scripts.py

  PMDS_TIMEOUT_SEC=300 PMDS_OUTPUT_DIR=outputs/chapters python tools/check_chapter_scripts.py

Environment variables:
- `PMDS_FILTER`: Substring to filter script filenames (e.g., `ch07`).
- `PMDS_TIMEOUT_SEC`: Per‑script timeout in seconds (default: 120).
- `PMDS_OUTPUT_DIR`: Where to save figures (default: `outputs/chapters`).

Notes:
- Figures are saved and windows never pop up.
- Execute from the repository root so relative paths in scripts resolve correctly.

## 2) Validate Notebooks

Script: `tools/check_notebooks.py`

What it does:
- Finds notebooks (default: `notebooks/*.ipynb`).
- Prints detailed, per‑notebook diagnostics: structure, imports scan/probe, execution, and total time.
- Executes with a headless Matplotlib backend and writable config/runtime directories.

Usage examples:

  python tools/check_notebooks.py --paths notebooks --timeout 180

  python tools/check_notebooks.py --paths notebooks --pattern 'ch02_*.ipynb' --timeout 60 --fail-fast

  python tools/check_notebooks.py --outdir outputs/executed_notebooks

Key options:
- `--paths`: Files or directories to search (default: `notebooks`).
- `--pattern`: Glob pattern within directories (default: `*.ipynb`).
- `--exclude`: Regex to skip certain notebooks (repeatable).
- `--timeout`: Per‑cell timeout seconds (default: 180).
- `--kernel`: Jupyter kernel name (default: `python3`).
- `--allow-errors`: Report but do not fail on cell exceptions.
- `--fail-fast`: Stop on the first failure.
- `--outdir`: Save executed notebooks under this directory (mirrors tree).
- `--normalize-ids`: Ensure missing/duplicate cell IDs are normalized.
- `--write-normalized`: Write normalized notebooks (with `--no-exec` or when not using `--outdir`).
- `--no-exec`: Only normalize, do not execute.

Notes:
- The validator configures `MPLBACKEND=Agg`, `MPLCONFIGDIR`, `IPYTHONDIR`, and `JUPYTER_RUNTIME_DIR` to writable locations automatically.
- If you run into missing packages, install them via `pip install -r requirements.txt`.

## Tips

- Prefer running from the repository root for consistent relative paths.
- For headless servers, no GUI is required. Figures are saved automatically.

