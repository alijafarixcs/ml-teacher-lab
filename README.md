# ML Teacher Lab

ML Teacher Lab is a practical, mathematics-first machine-learning course built as executable Jupyter notebooks. It connects plain-language intuition, hand calculations, NumPy implementations, production libraries, experiments, visualizations, and exercises across supervised learning, unsupervised learning, deep learning, and reinforcement learning.

## Who this is for

The course is for learners who can write basic Python—variables, functions, loops, lists, and imports—but are still building confidence with mathematics and machine learning. No calculus or linear-algebra course is assumed. Each mathematical idea is introduced when it becomes useful.

## Prerequisites

- Python 3.11 or newer
- Comfort running commands in PowerShell, Terminal, or a shell
- Basic Python syntax
- Roughly 5 GB of free disk space for the environment (PyTorch is the largest dependency)
- CPU-only hardware is sufficient; 8 GB RAM is recommended

## Installation

Clone or download the repository, then change into `ml-teacher-lab`.

Create the environment on every platform:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

If several Python versions are installed on Windows, use `py -3.11 -m venv .venv`.

## Recommended learning path

Work through each numbered directory and each notebook in filename order:

1. `01_supervised_learning` — predictions from labeled examples (estimated 18–24 hours)
2. `02_unsupervised_learning` — structure without target labels (12–16 hours)
3. `03_deep_learning` — neural networks from arrays to attention (24–32 hours)
4. `04_reinforcement_learning` — learning actions from rewards (18–24 hours)
5. `05_final_projects` — integrated, end-to-end practice (12–20 hours)

The full learning path is approximately 84–116 hours, including exercises and experiments.

## Repository structure

```text
ml-teacher-lab/
├── 01_supervised_learning/   # Regression, classification, evaluation, capstone
├── 02_unsupervised_learning/ # Clustering, PCA, anomalies, capstone
├── 03_deep_learning/         # NumPy foundations and CPU-first PyTorch models
├── 04_reinforcement_learning/# Tabular and small neural agents
├── 05_final_projects/        # Larger integrated projects
├── data/                     # Optional raw and derived local data
├── images/                   # Course images and exported figures
├── src/                      # Reusable data, metric, plotting, and notebook helpers
├── tests/                    # Utility and notebook-quality tests
├── requirements.txt
└── pyproject.toml
```

## Running the notebooks

Start JupyterLab from the repository root with `jupyter lab`. Open the first notebook in `01_supervised_learning` and use **Kernel → Restart Kernel and Run All Cells**. Each notebook is deterministic and designed to run independently from top to bottom.

To execute one notebook non-interactively:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace 01_supervised_learning/00_supervised_learning_overview.ipynb
```

To validate every authored notebook:

```bash
python tools/validate_notebooks.py --execute
```

## Running tests

```bash
pytest
```

The tests check shared calculations, deterministic data helpers, valid notebook JSON, nonempty code cells, forbidden placeholder markers, saved execution errors, and—when requested through the validation script—fresh top-to-bottom execution.

## Hardware and runtime

All required examples run on CPU. A GPU is optional and never assumed. Classical lessons generally finish in seconds; compact neural-network and reinforcement-learning lessons may take a few minutes. Dataset choices are built-in or synthetic by default, so the core course works without network access.

## Troubleshooting

- **PowerShell blocks activation:** run `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.
- **The notebook uses the wrong Python:** run `python -m ipykernel install --user --name ml-teacher-lab` and select that kernel in JupyterLab.
- **An import fails:** confirm the terminal shows `(.venv)` and rerun `pip install -r requirements.txt`.
- **A plot does not appear:** restart the kernel and run all cells; do not execute lessons out of order.
- **PyTorch installation is slow:** it is the largest package. The project uses the ordinary CPU-compatible wheel and does not require CUDA.
- **A remote dataset is unavailable:** lessons that mention remote data automatically use a documented built-in or synthetic fallback.

## Development status

The repository is built in controlled, sequential phases so every notebook can be executed and checked before the next group is added. The current status is recorded by the files present and by the validation suite; no empty notebook placeholders are used.

As of 2026-09-20, the complete 48-notebook curriculum is implemented: 11 supervised-learning notebooks, 9 unsupervised-learning notebooks, 13 deep-learning notebooks, 11 reinforcement-learning notebooks, and 4 final projects. Every notebook has saved outputs from a clean top-to-bottom execution, and the complete course is checked by the automated validation suite. The supervised section also includes an explicit required-mathematics coverage map.

## License suggestion

MIT is a practical default for educational code because it permits reuse and adaptation with attribution. Add a `LICENSE` file before public distribution and confirm that the chosen license fits any institutional requirements.
