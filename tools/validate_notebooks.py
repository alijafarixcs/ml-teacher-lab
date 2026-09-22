"""Validate notebook structure and optionally execute every notebook fresh."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_FOLDERS = (
    "01_supervised_learning",
    "02_unsupervised_learning",
    "03_deep_learning",
    "04_reinforcement_learning",
    "05_final_projects",
)
FORBIDDEN = (
    "TO" + "DO",
    "FIX" + "ME",
    "YOUR CODE" + " HERE",
    "raise NotImplemented" + "Error",
)


def discover_notebooks(selected_folder: str | None = None) -> list[Path]:
    """Return all course notebooks in deterministic learning order."""

    return [
        path
        for course_folder in COURSE_FOLDERS
        if selected_folder is None or course_folder == selected_folder
        for path in sorted((PROJECT_ROOT / course_folder).glob("*.ipynb"))
    ]


def validate_structure(path: Path) -> nbformat.NotebookNode:
    """Read and validate one notebook's schema and authored content."""

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if not notebook.cells:
        raise ValueError("notebook has no cells")
    source = "\n".join(cell.source for cell in notebook.cells)
    for marker in FORBIDDEN:
        if marker in source:
            raise ValueError(f"contains forbidden marker {marker!r}")
    if not any(cell.cell_type == "markdown" for cell in notebook.cells):
        raise ValueError("notebook has no Markdown cells")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and not cell.source.strip():
            raise ValueError(f"code cell {index} is empty")
    if re.search(r"[A-Za-z]:\\\\", source):
        raise ValueError("contains a Windows absolute path")
    return notebook


def execute_notebook(path: Path, timeout: int) -> float:
    """Execute a notebook from a fresh kernel and save verified outputs."""

    notebook = nbformat.read(path, as_version=4)
    started = time.perf_counter()
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, path)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run each notebook in a fresh kernel")
    parser.add_argument("--timeout", type=int, default=300, help="per-cell timeout in seconds")
    parser.add_argument(
        "--folder",
        choices=COURSE_FOLDERS,
        help="limit validation to one numbered course folder",
    )
    arguments = parser.parse_args()

    notebooks = discover_notebooks(arguments.folder)
    if not notebooks:
        print("No notebooks found", file=sys.stderr)
        return 1

    failures = 0
    for path in notebooks:
        relative = path.relative_to(PROJECT_ROOT)
        try:
            validate_structure(path)
            if arguments.execute:
                duration = execute_notebook(path, arguments.timeout)
                validate_structure(path)
                print(f"PASS {relative} ({duration:.1f}s)")
            else:
                print(f"PASS {relative}")
        except Exception as error:  # validation should report all files in one run
            failures += 1
            print(f"FAIL {relative}: {error}", file=sys.stderr)

    print(f"Checked {len(notebooks)} notebook(s); failures: {failures}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
