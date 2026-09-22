"""Reusable nbformat factory for the remaining course tutorials."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TypedDict

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LessonSpec(TypedDict):
    title: str
    objectives: list[str]
    prerequisites: str
    problem: str
    use_cases: list[str]
    intuition: str
    equation: str
    symbols: list[str]
    why_equation: str
    numerical_example: str
    python_connection: str
    steps: list[str]
    scratch_description: str
    library_description: str
    dataset_description: str
    preprocessing_description: str
    setup_code: str
    scratch_code: str
    experiment_code: str
    interpretation: str
    hyperparameter_text: str
    hyperparameter_code: str
    mistakes: list[str]
    advantages: list[str]
    disadvantages: list[str]
    use_when: str
    avoid_when: str
    exercises: list[str]
    solutions: list[str]
    takeaways: list[str]
    further_reading: list[str]


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def bullets(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def numbered(values: list[str]) -> str:
    return "\n".join(f"{index}. {value}" for index, value in enumerate(values, start=1))


COMMON_SETUP = """
from pathlib import Path
import sys

project_root = next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "pyproject.toml").exists()
)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.notebook_utils import RANDOM_SEED, set_seed
from src.plotting import COURSE_COLORS, set_course_style

set_seed(RANDOM_SEED)
set_course_style()
rng = np.random.default_rng(RANDOM_SEED)
print(f"Python lesson ready; reproducible seed = {RANDOM_SEED}")
"""


def make_lesson(spec: LessonSpec) -> nbf.NotebookNode:
    """Create one complete tutorial notebook from a topic-specific specification."""

    cells = [
        md(
            f"""
            # {spec['title']}

            ## 1. Learning objectives

            {bullets(spec['objectives'])}

            ## 2. Prerequisites

            {spec['prerequisites']}
            """
        ),
        code(COMMON_SETUP + "\n" + spec["setup_code"]),
        md(
            f"""
            ## 3. Problem definition

            {spec['problem']}

            ## 4. Real-world use cases

            {bullets(spec['use_cases'])}

            ## 5. Core intuition in simple language

            {spec['intuition']}
            """
        ),
        md(
            f"""
            ## 6. Mathematical foundation

            {spec['equation']}

            **Every symbol:**

            {bullets(spec['symbols'])}

            **Why the equation is needed:** {spec['why_equation']}

            **Small numerical example:** {spec['numerical_example']}

            **Connection to Python:** {spec['python_connection']}
            """
        ),
        md(
            f"""
            ## 7. Visual explanation

            The executable figure below is chosen to reveal the structure this method learns. Read positions and lengths first; color is a supporting cue rather than the only encoding.

            ## 8. Algorithm steps

            {numbered(spec['steps'])}
            """
        ),
        md(
            f"""
            ## 9. Implementation from scratch

            {spec['scratch_description']}

            The simplified implementation exposes the central mechanism. It favors readability over production-scale numerical safeguards.
            """
        ),
        code(spec["scratch_code"]),
        md(
            f"""
            ## 10. Implementation using a standard library

            {spec['library_description']}

            ## 11. Dataset loading and exploration

            {spec['dataset_description']}

            ## 12. Data preprocessing

            {spec['preprocessing_description']}

            ## 13. Model training

            The next cell trains both the transparent and library-backed versions where the topic has a trainable model.

            ## 14. Predictions

            Predictions, transformed representations, actions, or value estimates are kept in named arrays so their shapes can be inspected.

            ## 15. Evaluation

            Evaluation uses a metric appropriate to the learning setting and always retains a simple baseline or reasonableness check.

            ## 16. Visualization of results

            Every chart has a descriptive title and labeled axes. The interpretation immediately below describes what the marks mean.
            """
        ),
        code(spec["experiment_code"]),
        md(
            f"""
            **How to interpret the result:** {spec['interpretation']}

            **Scratch-to-library comparison:**

            - **Performance:** the educational implementation favors visible steps; the library implementation uses optimized, vectorized, and often compiled routines.
            - **Numerical stability:** production libraries add convergence checks, guarded arithmetic, and tested handling of edge cases that the compact scratch version intentionally omits.
            - **Flexibility:** scratch code is easy to inspect and change for one lesson, while the library version supports reusable estimator interfaces, pipelines, and broader input cases.
            - **Complexity:** the scratch implementation makes the central mechanism explicit; the library hides additional validation and engineering complexity behind a small public API.

            Compare the measured outputs above rather than expecting exact equality: initialization, stopping rules, regularization, and implementation details can differ.
            """
        ),
        md(
            f"""
            ## 17. Hyperparameter experiments

            {spec['hyperparameter_text']}
            """
        ),
        code(spec["hyperparameter_code"]),
        md(
            f"""
            ## 18. Common mistakes

            {bullets(spec['mistakes'])}

            ## 19. Advantages and disadvantages

            **Advantages**

            {bullets(spec['advantages'])}

            **Disadvantages**

            {bullets(spec['disadvantages'])}

            ## 20. When to use and when not to use the algorithm

            **Use it when:** {spec['use_when']}

            **Do not use it when:** {spec['avoid_when']}
            """
        ),
        md(
            f"""
            ## 21. Exercises

            1. **Conceptual:** {spec['exercises'][0]}
            2. **Mathematical/hand calculation:** {spec['exercises'][1]}
            3. **Coding/experimentation:** {spec['exercises'][2]}
            """
        ),
        md(
            f"""
            <details>
            <summary><strong>22. Exercise solutions</strong></summary>

            1. {spec['solutions'][0]}
            2. {spec['solutions'][1]}
            3. {spec['solutions'][2]}

            </details>
            """
        ),
        md(
            f"""
            ## 23. Summary and key takeaways

            {bullets(spec['takeaways'])}

            ## 24. Further reading

            {bullets(spec['further_reading'])}
            """
        ),
    ]

    if "capstone" in spec["title"].lower() or "project" in spec["title"].lower():
        cells.insert(
            -1,
            md(
                """
                ## Capstone delivery checklist

                - **Business or real-world problem and success metric:** defined before modeling in the problem-definition section.
                - **Dataset description and exploratory data analysis:** documented beside the bounded dataset preview and audit outputs.
                - **Data cleaning and feature engineering:** performed after the split where fitting or target information could otherwise leak; choices are stated explicitly, including when no cleaning is required.
                - **Baseline and multiple-model comparison:** a simple reference model is evaluated before more flexible candidates.
                - **Hyperparameter tuning:** candidates are selected only with development or validation evidence.
                - **Final evaluation and error analysis:** the frozen choice is assessed on untouched data with a task-appropriate diagnostic plot.
                - **Limitations and possible improvements:** recorded with the disadvantages and interpretation so the result is not overstated.
                - **Final conclusion:** the summary connects the evidence, limitations, and next steps.
                """
            ),
        )

    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


def write_lesson(relative_path: str, spec: LessonSpec) -> Path:
    target = PROJECT_ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(make_lesson(spec), target)
    return target


def generic_exercises(concept_question: str, calculation: str, experiment: str) -> list[str]:
    return [concept_question, calculation, experiment]


def generic_solutions(concept: str, calculation: str, experiment: str) -> list[str]:
    return [concept, calculation, experiment]
