import base64
import re
import struct
from pathlib import Path

import nbformat
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = [
    "01_supervised_learning",
    "02_unsupervised_learning",
    "03_deep_learning",
    "04_reinforcement_learning",
    "05_final_projects",
]
EXPECTED_PARTS_2_TO_5 = {
    "02_unsupervised_learning": {
        "00_unsupervised_learning_overview.ipynb",
        "01_k_means_clustering.ipynb",
        "02_hierarchical_clustering.ipynb",
        "03_dbscan.ipynb",
        "04_gaussian_mixture_models.ipynb",
        "05_principal_component_analysis.ipynb",
        "06_anomaly_detection.ipynb",
        "07_clustering_evaluation.ipynb",
        "08_unsupervised_capstone_project.ipynb",
    },
    "03_deep_learning": {
        "00_deep_learning_overview.ipynb",
        "01_tensors_and_linear_algebra.ipynb",
        "02_perceptron.ipynb",
        "03_neural_network_from_scratch.ipynb",
        "04_gradient_descent_and_backpropagation.ipynb",
        "05_pytorch_fundamentals.ipynb",
        "06_multilayer_perceptron.ipynb",
        "07_convolutional_neural_network.ipynb",
        "08_recurrent_neural_network.ipynb",
        "09_lstm_and_gru.ipynb",
        "10_attention_and_transformers.ipynb",
        "11_regularization_and_optimization.ipynb",
        "12_deep_learning_capstone_project.ipynb",
    },
    "04_reinforcement_learning": {
        "00_reinforcement_learning_overview.ipynb",
        "01_multi_armed_bandits.ipynb",
        "02_markov_decision_processes.ipynb",
        "03_dynamic_programming.ipynb",
        "04_monte_carlo_methods.ipynb",
        "05_q_learning.ipynb",
        "06_sarsa.ipynb",
        "07_deep_q_network.ipynb",
        "08_policy_gradients.ipynb",
        "09_actor_critic.ipynb",
        "10_reinforcement_learning_capstone.ipynb",
    },
    "05_final_projects": {
        "01_end_to_end_tabular_project.ipynb",
        "02_customer_segmentation_project.ipynb",
        "03_image_classification_project.ipynb",
        "04_reinforcement_learning_agent.ipynb",
    },
}
EXPECTED_SUPERVISED = {
    "00_supervised_learning_overview.ipynb",
    "01_linear_regression.ipynb",
    "02_logistic_regression.ipynb",
    "03_k_nearest_neighbors.ipynb",
    "04_naive_bayes.ipynb",
    "05_decision_tree.ipynb",
    "06_random_forest.ipynb",
    "07_support_vector_machine.ipynb",
    "08_gradient_boosting.ipynb",
    "09_model_evaluation.ipynb",
    "10_supervised_capstone_project.ipynb",
}
REQUIRED_MATHEMATICS = (
    "Scalars, vectors, matrices, and tensors",
    "Matrix multiplication",
    "Dot products",
    "Mean, variance, and standard deviation",
    "Probability and conditional probability",
    "Bayes' theorem",
    "Derivatives and partial derivatives",
    "Gradients",
    "Chain rule",
    "Loss and cost functions",
    "Optimization",
    "Eigenvalues and eigenvectors",
    "Distance metrics",
    "Entropy and information gain",
    "Probability distributions",
    "Expected value",
    "Markov property",
    "Bellman equations",
    "Discounted returns",
)
FORBIDDEN_MARKERS = (
    "TO" + "DO",
    "FIX" + "ME",
    "YOUR CODE" + " HERE",
    "raise NotImplemented" + "Error",
)
EXPECTED_PROJECT_PATHS = (
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    "data/raw",
    "data/processed",
    "data/README.md",
    "images",
    "src/__init__.py",
    "src/data_utils.py",
    "src/plotting.py",
    "src/metrics.py",
    "src/notebook_utils.py",
    "tests/test_data_utils.py",
    "tests/test_metrics.py",
    "tests/test_notebooks.py",
)
DEEP_MODEL_NOTEBOOKS = (
    "02_perceptron.ipynb",
    "03_neural_network_from_scratch.ipynb",
    "05_pytorch_fundamentals.ipynb",
    "06_multilayer_perceptron.ipynb",
    "07_convolutional_neural_network.ipynb",
    "08_recurrent_neural_network.ipynb",
    "09_lstm_and_gru.ipynb",
    "11_regularization_and_optimization.ipynb",
    "12_deep_learning_capstone_project.ipynb",
)
CAPSTONE_NOTEBOOKS = (
    "01_supervised_learning/10_supervised_capstone_project.ipynb",
    "02_unsupervised_learning/08_unsupervised_capstone_project.ipynb",
    "03_deep_learning/12_deep_learning_capstone_project.ipynb",
    "04_reinforcement_learning/10_reinforcement_learning_capstone.ipynb",
    "05_final_projects/01_end_to_end_tabular_project.ipynb",
    "05_final_projects/02_customer_segmentation_project.ipynb",
    "05_final_projects/03_image_classification_project.ipynb",
    "05_final_projects/04_reinforcement_learning_agent.ipynb",
)


def notebook_paths() -> list[Path]:
    return sorted(
        path
        for directory in NOTEBOOK_DIRS
        for path in (PROJECT_ROOT / directory).glob("*.ipynb")
    )


def test_course_contains_notebooks():
    assert notebook_paths(), "No notebooks were found"


def test_required_project_structure_exists():
    for relative_path in EXPECTED_PROJECT_PATHS:
        assert (PROJECT_ROOT / relative_path).exists(), f"Missing {relative_path}"


def test_parts_2_to_5_are_complete():
    for folder, expected_names in EXPECTED_PARTS_2_TO_5.items():
        actual_names = {
            path.name for path in (PROJECT_ROOT / folder).glob("*.ipynb")
        }
        assert actual_names == expected_names, (
            f"{folder} mismatch: missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )


def test_supervised_learning_is_complete():
    actual_names = {
        path.name for path in (PROJECT_ROOT / "01_supervised_learning").glob("*.ipynb")
    }
    assert actual_names == EXPECTED_SUPERVISED, (
        f"01_supervised_learning mismatch: "
        f"missing={sorted(EXPECTED_SUPERVISED - actual_names)}, "
        f"unexpected={sorted(actual_names - EXPECTED_SUPERVISED)}"
    )


def test_required_mathematics_has_an_explicit_coverage_map():
    map_text = (
        PROJECT_ROOT / "01_supervised_learning" / "MATHEMATICS_MAP.md"
    ).read_text(encoding="utf-8")
    for topic in REQUIRED_MATHEMATICS:
        assert topic in map_text, f"Mathematics map is missing {topic!r}"


@pytest.mark.parametrize("path", notebook_paths(), ids=lambda path: path.name)
def test_notebook_json_and_content(path: Path):
    notebook = nbformat.read(path, as_version=4)
    assert notebook.cells, f"{path} has no cells"
    assert any(cell.cell_type == "code" for cell in notebook.cells)
    assert any(cell.cell_type == "markdown" for cell in notebook.cells)
    combined_source = "\n".join(cell.source for cell in notebook.cells)
    for marker in FORBIDDEN_MARKERS:
        assert marker not in combined_source, f"{path} contains {marker!r}"
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.source.strip(), f"{path} contains an empty code cell"
    assert not re.search(
        r"(?m)^\s*pass\s*(?:#.*)?$", combined_source
    ), f"{path} contains a bare pass statement"
    assert not re.search(
        r"[\"'][A-Za-z]:[\\/]", combined_source
    ), f"{path} contains a Windows absolute path"


@pytest.mark.parametrize("path", notebook_paths(), ids=lambda path: path.name)
def test_executed_notebooks_have_no_error_outputs(path: Path):
    notebook = nbformat.read(path, as_version=4)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert not errors, f"{path} contains saved execution errors"
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    execution_counts = [cell.execution_count for cell in code_cells]
    assert all(count is not None for count in execution_counts), (
        f"{path} contains unexecuted code cells"
    )
    assert execution_counts == sorted(execution_counts), (
        f"{path} was not saved in top-to-bottom execution order"
    )


@pytest.mark.parametrize("path", notebook_paths(), ids=lambda path: path.name)
def test_notebooks_have_readable_titles_math_and_rendered_figures(path: Path):
    notebook = nbformat.read(path, as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    assert re.search(r"(?m)^#\s+\S", markdown), f"{path} has no title"
    assert re.search(r"\$[^$]+\$|\\\[|\\begin\{", markdown), (
        f"{path} has no rendered-mathematics source"
    )
    figures = [
        output.data["image/png"]
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
    ]
    assert figures, f"{path} has no saved rendered figure"
    for encoded_figure in figures:
        raw_figure = base64.b64decode(encoded_figure)
        assert raw_figure.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", raw_figure[16:24])
        assert width >= 600 and height >= 300, (
            f"{path} contains an undersized figure ({width}x{height})"
        )


@pytest.mark.parametrize(
    "path",
    [
        PROJECT_ROOT / "01_supervised_learning" / name
        for name in sorted(EXPECTED_SUPERVISED - {"00_supervised_learning_overview.ipynb"})
    ]
    + [
        PROJECT_ROOT / folder / name
        for folder, names in EXPECTED_PARTS_2_TO_5.items()
        for name in sorted(names)
    ],
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_algorithm_lessons_have_complete_teaching_structure_and_figures(path: Path):
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    required_sections = (
        "Learning objectives",
        "Prerequisites",
        "Problem definition",
        "Real-world use cases",
        "Core intuition",
        "Mathematical foundation",
        "Visual explanation",
        "Algorithm steps",
        "Implementation from scratch",
        "Implementation using a standard library",
        "Dataset loading and exploration",
        "Data preprocessing",
        "Model training",
        "Predictions",
        "Evaluation",
        "Visualization of results",
        "Hyperparameter experiments",
        "Common mistakes",
        "Advantages and disadvantages",
        "When to use and when not to use",
        "Exercises",
        "Exercise solutions",
        "Summary and key takeaways",
        "Further reading",
    )
    for section in required_sections:
        assert section in source, f"{path} is missing section {section!r}"
    for explanation in (
        "Every symbol",
        "Why the equation is needed",
        "Small numerical example",
        "Connection to Python",
        "Scratch-to-library comparison",
        "Performance",
        "Numerical stability",
        "Flexibility",
        "Complexity",
    ):
        assert explanation in source, f"{path} is missing {explanation!r}"
    for exercise_type in (
        "**Conceptual:**",
        "**Mathematical/hand calculation:**",
        "**Coding/experimentation:**",
    ):
        assert exercise_type in source, f"{path} is missing {exercise_type!r}"
    figure_outputs = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    assert figure_outputs, f"{path} has no saved rendered figure"


def test_required_course_topics_are_present_in_authored_notebooks():
    source = "\n".join(
        cell.source
        for path in notebook_paths()
        for cell in nbformat.read(path, as_version=4).cells
    )
    required_patterns = {
        "scalars, vectors, matrices, tensors": r"scalar.*vector.*matri.*tensor",
        "matrix multiplication": r"matrix multiplication|matrix product",
        "dot products": r"dot product",
        "mean, variance, standard deviation": r"mean[\s\S]{0,120}variance[\s\S]{0,120}standard deviation",
        "conditional probability": r"conditional probability|P\(.+\|.+\)",
        "Bayes theorem": r"Bayes.? theorem",
        "partial derivatives": r"partial derivative|\\partial",
        "gradients and chain rule": r"chain rule",
        "optimization": r"optimization|optimizer",
        "eigenvalues and eigenvectors": r"eigenvalue.*eigenvector|eigenvector.*eigenvalue",
        "distance metrics": r"Euclidean.*Manhattan|Manhattan.*Euclidean",
        "entropy and information gain": r"entropy.*information gain|information gain.*entropy",
        "probability distributions": r"probability distribution",
        "expected value": r"expected value|expectation",
        "Markov property": r"Markov property",
        "Bellman equations": r"Bellman(?: expectation| optimality)? equation",
        "discounted returns": r"discounted return",
    }
    for topic, pattern in required_patterns.items():
        assert re.search(pattern, source, re.IGNORECASE), f"Missing topic: {topic}"


@pytest.mark.parametrize("name", DEEP_MODEL_NOTEBOOKS)
def test_trained_deep_models_report_all_required_diagnostics(name: str):
    path = PROJECT_ROOT / "03_deep_learning" / name
    source = "\n".join(cell.source for cell in nbformat.read(path, 4).cells)
    for field in (
        "input_shape",
        "intermediate_shape",
        "output_shape",
        "trainable_parameters",
        "final_training_loss",
        "final_validation_loss",
        "validation_",
    ):
        assert field in source, f"{path} is missing deep-model diagnostic {field!r}"


def test_attention_example_declares_its_forward_only_scope():
    path = PROJECT_ROOT / "03_deep_learning" / "10_attention_and_transformers.ipynb"
    source = "\n".join(cell.source for cell in nbformat.read(path, 4).cells)
    assert "forward-only educational attention example" in source


@pytest.mark.parametrize(
    "name",
    sorted(EXPECTED_PARTS_2_TO_5["04_reinforcement_learning"]),
)
def test_rl_notebooks_render_all_learning_diagnostics(name: str):
    path = PROJECT_ROOT / "04_reinforcement_learning" / name
    source = "\n".join(cell.source for cell in nbformat.read(path, 4).cells)
    assert source.count("plot_rl_metrics(") >= 2, (
        f"{path} does not call the shared reward/exploration/length diagnostic"
    )


@pytest.mark.parametrize(
    "relative_path",
    (
        "04_reinforcement_learning/00_reinforcement_learning_overview.ipynb",
        "04_reinforcement_learning/05_q_learning.ipynb",
        "04_reinforcement_learning/06_sarsa.ipynb",
        "04_reinforcement_learning/10_reinforcement_learning_capstone.ipynb",
        "05_final_projects/04_reinforcement_learning_agent.ipynb",
    ),
)
def test_tabular_rl_shows_q_tables_before_during_and_after(relative_path: str):
    source = "\n".join(
        cell.source
        for cell in nbformat.read(PROJECT_ROOT / relative_path, as_version=4).cells
    )
    assert re.search(r"Q-table (?:before|initial)", source, re.IGNORECASE)
    assert re.search(r"Q-table (?:during|after \d+)", source, re.IGNORECASE)
    assert re.search(r"Q-table after (?:training|\d+)", source, re.IGNORECASE)


@pytest.mark.parametrize("relative_path", CAPSTONE_NOTEBOOKS)
def test_capstones_include_the_complete_delivery_checklist(relative_path: str):
    source = "\n".join(
        cell.source
        for cell in nbformat.read(PROJECT_ROOT / relative_path, as_version=4).cells
    )
    for item in (
        "Capstone delivery checklist",
        "Business or real-world problem and success metric",
        "Dataset description and exploratory data analysis",
        "Data cleaning and feature engineering",
        "Baseline and multiple-model comparison",
        "Hyperparameter tuning",
        "Final evaluation and error analysis",
        "Limitations and possible improvements",
        "Final conclusion",
    ):
        assert item in source, f"{relative_path} is missing {item!r}"
