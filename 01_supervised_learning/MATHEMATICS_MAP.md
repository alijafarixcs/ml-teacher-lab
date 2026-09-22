# Required Mathematics Coverage Map

The course introduces each mathematical idea where it becomes useful. Start with the supervised lessons listed here, then follow the links into later sections for topics that naturally belong to unsupervised learning, deep learning, or reinforcement learning.

| Required topic | First focused treatment | Reinforcement or extension |
|---|---|---|
| Scalars, vectors, matrices, and tensors | `01_linear_regression.ipynb` introduces feature vectors, parameter vectors, and the design matrix | `03_deep_learning/01_tensors_and_linear_algebra.ipynb` extends arrays to tensors |
| Matrix multiplication | `01_linear_regression.ipynb` uses the matrix-vector product \(Xw\) | Deep-learning forward propagation reuses batched matrix multiplication |
| Dot products | `01_linear_regression.ipynb` connects one prediction to a weighted dot product | `07_support_vector_machine.ipynb` uses \(w^Tx\) for a separating boundary |
| Mean, variance, and standard deviation | `01_linear_regression.ipynb` defines regression baselines and error summaries | `04_naive_bayes.ipynb` estimates class-conditional Gaussian parameters |
| Probability and conditional probability | `02_logistic_regression.ipynb` interprets predicted class probabilities | `04_naive_bayes.ipynb` derives posterior class probabilities |
| Bayes' theorem | `04_naive_bayes.ipynb` includes a numerical Bayes calculation and Gaussian Naive Bayes | Gaussian mixture and Bayesian updates reinforce the idea |
| Derivatives and partial derivatives | `01_linear_regression.ipynb` derives per-parameter derivatives of mean-squared error | Deep-learning lessons build toward backpropagation |
| Gradients | `01_linear_regression.ipynb` implements batch gradient descent | `02_logistic_regression.ipynb` and the deep-learning section reuse gradient updates |
| Chain rule | Introduced conceptually when logistic loss is differentiated | `03_deep_learning/04_gradient_descent_and_backpropagation.ipynb` gives the focused treatment |
| Loss and cost functions | `01_linear_regression.ipynb` uses MSE; `02_logistic_regression.ipynb` uses binary cross-entropy | Later sections compare task-specific objectives |
| Optimization | `01_linear_regression.ipynb` implements gradient descent and learning-rate experiments | Deep learning compares optimizers and regularization |
| Eigenvalues and eigenvectors | `02_unsupervised_learning/05_principal_component_analysis.ipynb` introduces eigen-directions only when PCA needs them | The tensors/linear-algebra lesson provides supporting intuition |
| Distance metrics | `03_k_nearest_neighbors.ipynb` implements Euclidean and Manhattan distance | Clustering lessons compare geometry and scale sensitivity |
| Entropy and information gain | `05_decision_tree.ipynb` computes both from scratch | Random forests reuse randomized tree splits |
| Probability distributions | `04_naive_bayes.ipynb` uses Gaussian class-conditional distributions | Gaussian mixture models deepen distribution modeling |
| Expected value | `06_random_forest.ipynb` motivates averaging many randomized estimators | Bandits and value functions interpret long-run expected reward |
| Markov property | `04_reinforcement_learning/02_markov_decision_processes.ipynb` introduces state sufficiency | Dynamic programming and temporal-difference methods depend on it |
| Bellman equations | `04_reinforcement_learning/02_markov_decision_processes.ipynb` introduces the recursive value relation | Dynamic programming, Q-learning, and SARSA implement Bellman-style backups |
| Discounted returns | `04_reinforcement_learning/02_markov_decision_processes.ipynb` defines finite and continuing returns | Monte Carlo and policy-gradient lessons estimate those returns |

Every focused lesson follows the same learning pattern: intuition, symbols, a hand-worked numerical example, code connection, from-scratch implementation, standard-library implementation, visual experiment, exercises, and solutions.
