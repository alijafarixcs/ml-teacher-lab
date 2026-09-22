"""Build all unsupervised-learning notebooks."""

from __future__ import annotations

from course_factory import LessonSpec, write_lesson


CLUSTERING_GUIDE = "[scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html)"


def base_spec(**overrides) -> LessonSpec:
    spec = {
        "title": "Unsupervised Learning",
        "objectives": ["Explain the method intuitively", "Implement its core calculation", "Evaluate and visualize the result"],
        "prerequisites": "Basic NumPy arrays, means, and two-dimensional scatter plots.",
        "problem": "Discover structure in feature vectors when no target labels are supplied to the fitting algorithm.",
        "use_cases": ["Exploratory customer grouping", "Representation learning", "Quality-control pattern discovery"],
        "intuition": "The algorithm organizes observations using regularities in the features rather than an answer column.",
        "equation": r"$$d(\mathbf{x},\mathbf{c})=\sqrt{\sum_{j=1}^{p}(x_j-c_j)^2}.$$",
        "symbols": [r"$\mathbf{x}$ is one observation.", r"$\mathbf{c}$ is a reference point such as a centroid.", r"$p$ is the number of features.", r"$j$ indexes a feature.", r"$d$ is Euclidean distance."],
        "why_equation": "Distance turns similarity into a quantity the algorithm can compare.",
        "numerical_example": r"For $\mathbf{x}=(1,2)$ and $\mathbf{c}=(4,6)$, $d=\sqrt{3^2+4^2}=5$.",
        "python_connection": "np.linalg.norm(x - c) evaluates exactly this distance.",
        "steps": ["Represent each observation as a feature vector", "Scale features when units differ", "Fit the structure-discovery rule", "Evaluate stability and usefulness", "Inspect the result visually and with domain knowledge"],
        "scratch_description": "We implement the smallest useful version in NumPy so every update remains visible.",
        "library_description": "Scikit-learn adds input validation, efficient routines, and a consistent estimator API.",
        "dataset_description": "The lesson uses a small built-in or synthetic dataset with a fixed seed and no network dependency.",
        "preprocessing_description": "Features are standardized when distance or covariance would otherwise be dominated by measurement units.",
        "setup_code": "",
        "scratch_code": "def euclidean_distance(a, b):\n    return float(np.sqrt(np.sum((np.asarray(a) - np.asarray(b)) ** 2)))\n\nassert np.isclose(euclidean_distance([1, 2], [4, 6]), 5.0)",
        "experiment_code": "values = np.array([[1.0, 2.0], [4.0, 6.0]])\nprint('Distance:', euclidean_distance(values[0], values[1]))",
        "interpretation": "Smaller distance means greater similarity under the chosen feature representation.",
        "hyperparameter_text": "We vary the main complexity control and compare a quantitative score with the visual result.",
        "hyperparameter_code": "candidate_values = np.arange(2, 7)\nfigure, axis = plt.subplots()\naxis.plot(candidate_values, 1 / candidate_values, marker='o', color=COURSE_COLORS['blue'], label='Illustrative complexity')\naxis.set(title='Complexity changes with the hyperparameter', xlabel='Candidate value', ylabel='Illustrative score')\naxis.legend(); plt.show()",
        "mistakes": ["Treating cluster identifiers as ordered numbers", "Skipping scaling for distance-based methods", "Assuming an attractive grouping is automatically useful"],
        "advantages": ["Does not require labels", "Can expose previously unknown structure"],
        "disadvantages": ["Evaluation is indirect", "Results can be sensitive to representation and hyperparameters"],
        "use_when": "exploration, compression, or structure discovery is the goal and trustworthy labels are unavailable.",
        "avoid_when": "a supervised target directly represents the decision or the learned grouping has no domain interpretation.",
        "exercises": ["Why can cluster labels 0 and 1 not be interpreted as low and high?", "Compute the Euclidean distance between (0, 0) and (3, 4).", "Change the random seed and check whether the structure is stable."],
        "solutions": ["Cluster IDs are arbitrary names; permuting them does not change the grouping.", r"$\sqrt{3^2+4^2}=5$.", "Rerun with several seeds and compare scores plus assignments; instability is evidence that conclusions need caution."],
        "takeaways": ["Unsupervised methods learn from features without fitting to labels", "Scaling and representation strongly affect results", "Metrics, visualization, stability, and domain usefulness must be considered together"],
        "further_reading": [CLUSTERING_GUIDE],
    }
    spec.update(overrides)
    return spec


KMEANS_SCRATCH = r"""
def kmeans_scratch(features, n_clusters, max_iter=100, seed=42):
    generator = np.random.default_rng(seed)
    centroids = features[generator.choice(len(features), n_clusters, replace=False)].copy()
    for _ in range(max_iter):
        distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new_centroids = np.vstack([
            features[labels == cluster].mean(axis=0)
            if np.any(labels == cluster) else centroids[cluster]
            for cluster in range(n_clusters)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    inertia = float(np.sum((features - centroids[labels]) ** 2))
    return labels, centroids, inertia

tiny = np.array([[0., 0.], [0., 1.], [5., 5.], [5., 6.]])
tiny_labels, tiny_centroids, tiny_inertia = kmeans_scratch(tiny, 2)
print("Tiny centroids:", tiny_centroids.round(2).tolist(), "inertia:", round(tiny_inertia, 2))
"""


KMEANS_EXPERIMENT = r"""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

X, hidden_labels = make_blobs(n_samples=300, centers=3, cluster_std=1.05, random_state=RANDOM_SEED)
X_scaled = StandardScaler().fit_transform(X)
scratch_labels, scratch_centroids, scratch_inertia = kmeans_scratch(X_scaled, 3)
library_model = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_SEED).fit(X_scaled)
print("Shape:", X_scaled.shape)
print("Scratch/library agreement (ARI):", round(adjusted_rand_score(scratch_labels, library_model.labels_), 3))
print("Library silhouette:", round(silhouette_score(X_scaled, library_model.labels_), 3))

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for axis, labels, title in zip(
    axes,
    [scratch_labels, library_model.labels_],
    ["NumPy K-means", "scikit-learn K-means"],
):
    axis.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="cividis", s=28, alpha=.8)
    axis.set(title=title, xlabel="Standardized feature 1", ylabel="Standardized feature 2")
figure.tight_layout()
plt.show()
"""


KMEANS_HYPER = r"""
rows = []
for k in range(2, 7):
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(X_scaled)
    rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(X_scaled, model.labels_)})
scores = pd.DataFrame(rows)
display(scores.round(3))
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(scores["k"], scores["inertia"], marker="o", label="Inertia", color=COURSE_COLORS["blue"])
axes[0].set(title="Elbow method", xlabel="Number of clusters k", ylabel="Within-cluster squared distance")
axes[0].legend()
axes[1].plot(scores["k"], scores["silhouette"], marker="o", label="Silhouette", color=COURSE_COLORS["orange"])
axes[1].set(title="Silhouette by cluster count", xlabel="Number of clusters k", ylabel="Silhouette score")
axes[1].legend()
figure.tight_layout()
plt.show()
"""


def overview_spec() -> LessonSpec:
    return base_spec(
        title="Unsupervised Learning Overview",
        objectives=["Distinguish supervised and unsupervised learning", "Explain distance, centroids, density, and representations", "Evaluate a grouping without using hidden labels during fitting"],
        problem="Find useful patterns, groups, compact representations, or unusual observations using only feature columns.",
        intuition="Imagine sorting an unlabeled box of objects. Shape, size, and color suggest possible organizations, but no answer key says which organization is best.",
        scratch_code=KMEANS_SCRATCH,
        experiment_code=KMEANS_EXPERIMENT,
        hyperparameter_code=KMEANS_HYPER,
        interpretation="Both implementations recover the same broad groups. Hidden synthetic labels are used only for a post-hoc teaching check, never to fit K-means.",
        further_reading=[CLUSTERING_GUIDE, "[scikit-learn decomposition guide](https://scikit-learn.org/stable/modules/decomposition.html)"],
    )


def kmeans_spec() -> LessonSpec:
    return base_spec(
        title="K-means Clustering",
        objectives=["Derive assignment and centroid updates", "Implement K-means with NumPy", "Choose k with inertia and silhouette evidence"],
        problem="Partition observations into a chosen number of compact groups so each observation is close to its assigned centroid.",
        use_cases=["Customer segmentation", "Image color quantization", "Prototype discovery"],
        intuition="Place k movable pins in the data, assign every point to its nearest pin, move each pin to the mean of its assigned points, and repeat.",
        equation=r"$$J=\sum_{i=1}^{n}\lVert\mathbf{x}_i-\boldsymbol{\mu}_{z_i}\rVert_2^2.$$",
        symbols=[r"$J$ is inertia, the objective to minimize.", r"$n$ is the number of observations.", r"$\mathbf{x}_i$ is observation $i$.", r"$z_i$ is its assigned cluster.", r"$\boldsymbol{\mu}_{z_i}$ is that cluster's centroid."],
        why_equation="It measures total within-cluster compactness and directly produces the assignment and mean-update rules.",
        numerical_example="For points 1 and 3 assigned to centroid 2, their contribution is (1-2)^2+(3-2)^2=2.",
        python_connection="np.sum((X - centroids[labels]) ** 2) computes inertia.",
        scratch_code=KMEANS_SCRATCH,
        experiment_code=KMEANS_EXPERIMENT,
        hyperparameter_text="The elbow in inertia and the peak silhouette provide complementary evidence for k; neither proves business usefulness.",
        hyperparameter_code=KMEANS_HYPER,
        interpretation="Near-identical partitions show that the scratch update captures K-means' core mechanism; cluster numbers themselves remain arbitrary.",
    )


def hierarchical_spec() -> LessonSpec:
    scratch = r"""
from scipy.spatial.distance import cdist

def single_linkage_scratch(features, target_clusters):
    clusters = [[index] for index in range(len(features))]
    while len(clusters) > target_clusters:
        best = (np.inf, 0, 1)
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                distance = cdist(features[clusters[left]], features[clusters[right]]).min()
                if distance < best[0]:
                    best = (distance, left, right)
        _, left, right = best
        clusters[left] = clusters[left] + clusters[right]
        clusters.pop(right)
    labels = np.empty(len(features), dtype=int)
    for cluster_id, members in enumerate(clusters):
        labels[members] = cluster_id
    return labels

tiny = np.array([[0., 0.], [.2, .1], [4., 4.], [4.2, 4.1]])
print("Tiny single-link labels:", single_linkage_scratch(tiny, 2))
"""
    experiment = r"""
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

X, hidden_labels = make_blobs(n_samples=140, centers=3, cluster_std=.75, random_state=RANDOM_SEED)
X_scaled = StandardScaler().fit_transform(X)
scratch_labels = single_linkage_scratch(X_scaled, 3)
model = AgglomerativeClustering(n_clusters=3, linkage="single").fit(X_scaled)
print("Scratch/library ARI:", round(adjusted_rand_score(scratch_labels, model.labels_), 3))
print("Silhouette:", round(silhouette_score(X_scaled, model.labels_), 3))

figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=model.labels_, cmap="cividis", s=28)
axes[0].set(title="Agglomerative clusters", xlabel="Standardized feature 1", ylabel="Standardized feature 2")
dendrogram(linkage(X_scaled, method="ward"), truncate_mode="lastp", p=18, ax=axes[1], color_threshold=None)
axes[1].set(title="Truncated Ward dendrogram", xlabel="Merged group", ylabel="Merge distance")
figure.tight_layout()
plt.show()
"""
    hyper = r"""
rows = []
for linkage_name in ["single", "complete", "average", "ward"]:
    labels = AgglomerativeClustering(n_clusters=3, linkage=linkage_name).fit_predict(X_scaled)
    rows.append({"linkage": linkage_name, "silhouette": silhouette_score(X_scaled, labels)})
comparison = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
display(comparison.round(3))
figure, axis = plt.subplots()
axis.bar(comparison["linkage"], comparison["silhouette"], color=COURSE_COLORS["blue"])
axis.set(title="Linkage choice changes cluster compactness", xlabel="Linkage method", ylabel="Silhouette score")
plt.show()
"""
    return base_spec(
        title="Hierarchical Clustering",
        objectives=["Explain agglomerative merging", "Implement single linkage", "Read a dendrogram and compare linkage rules"],
        problem="Build a nested hierarchy of groups rather than committing immediately to one flat partition.",
        intuition="Begin with one cluster per observation and repeatedly join the two closest clusters; the full merge history forms a tree.",
        equation=r"$$D_{single}(A,B)=\min_{\mathbf{x}\in A,\mathbf{y}\in B}d(\mathbf{x},\mathbf{y}).$$",
        symbols=[r"$A$ and $B$ are clusters.", r"$\mathbf{x}$ and $\mathbf{y}$ are member observations.", r"$d$ is point distance.", r"$D_{single}$ is the closest cross-cluster distance."],
        why_equation="A linkage rule defines what closest clusters means.",
        numerical_example="If cross-cluster distances are 2, 5, 7, and 9, single linkage uses 2.",
        python_connection="cdist(X[A], X[B]).min() is the direct translation.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Single, complete, average, and Ward linkage make different geometric assumptions.",
        hyperparameter_code=hyper,
        interpretation="The scatter plot shows one cut; the dendrogram retains the larger hierarchy and merge distances.",
        exercises=["Why can single linkage create chains?", "Given cross-distances [2, 4, 6], find single and complete linkage.", "Change the dendrogram cut and compare the number of groups."],
        solutions=["One close pair can connect two otherwise distant groups, and repeated bridges form a chain.", "Single is 2; complete is 6.", "Choose a horizontal cut, count intersected branches, then check stability and domain usefulness."],
    )


def dbscan_spec() -> LessonSpec:
    scratch = r"""
def dbscan_scratch(features, eps, min_samples):
    labels = np.full(len(features), -99, dtype=int)
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    neighbors = [np.flatnonzero(distances[index] <= eps) for index in range(len(features))]
    cluster_id = 0
    for point in range(len(features)):
        if labels[point] != -99:
            continue
        if len(neighbors[point]) < min_samples:
            labels[point] = -1
            continue
        labels[point] = cluster_id
        seeds = list(map(int, neighbors[point]))
        cursor = 0
        while cursor < len(seeds):
            candidate = seeds[cursor]
            if labels[candidate] == -1:
                labels[candidate] = cluster_id
            if labels[candidate] == -99:
                labels[candidate] = cluster_id
                if len(neighbors[candidate]) >= min_samples:
                    seeds.extend(int(value) for value in neighbors[candidate] if int(value) not in seeds)
            cursor += 1
        cluster_id += 1
    labels[labels == -99] = -1
    return labels

print("Scratch DBSCAN defined; -1 is the noise label.")
"""
    experiment = r"""
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

X, hidden_labels = make_moons(n_samples=260, noise=.07, random_state=RANDOM_SEED)
X_scaled = StandardScaler().fit_transform(X)
scratch_labels = dbscan_scratch(X_scaled, eps=.25, min_samples=5)
library_labels = DBSCAN(eps=.25, min_samples=5).fit_predict(X_scaled)
cluster_mask = library_labels != -1
print("Scratch/library ARI:", round(adjusted_rand_score(scratch_labels, library_labels), 3))
print("Clusters:", len(set(library_labels)) - (-1 in library_labels), "noise:", int((library_labels == -1).sum()))
print("Non-noise silhouette:", round(silhouette_score(X_scaled[cluster_mask], library_labels[cluster_mask]), 3))

figure, axis = plt.subplots()
axis.scatter(X_scaled[:, 0], X_scaled[:, 1], c=library_labels, cmap="cividis", s=30)
axis.scatter(X_scaled[~cluster_mask, 0], X_scaled[~cluster_mask, 1], facecolors="none", edgecolors=COURSE_COLORS["orange"], s=70, label="Noise")
axis.set(title="DBSCAN finds curved dense regions", xlabel="Standardized feature 1", ylabel="Standardized feature 2")
axis.legend()
plt.show()
"""
    hyper = r"""
rows = []
for eps in [.15, .20, .25, .30, .40]:
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_scaled)
    rows.append({"eps": eps, "clusters": len(set(labels)) - (-1 in labels), "noise_points": int((labels == -1).sum())})
results = pd.DataFrame(rows)
display(results)
figure, axis = plt.subplots()
axis.plot(results["eps"], results["noise_points"], marker="o", color=COURSE_COLORS["orange"], label="Noise points")
axis.set(title="Neighborhood radius controls noise labeling", xlabel="eps radius", ylabel="Number of noise points")
axis.legend()
plt.show()
"""
    return base_spec(
        title="DBSCAN: Density-Based Clustering",
        objectives=["Define core, border, and noise points", "Implement density expansion", "Tune neighborhood radius and minimum samples"],
        problem="Find arbitrarily shaped dense regions while labeling isolated observations as noise.",
        intuition="A point starts or expands a cluster when enough neighbors fit inside a small radius; sparse points remain unassigned.",
        equation=r"$$N_{\varepsilon}(\mathbf{x})=\{\mathbf{y}:d(\mathbf{x},\mathbf{y})\leq\varepsilon\}.$$",
        symbols=[r"$\mathbf{x}$ is the current point.", r"$\mathbf{y}$ is another observation.", r"$d$ is distance.", r"$\varepsilon$ is the radius.", r"$N_\varepsilon$ is the neighbor set."],
        why_equation="Counting this set determines whether a point is a core point.",
        numerical_example="If distances are [0, .1, .3, .8] and epsilon=.35, the first three points are neighbors.",
        python_connection="np.flatnonzero(distances <= eps) returns the neighborhood.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Smaller epsilon marks more points as noise; larger epsilon can merge distinct dense regions.",
        hyperparameter_code=hyper,
        interpretation="DBSCAN follows the two-moons geometry without requiring a cluster count; outlined observations are noise.",
        advantages=["Finds non-spherical shapes", "Identifies noise explicitly", "Does not require a cluster count"],
        disadvantages=["Sensitive to scaling and density variation", "Neighborhood searches are difficult in high dimensions"],
    )


def gmm_spec() -> LessonSpec:
    scratch = r"""
from scipy.stats import norm

def gmm_1d_em(values, iterations=50):
    means = np.array([np.percentile(values, 25), np.percentile(values, 75)], dtype=float)
    variances = np.array([values.var(), values.var()], dtype=float)
    weights = np.array([.5, .5])
    for _ in range(iterations):
        likelihood = np.column_stack([
            weights[k] * norm.pdf(values, means[k], np.sqrt(variances[k]) + 1e-9)
            for k in range(2)
        ])
        responsibilities = likelihood / likelihood.sum(axis=1, keepdims=True)
        counts = responsibilities.sum(axis=0)
        weights = counts / len(values)
        means = (responsibilities * values[:, None]).sum(axis=0) / counts
        variances = (responsibilities * (values[:, None] - means) ** 2).sum(axis=0) / counts
    return weights, means, variances, responsibilities

small_values = np.array([-2.1, -1.9, -1.7, 1.8, 2.0, 2.2])
print("Scratch means:", gmm_1d_em(small_values)[1].round(2))
"""
    experiment = r"""
from sklearn.mixture import GaussianMixture

values = np.concatenate([rng.normal(-2, .55, 160), rng.normal(2.2, .8, 240)])
weights, means, variances, responsibilities = gmm_1d_em(values)
model = GaussianMixture(n_components=2, random_state=RANDOM_SEED).fit(values[:, None])
print("Scratch means:", np.sort(means).round(3))
print("Library means:", np.sort(model.means_.ravel()).round(3))
print("Responsibility row sums:", responsibilities[:3].sum(axis=1).round(6))

grid = np.linspace(values.min() - 1, values.max() + 1, 400)
density = np.exp(model.score_samples(grid[:, None]))
figure, axis = plt.subplots()
axis.hist(values, bins=32, density=True, alpha=.35, color=COURSE_COLORS["blue"], label="Observed values")
axis.plot(grid, density, color=COURSE_COLORS["orange"], label="Fitted mixture density")
axis.set(title="A Gaussian mixture models overlapping subpopulations", xlabel="Observed value", ylabel="Probability density")
axis.legend()
plt.show()
"""
    hyper = r"""
rows = []
for components in range(1, 6):
    candidate = GaussianMixture(n_components=components, random_state=RANDOM_SEED).fit(values[:, None])
    rows.append({"components": components, "BIC": candidate.bic(values[:, None]), "AIC": candidate.aic(values[:, None])})
criteria = pd.DataFrame(rows)
display(criteria.round(1))
figure, axis = plt.subplots()
axis.plot(criteria["components"], criteria["BIC"], marker="o", label="BIC", color=COURSE_COLORS["blue"])
axis.plot(criteria["components"], criteria["AIC"], marker="s", linestyle="--", label="AIC", color=COURSE_COLORS["orange"])
axis.set(title="Information criteria penalize unnecessary components", xlabel="Gaussian components", ylabel="Criterion (lower is better)")
axis.legend()
plt.show()
"""
    return base_spec(
        title="Gaussian Mixture Models",
        objectives=["Interpret mixture weights and responsibilities", "Implement one-dimensional EM", "Select component count with AIC and BIC"],
        problem="Represent overlapping subpopulations with a weighted sum of probability distributions and soft memberships.",
        intuition="Each observation may partly belong to several bell-shaped groups; expectation estimates memberships and maximization updates groups.",
        equation=r"$$p(x)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x\mid\mu_k,\sigma_k^2).$$",
        symbols=[r"$x$ is an observation.", r"$K$ is the component count.", r"$\pi_k$ is a nonnegative weight and weights sum to one.", r"$\mu_k$ is the mean.", r"$\sigma_k^2$ is the variance.", r"$\mathcal{N}$ is the Gaussian density."],
        why_equation="The weighted density gives both a likelihood and soft component probabilities.",
        numerical_example="With weights .4 and .6 and component densities .2 and .5, p(x)=.4(.2)+.6(.5)=.38.",
        python_connection="np.exp(model.score_samples(X)) evaluates the fitted density.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="AIC and BIC trade fit against complexity; BIC penalizes extra parameters more strongly as sample size grows.",
        hyperparameter_code=hyper,
        interpretation="The density curve overlaps both populations, while responsibilities preserve uncertainty near their boundary.",
    )


def pca_spec() -> LessonSpec:
    scratch = r"""
def pca_scratch(features, components=2):
    mean = features.mean(axis=0)
    centered = features - mean
    covariance = centered.T @ centered / (len(features) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    directions = eigenvectors[:, order[:components]]
    transformed = centered @ directions
    explained_ratio = eigenvalues[order[:components]] / eigenvalues.sum()
    return transformed, directions, explained_ratio

tiny = np.array([[1., 1.], [2., 2.], [3., 3.]])
print("Tiny first-component scores:", pca_scratch(tiny, 1)[0].round(3).ravel())
"""
    experiment = r"""
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target
scratch_projection, directions, scratch_ratio = pca_scratch(X, 2)
model = PCA(n_components=2).fit(X)
library_projection = model.transform(X)
correlations = [abs(np.corrcoef(scratch_projection[:, i], library_projection[:, i])[0, 1]) for i in range(2)]
print("Input/projected shapes:", X.shape, library_projection.shape)
print("Explained variance:", model.explained_variance_ratio_.round(3), "component agreement:", np.round(correlations, 3))

figure, axis = plt.subplots()
scatter = axis.scatter(library_projection[:, 0], library_projection[:, 1], c=y, cmap="tab10", s=12, alpha=.7)
axis.set(title="Digits projected onto two principal components", xlabel="Principal component 1", ylabel="Principal component 2")
axis.legend(*scatter.legend_elements(), title="Digit", ncol=2, loc="best")
plt.show()
"""
    hyper = r"""
full_pca = PCA().fit(X)
cumulative = np.cumsum(full_pca.explained_variance_ratio_)
components_90 = int(np.searchsorted(cumulative, .90) + 1)
figure, axis = plt.subplots()
axis.plot(np.arange(1, len(cumulative) + 1), cumulative, color=COURSE_COLORS["blue"], label="Cumulative explained variance")
axis.axhline(.90, color=COURSE_COLORS["orange"], linestyle="--", label="90% target")
axis.axvline(components_90, color=COURSE_COLORS["olive"], linestyle=":", label=f"{components_90} components")
axis.set(title="PCA compression tradeoff", xlabel="Number of components", ylabel="Cumulative explained variance ratio")
axis.legend()
plt.show()
"""
    return base_spec(
        title="Principal Component Analysis",
        objectives=["Connect covariance with directions of variation", "Implement PCA using eigenvectors", "Choose a component count with explained variance"],
        problem="Compress correlated features into fewer orthogonal coordinates while retaining as much variance as possible.",
        intuition="Rotate the coordinate system so the first axis follows the widest spread and later axes follow remaining spread.",
        equation=r"$$\mathbf{C}\mathbf{v}=\lambda\mathbf{v}.$$",
        symbols=[r"$\mathbf{C}$ is the covariance matrix.", r"$\mathbf{v}$ is an eigenvector and principal direction.", r"$\lambda$ is its eigenvalue and captured variance."],
        why_equation="Covariance eigenvectors identify directions whose projected variance is measured by the eigenvalue.",
        numerical_example="For covariance diag(9,1), the first axis explains 9/(9+1)=90% of variance.",
        python_connection="np.linalg.eigh(covariance) returns eigenvalues and directions.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="More components preserve more information but reduce less dimensionality.",
        hyperparameter_code=hyper,
        interpretation="Nearby projected points share dominant pixel patterns, but overlap warns that two dimensions cannot preserve all information.",
        further_reading=["[scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)"],
    )


def anomaly_spec() -> LessonSpec:
    scratch = r"""
def zscore_anomalies(features, threshold=3.0):
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-12
    z_scores = np.abs((features - mean) / std)
    return (z_scores.max(axis=1) > threshold).astype(int), z_scores

example = np.array([[0.], [.1], [-.1], [8.]])
flags, scores = zscore_anomalies(example, threshold=1.5)
print("Example z-scores:", scores.ravel().round(2), "flags:", flags)
"""
    experiment = r"""
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support

normal = rng.normal(0, 1, size=(280, 2))
outliers = rng.uniform(-7, 7, size=(20, 2))
X = np.vstack([normal, outliers])
y = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(outliers), dtype=int)]
z_flags, z_values = zscore_anomalies(X, threshold=3.0)
model = IsolationForest(contamination=len(outliers) / len(X), random_state=RANDOM_SEED).fit(X)
forest_flags = (model.predict(X) == -1).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(y, forest_flags, average="binary", zero_division=0)
print(f"Isolation Forest precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for axis, flags, title in zip(axes, [z_flags, forest_flags], ["Z-score rule", "Isolation Forest"]):
    axis.scatter(X[:, 0], X[:, 1], c=flags, cmap="cividis", s=26)
    axis.set(title=title, xlabel="Feature 1", ylabel="Feature 2")
figure.tight_layout()
plt.show()
"""
    hyper = r"""
rows = []
for contamination in [.02, .05, .067, .10, .15]:
    candidate = IsolationForest(contamination=contamination, random_state=RANDOM_SEED).fit(X)
    flags = (candidate.predict(X) == -1).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, flags, average="binary", zero_division=0)
    rows.append({"contamination": contamination, "precision": precision, "recall": recall, "F1": f1})
results = pd.DataFrame(rows)
display(results.round(3))
figure, axis = plt.subplots()
axis.plot(results["contamination"], results["precision"], marker="o", label="Precision", color=COURSE_COLORS["blue"])
axis.plot(results["contamination"], results["recall"], marker="s", linestyle="--", label="Recall", color=COURSE_COLORS["orange"])
axis.set(title="Expected anomaly rate changes the error tradeoff", xlabel="Contamination", ylabel="Score")
axis.legend()
plt.show()
"""
    return base_spec(
        title="Anomaly Detection and Isolation Forest",
        objectives=["Define anomaly scores and thresholds", "Implement a z-score detector", "Evaluate Isolation Forest under rare positives"],
        problem="Identify unusual observations when examples of every abnormal pattern are scarce or unavailable.",
        intuition="Z-scores flag extreme coordinates; Isolation Forest asks how few random splits are needed to isolate a point.",
        equation=r"$$z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}.$$",
        symbols=[r"$x_{ij}$ is observation i's value for feature j.", r"$\mu_j$ is the feature mean.", r"$\sigma_j$ is its standard deviation.", r"$z_{ij}$ is standardized distance from the mean."],
        why_equation="It makes deviations comparable across features with different units.",
        numerical_example="If x=16, mean=10, and standard deviation=2, z=(16-10)/2=3.",
        python_connection="(X - X.mean(0)) / X.std(0) computes all z-scores.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Contamination controls the expected anomaly fraction and therefore changes precision and recall.",
        hyperparameter_code=hyper,
        interpretation="Isolation Forest captures jointly unusual points that may not exceed one coordinate's z-score threshold.",
        further_reading=["[scikit-learn outlier detection guide](https://scikit-learn.org/stable/modules/outlier_detection.html)"],
    )


def evaluation_spec() -> LessonSpec:
    scratch = r"""
def silhouette_scratch(features, labels):
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    values = []
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        a = distances[index, same].mean() if same.any() else 0.0
        b = min(
            distances[index, labels == other].mean()
            for other in np.unique(labels) if other != label
        )
        values.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(values))

tiny = np.array([[0., 0.], [0., 1.], [5., 5.], [5., 6.]])
print("Tiny silhouette:", round(silhouette_scratch(tiny, np.array([0, 0, 1, 1])), 3))
"""
    experiment = r"""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=220, centers=4, cluster_std=.8, random_state=RANDOM_SEED)
X_scaled = StandardScaler().fit_transform(X)
labels = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_SEED).fit_predict(X_scaled)
scratch = silhouette_scratch(X_scaled, labels)
library = silhouette_score(X_scaled, labels)
db = davies_bouldin_score(X_scaled, labels)
print(f"Silhouette scratch={scratch:.4f}, library={library:.4f}; Davies-Bouldin={db:.4f}")

figure, axis = plt.subplots()
axis.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="cividis", s=28)
axis.set(title="Partition being evaluated", xlabel="Standardized feature 1", ylabel="Standardized feature 2")
plt.show()
"""
    hyper = r"""
rows = []
for k in range(2, 8):
    labels_k = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit_predict(X_scaled)
    rows.append({"k": k, "silhouette": silhouette_score(X_scaled, labels_k), "Davies-Bouldin": davies_bouldin_score(X_scaled, labels_k)})
scores = pd.DataFrame(rows)
display(scores.round(3))
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(scores["k"], scores["silhouette"], marker="o", color=COURSE_COLORS["blue"], label="Higher is better")
axes[0].set(title="Silhouette comparison", xlabel="k", ylabel="Silhouette")
axes[0].legend()
axes[1].plot(scores["k"], scores["Davies-Bouldin"], marker="s", color=COURSE_COLORS["orange"], label="Lower is better")
axes[1].set(title="Davies-Bouldin comparison", xlabel="k", ylabel="Davies-Bouldin")
axes[1].legend()
figure.tight_layout()
plt.show()
"""
    return base_spec(
        title="Clustering Evaluation",
        objectives=["Compute silhouette from pairwise distances", "Interpret Davies-Bouldin and inertia", "Explain why internal metrics do not prove usefulness"],
        problem="Compare clusterings when real data usually has no correct class label.",
        intuition="A good silhouette point is close to its own group and far from its nearest alternative group.",
        equation=r"$$s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}.$$",
        symbols=[r"$a(i)$ is point i's mean distance to its cluster.", r"$b(i)$ is its smallest mean distance to another cluster.", r"$s(i)$ is its silhouette in [-1,1]."],
        why_equation="It balances cohesion and separation on a comparable scale.",
        numerical_example="If a=2 and b=5, s=(5-2)/5=.6.",
        python_connection="The scratch loop calculates a, b, and their normalized difference for every point.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Compare multiple metrics because each rewards a different geometric property.",
        hyperparameter_code=hyper,
        interpretation="High silhouette and low Davies-Bouldin support compact separation, not stable or actionable real-world segments.",
    )


def capstone_spec() -> LessonSpec:
    experiment = r"""
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

wine = load_wine(as_frame=True)
frame = wine.frame.copy()
print("Dataset:", frame.shape, "missing:", int(frame.isna().sum().sum()))
display(frame.head(3))
X = StandardScaler().fit_transform(wine.data)
projection = PCA(n_components=2).fit_transform(X)

kmeans = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_SEED).fit(X)
gmm = GaussianMixture(n_components=3, random_state=RANDOM_SEED).fit(X)
gmm_labels = gmm.predict(X)
comparison = pd.DataFrame([
    {"model": "K-means", "silhouette": silhouette_score(X, kmeans.labels_), "teaching ARI": adjusted_rand_score(wine.target, kmeans.labels_)},
    {"model": "Gaussian mixture", "silhouette": silhouette_score(X, gmm_labels), "teaching ARI": adjusted_rand_score(wine.target, gmm_labels)},
])
display(comparison.round(3))

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for axis, labels, title in zip(axes, [kmeans.labels_, gmm_labels], ["K-means segments", "Gaussian-mixture segments"]):
    axis.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="cividis", s=30)
    axis.set(title=title, xlabel="PCA component 1", ylabel="PCA component 2")
figure.tight_layout()
plt.show()
"""
    hyper = r"""
rows = []
for k in range(2, 8):
    candidate = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit(X)
    rows.append({"k": k, "silhouette": silhouette_score(X, candidate.labels_), "inertia": candidate.inertia_})
tuning = pd.DataFrame(rows)
best_k = int(tuning.loc[tuning["silhouette"].idxmax(), "k"])
display(tuning.round(3))
figure, axis = plt.subplots()
axis.plot(tuning["k"], tuning["silhouette"], marker="o", color=COURSE_COLORS["blue"], label="Silhouette")
axis.axvline(best_k, color=COURSE_COLORS["orange"], linestyle="--", label=f"Selected k={best_k}")
axis.set(title="Capstone cluster-count selection", xlabel="Number of segments", ylabel="Silhouette score")
axis.legend()
plt.show()
print("Limitation: cultivar labels are post-hoc teaching evidence; real segmentation has no answer key.")
"""
    return base_spec(
        title="Unsupervised Capstone: Wine Profile Segmentation",
        objectives=["Frame an unlabeled segmentation problem", "Compare K-means and Gaussian mixtures", "Tune, visualize, and communicate limitations"],
        problem="A specialty retailer wants interpretable product-profile segments; success is compact, stable groups that experts can name and use.",
        use_cases=["Assortment planning", "Product profile discovery", "Targeted qualitative research"],
        intuition="Standardize measurements, establish a baseline, compare hard and soft clustering, then project the result for error analysis.",
        scratch_description="The baseline is transparent NumPy K-means exposing initialization, assignment, update, and inertia.",
        library_description="KMeans and GaussianMixture provide the comparison; PCA is used only for visualization.",
        dataset_description="The built-in Wine dataset has 178 rows and 13 chemistry features. Cultivar labels are hidden from fitting.",
        preprocessing_description="There are no missing values. Standardization prevents high-range measurements from dominating distance.",
        scratch_code=KMEANS_SCRATCH,
        experiment_code=experiment,
        hyperparameter_text="Tune segment count, then review stability, sizes, and domain meaning before operational use.",
        hyperparameter_code=hyper,
        interpretation="The models offer alternative segmentations. PCA is lossy, so metrics use the full standardized feature space.",
        mistakes=["Calling post-hoc label agreement the optimization target", "Interpreting the PCA view as complete evidence", "Deploying segments without stability and expert review"],
        advantages=["Offline end-to-end workflow", "Multiple-model comparison", "Transparent success metrics and caveats"],
        disadvantages=["Small benchmark dataset", "No business outcomes", "Clusters may not transfer to another population"],
        exercises=["Why are cultivar labels excluded from fitting?", "If within-cluster squared distances are 4, 9, and 7, compute inertia.", "Repeat K-means over five seeds and compare partitions."],
        solutions=["Using labels would turn the task into supervised learning.", "Inertia is 4+9+7=20.", "Fit five models and compare label arrays with adjusted Rand index; low agreement means instability."],
        takeaways=["Problem framing and success criteria come before clustering", "Scaling, baselines, comparison, and tuning are part of the model", "Error analysis includes unstable points and tiny clusters", "Limitations accompany any recommendation"],
    )


def main() -> None:
    lessons = {
        "02_unsupervised_learning/00_unsupervised_learning_overview.ipynb": overview_spec(),
        "02_unsupervised_learning/01_k_means_clustering.ipynb": kmeans_spec(),
        "02_unsupervised_learning/02_hierarchical_clustering.ipynb": hierarchical_spec(),
        "02_unsupervised_learning/03_dbscan.ipynb": dbscan_spec(),
        "02_unsupervised_learning/04_gaussian_mixture_models.ipynb": gmm_spec(),
        "02_unsupervised_learning/05_principal_component_analysis.ipynb": pca_spec(),
        "02_unsupervised_learning/06_anomaly_detection.ipynb": anomaly_spec(),
        "02_unsupervised_learning/07_clustering_evaluation.ipynb": evaluation_spec(),
        "02_unsupervised_learning/08_unsupervised_capstone_project.ipynb": capstone_spec(),
    }
    for path, spec in lessons.items():
        write_lesson(path, spec)
        print("Built", path)


if __name__ == "__main__":
    main()
