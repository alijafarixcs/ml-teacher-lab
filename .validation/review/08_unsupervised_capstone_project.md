# Unsupervised Capstone: Wine Profile Segmentation

            ## 1. Learning objectives

            - Frame an unlabeled segmentation problem
- Compare K-means and Gaussian mixtures
- Tune, visualize, and communicate limitations

            ## 2. Prerequisites

            Basic NumPy arrays, means, and two-dimensional scatter plots.


```python
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
```

    Python lesson ready; reproducible seed = 42
    

## 3. Problem definition

            A specialty retailer wants interpretable product-profile segments; success is compact, stable groups that experts can name and use.

            ## 4. Real-world use cases

            - Assortment planning
- Product profile discovery
- Targeted qualitative research

            ## 5. Core intuition in simple language

            Standardize measurements, establish a baseline, compare hard and soft clustering, then project the result for error analysis.

## 6. Mathematical foundation

            $$d(\mathbf{x},\mathbf{c})=\sqrt{\sum_{j=1}^{p}(x_j-c_j)^2}.$$

            **Every symbol:**

            - $\mathbf{x}$ is one observation.
- $\mathbf{c}$ is a reference point such as a centroid.
- $p$ is the number of features.
- $j$ indexes a feature.
- $d$ is Euclidean distance.

            **Why the equation is needed:** Distance turns similarity into a quantity the algorithm can compare.

            **Small numerical example:** For $\mathbf{x}=(1,2)$ and $\mathbf{c}=(4,6)$, $d=\sqrt{3^2+4^2}=5$.

            **Connection to Python:** np.linalg.norm(x - c) evaluates exactly this distance.

## 7. Visual explanation

            The executable figure below is chosen to reveal the structure this method learns. Read positions and lengths first; color is a supporting cue rather than the only encoding.

            ## 8. Algorithm steps

            1. Represent each observation as a feature vector
2. Scale features when units differ
3. Fit the structure-discovery rule
4. Evaluate stability and usefulness
5. Inspect the result visually and with domain knowledge

## 9. Implementation from scratch

The baseline is transparent NumPy K-means exposing initialization, assignment, update, and inertia.

The simplified implementation exposes the central mechanism. It favors readability over production-scale numerical safeguards.


```python
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
```

    Tiny centroids: [[0.0, 0.5], [5.0, 5.5]] inertia: 1.0
    

## 10. Implementation using a standard library

KMeans and GaussianMixture provide the comparison; PCA is used only for visualization.

## 11. Dataset loading and exploration

The built-in Wine dataset has 178 rows and 13 chemistry features. Cultivar labels are hidden from fitting.

## 12. Data preprocessing

There are no missing values. Standardization prevents high-range measurements from dominating distance.

## 13. Model training

The next cell trains both the transparent and library-backed versions where the topic has a trainable model.

## 14. Predictions

Predictions, transformed representations, actions, or value estimates are kept in named arrays so their shapes can be inspected.

## 15. Evaluation

Evaluation uses a metric appropriate to the learning setting and always retains a simple baseline or reasonableness check.

## 16. Visualization of results

Every chart has a descriptive title and labeled axes. The interpretation immediately below describes what the marks mean.


```python
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
```

    Dataset: (178, 14) missing: 0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>silhouette</th>
      <th>teaching ARI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K-means</td>
      <td>0.285</td>
      <td>0.897</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gaussian mixture</td>
      <td>0.285</td>
      <td>0.897</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](08_unsupervised_capstone_project_files/08_unsupervised_capstone_project_8_3.png)
    


**How to interpret the result:** The models offer alternative segmentations. PCA is lossy, so metrics use the full standardized feature space.

## 17. Hyperparameter experiments

Tune segment count, then review stability, sizes, and domain meaning before operational use.


```python
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
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>silhouette</th>
      <th>inertia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.259</td>
      <td>1658.759</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.285</td>
      <td>1277.928</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.259</td>
      <td>1175.352</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.232</td>
      <td>1107.007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.237</td>
      <td>1046.002</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>0.204</td>
      <td>981.595</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](08_unsupervised_capstone_project_files/08_unsupervised_capstone_project_11_1.png)
    


    Limitation: cultivar labels are post-hoc teaching evidence; real segmentation has no answer key.
    

## 18. Common mistakes

            - Calling post-hoc label agreement the optimization target
- Interpreting the PCA view as complete evidence
- Deploying segments without stability and expert review

            ## 19. Advantages and disadvantages

            **Advantages**

            - Offline end-to-end workflow
- Multiple-model comparison
- Transparent success metrics and caveats

            **Disadvantages**

            - Small benchmark dataset
- No business outcomes
- Clusters may not transfer to another population

            ## 20. When to use and when not to use the algorithm

            **Use it when:** exploration, compression, or structure discovery is the goal and trustworthy labels are unavailable.

            **Do not use it when:** a supervised target directly represents the decision or the learned grouping has no domain interpretation.

## 21. Exercises

1. **Conceptual:** Why are cultivar labels excluded from fitting?
2. **Mathematical/hand calculation:** If within-cluster squared distances are 4, 9, and 7, compute inertia.
3. **Coding/experimentation:** Repeat K-means over five seeds and compare partitions.

<details>
<summary><strong>22. Exercise solutions</strong></summary>

1. Using labels would turn the task into supervised learning.
2. Inertia is 4+9+7=20.
3. Fit five models and compare label arrays with adjusted Rand index; low agreement means instability.

</details>

## 23. Summary and key takeaways

            - Problem framing and success criteria come before clustering
- Scaling, baselines, comparison, and tuning are part of the model
- Error analysis includes unstable points and tiny clusters
- Limitations accompany any recommendation

            ## 24. Further reading

            - [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html)
