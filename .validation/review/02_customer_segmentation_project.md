# Final Project 2: Customer-Style Segmentation

            ## 1. Learning objectives

            - Build a reproducible segmentation workflow
- Compare hard and probabilistic clustering
- Profile and critique the selected groups

            ## 2. Prerequisites

            Complete the corresponding numbered course section and understand leakage-safe train/validation/test roles.


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

            Use wine chemistry as a safe proxy for customer-style segmentation. Success requires compact groups plus interpretable profiles, not a hidden answer label.

            ## 4. Real-world use cases

            - Exploratory segmentation
- Persona research inputs
- Assortment analysis

            ## 5. Core intuition in simple language

            Standardize behavior-like features, compare plausible grouping assumptions, then name profiles only after quantitative and domain review.

## 6. Mathematical foundation

            $$\widehat{Risk}_{test}=\frac{1}{n_{test}}\sum_{i=1}^{n_{test}}L(y_i,\hat y_i).$$

            **Every symbol:**

            - $n_{test}$ is held-out test size.
- $L$ is the chosen per-example loss.
- $y_i$ is truth.
- $\hat y_i$ is the frozen model prediction.
- $\widehat{Risk}_{test}$ is estimated generalization error.

            **Why the equation is needed:** The held-out average connects the technical model to a predeclared success criterion.

            **Small numerical example:** If 3 of 100 equally weighted test cases are wrong, error is 3/100=.03 and accuracy is .97.

            **Connection to Python:** metric(y_test, model.predict(X_test)) evaluates the frozen pipeline once.

## 7. Visual explanation

            The executable figure below is chosen to reveal the structure this method learns. Read positions and lengths first; color is a supporting cue rather than the only encoding.

            ## 8. Algorithm steps

            1. Define decision and success metric
2. Describe and audit data
3. Split before fitted preprocessing
4. Create a simple baseline
5. Compare candidate models
6. Tune on development data
7. Evaluate once on test data
8. Analyze errors and limitations

## 9. Implementation from scratch

NumPy K-means exposes centroid assignment and update as the baseline.

The simplified implementation exposes the central mechanism. It favors readability over production-scale numerical safeguards.


```python
def kmeans_scratch(features,k,iterations=60):
    centroids=features[np.linspace(0,len(features)-1,k,dtype=int)].copy()
    for _ in range(iterations):
        labels=np.linalg.norm(features[:,None,:]-centroids[None,:,:],axis=2).argmin(1)
        updated=np.vstack([features[labels==group].mean(0) for group in range(k)])
        if np.allclose(updated,centroids): break
        centroids=updated
    return labels,centroids
print("Scratch K-means ready")
```

    Scratch K-means ready
    

## 10. Implementation using a standard library

Standard libraries provide reproducible pipelines and robust implementations; decisions remain visible in code.

## 11. Dataset loading and exploration

The bundled Wine dataset replaces private customer data and has no network dependency.

## 12. Data preprocessing

StandardScaler is fit on the complete exploratory feature set because this project describes that fixed set rather than predicts future labels.

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
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

wine=load_wine(as_frame=True)
print("Rows/features:",wine.data.shape,"missing:",int(wine.data.isna().sum().sum()))
display(wine.data.describe().loc[["mean","std"]].round(2))
scaler=StandardScaler(); X=scaler.fit_transform(wine.data)
scratch_labels,scratch_centroids=kmeans_scratch(X,3)
kmeans=KMeans(n_clusters=3,n_init=20,random_state=RANDOM_SEED).fit(X)
gmm=GaussianMixture(n_components=3,random_state=RANDOM_SEED).fit(X); gmm_labels=gmm.predict(X)
comparison=pd.DataFrame([
    {"model":"Scratch K-means","silhouette":silhouette_score(X,scratch_labels)},
    {"model":"Library K-means","silhouette":silhouette_score(X,kmeans.labels_)},
    {"model":"Gaussian mixture","silhouette":silhouette_score(X,gmm_labels)},
]).sort_values("silhouette",ascending=False)
display(comparison.round(3))
projection=PCA(n_components=2).fit_transform(X)
figure,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].scatter(projection[:,0],projection[:,1],c=kmeans.labels_,cmap="cividis"); axes[0].set(title="K-means customer-style segments",xlabel="PCA component 1",ylabel="PCA component 2")
standardized_frame=pd.DataFrame(X,columns=wine.feature_names).assign(segment=kmeans.labels_)
profile=standardized_frame.groupby("segment").mean()[["alcohol","color_intensity","proline"]]
profile.plot.bar(ax=axes[1],color=[COURSE_COLORS["blue"],COURSE_COLORS["orange"],COURSE_COLORS["olive"]])
axes[1].axhline(0,color=COURSE_COLORS["charcoal"],linewidth=1)
axes[1].set(title="Selected segment profiles",xlabel="Segment",ylabel="Mean standardized feature (z-score)"); axes[1].legend(title="Feature")
figure.tight_layout(); plt.show()
```

    Rows/features: (178, 13) missing: 0
    


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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>13.00</td>
      <td>2.34</td>
      <td>2.37</td>
      <td>19.49</td>
      <td>99.74</td>
      <td>2.30</td>
      <td>2.03</td>
      <td>0.36</td>
      <td>1.59</td>
      <td>5.06</td>
      <td>0.96</td>
      <td>2.61</td>
      <td>746.89</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.81</td>
      <td>1.12</td>
      <td>0.27</td>
      <td>3.34</td>
      <td>14.28</td>
      <td>0.63</td>
      <td>1.00</td>
      <td>0.12</td>
      <td>0.57</td>
      <td>2.32</td>
      <td>0.23</td>
      <td>0.71</td>
      <td>314.91</td>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Library K-means</td>
      <td>0.285</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gaussian mixture</td>
      <td>0.285</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Scratch K-means</td>
      <td>0.284</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](02_customer_segmentation_project_files/02_customer_segmentation_project_8_3.png)
    


**How to interpret the result:** PCA shows a lossy view of groups while the profile chart returns to original feature units for interpretation.

## 17. Hyperparameter experiments

The number of groups is compared with silhouette and inertia; interpretability and stability remain required.


```python
rows=[]
for k in range(2,8):
    model=KMeans(n_clusters=k,n_init=20,random_state=RANDOM_SEED).fit(X)
    rows.append({"k":k,"silhouette":silhouette_score(X,model.labels_),"inertia":model.inertia_})
tuning=pd.DataFrame(rows); display(tuning.round(3))
figure,axis=plt.subplots(); axis.plot(tuning["k"],tuning["silhouette"],marker="o",color=COURSE_COLORS["blue"],label="Silhouette")
axis.set(title="Segment-count selection",xlabel="Number of segments",ylabel="Silhouette"); axis.legend(); plt.show()
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



    
![png](02_customer_segmentation_project_files/02_customer_segmentation_project_11_1.png)
    


## 18. Common mistakes

            - Starting with a model before defining success
- Fitting preprocessing before splitting
- Selecting on test performance
- Hiding failure modes behind one aggregate score

            ## 19. Advantages and disadvantages

            **Advantages**

            - Reproducible end-to-end evidence
- Explicit comparison and limitations

            **Disadvantages**

            - Benchmark data does not prove deployment readiness
- Offline metrics omit operational costs

            ## 20. When to use and when not to use the algorithm

            **Use it when:** the dataset, metric, and decision are aligned and the result will be treated as evidence rather than certainty.

            **Do not use it when:** segments would drive consequential decisions without stability checks, domain review, privacy analysis, and outcome validation.

## 21. Exercises

1. **Conceptual:** Which decision does the model support?
2. **Mathematical/hand calculation:** Recalculate the primary metric on a tiny example by hand.
3. **Coding/experimentation:** Repeat the final comparison over multiple seeds or splits.

<details>
<summary><strong>22. Exercise solutions</strong></summary>

1. State the user, action, model output, and consequence in one sentence.
2. Apply the displayed metric to each example, sum, and divide by the correct denominator.
3. Keep preprocessing and candidate definitions fixed, report every run, and summarize mean plus variability.

</details>

## 23. Summary and key takeaways

            - The workflow is part of the model
- Baselines and data leakage checks precede tuning
- Final evaluation uses untouched data
- Error analysis, limitations, and possible improvements belong in the conclusion

            ## 24. Further reading

            - [scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)
- [PyTorch tutorials](https://pytorch.org/tutorials/)
