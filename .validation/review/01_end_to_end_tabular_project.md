# Final Project 1: End-to-End Tabular Classification

            ## 1. Learning objectives

            - Audit a health benchmark
- Compare a constant baseline and three model families
- Tune, evaluate, and analyze classification errors

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

            Predict the diagnostic class in the Breast Cancer Wisconsin benchmark. Primary success metric is F1; ROC-AUC and the confusion matrix are guardrails.

            ## 4. Real-world use cases

            - Educational diagnostic modeling
- Tabular pipeline comparison
- Error-cost discussion

            ## 5. Core intuition in simple language

            Start with a majority baseline, preserve the untouched test set, compare linear and tree-based candidates, then inspect false positives and negatives.

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

A transparent baseline verifies label and metric semantics before complex modeling.

The simplified implementation exposes the central mechanism. It favors readability over production-scale numerical safeguards.


```python
def majority_prediction(target,size):
    labels,counts=np.unique(target,return_counts=True)
    return np.full(size,labels[np.argmax(counts)])

example=np.array([0,1,1,1,0])
print("Majority predictions:",majority_prediction(example,3))
```

    Majority predictions: [1 1 1]
    

## 10. Implementation using a standard library

Standard libraries provide reproducible pipelines and robust implementations; decisions remain visible in code.

## 11. Dataset loading and exploration

The project uses a small bundled dataset so every result can be reproduced offline.

## 12. Data preprocessing

Cleaning and feature transformations are fitted only on development or training rows.

## 13. Model training

The next cell trains both the transparent and library-backed versions where the topic has a trainable model.

## 14. Predictions

Predictions, transformed representations, actions, or value estimates are kept in named arrays so their shapes can be inspected.

## 15. Evaluation

Evaluation uses a metric appropriate to the learning setting and always retains a simple baseline or reasonableness check.

## 16. Visualization of results

Every chart has a descriptive title and labeled axes. The interpretation immediately below describes what the marks mean.


```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data=load_breast_cancer(as_frame=True)
frame=data.frame.copy()
print("Rows/features:",data.data.shape,"missing:",int(frame.isna().sum().sum()),"duplicates:",int(frame.duplicated().sum()))
display(frame.head(3))
X_development,X_test,y_development,y_test=train_test_split(data.data,data.target,test_size=.2,stratify=data.target,random_state=RANDOM_SEED)
X_train,X_validation,y_train,y_validation=train_test_split(X_development,y_development,test_size=.25,stratify=y_development,random_state=RANDOM_SEED)
baseline=majority_prediction(y_train,len(y_validation))
candidates={
    "Logistic":make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=RANDOM_SEED)),
    "Random forest":RandomForestClassifier(n_estimators=160,max_depth=5,random_state=RANDOM_SEED),
    "Gradient boosting":GradientBoostingClassifier(random_state=RANDOM_SEED),
}
rows=[{"model":"Majority baseline","F1":f1_score(y_validation,baseline),"ROC-AUC":.5}]
for name,model in candidates.items():
    model.fit(X_train,y_train); prediction=model.predict(X_validation); probability=model.predict_proba(X_validation)[:,1]
    rows.append({"model":name,"F1":f1_score(y_validation,prediction),"ROC-AUC":roc_auc_score(y_validation,probability)})
comparison=pd.DataFrame(rows).sort_values("F1",ascending=False)
display(comparison.round(3))
best_name=comparison.iloc[0]["model"]; best_model=candidates[best_name]
best_model.fit(X_development,y_development)
test_prediction=best_model.predict(X_test); test_probability=best_model.predict_proba(X_test)[:,1]
print("Selected:",best_name,"test F1:",round(f1_score(y_test,test_prediction),3),"test ROC-AUC:",round(roc_auc_score(y_test,test_probability),3))
figure,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].bar(comparison["model"],comparison["F1"],color=COURSE_COLORS["blue"]); axes[0].set(title="Development model comparison",xlabel="Candidate",ylabel="Validation F1"); axes[0].tick_params(axis="x",rotation=20)
ConfusionMatrixDisplay.from_predictions(y_test,test_prediction,display_labels=data.target_names,cmap="Blues",colorbar=False,ax=axes[1]); axes[1].set_title("Final held-out error analysis")
figure.tight_layout(); plt.show()
```

    Rows/features: (569, 30) missing: 0 duplicates: 0
    


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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.8</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.6</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.9</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.8</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.0</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.5</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 31 columns</p>
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
      <th>F1</th>
      <th>ROC-AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Logistic</td>
      <td>0.993</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gradient boosting</td>
      <td>0.972</td>
      <td>0.993</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random forest</td>
      <td>0.958</td>
      <td>0.983</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Majority baseline</td>
      <td>0.768</td>
      <td>0.500</td>
    </tr>
  </tbody>
</table>
</div>


    Selected: Logistic test F1: 0.986 test ROC-AUC: 0.995
    


    
![png](01_end_to_end_tabular_project_files/01_end_to_end_tabular_project_8_4.png)
    


**How to interpret the result:** The comparison establishes whether complexity improves on the baseline; the confusion matrix exposes the exact final error types.

## 17. Hyperparameter experiments

Tree depth is varied using validation F1. The test set is not consulted during this experiment.


```python
depth_rows=[]
for depth in [2,3,5,8,None]:
    model=RandomForestClassifier(n_estimators=160,max_depth=depth,random_state=RANDOM_SEED).fit(X_train,y_train)
    depth_rows.append({"max_depth":str(depth),"validation_F1":f1_score(y_validation,model.predict(X_validation))})
depth_results=pd.DataFrame(depth_rows)
display(depth_results.round(3))
figure,axis=plt.subplots(); axis.plot(depth_results["max_depth"],depth_results["validation_F1"],marker="o",color=COURSE_COLORS["blue"],label="Random-forest F1")
axis.set(title="Tree-depth tuning on validation data",xlabel="Maximum depth",ylabel="Validation F1"); axis.legend(); plt.show()
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
      <th>max_depth</th>
      <th>validation_F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.958</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.958</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0.958</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>0.965</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>0.965</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](01_end_to_end_tabular_project_files/01_end_to_end_tabular_project_11_1.png)
    


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

            **Do not use it when:** used as a clinical system without prospective validation, calibration, subgroup audits, and medical governance.

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
