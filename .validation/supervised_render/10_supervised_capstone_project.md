# Supervised Learning Capstone: Diabetes Progression Regression

            ## 1. Learning objectives

            - Frame an end-to-end regression problem
- Audit, engineer, baseline, compare, and tune models
- Perform held-out evaluation and residual error analysis

            ## 2. Prerequisites

            Basic Python, NumPy arrays, means, and the supervised-learning overview.


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

            Predict the one-year disease-progression measure in the scikit-learn Diabetes benchmark. Success is lower RMSE than a mean baseline, with MAE and R² as guardrails.

            ## 4. Real-world use cases

            - Educational clinical-risk regression
- Tabular model comparison
- Residual-based error analysis

            ## 5. Core intuition in simple language

            A useful capstone earns complexity: begin with the mean, add an interpretable linear model, compare ensembles, tune without seeing test outcomes, and document limitations.

## 6. Mathematical foundation

            $$RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2},\qquad R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.$$

            **Every symbol:**

            - $n$ is evaluated row count.
- $y_i$ is true progression.
- $\hat y_i$ is prediction.
- $\bar y$ is mean target.
- $RMSE$ is error in target units.
- $R^2$ compares with the mean baseline.

            **Why the equation is needed:** RMSE defines primary success and emphasizes large misses; R² shows improvement relative to a constant prediction.

            **Small numerical example:** For errors [1,-2], RMSE=sqrt((1+4)/2)=1.581.

            **Connection to Python:** mean_squared_error(y, prediction)**.5 and r2_score(y, prediction) implement the metrics.

## 7. Visual explanation

            The executable figure below is chosen to reveal the structure this method learns. Read positions and lengths first; color is a supporting cue rather than the only encoding.

            ## 8. Algorithm steps

            1. Define target and success metric
2. Split observations before fitted preprocessing
3. Build a simple baseline
4. Fit a transparent implementation
5. Fit the library estimator
6. Tune with validation evidence
7. Evaluate once on untouched test data

## 9. Implementation from scratch

The mean regressor is implemented directly and establishes the minimum useful benchmark.

The simplified implementation exposes the central mechanism. It favors readability over production-scale numerical safeguards.


```python
class MeanRegressor:
    def fit(self,target):
        self.mean_=float(np.mean(target)); return self
    def predict(self,size):
        return np.full(size,self.mean_)

print("Mean-baseline example:",MeanRegressor().fit([2,4,6]).predict(3))
```

    Mean-baseline example: [4. 4. 4.]
    

## 10. Implementation using a standard library

Ridge, Random Forest, and Gradient Boosting represent linear, bagged-tree, and sequential-tree families.

## 11. Dataset loading and exploration

The bundled dataset has 442 observations, ten standardized baseline variables, and a numeric progression target.

## 12. Data preprocessing

A BMI-by-blood-pressure interaction is engineered before splitting columns; fitted scaling remains inside the Ridge pipeline.

## 13. Model training

The next cell trains both the transparent and library-backed versions where the topic has a trainable model.

## 14. Predictions

Predictions, transformed representations, actions, or value estimates are kept in named arrays so their shapes can be inspected.

## 15. Evaluation

Evaluation uses a metric appropriate to the learning setting and always retains a simple baseline or reasonableness check.

## 16. Visualization of results

Every chart has a descriptive title and labeled axes. The interpretation immediately below describes what the marks mean.


```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

data=load_diabetes(as_frame=True)
frame=data.frame.copy()
print("Dataset:",frame.shape,"missing:",int(frame.isna().sum().sum()),"duplicates:",int(frame.duplicated().sum()))
display(frame.head(3))
features=data.data.copy()
features["bmi_bp_interaction"]=features["bmi"]*features["bp"]
X_development,X_test,y_development,y_test=train_test_split(features,data.target,test_size=.2,random_state=RANDOM_SEED)
X_train,X_validation,y_train,y_validation=train_test_split(X_development,y_development,test_size=.25,random_state=RANDOM_SEED)
models={
    "Ridge":make_pipeline(StandardScaler(),Ridge(alpha=10)),
    "Random forest":RandomForestRegressor(n_estimators=180,max_depth=5,random_state=RANDOM_SEED),
    "Gradient boosting":GradientBoostingRegressor(n_estimators=100,max_depth=2,learning_rate=.05,random_state=RANDOM_SEED),
}
rows=[{"model":"Mean baseline","MAE":mean_absolute_error(y_validation,np.full(len(y_validation),y_train.mean())),"RMSE":mean_squared_error(y_validation,np.full(len(y_validation),y_train.mean()))**.5,"R2":r2_score(y_validation,np.full(len(y_validation),y_train.mean()))}]
for name,model in models.items():
    model.fit(X_train,y_train); prediction=model.predict(X_validation)
    rows.append({"model":name,"MAE":mean_absolute_error(y_validation,prediction),"RMSE":mean_squared_error(y_validation,prediction)**.5,"R2":r2_score(y_validation,prediction)})
comparison=pd.DataFrame(rows).sort_values("RMSE"); display(comparison.round(3))
best_name=comparison.iloc[0]["model"]; best_model=models[best_name]; best_model.fit(X_development,y_development); test_prediction=best_model.predict(X_test)
print("Selected:",best_name,"held-out MAE/RMSE/R2:",round(mean_absolute_error(y_test,test_prediction),2),round(mean_squared_error(y_test,test_prediction)**.5,2),round(r2_score(y_test,test_prediction),3))
residuals=y_test-test_prediction
figure,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].bar(comparison["model"],comparison["RMSE"],color=COURSE_COLORS["blue"]); axes[0].set(title="Development model comparison",xlabel="Candidate",ylabel="Validation RMSE"); axes[0].tick_params(axis="x",rotation=20)
axes[1].scatter(test_prediction,residuals,color=COURSE_COLORS["orange"],alpha=.7,label="Test residuals"); axes[1].axhline(0,color=COURSE_COLORS["charcoal"],linestyle="--"); axes[1].set(title="Held-out residual error analysis",xlabel="Predicted progression",ylabel="True minus predicted"); axes[1].legend()
figure.tight_layout(); plt.show()
```

    Dataset: (442, 11) missing: 0 duplicates: 0
    


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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019907</td>
      <td>-0.017646</td>
      <td>151.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068332</td>
      <td>-0.092204</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005670</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002861</td>
      <td>-0.025930</td>
      <td>141.0</td>
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
      <th>MAE</th>
      <th>RMSE</th>
      <th>R2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Ridge</td>
      <td>42.362</td>
      <td>51.411</td>
      <td>0.518</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gradient boosting</td>
      <td>43.534</td>
      <td>53.622</td>
      <td>0.475</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random forest</td>
      <td>43.752</td>
      <td>54.188</td>
      <td>0.464</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Mean baseline</td>
      <td>64.129</td>
      <td>75.639</td>
      <td>-0.044</td>
    </tr>
  </tbody>
</table>
</div>


    Selected: Ridge held-out MAE/RMSE/R2: 41.98 52.36 0.483
    


    
![png](10_supervised_capstone_project_files/10_supervised_capstone_project_8_4.png)
    


**How to interpret the result:** The comparison tests whether nonlinear ensembles beat the baseline; residuals expose target ranges where systematic under- or overprediction remains.

## 17. Hyperparameter experiments

Gradient-boosting depth and stage count are compared only on validation RMSE. The test set is reserved for the frozen winner.


```python
rows=[]
for depth in [1,2,3]:
    for estimators in [50,100,180]:
        candidate=GradientBoostingRegressor(max_depth=depth,n_estimators=estimators,learning_rate=.05,random_state=RANDOM_SEED).fit(X_train,y_train)
        rows.append({"depth":depth,"estimators":estimators,"validation_RMSE":mean_squared_error(y_validation,candidate.predict(X_validation))**.5})
tuning=pd.DataFrame(rows); display(tuning.round(2))
pivot=tuning.pivot(index="depth",columns="estimators",values="validation_RMSE")
figure,axis=plt.subplots(); sns.heatmap(pivot,annot=True,fmt=".1f",cmap="Blues_r",ax=axis); axis.set(title="Capstone hyperparameter search",xlabel="Boosting stages",ylabel="Tree depth"); plt.show()
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
      <th>depth</th>
      <th>estimators</th>
      <th>validation_RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>50</td>
      <td>55.64</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>100</td>
      <td>54.05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>180</td>
      <td>53.48</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>50</td>
      <td>54.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>100</td>
      <td>53.62</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>180</td>
      <td>53.49</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>50</td>
      <td>55.67</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>100</td>
      <td>55.69</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>180</td>
      <td>57.23</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](10_supervised_capstone_project_files/10_supervised_capstone_project_11_1.png)
    


## 18. Common mistakes

            - Treating the benchmark as deployment-ready medicine
- Tuning on test RMSE
- Ignoring target-range error patterns
- Claiming the engineered interaction is causal

            ## 19. Advantages and disadvantages

            **Advantages**

            - Complete offline workflow
- Multiple model families and explicit baseline
- Held-out residual analysis

            **Disadvantages**

            - Small benchmark sample
- No temporal or external validation
- No uncertainty intervals or clinical utility analysis

            ## 20. When to use and when not to use the algorithm

            **Use it when:** learning reproducible supervised workflows or benchmarking methods on small continuous-target data.

            **Do not use it when:** making clinical decisions without prospective validation, governance, calibration, uncertainty, and subgroup analysis.

## 21. Exercises

1. **Conceptual:** Why use both RMSE and MAE?
2. **Mathematical/hand calculation:** For true [2,4] and predicted [1,7], compute RMSE.
3. **Coding/experimentation:** Repeat the comparison across five splits and report mean plus standard deviation.

<details>
<summary><strong>22. Exercise solutions</strong></summary>

1. RMSE emphasizes large errors while MAE expresses a robust typical absolute miss; disagreement is diagnostically useful.
2. Errors [1,-3] give squared errors [1,9], MSE=5, RMSE=sqrt(5)=2.236.
3. Loop over fixed seeds, repeat every split and fitted preprocessing step, retain every score, and summarize variability without selecting the best split.

</details>

## 23. Summary and key takeaways

            - Business framing and success metrics precede model choice
- Feature engineering and preprocessing must respect split boundaries
- Baselines make added complexity accountable
- Final evaluation, residual analysis, limitations, and improvements form one conclusion

            ## 24. Further reading

            - [scikit-learn supervised learning guide](https://scikit-learn.org/stable/supervised_learning.html)
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
