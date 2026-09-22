# Final Project 3: Image Classification

            ## 1. Learning objectives

            - Compare a linear pixel baseline and CNN
- Tune convolution channels on validation data
- Inspect feature shapes, learning curves, and test errors

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

            Classify 8-by-8 handwritten digits. Success is held-out accuracy with class-level error analysis under a CPU budget.

            ## 4. Real-world use cases

            - Document processing prototypes
- Educational visual recognition
- Compact edge-model experiments

            ## 5. Core intuition in simple language

            A linear baseline tests raw pixels; a CNN earns complexity only if local shared filters improve validation performance.

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

Manual softmax checks probability semantics before library loss functions.

The simplified implementation exposes the central mechanism. It favors readability over production-scale numerical safeguards.


```python
def softmax(values):
    shifted=values-values.max(axis=1,keepdims=True)
    exp=np.exp(shifted)
    return exp/exp.sum(axis=1,keepdims=True)
print("Softmax check:",softmax(np.array([[1.,2.,3.]])).round(3))
```

    Softmax check: [[0.09  0.245 0.665]]
    

## 10. Implementation using a standard library

Standard libraries provide reproducible pipelines and robust implementations; decisions remain visible in code.

## 11. Dataset loading and exploration

The bundled Digits dataset contains 1,797 small images and needs no remote download.

## 12. Data preprocessing

Development/test and train/validation splits precede any fitted scaling; CNN pixels use the known 0–16 range.

## 13. Model training

The next cell trains both the transparent and library-backed versions where the topic has a trainable model.

## 14. Predictions

Predictions, transformed representations, actions, or value estimates are kept in named arrays so their shapes can be inspected.

## 15. Evaluation

Evaluation uses a metric appropriate to the learning setting and always retains a simple baseline or reasonableness check.

## 16. Visualization of results

Every chart has a descriptive title and labeled axes. The interpretation immediately below describes what the marks mean.


```python
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
torch.set_num_threads(1)

digits=load_digits()
X_development,X_test,y_development,y_test=train_test_split(digits.images,digits.target,test_size=.2,stratify=digits.target,random_state=RANDOM_SEED)
X_train,X_validation,y_train,y_validation=train_test_split(X_development,y_development,test_size=.25,stratify=y_development,random_state=RANDOM_SEED)
scaler=StandardScaler(); flat_train=scaler.fit_transform(X_train.reshape(len(X_train),-1)); flat_validation=scaler.transform(X_validation.reshape(len(X_validation),-1))
baseline=LogisticRegression(max_iter=1000,random_state=RANDOM_SEED).fit(flat_train,y_train)

class SmallCNN(nn.Module):
    def __init__(self,channels=8):
        super().__init__(); self.features=nn.Sequential(nn.Conv2d(1,channels,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)); self.output=nn.Linear(channels*4*4,10)
    def forward(self,values):
        maps=self.features(values)
        return self.output(maps.flatten(1)),maps

train_X=torch.tensor((X_train/16)[:,None],dtype=torch.float32); train_y=torch.tensor(y_train,dtype=torch.long)
validation_X=torch.tensor((X_validation/16)[:,None],dtype=torch.float32); validation_y=torch.tensor(y_validation,dtype=torch.long)
models={}
for channels in [4,8,16]:
    model=SmallCNN(channels); optimizer=torch.optim.Adam(model.parameters(),lr=.01); loss_fn=nn.CrossEntropyLoss(); train_history=[]; validation_history=[]
    for _ in range(10):
        model.train(); optimizer.zero_grad(); logits,maps=model(train_X); loss=loss_fn(logits,train_y); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_logits,_=model(validation_X); val_loss=loss_fn(val_logits,validation_y)
        train_history.append(loss.item()); validation_history.append(val_loss.item())
    models[channels]=(model,train_history,validation_history,float((val_logits.argmax(1)==validation_y).float().mean()))
best_channels=max(models,key=lambda value:models[value][3]); best_model,train_losses,validation_losses,best_accuracy=models[best_channels]
print("Baseline/CNN validation accuracy:",round(baseline.score(flat_validation,y_validation),3),round(best_accuracy,3))
print("Input/maps/output:",tuple(train_X.shape),tuple(maps.shape),tuple(logits.shape),"parameters:",sum(p.numel() for p in best_model.parameters()))

test_X=torch.tensor((X_test/16)[:,None],dtype=torch.float32)
with torch.no_grad(): test_logits,test_maps=best_model(test_X); test_prediction=test_logits.argmax(1).numpy()
print("Held-out test accuracy:",round(accuracy_score(y_test,test_prediction),3))
figure,axes=plt.subplots(1,3,figsize=(13,4))
axes[0].plot(train_losses,label="Train",color=COURSE_COLORS["blue"]); axes[0].plot(validation_losses,label="Validation",color=COURSE_COLORS["orange"],linestyle="--"); axes[0].set(title="Selected CNN learning curves",xlabel="Epoch",ylabel="Cross-entropy"); axes[0].legend()
axes[1].imshow(test_X[0,0],cmap="gray"); axes[1].set(title=f"Test digit: true {y_test[0]}, predicted {test_prediction[0]}",xlabel="Pixel column",ylabel="Pixel row")
ConfusionMatrixDisplay.from_predictions(y_test,test_prediction,cmap="Blues",colorbar=False,ax=axes[2]); axes[2].set_title("Held-out confusion matrix")
figure.tight_layout(); plt.show()
```

    Baseline/CNN validation accuracy: 0.972 0.886
    Input/maps/output: (1077, 1, 8, 8) (1077, 16, 4, 4) (1077, 10) parameters: 1370
    Held-out test accuracy: 0.867
    


    
![png](03_image_classification_project_files/03_image_classification_project_8_1.png)
    


**How to interpret the result:** Learning curves show fit, the example image connects output to input, and the confusion matrix reveals systematic class errors.

## 17. Hyperparameter experiments

Convolution channel width is selected on validation accuracy; parameter count exposes its cost.


```python
comparison=pd.DataFrame([{"channels":channels,"validation_accuracy":values[3],"parameters":sum(p.numel() for p in values[0].parameters())} for channels,values in models.items()])
display(comparison.round(3))
figure,axis=plt.subplots(); axis.plot(comparison["channels"],comparison["validation_accuracy"],marker="o",color=COURSE_COLORS["blue"],label="Validation accuracy")
axis.set(title="CNN channel-width tuning",xlabel="Convolution channels",ylabel="Validation accuracy"); axis.legend(); plt.show()
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
      <th>channels</th>
      <th>validation_accuracy</th>
      <th>parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0.758</td>
      <td>690</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>0.886</td>
      <td>1370</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>0.872</td>
      <td>2730</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](03_image_classification_project_files/03_image_classification_project_11_1.png)
    


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

            **Do not use it when:** higher-resolution, shifted, or safety-critical images are assumed to behave like this small benchmark without external validation.

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
