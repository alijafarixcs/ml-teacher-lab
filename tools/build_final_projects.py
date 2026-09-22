"""Build the four integrated final-project notebooks."""

from __future__ import annotations

from course_factory import LessonSpec, write_lesson


def project_base(**overrides) -> LessonSpec:
    spec = {
        "title": "End-to-End Machine-Learning Project",
        "objectives": ["Frame a measurable real-world problem", "Build and tune multiple baselines", "Evaluate errors, limitations, and improvements"],
        "prerequisites": "Complete the corresponding numbered course section and understand leakage-safe train/validation/test roles.",
        "problem": "Turn a practical question into a reproducible modeling workflow with an explicit success metric.",
        "use_cases": ["Decision support", "Model comparison", "Reproducible technical handoff"],
        "intuition": "A project is not only an algorithm: problem definition, data quality, preprocessing, baselines, selection, error analysis, and limitations determine whether results are useful.",
        "equation": r"$$\widehat{Risk}_{test}=\frac{1}{n_{test}}\sum_{i=1}^{n_{test}}L(y_i,\hat y_i).$$",
        "symbols": [r"$n_{test}$ is held-out test size.", r"$L$ is the chosen per-example loss.", r"$y_i$ is truth.", r"$\hat y_i$ is the frozen model prediction.", r"$\widehat{Risk}_{test}$ is estimated generalization error."],
        "why_equation": "The held-out average connects the technical model to a predeclared success criterion.",
        "numerical_example": "If 3 of 100 equally weighted test cases are wrong, error is 3/100=.03 and accuracy is .97.",
        "python_connection": "metric(y_test, model.predict(X_test)) evaluates the frozen pipeline once.",
        "steps": ["Define decision and success metric", "Describe and audit data", "Split before fitted preprocessing", "Create a simple baseline", "Compare candidate models", "Tune on development data", "Evaluate once on test data", "Analyze errors and limitations"],
        "scratch_description": "A transparent baseline verifies label and metric semantics before complex modeling.",
        "library_description": "Standard libraries provide reproducible pipelines and robust implementations; decisions remain visible in code.",
        "dataset_description": "The project uses a small bundled dataset so every result can be reproduced offline.",
        "preprocessing_description": "Cleaning and feature transformations are fitted only on development or training rows.",
        "setup_code": "",
        "scratch_code": "def majority_prediction(target,size):\n    labels,counts=np.unique(target,return_counts=True)\n    return np.full(size,labels[np.argmax(counts)])\nprint('Baseline helper ready')",
        "experiment_code": "print('Project experiment is defined by the specialization.')",
        "interpretation": "Compare the result with the baseline and success metric, then inspect where it fails.",
        "hyperparameter_text": "Tuning is limited, explicit, and based on development evidence rather than the held-out test set.",
        "hyperparameter_code": "values=[1,2,3]; scores=[.6,.7,.68]\nfigure,axis=plt.subplots(); axis.plot(values,scores,marker='o',color=COURSE_COLORS['blue'],label='Validation score'); axis.set(title='Illustrative tuning curve',xlabel='Candidate',ylabel='Validation score'); axis.legend(); plt.show()",
        "mistakes": ["Starting with a model before defining success", "Fitting preprocessing before splitting", "Selecting on test performance", "Hiding failure modes behind one aggregate score"],
        "advantages": ["Reproducible end-to-end evidence", "Explicit comparison and limitations"],
        "disadvantages": ["Benchmark data does not prove deployment readiness", "Offline metrics omit operational costs"],
        "use_when": "the dataset, metric, and decision are aligned and the result will be treated as evidence rather than certainty.",
        "avoid_when": "labels or rewards do not represent the desired outcome, or deployment risks cannot be bounded.",
        "exercises": ["Which decision does the model support?", "Recalculate the primary metric on a tiny example by hand.", "Repeat the final comparison over multiple seeds or splits."],
        "solutions": ["State the user, action, model output, and consequence in one sentence.", "Apply the displayed metric to each example, sum, and divide by the correct denominator.", "Keep preprocessing and candidate definitions fixed, report every run, and summarize mean plus variability."],
        "takeaways": ["The workflow is part of the model", "Baselines and data leakage checks precede tuning", "Final evaluation uses untouched data", "Error analysis, limitations, and possible improvements belong in the conclusion"],
        "further_reading": ["[scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)", "[PyTorch tutorials](https://pytorch.org/tutorials/)"],
    }
    spec.update(overrides)
    return spec


def tabular_spec() -> LessonSpec:
    scratch = r"""
def majority_prediction(target,size):
    labels,counts=np.unique(target,return_counts=True)
    return np.full(size,labels[np.argmax(counts)])

example=np.array([0,1,1,1,0])
print("Majority predictions:",majority_prediction(example,3))
"""
    experiment = r"""
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
"""
    hyper = r"""
depth_rows=[]
for depth in [2,3,5,8,None]:
    model=RandomForestClassifier(n_estimators=160,max_depth=depth,random_state=RANDOM_SEED).fit(X_train,y_train)
    depth_rows.append({"max_depth":str(depth),"validation_F1":f1_score(y_validation,model.predict(X_validation))})
depth_results=pd.DataFrame(depth_rows)
display(depth_results.round(3))
figure,axis=plt.subplots(); axis.plot(depth_results["max_depth"],depth_results["validation_F1"],marker="o",color=COURSE_COLORS["blue"],label="Random-forest F1")
axis.set(title="Tree-depth tuning on validation data",xlabel="Maximum depth",ylabel="Validation F1"); axis.legend(); plt.show()
"""
    return project_base(
        title="Final Project 1: End-to-End Tabular Classification",
        objectives=["Audit a health benchmark", "Compare a constant baseline and three model families", "Tune, evaluate, and analyze classification errors"],
        problem="Predict the diagnostic class in the Breast Cancer Wisconsin benchmark. Primary success metric is F1; ROC-AUC and the confusion matrix are guardrails.",
        use_cases=["Educational diagnostic modeling", "Tabular pipeline comparison", "Error-cost discussion"],
        intuition="Start with a majority baseline, preserve the untouched test set, compare linear and tree-based candidates, then inspect false positives and negatives.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Tree depth is varied using validation F1. The test set is not consulted during this experiment.",
        hyperparameter_code=hyper,
        interpretation="The comparison establishes whether complexity improves on the baseline; the confusion matrix exposes the exact final error types.",
        avoid_when="used as a clinical system without prospective validation, calibration, subgroup audits, and medical governance.",
    )


def segmentation_spec() -> LessonSpec:
    scratch = r"""
def kmeans_scratch(features,k,iterations=60):
    centroids=features[np.linspace(0,len(features)-1,k,dtype=int)].copy()
    for _ in range(iterations):
        labels=np.linalg.norm(features[:,None,:]-centroids[None,:,:],axis=2).argmin(1)
        updated=np.vstack([features[labels==group].mean(0) for group in range(k)])
        if np.allclose(updated,centroids): break
        centroids=updated
    return labels,centroids
print("Scratch K-means ready")
"""
    experiment = r"""
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
"""
    hyper = r"""
rows=[]
for k in range(2,8):
    model=KMeans(n_clusters=k,n_init=20,random_state=RANDOM_SEED).fit(X)
    rows.append({"k":k,"silhouette":silhouette_score(X,model.labels_),"inertia":model.inertia_})
tuning=pd.DataFrame(rows); display(tuning.round(3))
figure,axis=plt.subplots(); axis.plot(tuning["k"],tuning["silhouette"],marker="o",color=COURSE_COLORS["blue"],label="Silhouette")
axis.set(title="Segment-count selection",xlabel="Number of segments",ylabel="Silhouette"); axis.legend(); plt.show()
"""
    return project_base(
        title="Final Project 2: Customer-Style Segmentation",
        objectives=["Build a reproducible segmentation workflow", "Compare hard and probabilistic clustering", "Profile and critique the selected groups"],
        problem="Use wine chemistry as a safe proxy for customer-style segmentation. Success requires compact groups plus interpretable profiles, not a hidden answer label.",
        use_cases=["Exploratory segmentation", "Persona research inputs", "Assortment analysis"],
        intuition="Standardize behavior-like features, compare plausible grouping assumptions, then name profiles only after quantitative and domain review.",
        scratch_description="NumPy K-means exposes centroid assignment and update as the baseline.",
        dataset_description="The bundled Wine dataset replaces private customer data and has no network dependency.",
        preprocessing_description="StandardScaler is fit on the complete exploratory feature set because this project describes that fixed set rather than predicts future labels.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="The number of groups is compared with silhouette and inertia; interpretability and stability remain required.",
        hyperparameter_code=hyper,
        interpretation="PCA shows a lossy view of groups while the profile chart returns to original feature units for interpretation.",
        avoid_when="segments would drive consequential decisions without stability checks, domain review, privacy analysis, and outcome validation.",
    )


def image_spec() -> LessonSpec:
    scratch = r"""
def softmax(values):
    shifted=values-values.max(axis=1,keepdims=True)
    exp=np.exp(shifted)
    return exp/exp.sum(axis=1,keepdims=True)
print("Softmax check:",softmax(np.array([[1.,2.,3.]])).round(3))
"""
    experiment = r"""
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
"""
    hyper = r"""
comparison=pd.DataFrame([{"channels":channels,"validation_accuracy":values[3],"parameters":sum(p.numel() for p in values[0].parameters())} for channels,values in models.items()])
display(comparison.round(3))
figure,axis=plt.subplots(); axis.plot(comparison["channels"],comparison["validation_accuracy"],marker="o",color=COURSE_COLORS["blue"],label="Validation accuracy")
axis.set(title="CNN channel-width tuning",xlabel="Convolution channels",ylabel="Validation accuracy"); axis.legend(); plt.show()
"""
    return project_base(
        title="Final Project 3: Image Classification",
        objectives=["Compare a linear pixel baseline and CNN", "Tune convolution channels on validation data", "Inspect feature shapes, learning curves, and test errors"],
        problem="Classify 8-by-8 handwritten digits. Success is held-out accuracy with class-level error analysis under a CPU budget.",
        use_cases=["Document processing prototypes", "Educational visual recognition", "Compact edge-model experiments"],
        intuition="A linear baseline tests raw pixels; a CNN earns complexity only if local shared filters improve validation performance.",
        scratch_description="Manual softmax checks probability semantics before library loss functions.",
        dataset_description="The bundled Digits dataset contains 1,797 small images and needs no remote download.",
        preprocessing_description="Development/test and train/validation splits precede any fitted scaling; CNN pixels use the known 0–16 range.",
        setup_code="",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Convolution channel width is selected on validation accuracy; parameter count exposes its cost.",
        hyperparameter_code=hyper,
        interpretation="Learning curves show fit, the example image connects output to input, and the confusion matrix reveals systematic class errors.",
        avoid_when="higher-resolution, shifted, or safety-critical images are assumed to behave like this small benchmark without external validation.",
    )


def rl_project_spec() -> LessonSpec:
    scratch = r"""
def td_update(old,target,alpha):
    return old+alpha*(target-old)
print("TD update check:",td_update(.2,.8,.5))
"""
    experiment = r"""
class GridLine:
    def __init__(self,length=10,max_steps=40,generator=None):
        self.length=length; self.max_steps=max_steps; self.generator=generator or np.random.default_rng(RANDOM_SEED)
    def reset(self): self.state=0; self.steps=0; return self.state
    def step(self,action):
        self.steps+=1
        if self.generator.random()<.08: action=1-action
        self.state=int(np.clip(self.state+(-1 if action==0 else 1),0,self.length-1))
        terminated=self.state==self.length-1; truncated=self.steps>=self.max_steps
        return self.state,(1.0 if terminated else -.01),terminated,truncated

def train(method,seed,episodes=300):
    local=np.random.default_rng(seed); environment=GridLine(generator=local); Q=np.zeros((environment.length,2)); rewards=[]; epsilons=[]; lengths=[]; snapshots={0:Q.copy()}
    for episode in range(episodes):
        state=environment.reset(); epsilon=max(.03,.9*(.985**episode)); action=int(local.integers(2)) if local.random()<epsilon else int(np.argmax(Q[state])); total=0
        for step in range(environment.max_steps):
            next_state,reward,terminated,truncated=environment.step(action)
            next_action=int(local.integers(2)) if local.random()<epsilon else int(np.argmax(Q[next_state]))
            bootstrap=Q[next_state,next_action] if method=="SARSA" else Q[next_state].max()
            target=reward+(0 if terminated else .97*bootstrap)
            Q[state,action]=td_update(Q[state,action],target,.25)
            state=next_state; action=next_action; total+=reward
            if terminated or truncated: break
        rewards.append(total); epsilons.append(epsilon); lengths.append(step+1)
        if episode+1==episodes//2: snapshots[episode+1]=Q.copy()
    snapshots[episodes]=Q.copy()
    return Q,np.array(rewards),np.array(epsilons),np.array(lengths),snapshots

rows=[]; runs={}
for method in ["Q-learning","SARSA"]:
    for seed in [7,21,42]:
        runs[(method,seed)]=train(method,seed)
        rows.append({"method":method,"seed":seed,"final_reward":runs[(method,seed)][1][-30:].mean(),"final_length":runs[(method,seed)][3][-30:].mean()})
summary=pd.DataFrame(rows); display(summary.round(3)); display(summary.groupby("method")[["final_reward","final_length"]].agg(["mean","std"]).round(3))
selected=summary.groupby("method")["final_reward"].mean().idxmax(); Q,rewards,epsilons,lengths,snapshots=runs[(selected,42)]
print("Selected:",selected)
print("Q-table before training:\n",snapshots[0].round(2))
print("Q-table during training:\n",snapshots[150].round(2))
print("Q-table after training:\n",snapshots[300].round(2))
window=20; moving=np.convolve(rewards,np.ones(window)/window,mode="valid"); episodes=np.arange(1,len(rewards)+1)
figure,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
axes[0].plot(episodes,rewards,alpha=.3,label="Reward",color=COURSE_COLORS["blue"]); axes[0].plot(episodes[window-1:],moving,label="Moving average",color=COURSE_COLORS["orange"]); axes[0].set(title=f"Selected {selected} agent",ylabel="Reward"); axes[0].legend()
axes[1].plot(episodes,epsilons,color=COURSE_COLORS["olive"]); axes[1].set(ylabel="Exploration rate")
axes[2].plot(episodes,lengths,color=COURSE_COLORS["pink"]); axes[2].set(xlabel="Episode",ylabel="Episode length")
figure.tight_layout(); plt.show()
"""
    hyper = r"""
aggregate=summary.groupby("method")["final_reward"].agg(["mean","std"])
figure,axis=plt.subplots(); axis.bar(aggregate.index,aggregate["mean"],yerr=aggregate["std"],capsize=5,color=COURSE_COLORS["blue"])
axis.set(title="Three-seed final-agent comparison",xlabel="Algorithm",ylabel="Final mean reward"); plt.show()
print("Limitations: one-dimensional states, shaped step cost, fixed slip probability, and only three seeds.")
"""
    return project_base(
        title="Final Project 4: Reinforcement-Learning Agent",
        objectives=["Define a stochastic environment and success metric", "Compare Q-learning and SARSA across matched seeds", "Inspect Q-values and all required learning diagnostics"],
        problem="Reach a delayed goal efficiently despite an 8% action-slip probability. Success is high final reward and short episodes across seeds.",
        use_cases=["Sequential-control prototyping", "Algorithm comparison", "Reproducible RL evaluation"],
        intuition="A credible agent project documents the environment and reward, includes a baseline comparison, controls randomness, and reports noisy learning rather than one lucky score.",
        equation=r"$$\delta=R+\gamma Q_{next}-Q(S,A),\qquad Q(S,A)\leftarrow Q(S,A)+\alpha\delta.$$",
        symbols=[r"$\delta$ is temporal-difference error.", r"$R$ is reward.", r"$\gamma$ is discount.", r"$Q_{next}$ is the method-specific bootstrap.", r"$\alpha$ is learning rate."],
        why_equation="Both algorithms share the TD correction while differing in how next action value is chosen.",
        numerical_example="With old Q=.2, reward 0, gamma=.9, next=.8, alpha=.5, new Q=.46.",
        python_connection="td_update applies the shared correction; the training function selects max-Q for Q-learning or the behavior action for SARSA.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Matched-seed means and standard deviations guide selection; limitations remain visible alongside the comparison.",
        hyperparameter_code=hyper,
        interpretation="Reward, moving reward, exploration, and length explain both performance and how behavior changes; the Q-table makes the learned policy inspectable.",
        avoid_when="unsafe real-world exploration would occur without simulation, constraints, monitoring, and staged evaluation.",
    )


def main() -> None:
    lessons = {
        "05_final_projects/01_end_to_end_tabular_project.ipynb": tabular_spec(),
        "05_final_projects/02_customer_segmentation_project.ipynb": segmentation_spec(),
        "05_final_projects/03_image_classification_project.ipynb": image_spec(),
        "05_final_projects/04_reinforcement_learning_agent.ipynb": rl_project_spec(),
    }
    for path,spec in lessons.items():
        write_lesson(path,spec)
        print("Built",path)


if __name__=="__main__":
    main()
