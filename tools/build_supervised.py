"""Build the complete supervised-learning notebook sequence."""

from __future__ import annotations

from course_factory import LessonSpec, write_lesson


def base_spec(**overrides) -> LessonSpec:
    spec = {
        "title": "Supervised Learning",
        "objectives": ["Explain the algorithm intuitively and mathematically", "Implement its central mechanism with NumPy", "Compare it with a scikit-learn implementation"],
        "prerequisites": "Basic Python, NumPy arrays, means, and the supervised-learning overview.",
        "problem": "Learn a mapping from measured features to known targets and generalize to unseen examples.",
        "use_cases": ["Risk classification", "Demand prediction", "Decision-support ranking"],
        "intuition": "Examples pair inputs with correct outputs; a model adjusts parameters or stores structure so future inputs receive useful predictions.",
        "equation": r"$$\hat y=f(\mathbf{x};\boldsymbol{\theta}).$$",
        "symbols": [r"$\mathbf{x}$ is one feature vector.", r"$\boldsymbol{\theta}$ contains learned parameters.", r"$f$ is the model.", r"$\hat y$ is the prediction."],
        "why_equation": "It separates observed inputs, learned quantities, and model outputs.",
        "numerical_example": "If f(x)=2x+1 and x=3, the prediction is 7.",
        "python_connection": "model.predict(X) evaluates the learned mapping for every row.",
        "steps": ["Define target and success metric", "Split observations before fitted preprocessing", "Build a simple baseline", "Fit a transparent implementation", "Fit the library estimator", "Tune with validation evidence", "Evaluate once on untouched test data"],
        "scratch_description": "A readable NumPy version exposes the mechanism and is checked against a tiny hand calculation.",
        "library_description": "Scikit-learn adds efficient numerical routines, validation, pipelines, and model-selection compatibility.",
        "dataset_description": "The lesson uses a bundled or generated scikit-learn dataset with a fixed seed and no network access.",
        "preprocessing_description": "Train/validation/test roles are separated before scaling or other fitted transformations to prevent data leakage.",
        "setup_code": "",
        "scratch_code": "def squared_error(y_true,y_pred):\n    return float(np.mean((np.asarray(y_true)-np.asarray(y_pred))**2))\nprint('MSE check:',squared_error([1,3],[2,2]))",
        "experiment_code": "print('Topic-specific training experiment follows.')",
        "interpretation": "Compare scratch and library results, then interpret performance relative to the baseline and error costs.",
        "hyperparameter_text": "Vary one meaningful model-control parameter while keeping the validation data fixed.",
        "hyperparameter_code": "values=[1,2,3,4]; scores=[.6,.72,.74,.7]\nfigure,axis=plt.subplots(); axis.plot(values,scores,marker='o',color=COURSE_COLORS['blue'],label='Validation score'); axis.set(title='Illustrative model selection',xlabel='Complexity',ylabel='Validation score'); axis.legend(); plt.show()",
        "mistakes": ["Fitting preprocessing before splitting", "Selecting hyperparameters on test data", "Reporting one metric without a baseline", "Confusing predictive association with causation"],
        "advantages": ["Direct objective from labeled examples", "Clear held-out evaluation"],
        "disadvantages": ["Labels can be costly or biased", "Future data may differ from training data"],
        "use_when": "a trustworthy target represents the decision and representative labeled examples are available.",
        "avoid_when": "labels do not represent the desired outcome or the goal is causal intervention rather than prediction.",
        "exercises": ["Why must the test set remain untouched during selection?", "Calculate the displayed equation on a small example.", "Change one hyperparameter and explain training-versus-validation behavior."],
        "solutions": ["Repeated test feedback leaks selection information and makes final performance optimistic.", "Substitute each number, compute in the displayed order, and retain units.", "Keep the split fixed, retrain each candidate, and prefer stable validation evidence rather than the best training score."],
        "takeaways": ["Data splitting and preprocessing are part of the model", "Scratch code explains the mechanism while the library version supports realistic use", "Metrics must match error costs", "Generalization—not memorization—is the objective"],
        "further_reading": ["[scikit-learn supervised learning guide](https://scikit-learn.org/stable/supervised_learning.html)", "[scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)"],
    }
    spec.update(overrides)
    return spec


def linear_regression_spec() -> LessonSpec:
    scratch = r"""
class LinearRegressionScratch:
    def __init__(self,learning_rate=.05,epochs=800,l2=0.0):
        self.learning_rate=learning_rate; self.epochs=epochs; self.l2=l2
    def fit(self,features,target):
        self.weights_=np.zeros(features.shape[1]); self.bias_=0.0; self.losses_=[]
        n=len(features)
        for _ in range(self.epochs):
            prediction=features@self.weights_+self.bias_
            error=prediction-target
            loss=np.mean(error**2)+self.l2*np.sum(self.weights_**2)
            gradient_w=2*features.T@error/n+2*self.l2*self.weights_
            gradient_b=2*error.mean()
            self.weights_-=self.learning_rate*gradient_w
            self.bias_-=self.learning_rate*gradient_b
            self.losses_.append(loss)
        return self
    def predict(self,features):
        return features@self.weights_+self.bias_

x_small=np.array([[1.],[2.]])
y_small=np.array([3.,5.])
model_small=LinearRegressionScratch(learning_rate=.05,epochs=1000).fit(x_small,y_small)
print("Tiny weight/bias:",model_small.weights_.round(3),round(model_small.bias_,3))
"""
    experiment = r"""
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X,y,truth=make_regression(n_samples=320,n_features=3,n_informative=3,noise=18,coef=True,random_state=RANDOM_SEED)
X_train,X_validation,y_train,y_validation=train_test_split(X,y,test_size=.25,random_state=RANDOM_SEED)
scaler=StandardScaler(); X_train_s=scaler.fit_transform(X_train); X_validation_s=scaler.transform(X_validation)
scratch=LinearRegressionScratch(learning_rate=.03,epochs=1200).fit(X_train_s,y_train)
library=LinearRegression().fit(X_train_s,y_train)
scratch_prediction=scratch.predict(X_validation_s); library_prediction=library.predict(X_validation_s)
rows=[]
for name,prediction in [("Mean baseline",np.full(len(y_validation),y_train.mean())),("NumPy gradient descent",scratch_prediction),("scikit-learn",library_prediction)]:
    rows.append({"model":name,"MAE":mean_absolute_error(y_validation,prediction),"MSE":mean_squared_error(y_validation,prediction),"RMSE":mean_squared_error(y_validation,prediction)**.5,"R2":r2_score(y_validation,prediction)})
comparison=pd.DataFrame(rows); display(comparison.round(3))
print("Feature matrix/vector shapes:",X_train_s.shape,y_train.shape)
print("Scratch/library weights:",scratch.weights_.round(2),library.coef_.round(2))
figure,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].plot(scratch.losses_,color=COURSE_COLORS["blue"],label="Training cost"); axes[0].set(title="Gradient descent reduces regularized MSE",xlabel="Iteration",ylabel="Cost"); axes[0].legend()
axes[1].scatter(y_validation,library_prediction,alpha=.7,color=COURSE_COLORS["blue"],label="Predictions"); limits=[min(y_validation.min(),library_prediction.min()),max(y_validation.max(),library_prediction.max())]; axes[1].plot(limits,limits,linestyle="--",color=COURSE_COLORS["orange"],label="Perfect prediction")
axes[1].set(title="Validation predictions versus targets",xlabel="True target",ylabel="Predicted target"); axes[1].legend()
figure.tight_layout(); plt.show()
"""
    hyper = r"""
alphas=np.logspace(-3,3,9)
rows=[]
for alpha in alphas:
    candidate=Ridge(alpha=alpha).fit(X_train_s,y_train)
    rows.append({"alpha":alpha,"validation_RMSE":mean_squared_error(y_validation,candidate.predict(X_validation_s))**.5,"coefficient_norm":np.linalg.norm(candidate.coef_)})
ridge_results=pd.DataFrame(rows); display(ridge_results.round(3))
figure,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].semilogx(ridge_results["alpha"],ridge_results["validation_RMSE"],marker="o",color=COURSE_COLORS["blue"],label="RMSE"); axes[0].set(title="Ridge validation error",xlabel="L2 strength alpha",ylabel="Validation RMSE"); axes[0].legend()
axes[1].semilogx(ridge_results["alpha"],ridge_results["coefficient_norm"],marker="s",color=COURSE_COLORS["orange"],label="Weight norm"); axes[1].set(title="Regularization shrinks coefficients",xlabel="L2 strength alpha",ylabel="Coefficient L2 norm"); axes[1].legend()
figure.tight_layout(); plt.show()
"""
    return base_spec(
        title="Linear and Multiple Linear Regression",
        objectives=["Use scalars, vectors, matrices, dot products, mean, variance, and standard deviation", "Derive MSE gradients and implement gradient descent", "Evaluate multiple regression and L2 regularization"],
        problem="Predict a continuous target as a weighted combination of one or more features.",
        use_cases=["Price estimation", "Demand forecasting", "Interpretable effect-sized prediction baselines"],
        intuition="Fit a flat surface through the feature space so vertical residuals are collectively small.",
        equation=r"$$\hat{\mathbf y}=\mathbf X\mathbf w+b,\qquad J(\mathbf w,b)=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2+\lambda\lVert\mathbf w\rVert_2^2.$$",
        symbols=[r"$\mathbf X$ is the n-by-p feature matrix.", r"$\mathbf w$ is a p-element weight vector.", r"$b$ is scalar intercept.", r"$\hat{\mathbf y}$ is prediction vector.", r"$J$ is cost.", r"$\lambda$ controls L2 regularization.", r"$n$ is row count."],
        why_equation="The matrix-vector dot product produces all predictions; MSE penalizes residuals and L2 discourages unstable large weights.",
        numerical_example="For x=[2,3], w=[4,-1], b=2, prediction=2(4)+3(-1)+2=7. If y=9, squared error is 4.",
        python_connection="X @ weights + bias is prediction; 2*X.T@(prediction-y)/n is the weight gradient.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Ridge alpha trades bias against variance. Stronger regularization shrinks weights and may first improve, then harm, validation error.",
        hyperparameter_code=hyper,
        interpretation="Scratch and library coefficients should be close. The diagonal plot reveals error spread; RMSE emphasizes large residuals while MAE is more robust.",
        exercises=["Why must features be arranged as an n-by-p matrix?", "For y=[2,4] and predictions [3,1], compute MAE, MSE, and RMSE.", "Add a redundant correlated feature and compare unregularized versus Ridge coefficient stability."],
        solutions=["Rows represent observations and columns features, making Xw one prediction per row.", "Absolute errors [1,3] give MAE=2; squared errors [1,9] give MSE=5 and RMSE=sqrt(5)=2.236.", "Duplicate a column with small noise, refit both models, and compare coefficient norms plus validation RMSE; Ridge usually distributes and shrinks weights more stably."],
    )


def logistic_regression_spec() -> LessonSpec:
    scratch = r"""
def sigmoid(values):
    return 1/(1+np.exp(-np.clip(values,-30,30)))

class LogisticRegressionScratch:
    def __init__(self,learning_rate=.1,epochs=1200):
        self.learning_rate=learning_rate; self.epochs=epochs
    def fit(self,features,target):
        self.weights_=np.zeros(features.shape[1]); self.bias_=0.0; self.losses_=[]
        for _ in range(self.epochs):
            probabilities=sigmoid(features@self.weights_+self.bias_)
            self.losses_.append(float(-np.mean(target*np.log(probabilities+1e-9)+(1-target)*np.log(1-probabilities+1e-9))))
            error=probabilities-target
            self.weights_-=self.learning_rate*(features.T@error/len(features))
            self.bias_-=self.learning_rate*error.mean()
        return self
    def predict_proba(self,features):
        return sigmoid(features@self.weights_+self.bias_)
    def predict(self,features,threshold=.5):
        return (self.predict_proba(features)>=threshold).astype(int)

print("Sigmoid at zero:",sigmoid(0.0))
"""
    experiment = r"""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay,PrecisionRecallDisplay,RocCurveDisplay,accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=load_breast_cancer()
X=data.data[:,[0,1]]; y=data.target
X_train,X_validation,y_train,y_validation=train_test_split(X,y,test_size=.25,stratify=y,random_state=RANDOM_SEED)
scaler=StandardScaler(); X_train_s=scaler.fit_transform(X_train); X_validation_s=scaler.transform(X_validation)
scratch=LogisticRegressionScratch(learning_rate=.1,epochs=1500).fit(X_train_s,y_train)
library=LogisticRegression(max_iter=2000,random_state=RANDOM_SEED).fit(X_train_s,y_train)
scratch_pred=scratch.predict(X_validation_s); library_pred=library.predict(X_validation_s); probability=library.predict_proba(X_validation_s)[:,1]
print("Scratch/library accuracy:",round(accuracy_score(y_validation,scratch_pred),3),round(accuracy_score(y_validation,library_pred),3))
print("Library F1:",round(f1_score(y_validation,library_pred),3),"input/output:",X_train_s.shape,probability.shape)
figure,axes=plt.subplots(1,3,figsize=(14,4))
axes[0].plot(scratch.losses_,color=COURSE_COLORS["blue"],label="Log loss"); axes[0].set(title="Scratch logistic-regression training",xlabel="Iteration",ylabel="Binary cross-entropy"); axes[0].legend()
RocCurveDisplay.from_predictions(y_validation,probability,ax=axes[1],curve_kwargs={"color":COURSE_COLORS["blue"]}); axes[1].set_title("Validation ROC curve")
PrecisionRecallDisplay.from_predictions(y_validation,probability,ax=axes[2],curve_kwargs={"color":COURSE_COLORS["orange"]}); axes[2].set_title("Validation precision-recall curve")
figure.tight_layout(); plt.show()
"""
    hyper = r"""
thresholds=np.linspace(.1,.9,9); rows=[]
for threshold in thresholds:
    prediction=(probability>=threshold).astype(int)
    rows.append({"threshold":threshold,"accuracy":accuracy_score(y_validation,prediction),"F1":f1_score(y_validation,prediction)})
threshold_results=pd.DataFrame(rows); display(threshold_results.round(3))
figure,axis=plt.subplots(); axis.plot(threshold_results["threshold"],threshold_results["accuracy"],marker="o",label="Accuracy",color=COURSE_COLORS["blue"]); axis.plot(threshold_results["threshold"],threshold_results["F1"],marker="s",label="F1",color=COURSE_COLORS["orange"])
axis.set(title="Decision threshold changes classification tradeoffs",xlabel="Positive-class threshold",ylabel="Score"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Logistic Regression",
        objectives=["Interpret logits, odds, probability, and conditional probability", "Derive binary cross-entropy gradients", "Tune decision thresholds with error costs"],
        problem="Estimate the conditional probability of a binary class and convert it into decisions.",
        intuition="A linear score moves examples along an S-shaped probability curve; thresholding separates the two predicted classes.",
        equation=r"$$P(Y=1\mid\mathbf x)=\sigma(\mathbf w^T\mathbf x+b)=\frac{1}{1+e^{-(\mathbf w^T\mathbf x+b)}}.$$",
        symbols=[r"$P(Y=1\mid\mathbf x)$ is conditional positive probability.", r"$\sigma$ is sigmoid.", r"$\mathbf w^T\mathbf x$ is a dot product.", r"$b$ is intercept.", r"$e$ is Euler's number."],
        why_equation="The sigmoid maps every real-valued linear score into (0,1), enabling probabilistic classification.",
        numerical_example="If the logit is 0, probability is 1/(1+1)=.5; logit ln(3) gives odds 3:1 and probability .75.",
        python_connection="sigmoid(X @ weights + bias) computes all probabilities; the log-loss gradient simplifies to X.T @ (p-y)/n.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="A threshold is a decision hyperparameter, not a learned probability parameter. Select it using validation costs.",
        hyperparameter_code=hyper,
        interpretation="ROC summarizes ranking across thresholds; precision-recall emphasizes the positive class; threshold selection changes deployed error counts.",
    )


def knn_spec() -> LessonSpec:
    scratch = r"""
class KNNScratch:
    def __init__(self,k=5,metric="euclidean"):
        self.k=k; self.metric=metric
    def fit(self,features,target):
        self.features_=np.asarray(features); self.target_=np.asarray(target); return self
    def _distance(self,row):
        difference=np.abs(self.features_-row)
        return np.sqrt(np.sum(difference**2,axis=1)) if self.metric=="euclidean" else np.sum(difference,axis=1)
    def predict(self,features):
        predictions=[]
        for row in features:
            nearest=np.argsort(self._distance(row))[:self.k]
            labels,counts=np.unique(self.target_[nearest],return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.asarray(predictions)

print("Euclidean/Manhattan:",np.sqrt((3-0)**2+(4-0)**2),abs(3)+abs(4))
"""
    experiment = r"""
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris=load_iris()
X=iris.data[:,[0,2]]; y=iris.target
X_train,X_validation,y_train,y_validation=train_test_split(X,y,test_size=.3,stratify=y,random_state=RANDOM_SEED)
scaler=StandardScaler(); X_train_s=scaler.fit_transform(X_train); X_validation_s=scaler.transform(X_validation)
scratch=KNNScratch(k=5).fit(X_train_s,y_train)
library=KNeighborsClassifier(n_neighbors=5).fit(X_train_s,y_train)
scratch_prediction=scratch.predict(X_validation_s); library_prediction=library.predict(X_validation_s)
print("Scratch/library accuracy:",accuracy_score(y_validation,scratch_prediction),accuracy_score(y_validation,library_prediction))

x_grid=np.linspace(X_train_s[:,0].min()-.5,X_train_s[:,0].max()+.5,180)
y_grid=np.linspace(X_train_s[:,1].min()-.5,X_train_s[:,1].max()+.5,180)
xx,yy=np.meshgrid(x_grid,y_grid); grid=np.c_[xx.ravel(),yy.ravel()]
regions=library.predict(grid).reshape(xx.shape)
figure,axis=plt.subplots(); axis.contourf(xx,yy,regions,alpha=.2,cmap="cividis"); scatter=axis.scatter(X_validation_s[:,0],X_validation_s[:,1],c=y_validation,cmap="cividis",edgecolor="white",label="Validation observations")
axis.set(title="KNN decision regions",xlabel="Standardized sepal length",ylabel="Standardized petal length"); axis.legend(*scatter.legend_elements(),title="Iris class"); plt.show()
"""
    hyper = r"""
rows=[]
for k in range(1,26,2):
    candidate=KNeighborsClassifier(n_neighbors=k).fit(X_train_s,y_train)
    rows.append({"k":k,"train_accuracy":candidate.score(X_train_s,y_train),"validation_accuracy":candidate.score(X_validation_s,y_validation)})
results=pd.DataFrame(rows); display(results.round(3))
figure,axis=plt.subplots(); axis.plot(results["k"],results["train_accuracy"],marker="o",label="Training",color=COURSE_COLORS["blue"]); axis.plot(results["k"],results["validation_accuracy"],marker="s",label="Validation",color=COURSE_COLORS["orange"])
axis.set(title="Neighbor count controls bias and variance",xlabel="Number of neighbors k",ylabel="Accuracy"); axis.legend(); plt.show()
"""
    return base_spec(
        title="K-Nearest Neighbors",
        objectives=["Calculate Euclidean and Manhattan distance", "Implement neighbor voting", "Explain scaling and the curse of dimensionality"],
        problem="Predict from the labels of the most similar stored training observations.",
        intuition="Find nearby examples and let them vote; small k follows local detail while large k smooths broadly.",
        equation=r"$$d_2(\mathbf x,\mathbf z)=\sqrt{\sum_{j=1}^{p}(x_j-z_j)^2},\qquad d_1(\mathbf x,\mathbf z)=\sum_{j=1}^{p}|x_j-z_j|.$$",
        symbols=[r"$d_2$ is Euclidean distance.", r"$d_1$ is Manhattan distance.", r"$\mathbf x,\mathbf z$ are feature vectors.", r"$p$ is feature count.", r"$j$ indexes features."],
        why_equation="Nearest-neighbor predictions require a precise definition of similarity.",
        numerical_example="Between (0,0) and (3,4), Euclidean distance is 5 and Manhattan distance is 7.",
        python_connection="np.sqrt(np.sum((X-row)**2,axis=1)) and np.sum(np.abs(X-row),axis=1) implement the metrics.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Small k has low bias and high variance; large k has higher bias and lower variance. Validation selects the compromise.",
        hyperparameter_code=hyper,
        interpretation="Regions show local voting. Scaling is essential because a large-unit feature would otherwise dominate every distance.",
        disadvantages=["Prediction can be slow and memory-heavy", "Distances become less informative as dimensionality grows", "Sensitive to scaling and irrelevant features"],
    )


def naive_bayes_spec() -> LessonSpec:
    scratch = r"""
class GaussianNBScratch:
    def fit(self,features,target):
        self.classes_=np.unique(target)
        self.means_=np.vstack([features[target==label].mean(0) for label in self.classes_])
        self.variances_=np.vstack([features[target==label].var(0)+1e-9 for label in self.classes_])
        self.priors_=np.array([(target==label).mean() for label in self.classes_])
        return self
    def predict_log_proba(self,features):
        scores=[]
        for class_index in range(len(self.classes_)):
            log_prior=np.log(self.priors_[class_index])
            mean=self.means_[class_index]; variance=self.variances_[class_index]
            log_likelihood=-.5*np.sum(np.log(2*np.pi*variance)+(features-mean)**2/variance,axis=1)
            scores.append(log_prior+log_likelihood)
        return np.column_stack(scores)
    def predict(self,features):
        return self.classes_[self.predict_log_proba(features).argmax(1)]

print("Bayes numerical example P(disease|positive):",round((.9*.01)/(.9*.01+.05*.99),3))
"""
    experiment = r"""
from sklearn.datasets import load_wine
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

wine=load_wine()
X=wine.data; y=wine.target
X_train,X_validation,y_train,y_validation=train_test_split(X,y,test_size=.3,stratify=y,random_state=RANDOM_SEED)
scratch=GaussianNBScratch().fit(X_train,y_train)
library=GaussianNB().fit(X_train,y_train)
scratch_prediction=scratch.predict(X_validation); library_prediction=library.predict(X_validation)
print("Means/variances shapes:",scratch.means_.shape,scratch.variances_.shape)
print("Scratch/library accuracy:",round(accuracy_score(y_validation,scratch_prediction),3),round(accuracy_score(y_validation,library_prediction),3))
figure,axes=plt.subplots(1,2,figsize=(10,4))
for label,name,color in zip(np.unique(y),wine.target_names,[COURSE_COLORS["blue"],COURSE_COLORS["orange"],COURSE_COLORS["olive"]]):
    axes[0].hist(X_train[y_train==label,0],bins=14,alpha=.45,label=name,color=color,density=True)
axes[0].set(title="Class-conditional alcohol distributions",xlabel="Alcohol",ylabel="Probability density"); axes[0].legend()
ConfusionMatrixDisplay.from_predictions(y_validation,library_prediction,display_labels=wine.target_names,cmap="Blues",colorbar=False,ax=axes[1]); axes[1].set_title("Gaussian Naive Bayes validation errors")
figure.tight_layout(); plt.show()
"""
    hyper = r"""
smoothing=np.logspace(-12,-6,7); rows=[]
for value in smoothing:
    candidate=GaussianNB(var_smoothing=value).fit(X_train,y_train)
    rows.append({"var_smoothing":value,"validation_accuracy":candidate.score(X_validation,y_validation)})
smooth_results=pd.DataFrame(rows); display(smooth_results)
figure,axis=plt.subplots(); axis.semilogx(smooth_results["var_smoothing"],smooth_results["validation_accuracy"],marker="o",color=COURSE_COLORS["blue"],label="Accuracy")
axis.set(title="Variance smoothing experiment",xlabel="var_smoothing",ylabel="Validation accuracy"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Naive Bayes",
        objectives=["Apply probability, conditional probability, Bayes' theorem, and Gaussian distributions", "Implement Gaussian Naive Bayes", "Explain the conditional-independence assumption"],
        problem="Classify by comparing how probable the observed features are under each class.",
        intuition="Begin with each class's prior plausibility, multiply by how compatible each feature is with that class, then normalize conceptually.",
        equation=r"$$P(C_k\mid\mathbf x)=\frac{P(\mathbf x\mid C_k)P(C_k)}{P(\mathbf x)},\qquad P(\mathbf x\mid C_k)\approx\prod_{j=1}^{p}P(x_j\mid C_k).$$",
        symbols=[r"$C_k$ is class k.", r"$P(C_k)$ is prior probability.", r"$P(\mathbf x\mid C_k)$ is likelihood.", r"$P(\mathbf x)$ is evidence.", r"$P(C_k\mid\mathbf x)$ is posterior.", r"$p$ is feature count."],
        why_equation="Bayes' theorem reverses likelihood into the class probability needed for prediction; the naive product makes estimation practical.",
        numerical_example="With sensitivity .9, prevalence .01, and false-positive rate .05, P(disease|positive)=.009/(.009+.0495)=.154.",
        python_connection="The scratch model adds log prior and Gaussian log likelihoods, avoiding unstable multiplication of tiny probabilities.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Variance smoothing prevents division by extremely small estimated variances and improves numerical stability.",
        hyperparameter_code=hyper,
        interpretation="Overlapping class-conditional distributions create errors. Agreement with scikit-learn validates the simplified log-probability implementation.",
    )


TREE_SCRATCH = r"""
def entropy(target):
    _,counts=np.unique(target,return_counts=True)
    probabilities=counts/counts.sum()
    return float(-np.sum(probabilities*np.log2(probabilities+1e-12)))

class DecisionStumpScratch:
    def fit(self,features,target):
        best=(-np.inf,None,None,None,None)
        parent=entropy(target)
        for feature in range(features.shape[1]):
            for threshold in np.unique(features[:,feature]):
                left=features[:,feature]<=threshold
                if left.all() or (~left).all(): continue
                child=(left.mean()*entropy(target[left])+(~left).mean()*entropy(target[~left]))
                gain=parent-child
                left_label=np.bincount(target[left]).argmax(); right_label=np.bincount(target[~left]).argmax()
                if gain>best[0]: best=(gain,feature,threshold,left_label,right_label)
        self.gain_,self.feature_,self.threshold_,self.left_label_,self.right_label_=best
        return self
    def predict(self,features):
        return np.where(features[:,self.feature_]<=self.threshold_,self.left_label_,self.right_label_)

print("Entropy balanced/pure:",round(entropy(np.array([0,0,1,1])),3),round(entropy(np.array([1,1,1])),3))
"""


def decision_tree_spec() -> LessonSpec:
    experiment = r"""
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree

iris=load_iris()
X=iris.data; y=iris.target
X_train,X_validation,y_train,y_validation=train_test_split(X,y,test_size=.3,stratify=y,random_state=RANDOM_SEED)
stump=DecisionStumpScratch().fit(X_train,y_train)
tree=DecisionTreeClassifier(max_depth=3,random_state=RANDOM_SEED).fit(X_train,y_train)
print("Stump split:",iris.feature_names[stump.feature_],"<=",round(stump.threshold_,2),"gain:",round(stump.gain_,3))
print("Stump/tree accuracy:",round(accuracy_score(y_validation,stump.predict(X_validation)),3),round(tree.score(X_validation,y_validation),3))
figure,axes=plt.subplots(1,2,figsize=(14,5))
plot_tree(tree,feature_names=iris.feature_names,class_names=iris.target_names,filled=True,rounded=True,ax=axes[0]); axes[0].set_title("Depth-3 decision tree")
importance=pd.Series(tree.feature_importances_,index=iris.feature_names).sort_values()
axes[1].barh(importance.index,importance.values,color=COURSE_COLORS["blue"]); axes[1].set(title="Impurity-based feature importance",xlabel="Total normalized impurity decrease",ylabel="Feature")
figure.tight_layout(); plt.show()
"""
    hyper = r"""
rows=[]
for depth in range(1,11):
    candidate=DecisionTreeClassifier(max_depth=depth,random_state=RANDOM_SEED).fit(X_train,y_train)
    rows.append({"depth":depth,"train_accuracy":candidate.score(X_train,y_train),"validation_accuracy":candidate.score(X_validation,y_validation)})
depth_results=pd.DataFrame(rows); display(depth_results.round(3))
figure,axis=plt.subplots(); axis.plot(depth_results["depth"],depth_results["train_accuracy"],marker="o",label="Training",color=COURSE_COLORS["blue"]); axis.plot(depth_results["depth"],depth_results["validation_accuracy"],marker="s",label="Validation",color=COURSE_COLORS["orange"])
axis.set(title="Tree depth reveals overfitting",xlabel="Maximum depth",ylabel="Accuracy"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Decision Trees",
        objectives=["Calculate entropy and information gain", "Implement a decision stump", "Diagnose depth-driven overfitting"],
        problem="Partition feature space with interpretable if-then rules.",
        intuition="Ask the question that most reduces class uncertainty, then repeat inside each resulting group.",
        equation=r"$$H(Y)=-\sum_{k=1}^{K}p_k\log_2p_k,\qquad IG=H(parent)-\sum_m\frac{n_m}{n}H(child_m).$$",
        symbols=[r"$H$ is entropy.", r"$p_k$ is class-k proportion.", r"$K$ is class count.", r"$IG$ is information gain.", r"$m$ indexes children.", r"$n_m/n$ weights child size."],
        why_equation="Entropy quantifies uncertainty; information gain chooses the split that reduces it most.",
        numerical_example="A 50/50 binary node has entropy 1 bit; a pure child has entropy 0.",
        python_connection="The scratch code computes class proportions, weighted child entropy, and selects maximum gain.",
        scratch_code=TREE_SCRATCH,
        experiment_code=experiment,
        hyperparameter_text="Depth increases flexibility: training accuracy rises monotonically while validation performance can peak and decline.",
        hyperparameter_code=hyper,
        interpretation="The plotted rules are directly readable. Feature importance records impurity reduction, not causal importance.",
    )


def random_forest_spec() -> LessonSpec:
    scratch = TREE_SCRATCH + r"""
class BaggedStumpsScratch:
    def __init__(self,n_estimators=31):
        self.n_estimators=n_estimators
    def fit(self,features,target,seed=42):
        generator=np.random.default_rng(seed); self.models_=[]
        for _ in range(self.n_estimators):
            indices=generator.integers(0,len(features),len(features))
            feature_subset=generator.choice(features.shape[1],max(1,int(np.sqrt(features.shape[1]))),replace=False)
            stump=DecisionStumpScratch().fit(features[indices][:,feature_subset],target[indices])
            self.models_.append((stump,feature_subset))
        return self
    def predict(self,features):
        votes=np.column_stack([model.predict(features[:,subset]) for model,subset in self.models_])
        return np.apply_along_axis(lambda row:np.bincount(row).argmax(),1,votes)
"""
    experiment = r"""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data=load_breast_cancer()
X=data.data; y=data.target
X_train,X_validation,y_train,y_validation=train_test_split(X,y,test_size=.25,stratify=y,random_state=RANDOM_SEED)
scratch=BaggedStumpsScratch(n_estimators=41).fit(X_train,y_train)
forest=RandomForestClassifier(n_estimators=180,max_features="sqrt",oob_score=True,random_state=RANDOM_SEED).fit(X_train,y_train)
print("Scratch forest/library accuracy:",round(accuracy_score(y_validation,scratch.predict(X_validation)),3),round(forest.score(X_validation,y_validation),3))
print("OOB score:",round(forest.oob_score_,3),"trees:",len(forest.estimators_))
importance=permutation_importance(forest,X_validation,y_validation,n_repeats=5,random_state=RANDOM_SEED)
top=np.argsort(importance.importances_mean)[-10:]
figure,axes=plt.subplots(1,2,figsize=(12,4.5))
axes[0].hist([estimator.tree_.max_depth for estimator in forest.estimators_],bins=12,color=COURSE_COLORS["blue"],edgecolor="white",label="Tree depths"); axes[0].set(title="Diversity of fitted tree depths",xlabel="Tree depth",ylabel="Number of trees"); axes[0].legend()
axes[1].barh(np.array(data.feature_names)[top],importance.importances_mean[top],color=COURSE_COLORS["orange"]); axes[1].set(title="Validation permutation importance",xlabel="Mean accuracy decrease",ylabel="Feature")
figure.tight_layout(); plt.show()
"""
    hyper = r"""
counts=[1,5,15,40,100,180]; rows=[]
for count in counts:
    candidate=RandomForestClassifier(n_estimators=count,max_features="sqrt",random_state=RANDOM_SEED).fit(X_train,y_train)
    rows.append({"trees":count,"validation_accuracy":candidate.score(X_validation,y_validation)})
forest_results=pd.DataFrame(rows); display(forest_results.round(3))
figure,axis=plt.subplots(); axis.plot(forest_results["trees"],forest_results["validation_accuracy"],marker="o",color=COURSE_COLORS["blue"],label="Validation accuracy")
axis.set(title="More trees reduce Monte Carlo variance",xlabel="Number of trees",ylabel="Validation accuracy"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Random Forests",
        objectives=["Explain bootstrap sampling and random feature subsets", "Build a bagged-stump forest", "Connect averaging with variance reduction"],
        problem="Improve unstable decision trees by averaging diverse fitted trees.",
        intuition="Train many trees on slightly different rows and features; their uncorrelated mistakes partly cancel in the vote.",
        equation=r"$$\hat p_c(\mathbf x)=\frac{1}{B}\sum_{b=1}^{B}\mathbb{1}[\hat y_b(\mathbf x)=c].$$",
        symbols=[r"$B$ is tree count.", r"$b$ indexes trees.", r"$\hat y_b$ is tree b's prediction.", r"$\mathbb{1}$ is an indicator.", r"$\hat p_c$ is vote share for class c."],
        why_equation="The vote is an empirical expected value across randomized trees; averaging reduces variance when errors are not perfectly correlated.",
        numerical_example="If 7 of 10 trees vote positive, estimated vote probability is .7.",
        python_connection="np.mean(tree_predictions == class_label, axis=1) computes vote shares.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Additional trees stabilize the Monte Carlo average but eventually give diminishing returns and cost more compute.",
        hyperparameter_code=hyper,
        interpretation="Bootstrap and feature randomness produce diverse depths. Permutation importance measures validation-score sensitivity, not causation.",
        disadvantages=["Less interpretable than one tree", "Larger memory and prediction cost", "Impurity importance can be biased"],
    )


def svm_spec() -> LessonSpec:
    scratch = r"""
class LinearSVMScratch:
    def __init__(self,learning_rate=.01,epochs=800,regularization=.01):
        self.learning_rate=learning_rate; self.epochs=epochs; self.regularization=regularization
    def fit(self,features,target):
        self.weights_=np.zeros(features.shape[1]); self.bias_=0.0; self.losses_=[]
        signed=np.where(target==1,1.0,-1.0)
        for _ in range(self.epochs):
            margins=signed*(features@self.weights_+self.bias_)
            active=margins<1
            gradient_w=self.regularization*self.weights_-(features[active].T@signed[active])/len(features)
            gradient_b=-signed[active].sum()/len(features)
            self.weights_-=self.learning_rate*gradient_w; self.bias_-=self.learning_rate*gradient_b
            self.losses_.append(.5*self.regularization*np.sum(self.weights_**2)+np.maximum(0,1-margins).mean())
        return self
    def predict(self,features):
        return (features@self.weights_+self.bias_>=0).astype(int)

print("Hinge losses for margins [-1,.5,2]:",np.maximum(0,1-np.array([-1,.5,2.])))
"""
    experiment = r"""
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X,y=make_moons(n_samples=320,noise=.18,random_state=RANDOM_SEED)
X_train,X_validation,y_train,y_validation=train_test_split(X,y,test_size=.3,stratify=y,random_state=RANDOM_SEED)
scaler=StandardScaler(); X_train_s=scaler.fit_transform(X_train); X_validation_s=scaler.transform(X_validation)
scratch=LinearSVMScratch(learning_rate=.02,epochs=1200).fit(X_train_s,y_train)
linear=SVC(kernel="linear",C=1).fit(X_train_s,y_train)
rbf=SVC(kernel="rbf",C=1,gamma="scale").fit(X_train_s,y_train)
print("Scratch linear/library linear/RBF accuracy:",round(accuracy_score(y_validation,scratch.predict(X_validation_s)),3),round(linear.score(X_validation_s,y_validation),3),round(rbf.score(X_validation_s,y_validation),3))
x_values=np.linspace(X_train_s[:,0].min()-.5,X_train_s[:,0].max()+.5,180); y_values=np.linspace(X_train_s[:,1].min()-.5,X_train_s[:,1].max()+.5,180); xx,yy=np.meshgrid(x_values,y_values); grid=np.c_[xx.ravel(),yy.ravel()]
figure,axes=plt.subplots(1,2,figsize=(11,4))
for axis,model,title in zip(axes,[linear,rbf],["Linear maximum-margin boundary","RBF kernel boundary"]):
    regions=model.predict(grid).reshape(xx.shape); axis.contourf(xx,yy,regions,alpha=.2,cmap="cividis"); axis.scatter(X_validation_s[:,0],X_validation_s[:,1],c=y_validation,cmap="cividis",edgecolor="white"); axis.set(title=title,xlabel="Standardized feature 1",ylabel="Standardized feature 2")
figure.tight_layout(); plt.show()
"""
    hyper = r"""
c_values=np.logspace(-2,2,7); rows=[]
for value in c_values:
    candidate=SVC(kernel="rbf",C=value,gamma="scale").fit(X_train_s,y_train)
    rows.append({"C":value,"training_accuracy":candidate.score(X_train_s,y_train),"validation_accuracy":candidate.score(X_validation_s,y_validation)})
svm_results=pd.DataFrame(rows); display(svm_results.round(3))
figure,axis=plt.subplots(); axis.semilogx(svm_results["C"],svm_results["training_accuracy"],marker="o",label="Training",color=COURSE_COLORS["blue"]); axis.semilogx(svm_results["C"],svm_results["validation_accuracy"],marker="s",label="Validation",color=COURSE_COLORS["orange"])
axis.set(title="SVM regularization tradeoff",xlabel="C (larger penalizes violations more)",ylabel="Accuracy"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Support Vector Machines",
        objectives=["Connect dot products with hyperplanes and margins", "Optimize hinge loss with a NumPy subgradient", "Compare linear and RBF kernels"],
        problem="Find a wide-margin decision boundary, optionally in a nonlinear feature space.",
        intuition="Choose a separating boundary that leaves the widest street between classes; points touching or violating the street guide the fit.",
        equation=r"$$\min_{\mathbf w,b}\frac{1}{2}\lVert\mathbf w\rVert^2+C\sum_i\max(0,1-y_i(\mathbf w^T\mathbf x_i+b)).$$",
        symbols=[r"$\mathbf w,b$ define the hyperplane.", r"$C$ penalizes margin violations.", r"$y_i\in\{-1,+1\}$ is signed class.", r"$\mathbf w^T\mathbf x_i+b$ is score.", r"$\max(0,\cdot)$ is hinge loss."],
        why_equation="Weight norm rewards a wide margin while hinge loss permits controlled violations for noisy data.",
        numerical_example="Margins -1, .5, and 2 produce hinge losses 2, .5, and 0.",
        python_connection="np.maximum(0, 1 - signed_y * score) computes hinge losses; active examples contribute the subgradient.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Small C accepts more violations for a wider, smoother margin; large C fits training boundaries more strictly.",
        hyperparameter_code=hyper,
        interpretation="The linear model cannot follow two moons, while the RBF kernel represents the curved boundary; validation controls added flexibility.",
    )


def gradient_boosting_spec() -> LessonSpec:
    scratch = r"""
class RegressionStump:
    def fit(self,features,target):
        best=(np.inf,None,None,None,None)
        for feature in range(features.shape[1]):
            for threshold in np.unique(features[:,feature]):
                left=features[:,feature]<=threshold
                if left.all() or (~left).all(): continue
                left_value=target[left].mean(); right_value=target[~left].mean()
                prediction=np.where(left,left_value,right_value)
                error=np.mean((target-prediction)**2)
                if error<best[0]: best=(error,feature,threshold,left_value,right_value)
        _,self.feature_,self.threshold_,self.left_value_,self.right_value_=best
        return self
    def predict(self,features):
        return np.where(features[:,self.feature_]<=self.threshold_,self.left_value_,self.right_value_)

class GradientBoostingScratch:
    def __init__(self,n_estimators=40,learning_rate=.1):
        self.n_estimators=n_estimators; self.learning_rate=learning_rate
    def fit(self,features,target):
        self.initial_=target.mean(); prediction=np.full(len(target),self.initial_); self.models_=[]; self.losses_=[]
        for _ in range(self.n_estimators):
            residual=target-prediction
            stump=RegressionStump().fit(features,residual)
            prediction+=self.learning_rate*stump.predict(features)
            self.models_.append(stump); self.losses_.append(np.mean((target-prediction)**2))
        return self
    def predict(self,features):
        prediction=np.full(len(features),self.initial_)
        for stump in self.models_: prediction+=self.learning_rate*stump.predict(features)
        return prediction

print("Negative MSE gradient is proportional to residual y-prediction.")
"""
    experiment = r"""
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

data=load_diabetes()
X_train,X_validation,y_train,y_validation=train_test_split(data.data,y:=data.target,test_size=.25,random_state=RANDOM_SEED)
scratch=GradientBoostingScratch(n_estimators=60,learning_rate=.1).fit(X_train,y_train)
library=GradientBoostingRegressor(n_estimators=100,learning_rate=.05,max_depth=2,random_state=RANDOM_SEED).fit(X_train,y_train)
rows=[]
for name,prediction in [("Mean baseline",np.full(len(y_validation),y_train.mean())),("Scratch boosting",scratch.predict(X_validation)),("scikit-learn boosting",library.predict(X_validation))]:
    rows.append({"model":name,"MAE":mean_absolute_error(y_validation,prediction),"RMSE":mean_squared_error(y_validation,prediction)**.5,"R2":r2_score(y_validation,prediction)})
comparison=pd.DataFrame(rows); display(comparison.round(3))
library_prediction=library.predict(X_validation)
figure,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].plot(scratch.losses_,color=COURSE_COLORS["blue"],label="Scratch training MSE"); axes[0].set(title="Sequential stumps reduce residual error",xlabel="Boosting stage",ylabel="Training MSE"); axes[0].legend()
axes[1].scatter(library_prediction,y_validation-library_prediction,color=COURSE_COLORS["orange"],alpha=.7,label="Residuals"); axes[1].axhline(0,color=COURSE_COLORS["charcoal"],linestyle="--"); axes[1].set(title="Validation residuals",xlabel="Predicted progression",ylabel="True minus predicted"); axes[1].legend()
figure.tight_layout(); plt.show()
"""
    hyper = r"""
rows=[]
for rate in [.02,.05,.1,.2]:
    for estimators in [30,80,160]:
        candidate=GradientBoostingRegressor(learning_rate=rate,n_estimators=estimators,max_depth=2,random_state=RANDOM_SEED).fit(X_train,y_train)
        rows.append({"learning_rate":rate,"estimators":estimators,"validation_RMSE":mean_squared_error(y_validation,candidate.predict(X_validation))**.5})
tuning=pd.DataFrame(rows); display(tuning.round(2))
pivot=tuning.pivot(index="learning_rate",columns="estimators",values="validation_RMSE")
figure,axis=plt.subplots(); sns.heatmap(pivot,annot=True,fmt=".1f",cmap="Blues_r",ax=axis); axis.set(title="Boosting shrinkage-stage tradeoff",xlabel="Number of estimators",ylabel="Learning rate"); plt.show()
"""
    return base_spec(
        title="Gradient Boosting",
        objectives=["Connect residual fitting to derivatives and gradients", "Implement regression boosting with stumps", "Tune learning rate and stage count jointly"],
        problem="Build a strong predictor by sequentially correcting errors from an existing ensemble.",
        intuition="Each new weak learner focuses on what the current ensemble still gets wrong, then contributes a small correction.",
        equation=r"$$F_m(\mathbf x)=F_{m-1}(\mathbf x)+\eta h_m(\mathbf x),\qquad h_m\approx-\frac{\partial L(y,F)}{\partial F}\bigg|_{F=F_{m-1}}.$$",
        symbols=[r"$F_m$ is ensemble after stage m.", r"$h_m$ is new weak learner.", r"$\eta$ is learning rate.", r"$L$ is loss.", r"$-\partial L/\partial F$ is the negative functional gradient."],
        why_equation="It views boosting as optimization in function space; for squared error the negative gradient is proportional to the residual.",
        numerical_example="If current prediction is 6, true target 10, stump predicts residual 3, and eta=.1, new prediction is 6.3.",
        python_connection="residual = y - prediction; prediction += learning_rate * stump.predict(X).",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Small learning rates usually need more stages; too many high-rate stages can overfit. Tune the pair together.",
        hyperparameter_code=hyper,
        interpretation="Falling scratch loss shows sequential correction. Residual structure reveals remaining bias that a single RMSE cannot show.",
        disadvantages=["Sequential training is less parallel than random forests", "Sensitive to interacting hyperparameters", "Can overfit noisy targets"],
    )


def model_evaluation_spec() -> LessonSpec:
    scratch = r"""
def confusion_counts(y_true,y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    return {
        "TP":int(np.sum((y_true==1)&(y_pred==1))),
        "TN":int(np.sum((y_true==0)&(y_pred==0))),
        "FP":int(np.sum((y_true==0)&(y_pred==1))),
        "FN":int(np.sum((y_true==1)&(y_pred==0))),
    }

def classification_metrics_scratch(y_true,y_pred):
    counts=confusion_counts(y_true,y_pred); tp,tn,fp,fn=[counts[key] for key in ["TP","TN","FP","FN"]]
    accuracy=(tp+tn)/(tp+tn+fp+fn); precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    return counts|{"accuracy":accuracy,"precision":precision,"recall":recall,"F1":f1}

display(pd.Series(classification_metrics_scratch([1,1,0,0],[1,0,1,0])))
"""
    experiment = r"""
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay,PrecisionRecallDisplay,RocCurveDisplay,average_precision_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold,cross_validate,learning_curve,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X,y=make_classification(n_samples=900,n_features=10,n_informative=5,weights=[.9,.1],class_sep=1.0,random_state=RANDOM_SEED)
X_development,X_test,y_development,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=RANDOM_SEED)
X_train,X_validation,y_train,y_validation=train_test_split(X_development,y_development,test_size=.25,stratify=y_development,random_state=RANDOM_SEED)
baseline=DummyClassifier(strategy="most_frequent").fit(X_train,y_train)
model=make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=2000,random_state=RANDOM_SEED)).fit(X_train,y_train)
prediction=model.predict(X_validation); probability=model.predict_proba(X_validation)[:,1]
display(pd.DataFrame([classification_metrics_scratch(y_validation,baseline.predict(X_validation)),classification_metrics_scratch(y_validation,prediction)],index=["Majority baseline","Balanced logistic"]).round(3))
print("ROC-AUC:",round(roc_auc_score(y_validation,probability),3),"Average precision:",round(average_precision_score(y_validation,probability),3))
cv=StratifiedKFold(5,shuffle=True,random_state=RANDOM_SEED)
cv_results=cross_validate(model,X_development,y_development,cv=cv,scoring=["roc_auc","average_precision"])
print("5-fold ROC-AUC mean/std:",round(cv_results["test_roc_auc"].mean(),3),round(cv_results["test_roc_auc"].std(),3))
figure,axes=plt.subplots(1,3,figsize=(14,4))
ConfusionMatrixDisplay.from_predictions(y_validation,prediction,cmap="Blues",colorbar=False,ax=axes[0]); axes[0].set_title("Validation confusion matrix")
RocCurveDisplay.from_predictions(y_validation,probability,ax=axes[1],curve_kwargs={"color":COURSE_COLORS["blue"]}); axes[1].set_title("ROC curve on imbalanced data")
PrecisionRecallDisplay.from_predictions(y_validation,probability,ax=axes[2],curve_kwargs={"color":COURSE_COLORS["orange"]}); axes[2].set_title("Precision-recall curve on imbalanced data")
figure.tight_layout(); plt.show()
"""
    hyper = r"""
train_sizes,train_scores,validation_scores=learning_curve(model,X_development,y_development,cv=cv,scoring="roc_auc",train_sizes=np.linspace(.15,1,6),n_jobs=1)
train_mean=train_scores.mean(1); validation_mean=validation_scores.mean(1)
figure,axis=plt.subplots(); axis.plot(train_sizes,train_mean,marker="o",label="Training ROC-AUC",color=COURSE_COLORS["blue"]); axis.plot(train_sizes,validation_mean,marker="s",label="Cross-validation ROC-AUC",color=COURSE_COLORS["orange"])
axis.set(title="Learning curve diagnoses bias and variance",xlabel="Training observations",ylabel="ROC-AUC"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Model Evaluation and Selection",
        objectives=["Calculate regression and classification metrics", "Use train/validation/test sets and cross-validation", "Diagnose imbalance, leakage, underfitting, overfitting, bias, and variance"],
        problem="Estimate how a model will behave on unseen data and choose among candidates without contaminating the final test.",
        intuition="Evaluation is an experiment: control the split, compare against a baseline, quantify variability, and inspect error types.",
        equation=r"$$Precision=\frac{TP}{TP+FP},\quad Recall=\frac{TP}{TP+FN},\quad F1=2\frac{Precision\cdot Recall}{Precision+Recall}.$$",
        symbols=[r"$TP$ counts found positives.", r"$FP$ counts false alarms.", r"$FN$ counts missed positives.", r"$Precision$ measures positive-prediction reliability.", r"$Recall$ measures positive coverage.", r"$F1$ is their harmonic mean."],
        why_equation="On imbalanced data, accuracy can be high while the minority class is completely missed.",
        numerical_example="With TP=8, FP=2, FN=4, precision=.8, recall=.667, and F1=.727.",
        python_connection="The scratch function counts Boolean combinations and applies each fraction with safe zero division.",
        scratch_description="We compute the confusion counts, accuracy, precision, recall, and F1 directly before using scikit-learn displays.",
        library_description="Pipelines prevent leakage; stratified cross-validation estimates variability; ROC and precision-recall displays study thresholds.",
        dataset_description="A generated 90/10 imbalanced binary dataset makes misleading accuracy behavior visible.",
        preprocessing_description="Scaling lives inside a Pipeline, so every cross-validation fold fits its own scaler only on that fold's training rows.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="A learning curve varies training-set size. A persistent train-validation gap suggests variance; similarly low curves suggest bias.",
        hyperparameter_code=hyper,
        interpretation="The majority baseline can have high accuracy yet zero minority recall. Precision-recall is especially informative when positives are rare.",
        mistakes=["Using ordinary K-fold for imbalanced classes", "Oversampling before splitting", "Tuning on test data", "Comparing metrics with different positive-label definitions"],
        advantages=["Makes model selection auditable", "Separates ranking, threshold, and calibration questions"],
        disadvantages=["One split has sampling noise", "Cross-validation costs more compute", "Offline metrics may omit deployment effects"],
    )


def capstone_spec() -> LessonSpec:
    scratch = r"""
class MeanRegressor:
    def fit(self,target):
        self.mean_=float(np.mean(target)); return self
    def predict(self,size):
        return np.full(size,self.mean_)

print("Mean-baseline example:",MeanRegressor().fit([2,4,6]).predict(3))
"""
    experiment = r"""
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
"""
    hyper = r"""
rows=[]
for depth in [1,2,3]:
    for estimators in [50,100,180]:
        candidate=GradientBoostingRegressor(max_depth=depth,n_estimators=estimators,learning_rate=.05,random_state=RANDOM_SEED).fit(X_train,y_train)
        rows.append({"depth":depth,"estimators":estimators,"validation_RMSE":mean_squared_error(y_validation,candidate.predict(X_validation))**.5})
tuning=pd.DataFrame(rows); display(tuning.round(2))
pivot=tuning.pivot(index="depth",columns="estimators",values="validation_RMSE")
figure,axis=plt.subplots(); sns.heatmap(pivot,annot=True,fmt=".1f",cmap="Blues_r",ax=axis); axis.set(title="Capstone hyperparameter search",xlabel="Boosting stages",ylabel="Tree depth"); plt.show()
"""
    return base_spec(
        title="Supervised Learning Capstone: Diabetes Progression Regression",
        objectives=["Frame an end-to-end regression problem", "Audit, engineer, baseline, compare, and tune models", "Perform held-out evaluation and residual error analysis"],
        problem="Predict the one-year disease-progression measure in the scikit-learn Diabetes benchmark. Success is lower RMSE than a mean baseline, with MAE and R² as guardrails.",
        use_cases=["Educational clinical-risk regression", "Tabular model comparison", "Residual-based error analysis"],
        intuition="A useful capstone earns complexity: begin with the mean, add an interpretable linear model, compare ensembles, tune without seeing test outcomes, and document limitations.",
        equation=r"$$RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2},\qquad R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.$$",
        symbols=[r"$n$ is evaluated row count.", r"$y_i$ is true progression.", r"$\hat y_i$ is prediction.", r"$\bar y$ is mean target.", r"$RMSE$ is error in target units.", r"$R^2$ compares with the mean baseline."],
        why_equation="RMSE defines primary success and emphasizes large misses; R² shows improvement relative to a constant prediction.",
        numerical_example="For errors [1,-2], RMSE=sqrt((1+4)/2)=1.581.",
        python_connection="mean_squared_error(y, prediction)**.5 and r2_score(y, prediction) implement the metrics.",
        scratch_description="The mean regressor is implemented directly and establishes the minimum useful benchmark.",
        library_description="Ridge, Random Forest, and Gradient Boosting represent linear, bagged-tree, and sequential-tree families.",
        dataset_description="The bundled dataset has 442 observations, ten standardized baseline variables, and a numeric progression target.",
        preprocessing_description="A BMI-by-blood-pressure interaction is engineered before splitting columns; fitted scaling remains inside the Ridge pipeline.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Gradient-boosting depth and stage count are compared only on validation RMSE. The test set is reserved for the frozen winner.",
        hyperparameter_code=hyper,
        interpretation="The comparison tests whether nonlinear ensembles beat the baseline; residuals expose target ranges where systematic under- or overprediction remains.",
        mistakes=["Treating the benchmark as deployment-ready medicine", "Tuning on test RMSE", "Ignoring target-range error patterns", "Claiming the engineered interaction is causal"],
        advantages=["Complete offline workflow", "Multiple model families and explicit baseline", "Held-out residual analysis"],
        disadvantages=["Small benchmark sample", "No temporal or external validation", "No uncertainty intervals or clinical utility analysis"],
        use_when="learning reproducible supervised workflows or benchmarking methods on small continuous-target data.",
        avoid_when="making clinical decisions without prospective validation, governance, calibration, uncertainty, and subgroup analysis.",
        exercises=["Why use both RMSE and MAE?", "For true [2,4] and predicted [1,7], compute RMSE.", "Repeat the comparison across five splits and report mean plus standard deviation."],
        solutions=["RMSE emphasizes large errors while MAE expresses a robust typical absolute miss; disagreement is diagnostically useful.", "Errors [1,-3] give squared errors [1,9], MSE=5, RMSE=sqrt(5)=2.236.", "Loop over fixed seeds, repeat every split and fitted preprocessing step, retain every score, and summarize variability without selecting the best split."],
        takeaways=["Business framing and success metrics precede model choice", "Feature engineering and preprocessing must respect split boundaries", "Baselines make added complexity accountable", "Final evaluation, residual analysis, limitations, and improvements form one conclusion"],
    )


def main() -> None:
    lessons = {
        "01_supervised_learning/01_linear_regression.ipynb": linear_regression_spec(),
        "01_supervised_learning/02_logistic_regression.ipynb": logistic_regression_spec(),
        "01_supervised_learning/03_k_nearest_neighbors.ipynb": knn_spec(),
        "01_supervised_learning/04_naive_bayes.ipynb": naive_bayes_spec(),
        "01_supervised_learning/05_decision_tree.ipynb": decision_tree_spec(),
        "01_supervised_learning/06_random_forest.ipynb": random_forest_spec(),
        "01_supervised_learning/07_support_vector_machine.ipynb": svm_spec(),
        "01_supervised_learning/08_gradient_boosting.ipynb": gradient_boosting_spec(),
        "01_supervised_learning/09_model_evaluation.ipynb": model_evaluation_spec(),
        "01_supervised_learning/10_supervised_capstone_project.ipynb": capstone_spec(),
    }
    for path,spec in lessons.items():
        write_lesson(path,spec)
        print("Built",path)


if __name__=="__main__":
    main()
