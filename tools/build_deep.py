"""Build all deep-learning notebooks with small CPU-first experiments."""

from __future__ import annotations

from course_factory import LessonSpec, write_lesson


TORCH_SETUP = """
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)
device = torch.device("cpu")
print("PyTorch:", torch.__version__, "| device:", device)
"""


def base_spec(**overrides) -> LessonSpec:
    spec = {
        "title": "Deep Learning",
        "objectives": ["Explain the computation at three levels", "Inspect every important tensor shape", "Train and evaluate a small CPU model"],
        "prerequisites": "NumPy arrays, matrix multiplication, basic derivatives, and the supervised-learning workflow.",
        "problem": "Learn layered representations whose parameters are optimized from data.",
        "use_cases": ["Image recognition", "Sequence modeling", "Learned feature representations"],
        "intuition": "A neural network stacks simple transformations. Training adjusts their weights so useful intermediate features emerge.",
        "equation": r"$$\mathbf{h}=\phi(\mathbf{X}\mathbf{W}+\mathbf{b}).$$",
        "symbols": [r"$\mathbf{X}$ is a batch of inputs.", r"$\mathbf{W}$ is a learned weight matrix.", r"$\mathbf{b}$ is a learned bias vector.", r"$\phi$ is an activation function.", r"$\mathbf{h}$ is the resulting representation."],
        "why_equation": "Without the activation, stacked linear layers collapse to one linear transformation; the activation enables nonlinear patterns.",
        "numerical_example": "For x=[1,2], w=[.5,-.25], and b=.1, z=1(.5)+2(-.25)+.1=.1; ReLU(z)=.1.",
        "python_connection": "torch.relu(X @ W + b) implements the equation while autograd records the operations.",
        "steps": ["Create tensors with explicit batch and feature dimensions", "Run a forward pass", "Compute a loss against targets", "Backpropagate gradients", "Update parameters with an optimizer", "Evaluate on untouched validation data"],
        "scratch_description": "NumPy exposes the forward calculation and, where appropriate, the gradients.",
        "library_description": "PyTorch tensors, modules, autograd, losses, and optimizers provide the standard implementation.",
        "dataset_description": "Every lesson uses a generated sequence or a small dataset bundled with scikit-learn; no download is required.",
        "preprocessing_description": "Splits occur before fitted transformations. Inputs are converted to float32 tensors and class labels to int64.",
        "setup_code": TORCH_SETUP,
        "scratch_code": "def relu(values):\n    return np.maximum(values, 0.0)\n\nx = np.array([1.0, 2.0]); w = np.array([.5, -.25]); b = .1\nprint('Manual neuron:', relu(x @ w + b))",
        "experiment_code": "layer = nn.Linear(2, 1)\nexample = torch.tensor([[1.0, 2.0]])\noutput = torch.relu(layer(example))\nprint('Input/output shapes:', tuple(example.shape), tuple(output.shape))\nprint('Trainable parameters:', sum(p.numel() for p in layer.parameters() if p.requires_grad))",
        "interpretation": "The leading dimension is batch size; the last dimension is the learned feature size.",
        "hyperparameter_text": "We vary one capacity or optimization choice while keeping the data split fixed.",
        "hyperparameter_code": "widths = np.array([2, 4, 8, 16])\nparameters = widths * 2 + widths + widths + 1\nfigure, axis = plt.subplots()\naxis.plot(widths, parameters, marker='o', color=COURSE_COLORS['blue'], label='Parameter count')\naxis.set(title='Hidden width controls capacity', xlabel='Hidden units', ylabel='Trainable parameters')\naxis.legend(); plt.show()",
        "mistakes": ["Forgetting batch and feature dimensions", "Applying softmax before CrossEntropyLoss", "Evaluating on the training set", "Leaving gradients accumulated between optimizer steps"],
        "advantages": ["Learns features jointly with predictions", "Flexible building blocks for structured data"],
        "disadvantages": ["Needs careful optimization and validation", "Can be computationally expensive and difficult to interpret"],
        "use_when": "the data and task benefit from learned nonlinear representations and validation supports the added complexity.",
        "avoid_when": "a small tabular dataset is handled well by a simpler, easier-to-audit model.",
        "exercises": ["Why do hidden layers need nonlinear activations?", "Calculate ReLU([−2, 0, 3]).", "Double hidden width and compare parameter count and validation behavior."],
        "solutions": ["Without nonlinearities, multiple affine layers equal one affine layer.", "The result is [0, 0, 3].", "Change the layer width, count parameters, retrain with the same split, and compare training-versus-validation loss rather than training loss alone."],
        "takeaways": ["Tensor shapes are part of the model specification", "Forward propagation produces predictions and backpropagation produces gradients", "Training and validation curves reveal optimization and generalization"],
        "further_reading": ["[PyTorch tutorials](https://pytorch.org/tutorials/)", "[PyTorch neural-network modules](https://pytorch.org/docs/stable/nn.html)"],
    }
    spec.update(overrides)
    return spec


def overview_spec() -> LessonSpec:
    scratch = r"""
def sigmoid(values):
    return 1 / (1 + np.exp(-values))

X_np = np.array([[1.0, 2.0], [-1.0, 1.0]])
W_np = np.array([[.4, -.2], [.1, .3]])
b_np = np.array([.0, .1])
hidden_np = np.maximum(X_np @ W_np + b_np, 0)
output_np = sigmoid(hidden_np @ np.array([.7, -.5]))
print("Input:", X_np.shape, "hidden:", hidden_np.shape, "output:", output_np.shape)
print("NumPy outputs:", output_np.round(3))
"""
    experiment = r"""
model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid())
X = torch.tensor([[1., 2.], [-1., 1.]])
with torch.no_grad():
    hidden = model[1](model[0](X))
    predictions = model(X)
print("Input shape:", tuple(X.shape))
print("Intermediate shape:", tuple(hidden.shape))
print("Output shape:", tuple(predictions.shape))
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

activations = {
    "sigmoid": torch.sigmoid(torch.linspace(-5, 5, 200)),
    "tanh": torch.tanh(torch.linspace(-5, 5, 200)),
    "ReLU": torch.relu(torch.linspace(-5, 5, 200)),
}
figure, axis = plt.subplots()
grid = np.linspace(-5, 5, 200)
for name, values in activations.items():
    axis.plot(grid, values.numpy(), label=name)
axis.set(title="Common activation functions", xlabel="Pre-activation z", ylabel="Activation")
axis.legend()
plt.show()
"""
    hyper = r"""
depths = np.arange(1, 7)
linear_regions_proxy = 2 ** depths
figure, axis = plt.subplots()
axis.plot(depths, linear_regions_proxy, marker="o", color=COURSE_COLORS["blue"], label="Illustrative region capacity")
axis.set(title="Depth can increase representational capacity", xlabel="Number of nonlinear layers", ylabel="Illustrative capacity (not a guarantee)")
axis.legend()
plt.show()
"""
    return base_spec(
        title="Deep Learning Overview",
        objectives=["Describe neurons, layers, activations, loss, and optimization", "Trace input, hidden, and output shapes", "Build the same tiny network with NumPy and PyTorch"],
        problem="Learn useful intermediate representations and a prediction function together.",
        intuition="Each layer rewrites the data into a form the next layer can use; loss tells the network how the final answer should change.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Depth increases composition but also changes optimization, memory, and overfitting risk; the plotted capacity is illustrative rather than a measured accuracy.",
        hyperparameter_code=hyper,
        interpretation="Sigmoid and tanh saturate at large magnitudes; ReLU keeps a constant positive-side slope and is common in hidden layers.",
    )


def tensors_spec() -> LessonSpec:
    scratch = r"""
scalar = np.array(3.0)
vector = np.array([1.0, 2.0, 3.0])
matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
tensor = np.arange(24).reshape(2, 3, 4)
product = matrix @ np.array([[2.0], [-1.0]])
print("Ranks/shapes:", scalar.ndim, vector.shape, matrix.shape, tensor.shape)
print("Matrix product:", product.ravel())
"""
    experiment = r"""
a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = torch.tensor([[2.], [-1.]])
c = a @ b
loss = (c ** 2).mean()
loss.backward()
print("a:", tuple(a.shape), "b:", tuple(b.shape), "c:", tuple(c.shape))
print("Scalar loss:", loss.item())
print("Gradient shape:", tuple(a.grad.shape))

figure, axes = plt.subplots(1, 2, figsize=(9, 4))
sns.heatmap(a.detach().numpy(), annot=True, cmap="Blues", cbar=False, ax=axes[0])
axes[0].set(title="Input matrix A", xlabel="Column", ylabel="Row")
sns.heatmap(a.grad.numpy(), annot=True, cmap="Oranges", cbar=False, ax=axes[1])
axes[1].set(title="Gradient dLoss/dA", xlabel="Column", ylabel="Row")
figure.tight_layout()
plt.show()
"""
    hyper = r"""
batch_sizes = np.array([1, 8, 32, 128])
elements = batch_sizes * 64 * 64
figure, axis = plt.subplots()
axis.plot(batch_sizes, elements, marker="o", color=COURSE_COLORS["blue"], label="Tensor elements")
axis.set(title="Batch size scales activation memory", xlabel="Batch size", ylabel="Elements in a 64×64 activation")
axis.legend()
plt.show()
"""
    return base_spec(
        title="Tensors and Linear Algebra",
        objectives=["Distinguish scalars, vectors, matrices, and tensors", "Apply dot products, broadcasting, and matrix multiplication", "Use autograd on a computation graph"],
        problem="Represent batches and model parameters in arrays whose shapes make operations precise.",
        intuition="A tensor is a numbered box with any number of axes; matrix multiplication connects input features to output features.",
        equation=r"$$C_{ij}=\sum_{k=1}^{p}A_{ik}B_{kj}.$$",
        symbols=[r"$A$ has shape $n\times p$.", r"$B$ has shape $p\times m$.", r"$C$ has shape $n\times m$.", r"$k$ indexes the shared dimension."],
        why_equation="Every neural layer relies on compatible inner dimensions to combine features and weights.",
        numerical_example="For row [1,2] and column [2,-1], the dot product is 1(2)+2(-1)=0.",
        python_connection="A @ B performs matrix multiplication; * performs elementwise multiplication.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Batch size changes memory and gradient noise, even when the model architecture is unchanged.",
        hyperparameter_code=hyper,
        interpretation="The gradient has the same shape as the tensor it differentiates, so each input element receives a local sensitivity.",
    )


def perceptron_spec() -> LessonSpec:
    scratch = r"""
class PerceptronScratch:
    def __init__(self, learning_rate=.1, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, features, target):
        self.weights_ = np.zeros(features.shape[1])
        self.bias_ = 0.0
        self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            for row, correct in zip(features, target):
                prediction = int(row @ self.weights_ + self.bias_ >= 0)
                update = self.learning_rate * (correct - prediction)
                self.weights_ += update * row
                self.bias_ += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

    def predict(self, features):
        return (features @ self.weights_ + self.bias_ >= 0).astype(int)

and_X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
and_y = np.array([0,0,0,1])
scratch_model = PerceptronScratch().fit(and_X, and_y)
print("AND predictions:", scratch_model.predict(and_X))
"""
    experiment = r"""
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=240, centers=2, cluster_std=1.2, random_state=RANDOM_SEED)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=.25, stratify=y, random_state=RANDOM_SEED)
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=.05)
loss_fn = nn.BCEWithLogitsLoss()
Xt = torch.tensor(X_train, dtype=torch.float32)
yt = torch.tensor(y_train[:, None], dtype=torch.float32)
validation_Xt = torch.tensor(X_validation, dtype=torch.float32)
validation_yt = torch.tensor(y_validation[:, None], dtype=torch.float32)
loss_history = []; validation_loss_history = []
for epoch in range(40):
    model.train()
    optimizer.zero_grad()
    logits = model(Xt)
    loss = loss_fn(logits, yt)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        validation_logits = model(validation_Xt)
        validation_loss = loss_fn(validation_logits, validation_yt)
    loss_history.append(loss.item())
    validation_loss_history.append(validation_loss.item())
validation_pred = (validation_logits.squeeze() >= 0).numpy().astype(int)
validation_accuracy = accuracy_score(y_validation, validation_pred)
print("Model diagnostics:", {
    "input_shape": tuple(Xt.shape),
    "intermediate_shape": "not applicable (single affine layer)",
    "output_shape": tuple(logits.shape),
    "trainable_parameters": sum(p.numel() for p in model.parameters()),
    "final_training_loss": round(loss_history[-1], 4),
    "final_validation_loss": round(validation_loss_history[-1], 4),
    "validation_accuracy": round(validation_accuracy, 3),
})

figure, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(loss_history, color=COURSE_COLORS["blue"], label="Training loss")
axes[0].plot(validation_loss_history, color=COURSE_COLORS["orange"], linestyle="--", label="Validation loss")
axes[0].set(title="Perceptron-style linear classifier training", xlabel="Epoch", ylabel="Binary cross-entropy")
axes[0].legend()
axes[1].scatter(X_validation[:,0], X_validation[:,1], c=validation_pred, cmap="cividis")
axes[1].set(title="Validation predictions", xlabel="Feature 1", ylabel="Feature 2")
figure.tight_layout()
plt.show()
"""
    hyper = r"""
rates = [.005, .02, .05, .2]
final_losses = []
for rate in rates:
    candidate = nn.Linear(2, 1)
    opt = torch.optim.SGD(candidate.parameters(), lr=rate)
    for _ in range(30):
        opt.zero_grad(); candidate_loss = loss_fn(candidate(Xt), yt); candidate_loss.backward(); opt.step()
    final_losses.append(candidate_loss.item())
figure, axis = plt.subplots()
axis.semilogx(rates, final_losses, marker="o", color=COURSE_COLORS["orange"], label="Final training loss")
axis.set(title="Learning rate changes convergence", xlabel="Learning rate", ylabel="Loss after 30 epochs")
axis.legend()
plt.show()
"""
    return base_spec(
        title="The Perceptron",
        objectives=["Implement the perceptron update", "Relate a linear score to a decision boundary", "Train an equivalent PyTorch linear classifier"],
        problem="Separate two linearly separable classes with a weighted sum and threshold.",
        intuition="When an example is misclassified, nudge the separating line toward the correct side of that example.",
        equation=r"$$\mathbf{w}\leftarrow\mathbf{w}+\eta(y-\hat{y})\mathbf{x}.$$",
        symbols=[r"$\mathbf{w}$ is the weight vector.", r"$\eta$ is learning rate.", r"$y$ is the true binary label.", r"$\hat y$ is the prediction.", r"$\mathbf{x}$ is the input."],
        why_equation="The signed error determines update direction and the input scales each weight change.",
        numerical_example="With w=[0,0], eta=.1, y=1, prediction=0, and x=[1,2], new w=[.1,.2].",
        python_connection="weights += learning_rate * (correct - prediction) * row.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Too small a learning rate moves slowly; too large can overshoot useful parameters.",
        hyperparameter_code=hyper,
        interpretation="The loss curve shows optimization; the scatter shows that a single linear boundary is sufficient for this dataset.",
        disadvantages=["Cannot solve nonlinearly separable patterns such as XOR", "Hard threshold outputs are not calibrated probabilities"],
    )


def scratch_network_spec() -> LessonSpec:
    scratch = r"""
def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))

X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = np.array([[0.],[1.],[1.],[0.]])
W1 = rng.normal(0, .8, size=(2, 4)); b1 = np.zeros((1, 4))
W2 = rng.normal(0, .8, size=(4, 1)); b2 = np.zeros((1, 1))
learning_rate = .8
losses = []
for epoch in range(4000):
    hidden = np.tanh(X @ W1 + b1)
    prediction = sigmoid(hidden @ W2 + b2)
    loss = -np.mean(y*np.log(prediction+1e-9)+(1-y)*np.log(1-prediction+1e-9))
    losses.append(loss)
    output_gradient = (prediction - y) / len(X)
    dW2 = hidden.T @ output_gradient; db2 = output_gradient.sum(axis=0, keepdims=True)
    hidden_gradient = (output_gradient @ W2.T) * (1 - hidden**2)
    dW1 = X.T @ hidden_gradient; db1 = hidden_gradient.sum(axis=0, keepdims=True)
    W2 -= learning_rate*dW2; b2 -= learning_rate*db2
    W1 -= learning_rate*dW1; b1 -= learning_rate*db1
print("Shapes:", X.shape, hidden.shape, prediction.shape)
print("Final XOR predictions:", (prediction.ravel() >= .5).astype(int), "loss:", round(losses[-1], 4))
"""
    experiment = r"""
torch_model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
optimizer = torch.optim.Adam(torch_model.parameters(), lr=.08)
loss_fn = nn.BCEWithLogitsLoss()
training_features = np.repeat(X, 64, axis=0) + rng.normal(0, .04, size=(256, 2))
training_targets = np.repeat(y, 64, axis=0)
validation_features = np.repeat(X, 24, axis=0) + rng.normal(0, .04, size=(96, 2))
validation_targets = np.repeat(y, 24, axis=0)
Xt = torch.tensor(training_features, dtype=torch.float32)
yt = torch.tensor(training_targets, dtype=torch.float32)
Xv = torch.tensor(validation_features, dtype=torch.float32)
yv = torch.tensor(validation_targets, dtype=torch.float32)
torch_losses = []; torch_validation_losses = []
for epoch in range(800):
    torch_model.train()
    optimizer.zero_grad()
    logits = torch_model(Xt)
    loss = loss_fn(logits, yt)
    loss.backward()
    optimizer.step()
    torch_model.eval()
    with torch.no_grad():
        validation_logits = torch_model(Xv)
        validation_loss = loss_fn(validation_logits, yv)
    torch_losses.append(loss.item())
    torch_validation_losses.append(validation_loss.item())
with torch.no_grad():
    truth_table_logits = torch_model(torch.tensor(X, dtype=torch.float32))
    torch_prediction = (torch.sigmoid(truth_table_logits).ravel() >= .5).int().numpy()
    hidden_activations = torch_model[1](torch_model[0](Xt))
truth_table_accuracy = float(np.mean(torch_prediction == y.ravel()))
print("Model diagnostics:", {
    "input_shape": tuple(Xt.shape),
    "intermediate_shape": tuple(hidden_activations.shape),
    "output_shape": tuple(logits.shape),
    "trainable_parameters": sum(p.numel() for p in torch_model.parameters()),
    "final_training_loss": round(torch_losses[-1], 4),
    "final_validation_loss": round(torch_validation_losses[-1], 4),
    "truth_table_accuracy": round(truth_table_accuracy, 3),
})
print("Truth-table predictions:", torch_prediction)

figure, axis = plt.subplots()
axis.plot(losses, label="NumPy loss", color=COURSE_COLORS["blue"])
progress = np.linspace(0, len(losses)-1, len(torch_losses))
axis.plot(progress, torch_losses, label="PyTorch training loss", color=COURSE_COLORS["orange"], linestyle="--")
axis.plot(progress, torch_validation_losses, label="PyTorch validation loss", color=COURSE_COLORS["olive"], linestyle=":")
axis.set(title="Two-layer networks learn XOR", xlabel="Comparable training progress", ylabel="Binary cross-entropy")
axis.legend()
plt.show()
"""
    hyper = r"""
widths = [1, 2, 4, 8]
parameter_counts = [2*w + w + w + 1 for w in widths]
figure, axis = plt.subplots()
axis.bar([str(w) for w in widths], parameter_counts, color=COURSE_COLORS["blue"])
axis.set(title="Hidden width changes parameter count", xlabel="Hidden units", ylabel="Trainable parameters")
plt.show()
"""
    return base_spec(
        title="Neural Network from Scratch",
        objectives=["Implement forward propagation", "Derive and code two-layer backpropagation", "Compare NumPy and PyTorch on XOR"],
        problem="Learn a nonlinear decision function that a single perceptron cannot represent.",
        intuition="Hidden neurons create several intermediate boundaries; the output combines them into a nonlinear pattern.",
        equation=r"$$\frac{\partial L}{\partial W_1}=\mathbf{X}^{T}\left[\left(\frac{\partial L}{\partial z_2}W_2^T\right)\odot(1-\mathbf{h}^2)\right].$$",
        symbols=[r"$L$ is binary cross-entropy loss.", r"$W_1,W_2$ are layer weights.", r"$z_2$ is output pre-activation.", r"$\mathbf{h}$ is tanh hidden activation.", r"$\odot$ is elementwise multiplication."],
        why_equation="The chain rule carries output error backward through the second layer and tanh derivative.",
        numerical_example="If an upstream gradient is .4 and a tanh activation is .5, the local gradient multiplier is .4(1-.5^2)=.3.",
        python_connection="hidden_gradient = (output_gradient @ W2.T) * (1 - hidden**2).",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="A width of one cannot reliably form XOR's two disjoint positive regions; extra units add capacity and parameters.",
        hyperparameter_code=hyper,
        interpretation="Both implementations drive loss down and reproduce all four XOR labels, validating the manual gradients against autograd.",
    )


def backprop_spec() -> LessonSpec:
    scratch = r"""
x, target, weight, bias, learning_rate = 2.0, 5.0, 1.0, 0.0, .1
prediction = weight*x + bias
loss = (prediction-target)**2
d_loss_d_prediction = 2*(prediction-target)
d_loss_d_weight = d_loss_d_prediction*x
d_loss_d_bias = d_loss_d_prediction
new_weight = weight-learning_rate*d_loss_d_weight
new_bias = bias-learning_rate*d_loss_d_bias
print({"prediction": prediction, "loss": loss, "dW": d_loss_d_weight, "db": d_loss_d_bias, "new_w": new_weight, "new_b": new_bias})
"""
    experiment = r"""
x_t = torch.tensor(2.0)
target_t = torch.tensor(5.0)
weight_t = torch.tensor(1.0, requires_grad=True)
bias_t = torch.tensor(0.0, requires_grad=True)
prediction_t = weight_t*x_t + bias_t
loss_t = (prediction_t-target_t)**2
loss_t.backward()
print("Autograd dW/db:", weight_t.grad.item(), bias_t.grad.item())
assert np.isclose(weight_t.grad.item(), d_loss_d_weight)

grid = np.linspace(-1, 4, 200)
loss_surface = (grid*x-target)**2
figure, axis = plt.subplots()
axis.plot(grid, loss_surface, color=COURSE_COLORS["blue"], label="Loss(w)")
axis.scatter([weight, new_weight], [(weight*x-target)**2, (new_weight*x+new_bias-target)**2], color=COURSE_COLORS["orange"], label="Before/after update")
axis.set(title="One gradient step moves down the loss", xlabel="Weight w", ylabel="Squared error")
axis.legend()
plt.show()
"""
    hyper = r"""
rates = [.01, .05, .1, .3]
histories = {}
for rate in rates:
    current = 0.0
    history = []
    for _ in range(12):
        gradient = 2*(current*x-target)*x
        current -= rate*gradient
        history.append((current*x-target)**2)
    histories[rate] = history
figure, axis = plt.subplots()
for rate, history in histories.items():
    axis.plot(history, label=f"lr={rate}")
axis.set(title="Learning rate controls gradient-descent behavior", xlabel="Update", ylabel="Squared error")
axis.set_yscale("log")
axis.legend()
plt.show()
"""
    return base_spec(
        title="Gradient Descent and Backpropagation",
        objectives=["Calculate one gradient update by hand", "Apply the chain rule", "Verify manual derivatives with autograd"],
        problem="Determine how every parameter should change to reduce a composed loss function.",
        intuition="Backpropagation assigns responsibility backward; gradient descent takes a small step opposite that responsibility.",
        equation=r"$$w_{\text{new}}=w-\eta\frac{\partial L}{\partial w}.$$",
        symbols=[r"$w$ is a parameter.", r"$\eta$ is learning rate.", r"$L$ is loss.", r"$\partial L/\partial w$ is loss sensitivity to w."],
        why_equation="The negative gradient is the direction of steepest local loss decrease.",
        numerical_example="For prediction 2, target 5, x=2, squared-error gradient is 2(2-5)2=-12; with eta=.1, w changes from 1 to 2.2.",
        python_connection="optimizer.step() applies this update after loss.backward() fills parameter gradients.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Learning rate that is too small converges slowly; a large rate can oscillate or diverge.",
        hyperparameter_code=hyper,
        interpretation="The manual and autograd gradients agree exactly; the plotted update reduces the local loss.",
    )


def pytorch_spec() -> LessonSpec:
    scratch = r"""
X_np = rng.normal(size=(80, 2))
true_w = np.array([[2.0], [-1.0]])
y_np = X_np @ true_w + .5
w_np = np.zeros((2, 1)); b_np = 0.0
for _ in range(120):
    prediction = X_np @ w_np + b_np
    error = prediction - y_np
    w_np -= .05 * (2/len(X_np)) * X_np.T @ error
    b_np -= .05 * 2 * error.mean()
print("NumPy weights/bias:", w_np.ravel().round(3), round(float(b_np), 3))
"""
    experiment = r"""
X = torch.tensor(X_np[:60], dtype=torch.float32)
y = torch.tensor(y_np[:60], dtype=torch.float32)
validation_X = torch.tensor(X_np[60:], dtype=torch.float32)
validation_y = torch.tensor(y_np[60:], dtype=torch.float32)
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(RANDOM_SEED))
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=.05)
loss_fn = nn.MSELoss()
history = []; validation_history = []
for epoch in range(40):
    model.train()
    total = 0.0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        prediction = model(batch_X)
        loss = loss_fn(prediction, batch_y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch_X)
    history.append(total / len(X))
    model.eval()
    with torch.no_grad():
        validation_prediction = model(validation_X)
        validation_loss = loss_fn(validation_prediction, validation_y)
    validation_history.append(validation_loss.item())
validation_rmse = float(torch.sqrt(loss_fn(validation_prediction, validation_y)))
print("Model diagnostics:", {
    "input_shape": tuple(X.shape),
    "intermediate_shape": "not applicable (single affine layer)",
    "output_shape": tuple(model(X).shape),
    "trainable_parameters": sum(p.numel() for p in model.parameters()),
    "final_training_loss": round(history[-1], 6),
    "final_validation_loss": round(validation_history[-1], 6),
    "validation_RMSE": round(validation_rmse, 6),
})
print("Learned:", model.weight.detach().numpy().round(3), model.bias.detach().numpy().round(3))
figure, axis = plt.subplots()
axis.plot(history, color=COURSE_COLORS["blue"], label="Training MSE")
axis.plot(validation_history, color=COURSE_COLORS["orange"], linestyle="--", label="Validation MSE")
axis.set(title="Mini-batch PyTorch training", xlabel="Epoch", ylabel="Mean squared error")
axis.legend()
plt.show()
"""
    hyper = r"""
batch_sizes = [1, 8, 16, len(X)]
updates_per_epoch = [int(np.ceil(len(X)/size)) for size in batch_sizes]
figure, axis = plt.subplots()
axis.bar([str(value) for value in batch_sizes], updates_per_epoch, color=COURSE_COLORS["blue"])
axis.set(title="Batch size changes updates per epoch", xlabel="Batch size", ylabel="Optimizer updates")
plt.show()
"""
    return base_spec(
        title="PyTorch Fundamentals",
        objectives=["Create tensors and datasets", "Use modules, losses, optimizers, and DataLoader", "Inspect gradients and parameters"],
        problem="Express a reproducible training loop with automatic differentiation and mini-batches.",
        intuition="A DataLoader supplies batches; the module predicts; loss measures error; backward computes gradients; the optimizer changes parameters.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Batch size trades noisy frequent updates against stable less-frequent updates and memory use.",
        hyperparameter_code=hyper,
        interpretation="PyTorch recovers the known linear weights while recording each operation needed for gradients.",
    )


def mlp_spec() -> LessonSpec:
    scratch = r"""
def softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)

example_logits = np.array([[2.0, 1.0, 0.0]])
print("Softmax:", softmax(example_logits).round(3), "sum:", softmax(example_logits).sum())
"""
    experiment = r"""
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X_train, X_validation, y_train, y_validation = train_test_split(
    digits.data, digits.target, test_size=.25, stratify=digits.target, random_state=RANDOM_SEED
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
train_X = torch.tensor(X_train, dtype=torch.float32)
train_y = torch.tensor(y_train, dtype=torch.long)
validation_X = torch.tensor(X_validation, dtype=torch.float32)
model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=.01)
loss_fn = nn.CrossEntropyLoss()
train_losses = []; validation_losses = []
for epoch in range(12):
    model.train(); optimizer.zero_grad()
    logits = model(train_X); loss = loss_fn(logits, train_y); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        val_logits = model(validation_X)
        val_loss = loss_fn(val_logits, torch.tensor(y_validation, dtype=torch.long))
    train_losses.append(loss.item()); validation_losses.append(val_loss.item())
prediction = val_logits.argmax(dim=1).numpy()
validation_accuracy = accuracy_score(y_validation, prediction)
print("Model diagnostics:", {
    "input_shape": tuple(train_X.shape),
    "intermediate_shape": (len(train_X), 64),
    "output_shape": tuple(logits.shape),
    "trainable_parameters": sum(p.numel() for p in model.parameters()),
    "final_training_loss": round(train_losses[-1], 4),
    "final_validation_loss": round(validation_losses[-1], 4),
    "validation_accuracy": round(validation_accuracy, 3),
})
figure, axis = plt.subplots()
axis.plot(train_losses, label="Training loss", color=COURSE_COLORS["blue"])
axis.plot(validation_losses, label="Validation loss", color=COURSE_COLORS["orange"], linestyle="--")
axis.set(title="MLP learning curves on handwritten digits", xlabel="Epoch", ylabel="Cross-entropy")
axis.legend()
plt.show()
"""
    hyper = r"""
widths = [16, 32, 64, 128]
counts = [64*w+w+w*10+10 for w in widths]
figure, axis = plt.subplots()
axis.plot(widths, counts, marker="o", color=COURSE_COLORS["blue"], label="Trainable parameters")
axis.set(title="Hidden width controls MLP size", xlabel="Hidden units", ylabel="Parameter count")
axis.legend()
plt.show()
"""
    return base_spec(
        title="Multilayer Perceptron",
        objectives=["Use logits, softmax, and cross-entropy correctly", "Train a two-layer classifier", "Read training and validation curves"],
        problem="Classify images represented as fixed-length feature vectors with nonlinear hidden features.",
        intuition="Each hidden unit detects a weighted pattern; the output logits score the ten possible digits.",
        equation=r"$$p_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}},\qquad L=-\log p_y.$$",
        symbols=[r"$z_k$ is class k's logit.", r"$K$ is class count.", r"$p_k$ is class probability.", r"$y$ is the correct class.", r"$L$ is cross-entropy loss."],
        why_equation="Softmax converts relative scores into probabilities and cross-entropy rewards probability on the correct class.",
        numerical_example="For logits [0,0], probabilities are [.5,.5]; if class 0 is correct, loss is -log(.5)=.693.",
        python_connection="CrossEntropyLoss accepts raw logits and integer labels, combining log-softmax and negative log likelihood.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Wider layers add representational capacity and parameters; selection should use validation performance, not size alone.",
        hyperparameter_code=hyper,
        interpretation="Falling validation loss alongside training loss indicates learning that generalizes to the held-out split.",
    )


def cnn_spec() -> LessonSpec:
    scratch = r"""
def conv2d_single(image, kernel):
    out_h = image.shape[0] - kernel.shape[0] + 1
    out_w = image.shape[1] - kernel.shape[1] + 1
    output = np.empty((out_h, out_w))
    for row in range(out_h):
        for column in range(out_w):
            output[row, column] = np.sum(image[row:row+kernel.shape[0], column:column+kernel.shape[1]] * kernel)
    return output

image = np.arange(25, dtype=float).reshape(5, 5)
edge_kernel = np.array([[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]])
feature_map = conv2d_single(image, edge_kernel)
print("Image/kernel/output:", image.shape, edge_kernel.shape, feature_map.shape)
"""
    experiment = r"""
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

digits = load_digits()
X = (digits.images / 16.0).astype("float32")[:, None, :, :]
X_train, X_validation, y_train, y_validation = train_test_split(X, digits.target, test_size=.25, stratify=digits.target, random_state=RANDOM_SEED)
train_X = torch.tensor(X_train); train_y = torch.tensor(y_train, dtype=torch.long)
validation_X = torch.tensor(X_validation)
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.Flatten(),
    nn.Linear(16*4*4, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=.01)
loss_fn = nn.CrossEntropyLoss()
train_losses=[]; validation_losses=[]
for epoch in range(8):
    model.train(); optimizer.zero_grad(); logits=model(train_X); loss=loss_fn(logits, train_y); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        val_logits=model(validation_X); val_loss=loss_fn(val_logits, torch.tensor(y_validation, dtype=torch.long))
    train_losses.append(loss.item()); validation_losses.append(val_loss.item())
prediction=val_logits.argmax(1).numpy()
with torch.no_grad():
    first_map=model[1](model[0](validation_X[:1]))
validation_accuracy = accuracy_score(y_validation,prediction)
print("Model diagnostics:", {
    "input_shape": tuple(train_X.shape),
    "intermediate_shape": tuple(first_map.shape),
    "output_shape": tuple(logits.shape),
    "trainable_parameters": sum(p.numel() for p in model.parameters()),
    "final_training_loss": round(train_losses[-1], 4),
    "final_validation_loss": round(validation_losses[-1], 4),
    "validation_accuracy": round(validation_accuracy, 3),
})
figure, axes=plt.subplots(1,3,figsize=(11,3.5))
axes[0].imshow(validation_X[0,0], cmap="gray"); axes[0].set(title="Input digit", xlabel="Pixel column", ylabel="Pixel row")
axes[1].imshow(first_map[0,0], cmap="cividis"); axes[1].set(title="First convolution map", xlabel="Map column", ylabel="Map row")
axes[2].plot(train_losses,label="Train",color=COURSE_COLORS["blue"]); axes[2].plot(validation_losses,label="Validation",color=COURSE_COLORS["orange"],linestyle="--")
axes[2].set(title="CNN loss",xlabel="Epoch",ylabel="Cross-entropy"); axes[2].legend()
figure.tight_layout(); plt.show()
"""
    hyper = r"""
kernel_sizes=[1,3,5]
receptive_areas=[value**2 for value in kernel_sizes]
figure,axis=plt.subplots()
axis.bar([str(value) for value in kernel_sizes],receptive_areas,color=COURSE_COLORS["blue"])
axis.set(title="Kernel size changes local receptive area",xlabel="Square kernel width",ylabel="Input pixels per channel")
plt.show()
"""
    return base_spec(
        title="Convolutional Neural Networks",
        objectives=["Compute a convolution by hand", "Track channel and spatial dimensions", "Train a compact CNN on digits"],
        problem="Recognize local image patterns while sharing the same detector across spatial positions.",
        intuition="A small filter slides across an image; early filters detect edges and later layers combine local patterns.",
        equation=r"$$Y_{i,j}=\sum_{u=0}^{h-1}\sum_{v=0}^{w-1}X_{i+u,j+v}K_{u,v}.$$",
        symbols=[r"$X$ is the input image.", r"$K$ is an h-by-w kernel.", r"$Y_{i,j}$ is one output location.", r"$u,v$ index the local patch."],
        why_equation="Local weighted sums provide translation-aware feature detectors with far fewer parameters than a dense image layer.",
        numerical_example="For patch [[1,2],[3,4]] and kernel [[1,0],[0,-1]], output is 1-4=-3.",
        python_connection="nn.Conv2d vectorizes this operation across batches, channels, and learned kernels.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Kernel size changes local context and parameter count; stacking small kernels often provides efficient larger receptive fields.",
        hyperparameter_code=hyper,
        interpretation="The feature map shows one learned local response; the loss curves and validation accuracy evaluate the full classifier.",
    )


def rnn_spec() -> LessonSpec:
    scratch = r"""
def rnn_step(x_t, h_previous, W_xh, W_hh, bias):
    return np.tanh(x_t @ W_xh + h_previous @ W_hh + bias)

sequence = np.array([[.1], [.2], [.3]])
h = np.zeros((1, 2))
W_xh = np.array([[.5, -.4]]); W_hh = np.eye(2)*.3; bias=np.zeros(2)
states=[]
for value in sequence:
    h=rnn_step(value[None,:],h,W_xh,W_hh,bias); states.append(h.copy())
print("Sequence/state shapes:", sequence.shape, np.array(states).shape)
"""
    experiment = r"""
steps=240
signal=np.sin(np.linspace(0,12*np.pi,steps)).astype("float32")
window=12
X=np.stack([signal[i:i+window] for i in range(steps-window)])
y=np.array([signal[i+window] for i in range(steps-window)],dtype="float32")
split=170
train_X=torch.tensor(X[:split,:,None]); train_y=torch.tensor(y[:split,None])
validation_X=torch.tensor(X[split:,:,None]); validation_y=torch.tensor(y[split:,None])

class TinyRNN(nn.Module):
    def __init__(self):
        super().__init__(); self.rnn=nn.RNN(1,12,batch_first=True); self.output=nn.Linear(12,1)
    def forward(self,values):
        sequence_output,hidden=self.rnn(values)
        return self.output(sequence_output[:,-1]), sequence_output, hidden

model=TinyRNN(); optimizer=torch.optim.Adam(model.parameters(),lr=.02); loss_fn=nn.MSELoss()
train_losses=[]; validation_losses=[]
for epoch in range(40):
    model.train(); optimizer.zero_grad(); prediction,sequence_output,hidden=model(train_X)
    loss=loss_fn(prediction,train_y); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): val_prediction,_,_=model(validation_X); val_loss=loss_fn(val_prediction,validation_y)
    train_losses.append(loss.item()); validation_losses.append(val_loss.item())
validation_rmse = float(torch.sqrt(loss_fn(val_prediction, validation_y)))
print("Model diagnostics:", {
    "input_shape": tuple(train_X.shape),
    "intermediate_shapes": (tuple(sequence_output.shape), tuple(hidden.shape)),
    "output_shape": tuple(prediction.shape),
    "trainable_parameters": sum(p.numel() for p in model.parameters()),
    "final_training_loss": round(train_losses[-1], 5),
    "final_validation_loss": round(validation_losses[-1], 5),
    "validation_RMSE": round(validation_rmse, 5),
})
figure,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].plot(train_losses,label="Train",color=COURSE_COLORS["blue"]); axes[0].plot(validation_losses,label="Validation",color=COURSE_COLORS["orange"],linestyle="--")
axes[0].set(title="RNN learning curves",xlabel="Epoch",ylabel="MSE"); axes[0].legend()
axes[1].plot(validation_y.numpy()[:50],label="True",color=COURSE_COLORS["blue"]); axes[1].plot(val_prediction.numpy()[:50],label="Predicted",color=COURSE_COLORS["orange"],linestyle="--")
axes[1].set(title="Next-step sine prediction",xlabel="Validation example",ylabel="Signal value"); axes[1].legend()
figure.tight_layout(); plt.show()
"""
    hyper = r"""
lengths=[4,8,12,24]
dependencies=np.array(lengths)
figure,axis=plt.subplots()
axis.plot(lengths,dependencies,marker="o",color=COURSE_COLORS["blue"],label="Available context")
axis.set(title="Sequence window controls available history",xlabel="Input time steps",ylabel="Past observations visible")
axis.legend(); plt.show()
"""
    return base_spec(
        title="Recurrent Neural Networks",
        objectives=["Implement a recurrent state update", "Track batch, time, and hidden dimensions", "Train an RNN for next-step prediction"],
        problem="Process ordered observations while carrying information from earlier time steps.",
        intuition="The hidden state is a running notebook: each new input edits the notebook before the next step arrives.",
        equation=r"$$\mathbf{h}_t=\tanh(\mathbf{x}_tW_{xh}+\mathbf{h}_{t-1}W_{hh}+\mathbf{b}).$$",
        symbols=[r"$\mathbf{x}_t$ is input at time t.", r"$\mathbf{h}_{t-1}$ is prior state.", r"$W_{xh},W_{hh}$ are learned weights.", r"$\mathbf{b}$ is bias.", r"$\mathbf{h}_t$ is new state."],
        why_equation="Reusing the same update at every step supports variable-length ordered data.",
        numerical_example="If input contribution=.4 and recurrent contribution=.2, the state before tanh is .6 and tanh(.6)=.537.",
        python_connection="nn.RNN performs this recurrence for all steps and batches.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Window length controls available context but increases computation and can make gradient propagation harder.",
        hyperparameter_code=hyper,
        interpretation="The RNN learns short sine dynamics; divergence between train and validation loss would signal overfitting or distribution mismatch.",
    )


def lstm_gru_spec() -> LessonSpec:
    scratch = r"""
def sigmoid_np(values):
    return 1/(1+np.exp(-values))

candidate=.7
previous_cell=.4
forget_gate=sigmoid_np(1.0)
input_gate=sigmoid_np(.2)
new_cell=forget_gate*previous_cell+input_gate*candidate
print("Forget gate:",round(forget_gate,3),"input gate:",round(input_gate,3),"new cell:",round(new_cell,3))
"""
    experiment = r"""
samples=300; time_steps=16
X=rng.normal(size=(samples,time_steps,1)).astype("float32")
y=(X[:,:8,0].sum(axis=1)>X[:,8:,0].sum(axis=1)).astype("int64")
split=225
train_X=torch.tensor(X[:split]); train_y=torch.tensor(y[:split])
validation_X=torch.tensor(X[split:]); validation_y=torch.tensor(y[split:])

class GatedClassifier(nn.Module):
    def __init__(self,kind):
        super().__init__()
        recurrent=nn.LSTM if kind=="LSTM" else nn.GRU
        self.recurrent=recurrent(1,12,batch_first=True)
        self.output=nn.Linear(12,2)
    def forward(self,values):
        sequence,state=self.recurrent(values)
        return self.output(sequence[:,-1]),sequence,state

histories={}; models={}
for kind in ["LSTM","GRU"]:
    model=GatedClassifier(kind); optimizer=torch.optim.Adam(model.parameters(),lr=.02); loss_fn=nn.CrossEntropyLoss()
    train_history=[]; validation_history=[]
    for epoch in range(18):
        model.train(); optimizer.zero_grad(); logits,sequence,state=model(train_X)
        loss=loss_fn(logits,train_y); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_logits,_,_=model(validation_X); val_loss=loss_fn(val_logits,validation_y)
        train_history.append(loss.item()); validation_history.append(val_loss.item())
    histories[kind]=(train_history,validation_history); models[kind]=(model,val_logits)
    accuracy=(val_logits.argmax(1)==validation_y).float().mean().item()
    print(f"{kind} model diagnostics:", {
        "input_shape": tuple(train_X.shape),
        "intermediate_shape": tuple(sequence.shape),
        "output_shape": tuple(logits.shape),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "final_training_loss": round(train_history[-1], 4),
        "final_validation_loss": round(validation_history[-1], 4),
        "validation_accuracy": round(accuracy, 3),
    })

figure,axis=plt.subplots()
for kind,(train_history,validation_history) in histories.items():
    color=COURSE_COLORS["blue"] if kind=="LSTM" else COURSE_COLORS["orange"]
    axis.plot(train_history,label=f"{kind} training",color=color)
    axis.plot(validation_history,label=f"{kind} validation",color=color,linestyle="--")
axis.set(title="LSTM and GRU learning curves",xlabel="Epoch",ylabel="Cross-entropy"); axis.legend(); plt.show()
"""
    hyper = r"""
hidden_sizes=[4,8,12,24]
lstm_counts=[4*(1*h+h*h+h) + (2*h+2) for h in hidden_sizes]
gru_counts=[3*(1*h+h*h+h) + (2*h+2) for h in hidden_sizes]
figure,axis=plt.subplots()
axis.plot(hidden_sizes,lstm_counts,marker="o",label="LSTM",color=COURSE_COLORS["blue"])
axis.plot(hidden_sizes,gru_counts,marker="s",label="GRU",color=COURSE_COLORS["orange"])
axis.set(title="Gated-unit parameter growth",xlabel="Hidden size",ylabel="Approximate trainable parameters"); axis.legend(); plt.show()
"""
    return base_spec(
        title="LSTM and GRU Networks",
        objectives=["Explain gating and memory cells", "Compare LSTM and GRU shapes and parameters", "Train gated sequence classifiers"],
        problem="Preserve or forget information over longer sequences while reducing vanishing-gradient difficulty.",
        intuition="Gates are learned valves: forget controls retained memory, input controls new information, and output controls exposed state.",
        equation=r"$$\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t.$$",
        symbols=[r"$\mathbf{c}_t$ is the LSTM cell state.", r"$\mathbf{f}_t$ is forget gate.", r"$\mathbf{i}_t$ is input gate.", r"$\tilde{\mathbf{c}}_t$ is candidate memory.", r"$\odot$ is elementwise multiplication."],
        why_equation="An additive memory path lets gradients and information travel across more time steps.",
        numerical_example="With old cell .4, forget .73, input .55, candidate .7, new cell=.73(.4)+.55(.7)=.677.",
        python_connection="nn.LSTM returns both hidden and cell state; nn.GRU combines gating into one hidden state.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Hidden size controls memory capacity; LSTM generally has more gate parameters than GRU.",
        hyperparameter_code=hyper,
        interpretation="Validation curves compare optimization on the same sequence task; parameter counts clarify the complexity tradeoff.",
    )


def attention_spec() -> LessonSpec:
    scratch = r"""
def scaled_dot_product_attention(query,key,value):
    scores=query@key.T/np.sqrt(query.shape[-1])
    shifted=scores-scores.max(axis=-1,keepdims=True)
    weights=np.exp(shifted); weights/=weights.sum(axis=-1,keepdims=True)
    return weights@value,weights

Q=np.array([[1.,0.],[0.,1.]])
K=np.array([[1.,0.],[0.,1.],[1.,1.]])
V=np.array([[1.,0.],[0.,2.],[3.,3.]])
context,weights=scaled_dot_product_attention(Q,K,V)
print("Q/K/V/context:",Q.shape,K.shape,V.shape,context.shape)
print("Attention row sums:",weights.sum(axis=1))
"""
    experiment = r"""
embedding=nn.Embedding(6,8)
attention=nn.MultiheadAttention(embed_dim=8,num_heads=2,batch_first=True)
tokens=torch.tensor([[0,1,2,3]])
embedded=embedding(tokens)
attended,torch_weights=attention(embedded,embedded,embedded,need_weights=True,average_attn_weights=False)
print("Tokens/embeddings/output:",tuple(tokens.shape),tuple(embedded.shape),tuple(attended.shape))
print("Attention weights:",tuple(torch_weights.shape))
print("Parameters:",sum(p.numel() for p in embedding.parameters())+sum(p.numel() for p in attention.parameters()))
print("Loss/metric status: not applicable; this is the guide's forward-only educational attention example, not a trained model.")

figure,axes=plt.subplots(1,2,figsize=(10,4))
sns.heatmap(weights,annot=True,cmap="Blues",vmin=0,vmax=1,ax=axes[0])
axes[0].set(title="Manual scaled attention",xlabel="Key position",ylabel="Query position")
sns.heatmap(torch_weights[0,0].detach().numpy(),annot=True,cmap="Oranges",vmin=0,vmax=1,ax=axes[1])
axes[1].set(title="PyTorch attention head 1",xlabel="Key token",ylabel="Query token")
figure.tight_layout(); plt.show()
"""
    hyper = r"""
heads=[1,2,4,8]
head_dimensions=[8//value for value in heads]
figure,axis=plt.subplots()
axis.bar([str(value) for value in heads],head_dimensions,color=COURSE_COLORS["blue"])
axis.set(title="At fixed embedding size, more heads use smaller subspaces",xlabel="Attention heads",ylabel="Dimensions per head")
plt.show()
"""
    return base_spec(
        title="Attention and Transformer Fundamentals",
        objectives=["Compute scaled dot-product attention", "Interpret attention heatmaps", "Use embeddings and multi-head attention"],
        problem="Let each sequence position directly combine information from every relevant position.",
        intuition="A query asks what it needs, keys advertise available information, and values carry the information that matching queries collect.",
        equation=r"$$Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V.$$",
        symbols=[r"$Q$ contains queries.", r"$K$ contains keys.", r"$V$ contains values.", r"$d_k$ is key dimension.", r"$softmax$ creates row-wise weights summing to one."],
        why_equation="Dot products measure query-key compatibility; scaling prevents large dimensions from making softmax overly sharp.",
        numerical_example="If two scaled scores are equal, softmax assigns [.5,.5], so context is their values' average.",
        python_connection="The scratch function exposes scores, normalized weights, and weighted values; MultiheadAttention repeats this in learned subspaces.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="At fixed embedding width, head count trades the number of relationship subspaces against dimensions per head.",
        hyperparameter_code=hyper,
        interpretation="Each heatmap row sums to one. Bright cells indicate larger contribution to that query's context, not necessarily a causal explanation.",
    )


def regularization_spec() -> LessonSpec:
    scratch = r"""
def dropout_scratch(values, probability, seed=42):
    generator=np.random.default_rng(seed)
    mask=generator.random(values.shape)>=probability
    return values*mask/(1-probability),mask

values=np.ones(10)
dropped,mask=dropout_scratch(values,.5)
print("Mask:",mask.astype(int),"training mean:",round(dropped.mean(),2),"evaluation output:",values.mean())
"""
    experiment = r"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits=load_digits()
X_train,X_validation,y_train,y_validation=train_test_split(digits.data,digits.target,test_size=.3,stratify=digits.target,random_state=RANDOM_SEED)
scaler=StandardScaler(); X_train=scaler.fit_transform(X_train); X_validation=scaler.transform(X_validation)
train_X=torch.tensor(X_train,dtype=torch.float32); train_y=torch.tensor(y_train,dtype=torch.long)
validation_X=torch.tensor(X_validation,dtype=torch.float32); validation_y=torch.tensor(y_validation,dtype=torch.long)

def train_candidate(dropout,weight_decay):
    model=nn.Sequential(nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,10))
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=weight_decay); loss_fn=nn.CrossEntropyLoss()
    train_history=[]; validation_history=[]
    for _ in range(10):
        model.train(); optimizer.zero_grad(); logits=model(train_X); loss=loss_fn(logits,train_y); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_logits=model(validation_X); val_loss=loss_fn(val_logits,validation_y)
        train_history.append(loss.item()); validation_history.append(val_loss.item())
    accuracy=(val_logits.argmax(1)==validation_y).float().mean().item()
    return model,train_history,validation_history,accuracy

plain,plain_train,plain_val,plain_acc=train_candidate(0.0,0.0)
regularized,reg_train,reg_val,reg_acc=train_candidate(.3,1e-4)
regularized.eval()
with torch.no_grad():
    intermediate = regularized[2](regularized[1](regularized[0](train_X)))
print("Regularized model diagnostics:", {
    "input_shape": tuple(train_X.shape),
    "intermediate_shape": tuple(intermediate.shape),
    "output_shape": (len(train_X), 10),
    "trainable_parameters": sum(p.numel() for p in regularized.parameters()),
    "final_training_loss": round(reg_train[-1], 4),
    "final_validation_loss": round(reg_val[-1], 4),
    "validation_accuracy": round(reg_acc, 3),
})
print("Plain model final train/validation loss and accuracy:",round(plain_train[-1],4),round(plain_val[-1],4),round(plain_acc,3))
figure,axis=plt.subplots()
axis.plot(plain_train,label="Plain training",color=COURSE_COLORS["blue"],alpha=.6)
axis.plot(plain_val,label="Plain validation",color=COURSE_COLORS["blue"])
axis.plot(reg_train,label="Regularized training",color=COURSE_COLORS["orange"],alpha=.6)
axis.plot(reg_val,label="Regularized validation",color=COURSE_COLORS["orange"],linestyle="--")
axis.set(title="Regularization changes validation loss",xlabel="Epoch",ylabel="Cross-entropy"); axis.legend(); plt.show()
"""
    hyper = r"""
rates=[0,.1,.3,.5,.7]
expected_active=[64*(1-rate) for rate in rates]
figure,axis=plt.subplots()
axis.plot(rates,expected_active,marker="o",color=COURSE_COLORS["blue"],label="Expected active units")
axis.set(title="Dropout rate controls retained activations",xlabel="Dropout probability",ylabel="Expected active units out of 64")
axis.legend(); plt.show()
"""
    return base_spec(
        title="Regularization and Optimization",
        objectives=["Implement inverted dropout", "Use batch normalization and weight decay", "Compare regularized validation curves"],
        problem="Optimize efficiently while limiting the gap between training fit and unseen-data performance.",
        intuition="Regularization limits reliance on fragile details; optimizers decide how gradient information changes parameters.",
        equation=r"$$\tilde{h}_j=\frac{m_jh_j}{1-p},\qquad m_j\sim Bernoulli(1-p).$$",
        symbols=[r"$h_j$ is activation j.", r"$m_j$ is a random keep mask.", r"$p$ is dropout probability.", r"$\tilde h_j$ is training-time output."],
        why_equation="Division by keep probability preserves expected activation magnitude during training.",
        numerical_example="With h=2, p=.5, kept output is 2/(.5)=4 and dropped output is 0; expectation is 2.",
        python_connection="nn.Dropout applies inverted dropout in train mode and becomes identity in eval mode.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Higher dropout reduces active capacity; select it with validation results and use model.eval() during evaluation.",
        hyperparameter_code=hyper,
        interpretation="The comparison shows the generalization effect on one fixed split; repeated seeds are needed before a strong conclusion.",
    )


def capstone_spec() -> LessonSpec:
    scratch = r"""
def accuracy_from_logits(logits,target):
    return float((logits.argmax(axis=1)==target).mean())

example_logits=np.array([[2.,1.],[.2,.8]])
print("Manual accuracy:",accuracy_from_logits(example_logits,np.array([0,1])))
"""
    experiment = r"""
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits=load_digits()
X_development,X_test,y_development,y_test=train_test_split(digits.data,digits.target,test_size=.2,stratify=digits.target,random_state=RANDOM_SEED)
X_train,X_validation,y_train,y_validation=train_test_split(X_development,y_development,test_size=.25,stratify=y_development,random_state=RANDOM_SEED)
scaler=StandardScaler(); X_train_s=scaler.fit_transform(X_train); X_validation_s=scaler.transform(X_validation)
baseline=LogisticRegression(max_iter=1000,random_state=RANDOM_SEED).fit(X_train_s,y_train)
baseline_accuracy=baseline.score(X_validation_s,y_validation)

def fit_mlp(width):
    model=nn.Sequential(nn.Linear(64,width),nn.ReLU(),nn.Dropout(.2),nn.Linear(width,10))
    optimizer=torch.optim.Adam(model.parameters(),lr=.01); loss_fn=nn.CrossEntropyLoss()
    Xt=torch.tensor(X_train_s,dtype=torch.float32); yt=torch.tensor(y_train,dtype=torch.long)
    Xv=torch.tensor(X_validation_s,dtype=torch.float32); yv=torch.tensor(y_validation,dtype=torch.long)
    train_history=[]; validation_history=[]
    for _ in range(12):
        model.train(); optimizer.zero_grad(); logits=model(Xt); loss=loss_fn(logits,yt); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_logits=model(Xv); val_loss=loss_fn(val_logits,yv)
        train_history.append(loss.item()); validation_history.append(val_loss.item())
    return model,train_history,validation_history,float((val_logits.argmax(1)==yv).float().mean())

candidates={}
for width in [32,64,128]:
    candidates[width]=fit_mlp(width)
best_width=max(candidates,key=lambda value:candidates[value][3])
best_model,train_losses,validation_losses,best_validation_accuracy=candidates[best_width]
print("Dataset/missing:",digits.data.shape,int(np.isnan(digits.data).sum()))
print("Baseline accuracy:",round(baseline_accuracy,3),"best width/accuracy:",best_width,round(best_validation_accuracy,3))
best_model.eval()
with torch.no_grad():
    diagnostic_input=torch.tensor(X_train_s,dtype=torch.float32)
    diagnostic_hidden=best_model[1](best_model[0](diagnostic_input))
    diagnostic_output=best_model(diagnostic_input)
print("Selected MLP diagnostics:", {
    "input_shape": tuple(diagnostic_input.shape),
    "intermediate_shape": tuple(diagnostic_hidden.shape),
    "output_shape": tuple(diagnostic_output.shape),
    "trainable_parameters": sum(p.numel() for p in best_model.parameters()),
    "final_training_loss": round(train_losses[-1], 4),
    "final_validation_loss": round(validation_losses[-1], 4),
    "validation_accuracy": round(best_validation_accuracy, 3),
})

final_scaler=StandardScaler(); X_development_s=final_scaler.fit_transform(X_development); X_test_s=final_scaler.transform(X_test)
final_model=nn.Sequential(nn.Linear(64,best_width),nn.ReLU(),nn.Dropout(.2),nn.Linear(best_width,10))
optimizer=torch.optim.Adam(final_model.parameters(),lr=.01); loss_fn=nn.CrossEntropyLoss()
Xd=torch.tensor(X_development_s,dtype=torch.float32); yd=torch.tensor(y_development,dtype=torch.long)
for _ in range(12):
    final_model.train(); optimizer.zero_grad(); loss=loss_fn(final_model(Xd),yd); loss.backward(); optimizer.step()
final_model.eval()
with torch.no_grad(): test_prediction=final_model(torch.tensor(X_test_s,dtype=torch.float32)).argmax(1).numpy()
test_accuracy=accuracy_score(y_test,test_prediction)
print("Held-out test accuracy:",round(test_accuracy,3))
figure,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].plot(train_losses,label="Train",color=COURSE_COLORS["blue"]); axes[0].plot(validation_losses,label="Validation",color=COURSE_COLORS["orange"],linestyle="--")
axes[0].set(title="Selected MLP learning curves",xlabel="Epoch",ylabel="Cross-entropy"); axes[0].legend()
ConfusionMatrixDisplay.from_predictions(y_test,test_prediction,cmap="Blues",colorbar=False,ax=axes[1]); axes[1].set_title("Held-out digit errors")
figure.tight_layout(); plt.show()
"""
    hyper = r"""
comparison=pd.DataFrame([
    {"model":"Logistic baseline","validation_accuracy":baseline_accuracy,"parameters":"library-managed"},
    *[{"model":f"MLP width {width}","validation_accuracy":values[3],"parameters":sum(p.numel() for p in values[0].parameters())} for width,values in candidates.items()],
])
display(comparison.round(3))
figure,axis=plt.subplots()
axis.bar(comparison["model"],comparison["validation_accuracy"],color=COURSE_COLORS["blue"])
axis.set(title="Capstone model comparison",xlabel="Candidate",ylabel="Validation accuracy")
axis.tick_params(axis="x",rotation=20)
plt.show()
"""
    return base_spec(
        title="Deep Learning Capstone: Handwritten Digit Classification",
        objectives=["Build a leakage-safe end-to-end image classifier", "Compare a linear baseline with tuned MLPs", "Evaluate errors on an untouched test set"],
        problem="Classify 8-by-8 handwritten digits; success is held-out accuracy plus interpretable class-level error counts.",
        use_cases=["Document digit recognition", "Educational image pipelines", "Model comparison under CPU constraints"],
        intuition="Begin with a strong linear baseline, add nonlinear hidden features only if validation evidence supports them, then freeze the choice for test evaluation.",
        scratch_description="A transparent logits-to-accuracy calculation checks output semantics before model training.",
        library_description="Scikit-learn supplies the baseline and split/scaling tools; PyTorch supplies the MLP and training loop.",
        dataset_description="The bundled Digits dataset has 1,797 grayscale 8-by-8 images and ten balanced classes.",
        preprocessing_description="Stratified development/test and train/validation splits occur before StandardScaler fitting.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Hidden width is selected on validation accuracy; the test set remains untouched until the choice is frozen.",
        hyperparameter_code=hyper,
        interpretation="Learning curves reveal optimization; the confusion matrix exposes which classes remain difficult instead of hiding them behind one score.",
        mistakes=["Selecting width on test accuracy", "Scaling before splitting", "Reporting only training loss", "Ignoring systematic class confusions"],
        advantages=["Complete offline pipeline", "Explicit baseline and tuning", "Held-out error analysis"],
        disadvantages=["Small low-resolution benchmark", "No robustness or fairness audit", "One split gives limited uncertainty evidence"],
        exercises=["Why is the logistic model a useful baseline?", "If 342 of 360 test images are correct, calculate accuracy.", "Repeat the final training with three seeds and report mean and range."],
        solutions=["It tests whether nonlinear representation learning improves on a simpler, easier-to-audit decision boundary.", "342/360=.95 or 95%.", "Reset all seeds, rebuild and retrain three times, then report the three test scores, mean, and range without selecting the best run."],
        takeaways=["Success metrics and splits precede modeling", "A deep model should earn its complexity against a baseline", "Tuning uses validation data and final claims use untouched test data", "Error analysis and limitations are part of the conclusion"],
    )


def main() -> None:
    lessons = {
        "03_deep_learning/00_deep_learning_overview.ipynb": overview_spec(),
        "03_deep_learning/01_tensors_and_linear_algebra.ipynb": tensors_spec(),
        "03_deep_learning/02_perceptron.ipynb": perceptron_spec(),
        "03_deep_learning/03_neural_network_from_scratch.ipynb": scratch_network_spec(),
        "03_deep_learning/04_gradient_descent_and_backpropagation.ipynb": backprop_spec(),
        "03_deep_learning/05_pytorch_fundamentals.ipynb": pytorch_spec(),
        "03_deep_learning/06_multilayer_perceptron.ipynb": mlp_spec(),
        "03_deep_learning/07_convolutional_neural_network.ipynb": cnn_spec(),
        "03_deep_learning/08_recurrent_neural_network.ipynb": rnn_spec(),
        "03_deep_learning/09_lstm_and_gru.ipynb": lstm_gru_spec(),
        "03_deep_learning/10_attention_and_transformers.ipynb": attention_spec(),
        "03_deep_learning/11_regularization_and_optimization.ipynb": regularization_spec(),
        "03_deep_learning/12_deep_learning_capstone_project.ipynb": capstone_spec(),
    }
    for path, spec in lessons.items():
        write_lesson(path, spec)
        print("Built", path)


if __name__ == "__main__":
    main()
