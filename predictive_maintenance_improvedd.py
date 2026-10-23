# ============================================================
# AI-BASED PREDICTIVE MAINTENANCE
# Classical ML vs Quantum ML
# Dataset: AI4I 2020
# Quantum framework: PennyLane
# ============================================================

import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="autograd")

import numpy as np
import pandas as pd
import pennylane as qml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

CSV_PATH = r"C:\Users\jcsah\Downloads\archive\ai4i2020.csv"

df = pd.read_csv(CSV_PATH)

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features].copy()
y = df["Machine failure"].astype(int).copy()

print("\nDATASET")
print("Shape:", df.shape)
print("Class distribution:\n", y.value_counts().sort_index())

# ============================================================
# 2. SPLIT INTO TRAINING, VALIDATION AND TEST SETS
# ============================================================

# Test set remains untouched until final evaluation.
X_trainval_raw, X_test_raw, y_trainval, y_test = (
    train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

# 20% of the remaining data is used for validation.
X_train_raw, X_val_raw, y_train, y_val = (
    train_test_split(
        X_trainval_raw, y_trainval,
        test_size=0.20,
        random_state=42,
        stratify=y_trainval
    )
)

print("\nTraining:", len(X_train_raw))
print("Validation:", len(X_val_raw))
print("Testing:", len(X_test_raw))

# ============================================================
# 3. SCALE AND APPLY PCA
# Fit preprocessing ONLY on training data.
# ============================================================

standard_scaler = StandardScaler()

X_train_scaled = standard_scaler.fit_transform(X_train_raw)
X_val_scaled = standard_scaler.transform(X_val_raw)
X_test_scaled = standard_scaler.transform(X_test_raw)

pca = PCA(n_components=4, random_state=42)

X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(
    "\nPCA variance retained:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

# ============================================================
# 4. CLASSICAL BASELINE: RANDOM FOREST
# ============================================================

print("\nTRAINING RANDOM FOREST")

classical_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

start = time.perf_counter()
classical_model.fit(X_train_pca, y_train)
classical_train_time = time.perf_counter() - start

# Probabilities will be used for threshold tuning.
classical_val_scores = classical_model.predict_proba(
    X_val_pca
)[:, 1]

classical_test_scores = classical_model.predict_proba(
    X_test_pca
)[:, 1]

# ============================================================
# 5. SCALE PCA FEATURES FOR QUANTUM ANGLE ENCODING
# Fit on training data only.
# ============================================================

angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))

X_train_angle = angle_scaler.fit_transform(X_train_pca)
X_val_angle = angle_scaler.transform(X_val_pca)
X_test_angle = angle_scaler.transform(X_test_pca)

# ============================================================
# 6. BALANCE ONLY THE QUANTUM TRAINING SUBSET
# Validation and test distributions remain unchanged.
# ============================================================

train_df = pd.DataFrame(X_train_angle)
train_df["target"] = y_train.to_numpy()

class_0 = train_df[train_df["target"] == 0]
class_1 = train_df[train_df["target"] == 1]

# Up to 200 examples from each class.
n_per_class = min(200, len(class_0), len(class_1))

if n_per_class < 2:
    raise ValueError("Not enough examples in each training class.")

class_0_sample = resample(
    class_0,
    replace=False,
    n_samples=n_per_class,
    random_state=42
)

class_1_sample = resample(
    class_1,
    replace=False,
    n_samples=n_per_class,
    random_state=42
)

quantum_df = pd.concat(
    [class_0_sample, class_1_sample],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

X_train_q = quantum_df.drop(columns=["target"]).to_numpy()

y_train_q = quantum_df["target"].to_numpy()

# Convert labels 0/1 to -1/+1 for quantum regression-style loss.
y_train_signed = np.where(y_train_q == 0, -1.0, 1.0)

print("\nQUANTUM TRAINING")
print("Balanced samples:", len(X_train_q))
print("Samples per class:", n_per_class)

# ============================================================
# 7. FOUR-QUBIT VARIATIONAL QUANTUM CIRCUIT
# ============================================================

n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="autograd", diff_method="backprop")
def quantum_circuit(weights, x):

    qml.AngleEmbedding(
        features=x,
        wires=range(n_qubits),
        rotation="Y"
    )

    qml.StronglyEntanglingLayers(
        weights,
        wires=range(n_qubits)
    )

    return qml.expval(qml.PauliZ(0))

weight_shapes = qml.StronglyEntanglingLayers.shape(
    n_layers=n_layers,
    n_wires=n_qubits
)

rng = np.random.default_rng(42)

weights = qml.numpy.array(
    0.1 * rng.standard_normal(weight_shapes),
    requires_grad=True
)

# ============================================================
# 8. QUANTUM LOSS
# Circuit output is in [-1, +1].
# Target labels are also -1 or +1.
# ============================================================

def quantum_cost(weights, X_batch, y_signed):

    predictions = qml.numpy.stack([
        quantum_circuit(weights, x)
        for x in X_batch
    ])

    return qml.numpy.mean((predictions - y_signed) ** 2)

# ============================================================
# 9. TRAIN WITH ADAM
# ============================================================

print("\nTRAINING QUANTUM CLASSIFIER")

optimizer = qml.AdamOptimizer(stepsize=0.01)
epochs = 25

quantum_train_start = time.perf_counter()

for epoch in range(epochs):

    weights, loss = optimizer.step_and_cost(
        lambda w: quantum_cost(
            w, X_train_q, y_train_signed
        ),
        weights
    )

    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Loss: {float(loss):.5f}"
    )

quantum_train_time = time.perf_counter() - quantum_train_start

# ============================================================
# 10. GET CONTINUOUS SCORES FROM THE QUANTUM MODEL
# Higher expectation value means more evidence for class 1.
# ============================================================

def quantum_scores(X_data):
    return np.array([
        float(quantum_circuit(weights, x))
        for x in X_data
    ])

quantum_val_scores = quantum_scores(X_val_angle)
quantum_test_scores = quantum_scores(X_test_angle)

print(
    "\nQuantum validation score range:",
    quantum_val_scores.min(),
    "to",
    quantum_val_scores.max()
)

# ============================================================
# 11. SELECT THRESHOLDS USING VALIDATION DATA ONLY
# ============================================================

def best_f1_threshold(y_true, scores):

    precision, recall, thresholds = precision_recall_curve(
        y_true, scores
    )

    if len(thresholds) == 0:
        return 0.0

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])

rf_threshold = best_f1_threshold(y_val, classical_val_scores)

q_threshold = best_f1_threshold(y_val, quantum_val_scores)

print("\nValidation-selected thresholds")
print("Random Forest:", round(rf_threshold, 4))
print("Quantum:", round(q_threshold, 4))

# Apply fixed validation thresholds to the untouched test set.
classical_predictions = (
    classical_test_scores >= rf_threshold
).astype(int)

quantum_predictions = (
    quantum_test_scores >= q_threshold
).astype(int)

# ============================================================
# 12. EVALUATION
# ============================================================

def evaluate_model(name, y_true, y_pred):

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["No machine failure", "Machine failure"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

evaluate_model(
    "RANDOM FOREST",
    y_test,
    classical_predictions
)

evaluate_model(
    "QUANTUM VARIATIONAL CLASSIFIER",
    y_test,
    quantum_predictions
)

# ============================================================
# 13. FINAL COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": ["Random Forest", "Quantum Variational Classifier"],
    "Accuracy": [
        accuracy_score(y_test, classical_predictions),
        accuracy_score(y_test, quantum_predictions)
    ],
    "Precision": [
        precision_score(y_test, classical_predictions, zero_division=0),
        precision_score(y_test, quantum_predictions, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, classical_predictions, zero_division=0),
        recall_score(y_test, quantum_predictions, zero_division=0)
    ],
    "F1-score": [
        f1_score(y_test, classical_predictions, zero_division=0),
        f1_score(y_test, quantum_predictions, zero_division=0)
    ]
})

print("\nFINAL TEST COMPARISON")
print(results.round(4).to_string(index=False))

print("\nTraining time (seconds)")
print("Random Forest:", round(classical_train_time, 3))
print("Quantum:", round(quantum_train_time, 3))

print("\nExperiment completed.")
