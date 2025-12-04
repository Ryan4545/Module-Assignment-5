"""
breast_cancer_tree.py

Supervised learning case study:
- Predict benign vs malignant breast tumors using a decision tree.
- Generates basic metrics, figures, and example misclassifications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

RANDOM_STATE = 42


def load_data():
    """Load the Breast Cancer Wisconsin dataset as (X, y, feature_names, target_names, df)."""
    data = load_breast_cancer()
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign
    feature_names = data.feature_names
    target_names = data.target_names

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y

    return X, y, feature_names, target_names, df


def train_model(X, y, max_depth=4, test_size=0.2):
    """Train a DecisionTreeClassifier and return model + split data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)

    return tree, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, target_names):
    """Compute and print accuracy, confusion matrix, ROC-AUC, and classification report."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of benign (class 1)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("=== Test Performance ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC–AUC: {auc:.3f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            digits=3,
        )
    )

    return y_pred, y_proba, acc, auc, cm


def plot_roc_curve(y_test, y_proba, outpath="roc_curve.png"):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label="Decision Tree")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Breast Cancer (Benign vs Malignant)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[INFO] Saved ROC curve to {outpath}")


def plot_feature_importances(model, feature_names, outpath="feature_importances.png", top_k=15):
    """Plot and save top-k feature importances from the decision tree."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(top_features)), top_importances[::-1])
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_k} Feature Importances – Decision Tree")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[INFO] Saved feature importances to {outpath}")


def inspect_misclassifications(X_test, y_test, y_pred, feature_names, n_samples=5):
    """
    Return a DataFrame with up to n_samples misclassified points
    and print a short human-readable summary.
    """
    mis_idx = np.where(y_test != y_pred)[0]
    print(f"[INFO] Total misclassified test samples: {len(mis_idx)}")

    if len(mis_idx) == 0:
        print("[INFO] No misclassifications to inspect.")
        return pd.DataFrame()

    # Limit to first n_samples (reproducible due to fixed random state)
    mis_idx = mis_idx[:n_samples]

    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_err = df_test.iloc[mis_idx].copy()
    df_err["true_label"] = y_test[mis_idx]
    df_err["pred_label"] = y_pred[mis_idx]

    print("\n=== Example Misclassified Samples (first few) ===")
    for i, idx in enumerate(mis_idx):
        row = df_err.iloc[i]
        tl = int(row["true_label"])
        pl = int(row["pred_label"])

        print(f"\nSample #{i+1} (test index {idx}):")
        print(f"  True label: {tl} (0=malignant, 1=benign)")
        print(f"  Pred label: {pl}")
        print(
            f"  mean radius: {row['mean radius']:.2f}, "
            f"worst radius: {row['worst radius']:.2f}, "
            f"worst concave points: {row['worst concave points']:.4f}"
        )

    return df_err


def main():
    # 1. Load data
    X, y, feature_names, target_names, df = load_data()
    print("[INFO] Data loaded.")
    print(df.head())

    # 2. Train model
    model, X_train, X_test, y_train, y_test = train_model(X, y, max_depth=4, test_size=0.2)
    print("[INFO] Model trained (DecisionTreeClassifier, max_depth=4).")

    # 3. Evaluate
    y_pred, y_proba, acc, auc, cm = evaluate_model(model, X_test, y_test, target_names)

    # 4. Figures
    plot_roc_curve(y_test, y_proba, outpath="roc_curve.png")
    plot_feature_importances(model, feature_names, outpath="feature_importances.png", top_k=15)

    # 5. Misclassifications
    df_err = inspect_misclassifications(X_test, y_test, y_pred, feature_names, n_samples=5)

    # Optionally, save misclassified samples to CSV for further qualitative analysis
    if not df_err.empty:
        df_err.to_csv("misclassified_samples.csv", index=False)
        print("[INFO] Saved misclassified samples to misclassified_samples.csv")


if __name__ == "__main__":
    main()
