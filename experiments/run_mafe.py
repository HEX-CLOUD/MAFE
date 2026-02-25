import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from agents.transformer import TransformationAgent

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results"

# -----------------------------
# Load Adult Dataset
# -----------------------------
def load_adult():
    cols = [
        "age","workclass","fnlwgt","education","education_num",
        "marital_status","occupation","relationship","race","sex",
        "capital_gain","capital_loss","hours_per_week","native_country","income"
    ]

    df = pd.read_csv(
        DATA_DIR / "adult" / "adult.data",
        names=cols,
        na_values=" ?",
        skipinitialspace=True
    )

    df.dropna(inplace=True)
    df["income"] = (df["income"] == ">50K").astype(int)

    X = df.drop("income", axis=1)
    y = df["income"]
    return X, y

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return (
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
    )

# -----------------------------
# MAFE Round-1
# -----------------------------
if __name__ == "__main__":
    X, y = load_adult()

    # Baseline split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric = X.select_dtypes(include=np.number).columns
    categorical = X.select_dtypes(exclude=np.number).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Baseline
    model.fit(X_train, y_train)
    base_acc, base_f1, base_auc = evaluate(model, X_test, y_test)

    # --- Agent step ---
    agent = TransformationAgent()
    new_feats = agent.propose_features(X_train)

    X_train_aug = pd.concat([X_train.reset_index(drop=True),
                             new_feats.reset_index(drop=True)], axis=1)
    X_test_aug = pd.concat([
        X_test.reset_index(drop=True),
        agent.propose_features(X_test).reset_index(drop=True)
    ], axis=1)

    # Rebuild pipeline
    numeric_aug = X_train_aug.select_dtypes(include=np.number).columns
    categorical_aug = X_train_aug.select_dtypes(exclude=np.number).columns

    preprocessor_aug = ColumnTransformer([
        ("num", StandardScaler(), numeric_aug),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_aug)
    ])

    model_aug = Pipeline([
        ("prep", preprocessor_aug),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model_aug.fit(X_train_aug, y_train)
    acc, f1, auc = evaluate(model_aug, X_test_aug, y_test)

    print("\n=== MAFE Round-1 (Transformation Agent) ===")
    print(f"Baseline  -> acc={base_acc:.4f}, f1={base_f1:.4f}, auc={base_auc:.4f}")
    print(f"With Agent-> acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")

    # Save result
    out = pd.DataFrame([{
        "dataset": "Adult",
        "round": 1,
        "agent": "TransformationAgent",
        "features_added": new_feats.shape[1],
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    }])

    out.to_csv(RESULTS_DIR / "mafe_runs.csv", index=False)