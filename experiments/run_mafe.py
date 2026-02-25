import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from agents.transformer import TransformationAgent
from agents.interaction import InteractionAgent
from agents.coordinator import CoordinatorAgent

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load Adult Dataset
# -----------------------------
def load_adult():
    cols = [
        "age","workclass","fnlwgt","education","education_num",
        "marital_status","occupation","relationship","race","sex",
        "capital_gain","capital_loss","hours_per_week",
        "native_country","income"
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
# Main (MAFE Round-2: Multi-Agent)
# -----------------------------
if __name__ == "__main__":

    # Load data
    X, y = load_adult()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Baseline Model
    # -----------------------------
    numeric = X.select_dtypes(include=np.number).columns
    categorical = X.select_dtypes(exclude=np.number).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    baseline_model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    baseline_model.fit(X_train, y_train)
    base_acc, base_f1, base_auc = evaluate(baseline_model, X_test, y_test)

    # ✅ SAVE BASELINE MODEL
    joblib.dump(baseline_model, MODELS_DIR / "baseline_adult.pkl")

    # -----------------------------
    # Agents
    # -----------------------------
    t_agent = TransformationAgent()
    i_agent = InteractionAgent()
    coordinator = CoordinatorAgent(max_total_features=60)

    # Feature proposals
    t_feats_train = t_agent.propose_features(X_train)
    i_feats_train = i_agent.propose_features(X_train)

    selected = coordinator.select({
        "TransformationAgent": t_feats_train,
        "InteractionAgent": i_feats_train
    })

    # Apply selected features
    X_train_aug = X_train.reset_index(drop=True)
    X_test_aug = X_test.reset_index(drop=True)

    for agent_name, feats in selected.items():
        X_train_aug = pd.concat(
            [X_train_aug, feats.reset_index(drop=True)], axis=1
        )

        if agent_name == "TransformationAgent":
            test_feats = t_agent.propose_features(X_test)
        else:
            test_feats = i_agent.propose_features(X_test)

        X_test_aug = pd.concat(
            [X_test_aug, test_feats.reset_index(drop=True)], axis=1
        )

    # -----------------------------
    # Retrain with Augmented Features
    # -----------------------------
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

    # ✅ SAVE FINAL MAFE MODEL
    joblib.dump(model_aug, MODELS_DIR / "mafe_adult.pkl")

    # -----------------------------
    # Results
    # -----------------------------
    print("\n=== MAFE Round-2 (Multi-Agent) ===")
    print(f"Baseline   -> acc={base_acc:.4f}, f1={base_f1:.4f}, auc={base_auc:.4f}")
    print(f"With Agents-> acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")

    out = pd.DataFrame([{
        "dataset": "Adult",
        "round": 2,
        "agent": "Transformation+Interaction",
        "features_added": sum(df.shape[1] for df in selected.values()),
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    }])

    output_path = RESULTS_DIR / "mafe_runs.csv"

    if output_path.exists():
        out.to_csv(output_path, mode="a", header=False, index=False)
    else:
        out.to_csv(output_path, index=False)