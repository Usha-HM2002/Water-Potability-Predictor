# model_utils.py
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np

DATA_FILE = "water_potability.csv"
MODEL_PATH = "best_model.joblib"
TREE_FALLBACK_PATH = "tree_fallback_model.joblib"
METRICS_CSV = "metrics.csv"

# canonical feature ordering used across the app
FEATURES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']


# -----------------------------
# Load and preprocess data
# -----------------------------
def load_and_prepare_data(file_path=DATA_FILE):
    """
    Load CSV and drop NaNs. Returns DataFrame or None.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.dropna().reset_index(drop=True)
        # ensure expected columns present
        missing = [c for c in FEATURES + ['Potability'] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")
        return df
    except Exception as e:
        print(f"[model_utils] Error loading data: {e}")
        return None


# -----------------------------
# Train multiple models
# -----------------------------
def train_multiple_models(data):
    """
    Train RandomForest, GradientBoosting, LogisticRegression, VotingClassifier.
    Save best model to disk and also a tree-based fallback for permutation importance.
    Returns (best_model, metrics_df, feature_list)
    """
    X = data[FEATURES]
    y = data["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42)
    }

    metrics = []
    best_model = None
    best_score = -1
    tree_fallback_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        metrics.append({"Model": name, "Accuracy": acc})

        if acc > best_score:
            best_model = model
            best_score = acc

        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
            # keep the best-performing tree model as fallback
            if tree_fallback_model is None:
                tree_fallback_model = model
            else:
                prev_acc = accuracy_score(y_test, tree_fallback_model.predict(X_test))
                if acc > prev_acc:
                    tree_fallback_model = model

    # Voting ensemble
    voting = VotingClassifier(
        estimators=[("rf", models["RandomForest"]),
                    ("gb", models["GradientBoosting"]),
                    ("lr", models["LogisticRegression"])],
        voting="soft"
    )
    voting.fit(X_train, y_train)
    voting_acc = accuracy_score(y_test, voting.predict(X_test))
    metrics.append({"Model": "VotingClassifier", "Accuracy": voting_acc})

    if voting_acc > best_score:
        best_model = voting
        best_score = voting_acc

    # Persist models & metrics
    try:
        joblib.dump(best_model, MODEL_PATH)
        if tree_fallback_model:
            joblib.dump(tree_fallback_model, TREE_FALLBACK_PATH)
    except Exception as e:
        print(f"[model_utils] Failed to save models: {e}")

    metrics_df = pd.DataFrame(metrics).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    try:
        metrics_df.to_csv(METRICS_CSV, index=False)
    except Exception:
        pass

    return best_model, metrics_df, FEATURES


# -----------------------------
# Load models and metrics
# -----------------------------
def load_model_and_metrics():
    """
    Returns (model, features_list, metrics_df, tree_model)
    """
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

    try:
        tree_model = joblib.load(TREE_FALLBACK_PATH)
    except Exception:
        tree_model = None

    try:
        metrics_df = pd.read_csv(METRICS_CSV)
    except Exception:
        metrics_df = None

    return model, FEATURES, metrics_df, tree_model


# -----------------------------
# Permutation importance plot
# -----------------------------
def generate_permutation_importance(model, data, n_repeats=10, random_state=42):
    """
    Generate a bar chart (base64 PNG) of permutation importances.
    Returns base64 PNG string or None on failure.
    """
    try:
        if model is None:
            return None

        X = data[FEATURES]
        y = data["Potability"]

        # compute permutation importance on a small validation subset for speed
        X_sample = X.sample(min(200, len(X)), random_state=random_state)
        y_sample = y.loc[X_sample.index]

        r = permutation_importance(model, X_sample, y_sample,
                                   n_repeats=n_repeats, random_state=random_state, n_jobs=1)

        importances = r.importances_mean
        indices = importances.argsort()[::-1]
        labels = [FEATURES[i] for i in indices]
        values = importances[indices]

        # plot horizontal bar
        plt.figure(figsize=(8, 5))
        y_pos = np.arange(len(labels))
        plt.barh(y_pos, values[::-1], align='center')
        plt.yticks(y_pos, labels[::-1])
        plt.xlabel("Permutation Importance (mean)")
        plt.title("Feature Importance (Permutation)")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close()
        return img_b64
    except Exception as e:
        print(f"[model_utils] generate_permutation_importance failed: {e}")
        try:
            plt.close()
        except:
            pass
        return None


# -----------------------------
# Data viz helper
# -----------------------------
def create_data_viz_charts(data):
    """
    Generate distribution charts for ph, Hardness, Solids.
    Returns Base64 PNGs for embedding in HTML.
    """
    try:
        charts = {}
        features = {
            "ph": "pH Distribution",
            "Hardness": "Hardness Distribution",
            "Solids": "Solids Distribution"
        }

        for col, title in features.items():
            plt.figure(figsize=(6, 4))
            plt.hist(data[col].dropna(), bins=30, color="skyblue", edgecolor="black")
            plt.title(title)
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            charts[col] = base64.b64encode(buf.read()).decode()
            plt.close()

        return charts["ph"], charts["Hardness"], charts["Solids"]

    except Exception as e:
        print(f"[model_utils] create_data_viz_charts failed: {e}")
        return None, None, None



# -----------------------------
# Simple textual explanation (no SHAP)
# -----------------------------
def explain_prediction_alt(model, sample_df):
    """
    Produce a simple human-readable explanation for the prediction:
    - If model has feature_importances_: use top 3 features influencing prediction
    - If logistic regression: use coefficients
    - Otherwise: return a generic message.
    sample_df = single-row DataFrame with FEATURES columns
    """
    try:
        x = sample_df.iloc[0]
        medians = None
        # load dataset to compute medians for context (non-critical)
        try:
            data = load_and_prepare_data()
            if data is not None:
                medians = data[FEATURES].median()
        except:
            medians = None

        # Tree-based feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:3]
            parts = []
            for idx in top_idx:
                fname = FEATURES[idx]
                importance = importances[idx]
                val = x[fname]
                context = ""
                if medians is not None:
                    context = f" (median {medians[fname]:.2f})"
                parts.append(f"{fname}={val:.2f}{context} — importance {importance:.3f}")
            return "Top influences: " + "; ".join(parts)

        # Logistic regression: use coefficients
        if hasattr(model, "coef_"):
            coefs = model.coef_[0]
            top_idx = np.argsort(np.abs(coefs))[::-1][:3]
            parts = []
            for idx in top_idx:
                fname = FEATURES[idx]
                coef = coefs[idx]
                val = x[fname]
                influence = "increases" if coef > 0 else "decreases"
                parts.append(f"{fname}={val:.2f} — coef {coef:.3f} ({influence} potability)")
            return "Primary signals: " + "; ".join(parts)

        # VotingClassifier or unknown model: fallback
        return "No detailed explanation available for this model type. Inspect feature importance on the Dashboard."

    except Exception as e:
        print(f"[model_utils] explain_prediction_alt failed: {e}")
        return "Explanation generation failed."
