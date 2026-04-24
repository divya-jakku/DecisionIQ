"""
DecisionIQ - ML Training Script
Random Forest with feature importance, SHAP-style explanations, and archetype profiles.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json

# --- Load dataset ---
df = pd.read_csv("ml/decision_dataset.csv")

FEATURES = [
    "time_pressure", "stress_level", "confidence_level",
    "experience_level", "info_completeness", "num_alternatives"
]
X = df[FEATURES]
y = df["risk_label"]

# --- Train / test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model: Random Forest (best for explainability + small data) ---
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print(f"Test Accuracy:   {acc:.2%}")
print(f"CV Accuracy:     {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

# --- Feature importance ---
importance = dict(zip(FEATURES, model.feature_importances_.round(4)))
sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
print("\nFeature Importance:")
for f, v in sorted_imp:
    bar = "█" * int(v * 50)
    print(f"  {f:25s} {bar} {v:.3f}")

# --- Save model + metadata ---
joblib.dump(model, "ml/model.pkl")

# Save feature importance for use in API
with open("ml/feature_importance.json", "w") as f:
    json.dump(importance, f)

# --- Archetype profiles (for comparison in frontend) ---
# These represent "typical" decision-makers your user can compare against
archetypes = {
    "expert": {
        "label": "Expert Decision-Maker",
        "emoji": "🎯",
        "description": "Calm, experienced, well-informed. Makes decisions methodically.",
        "values": {
            "time_pressure": 3, "stress_level": 2, "confidence_level": 9,
            "experience_level": 9, "info_completeness": 88, "num_alternatives": 5
        }
    },
    "average": {
        "label": "Average Decision-Maker",
        "emoji": "🙂",
        "description": "Typical person making an everyday decision under moderate conditions.",
        "values": {
            "time_pressure": 5, "stress_level": 5, "confidence_level": 6,
            "experience_level": 5, "info_completeness": 60, "num_alternatives": 3
        }
    },
    "risky": {
        "label": "Risky Decision-Maker",
        "emoji": "🚨",
        "description": "High pressure, stressed, limited info. Classic recipe for poor outcomes.",
        "values": {
            "time_pressure": 9, "stress_level": 8, "confidence_level": 3,
            "experience_level": 2, "info_completeness": 25, "num_alternatives": 1
        }
    }
}

# Predict risk scores for archetypes
for key, arch in archetypes.items():
    vals = arch["values"]
    X_arch = np.array([[vals[f] for f in FEATURES]])
    pred_class = model.predict(X_arch)[0]
    pred_proba = model.predict_proba(X_arch)[0]
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    
    risk_score = round(
        vals["time_pressure"] * 0.22 + vals["stress_level"] * 0.20 +
        (10 - vals["confidence_level"]) * 0.18 + (10 - vals["experience_level"]) * 0.18 +
        (100 - vals["info_completeness"]) / 10 * 0.14 +
        (8 - vals["num_alternatives"]) / 7 * 0.08, 2
    )
    archetypes[key]["risk_score"] = risk_score
    archetypes[key]["risk_label"] = label_map[pred_class]
    archetypes[key]["success_prob"] = round(float(pred_proba[0]), 3)

with open("ml/archetypes.json", "w") as f:
    json.dump(archetypes, f, indent=2)

print(f"\nArchetype risk scores:")
for key, arch in archetypes.items():
    print(f"  {arch['label']:30s} → {arch['risk_label']:6s} ({arch['risk_score']:.1f}/10)")

print("\nModel + metadata saved!")
