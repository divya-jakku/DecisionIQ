"""
DecisionIQ - Flask Alternative Backend (simpler option)
Same logic as main.py but using Flask instead of FastAPI.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json, numpy as np
from typing import Optional

app = Flask(__name__)
CORS(app)

MODEL    = joblib.load("ml/model.pkl")
FEATURES = ["time_pressure","stress_level","confidence_level",
            "experience_level","info_completeness","num_alternatives"]
LABEL_MAP = {0:"Low", 1:"Medium", 2:"High"}

with open("ml/feature_importance.json") as f:
    GLOBAL_IMPORTANCE = json.load(f)
with open("ml/archetypes.json") as f:
    ARCHETYPES = json.load(f)

SESSION_HISTORY = []

def compute_risk_score(d):
    return round(
        d["time_pressure"]*0.22 + d["stress_level"]*0.20 +
        (10-d["confidence_level"])*0.18 + (10-d["experience_level"])*0.18 +
        (100-d["info_completeness"])/10*0.14 + (8-d["num_alternatives"])/7*0.08, 2
    )

@app.route("/predict", methods=["POST"])
def predict():
    d = request.get_json()
    for f in FEATURES:
        if f not in d:
            return jsonify({"error": f"Missing: {f}"}), 400
    
    X = np.array([[d[f] for f in FEATURES]])
    pred_class = MODEL.predict(X)[0]
    pred_proba = MODEL.predict_proba(X)[0]
    risk_label  = LABEL_MAP[pred_class]
    risk_score  = compute_risk_score(d)
    success_prob = round(float(pred_proba[0]), 3)

    raw = {}
    for feat in FEATURES:
        val = d[feat]
        w = GLOBAL_IMPORTANCE[feat]
        if feat in ["confidence_level","experience_level"]:
            raw[feat] = (10-val)/10*w
        elif feat == "info_completeness":
            raw[feat] = (100-val)/100*w
        elif feat == "num_alternatives":
            raw[feat] = (8-val)/7*w
        else:
            raw[feat] = val/10*w
        raw[feat] = max(raw[feat], 0)
    
    total = sum(raw.values()) or 1
    contributions = {k: round(v/total, 4) for k,v in raw.items()}
    top_factors = sorted(contributions, key=lambda k: -contributions[k])[:3]
    
    # Simple text explanation
    labels_text = {"time_pressure":"time pressure", "stress_level":"stress",
                   "confidence_level":"confidence", "experience_level":"experience",
                   "info_completeness":"information coverage", "num_alternatives":"alternatives"}
    
    top_label = labels_text[top_factors[0]]
    verdict = {
        "High":   f"High-risk decision. Primary driver: {top_label} ({contributions[top_factors[0]]*100:.0f}% of risk). Immediate action needed.",
        "Medium": f"Moderate risk. {top_label.title()} is your main concern. Addressable with targeted improvements.",
        "Low":    f"Low-risk conditions. {top_label.title()} is the minor residual factor to watch."
    }[risk_label]
    
    archetype_comparison = {
        k: {"label":v["label"], "emoji":v["emoji"], "risk_score":v["risk_score"],
            "risk_label":v["risk_label"], "delta": round(risk_score - v["risk_score"], 2)}
        for k, v in ARCHETYPES.items()
    }
    
    SESSION_HISTORY.append({"risk_label":risk_label, "risk_score":risk_score})
    
    return jsonify({
        "risk_label": risk_label,
        "risk_score": risk_score,
        "success_probability": success_prob,
        "feature_contributions": contributions,
        "verdict": verdict,
        "top_risk_factors": top_factors,
        "archetype_comparison": archetype_comparison
    })

@app.route("/simulate", methods=["POST"])
def simulate():
    d = request.get_json()
    X = np.array([[d[f] for f in FEATURES]])
    pred_class = MODEL.predict(X)[0]
    pred_proba = MODEL.predict_proba(X)[0]
    return jsonify({
        "risk_label": LABEL_MAP[pred_class],
        "risk_score": compute_risk_score(d),
        "success_probability": round(float(pred_proba[0]), 3)
    })

@app.route("/archetypes")
def get_archetypes():
    return jsonify(ARCHETYPES)

@app.route("/history")
def get_history():
    return jsonify({"count": len(SESSION_HISTORY), "history": SESSION_HISTORY})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
