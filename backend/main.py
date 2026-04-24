"""
DecisionIQ - FastAPI Backend v2.1
Fixed: model_copy → copy() for Pydantic v1 compatibility
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib, numpy as np, json
from typing import Optional

app = FastAPI(title="DecisionIQ API", version="2.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL = joblib.load("ml/model.pkl")
FEATURES = ["time_pressure","stress_level","confidence_level","experience_level","info_completeness","num_alternatives"]
LABEL_MAP = {0:"Low", 1:"Medium", 2:"High"}

with open("ml/feature_importance.json") as f:
    GLOBAL_IMPORTANCE = json.load(f)
with open("ml/archetypes.json") as f:
    ARCHETYPES = json.load(f)

SESSION_HISTORY = []

THRESHOLDS = {
    "time_pressure":    {"low":3,"high":7,"label":"Time Pressure",
        "high_msg":"You're under significant time pressure — rushed decisions skip critical evaluation steps.",
        "low_msg":"You have good time available — use it to gather more information and reflect.",
        "advice":"If possible, request a deadline extension or at least block 30 minutes of uninterrupted thinking time."},
    "stress_level":     {"low":3,"high":7,"label":"Stress Level",
        "high_msg":"High stress narrows your thinking and causes tunnel vision, making it easy to miss alternatives.",
        "low_msg":"Low stress is ideal — your cognitive clarity is good.",
        "advice":"Before deciding, try 5 minutes of deep breathing or a short walk. Research shows this measurably improves decision quality."},
    "confidence_level": {"low":4,"high":8,"label":"Confidence Level",
        "high_msg":"Your confidence looks solid.",
        "low_msg":"Low confidence often signals missing information or skills. Don't decide until you know why you're uncertain.",
        "advice":"Write down the 3 things you're most unsure about. Research or ask an expert on just those points."},
    "experience_level": {"low":4,"high":8,"label":"Experience Level",
        "high_msg":"Your experience level is a strong asset here.",
        "low_msg":"Limited experience means you may not see risks that aren't obvious. This calls for extra caution.",
        "advice":"Find someone who has made a similar decision before and ask them: 'What do you wish you knew?'"},
    "info_completeness":{"low":40,"high":70,"label":"Information Coverage",
        "high_msg":"You have good information coverage.",
        "low_msg":"You're deciding with less than half the picture — this is one of the strongest predictors of poor outcomes.",
        "advice":"Spend 20 minutes finding the 2-3 most critical missing facts. Even partial improvement here can shift your risk category."},
    "num_alternatives": {"low":2,"high":5,"label":"Alternatives Considered",
        "high_msg":"Considering multiple alternatives is great — it reduces anchoring bias.",
        "low_msg":"Considering only 1-2 options is a classic decision trap. You may be anchoring on the first idea.",
        "advice":"Force yourself to name at least one more alternative — even an extreme one. This alone broadens your perspective significantly."}
}

BIAS_RULES = [
    {"name":"Overconfidence Bias","emoji":"🎭",
     "condition":lambda d: d.confidence_level>=8 and d.experience_level<=3,
     "explanation":"You're highly confident but have limited experience — a classic overconfidence pattern. Novices often overestimate judgment accuracy by 30-40%."},
    {"name":"Time Pressure Tunnel Vision","emoji":"⏰",
     "condition":lambda d: d.time_pressure>=8 and d.num_alternatives<=2,
     "explanation":"Extreme time pressure + few alternatives: you're likely fixating on the first option rather than exploring better ones."},
    {"name":"Information Illusion","emoji":"🌫️",
     "condition":lambda d: d.confidence_level>=7 and d.info_completeness<50,
     "explanation":"You feel confident despite having less than 50% of the relevant information — a dangerous combination known as the 'illusion of knowing'."},
    {"name":"Stress-Induced Narrow Framing","emoji":"😤",
     "condition":lambda d: d.stress_level>=8 and d.num_alternatives<=2,
     "explanation":"High stress + few alternatives signals narrow framing — your brain under stress naturally reduces the option space you consider."},
    {"name":"Analysis Paralysis Risk","emoji":"🌀",
     "condition":lambda d: d.num_alternatives>=6 and d.confidence_level<=4,
     "explanation":"Many alternatives with low confidence can lead to analysis paralysis — where more options actually decrease decision quality."}
]

GOAL_RULES = {
    "Career":{
        "High":"Your current conditions (high stress, low information) are not suitable for a career-impacting decision. Career decisions made in this state have significantly lower success rates. Delay if at all possible.",
        "Medium":"Moderate risk for a career decision. These are high-stakes and often irreversible — consider improving your information coverage before committing.",
        "Low":"You're in a good state for a career-level decision. Conditions are favorable — proceed with intention."
    },
    "Academic":{
        "High":"Academic decisions under time pressure without enough experience tend to have poor outcomes. Consider delaying or getting external advice.",
        "Medium":"Moderate risk for an academic decision. Your information coverage is the key lever — spend 20 minutes on the biggest unknown.",
        "Low":"Good conditions for an academic decision. Your preparation level is solid."
    },
    "Personal":{
        "High":"Personal decisions made under high stress are often regretted. Give yourself a minimum 24-hour buffer before committing.",
        "Medium":"Moderate conditions for a personal decision. Reducing stress by even a few points can significantly improve your judgment quality.",
        "Low":"Your personal conditions are healthy for this decision. Proceed thoughtfully."
    },
    "Financial":{
        "High":"Financial decisions under these conditions carry a high risk of loss or regret. High stress, low information, and time pressure are the exact factors that cause financial mistakes. Do not commit today.",
        "Medium":"Moderate risk for a financial decision. Get a second opinion and verify your information before committing.",
        "Low":"Good conditions for a financial decision. Your preparation and calm state are your biggest assets."
    }
}

class DecisionInput(BaseModel):
    time_pressure:     int = Field(..., ge=1, le=10)
    stress_level:      int = Field(..., ge=1, le=10)
    confidence_level:  int = Field(..., ge=1, le=10)
    experience_level:  int = Field(..., ge=1, le=10)
    info_completeness: int = Field(..., ge=10, le=100)
    num_alternatives:  int = Field(..., ge=1, le=7)
    decision_context:  Optional[str] = None
    goal:              Optional[str] = None
    reversible:        Optional[bool] = None

def compute_risk_score(d) -> float:
    return round(
        d.time_pressure*0.22 + d.stress_level*0.20 +
        (10-d.confidence_level)*0.18 + (10-d.experience_level)*0.18 +
        (100-d.info_completeness)/10*0.14 + (8-d.num_alternatives)/7*0.08, 2
    )

def generate_explanation(data, risk_label, risk_score, feature_contributions):
    sorted_features = sorted(feature_contributions.items(), key=lambda x: -x[1])
    top_3 = sorted_features[:3]
    factor_analysis = []
    for feat, contrib in sorted_features:
        val = getattr(data, feat)
        thresh = THRESHOLDS[feat]
        if feat in ["confidence_level","experience_level","info_completeness","num_alternatives"]:
            is_risk = val < thresh["low"]
            is_safe = val >= thresh["high"]
        else:
            is_risk = val > thresh["high"]
            is_safe = val <= thresh["low"]
        status = "danger" if is_risk else ("safe" if is_safe else "neutral")
        message = thresh["low_msg"] if is_risk else thresh["high_msg"]
        factor_analysis.append({
            "feature": feat, "label": thresh["label"], "value": val,
            "contribution": round(contrib*100, 1), "status": status,
            "message": message, "advice": thresh["advice"] if is_risk else None
        })
    top_driver = top_3[0][0].replace("_"," ")
    if risk_label=="High":
        verdict = (f"High-risk decision. Primary driver: {top_driver} "
                   f"({top_3[0][1]*100:.0f}% of risk), followed by "
                   f"{top_3[1][0].replace('_',' ')} and {top_3[2][0].replace('_',' ')}. "
                   f"Risk score {risk_score}/10 — probability of poor outcome is significantly elevated.")
    elif risk_label=="Medium":
        verdict = (f"Moderate risk. Your {top_driver} is the primary concern "
                   f"({top_3[0][1]*100:.0f}% of risk). Clear steps exist to move into the low-risk zone.")
    else:
        verdict = (f"Decision conditions look relatively safe. Low risk driven by healthy "
                   f"{top_driver} and experience. Monitor {top_3[0][0].replace('_',' ')} "
                   f"as it contributes most to any residual risk.")
    detected_biases = [
        {"name":b["name"],"emoji":b["emoji"],"explanation":b["explanation"]}
        for b in BIAS_RULES if b["condition"](data)
    ]
    actions = [fa["advice"] for fa in factor_analysis if fa["advice"]][:3]
    quick_wins = []
    # Stress quick win
    d2 = data.copy(); d2.stress_level = max(1, data.stress_level-3)
    ns = compute_risk_score(d2)
    if ns < risk_score-0.3:
        quick_wins.append({"change":f"Reduce stress {data.stress_level}→{d2.stress_level}","new_score":round(ns,2),"delta":round(risk_score-ns,2)})
    # Info quick win
    d3 = data.copy(); d3.info_completeness = min(100, data.info_completeness+20)
    ns2 = compute_risk_score(d3)
    if ns2 < risk_score-0.2:
        quick_wins.append({"change":f"Gather 20% more info ({data.info_completeness}%→{d3.info_completeness}%)","new_score":round(ns2,2),"delta":round(risk_score-ns2,2)})
    # Alternatives quick win
    d4 = data.copy(); d4.num_alternatives = min(7, data.num_alternatives+2)
    ns3 = compute_risk_score(d4)
    if ns3 < risk_score-0.15:
        quick_wins.append({"change":f"Explore {d4.num_alternatives} alternatives instead of {data.num_alternatives}","new_score":round(ns3,2),"delta":round(risk_score-ns3,2)})
    # Reversibility adjustment
    rev_label = risk_label
    rev_note = None
    if data.reversible is not None:
        if data.reversible and risk_label=="High":
            rev_label = "High (Recoverable)"
            rev_note = "This is high-risk but reversible — a bad outcome is recoverable. Caution still advised."
        elif not data.reversible and risk_label in ["Medium","High"]:
            rev_note = "This decision is irreversible — treat it as one category higher. Extra care is warranted."
    # Goal support
    goal_message = None
    if data.goal and data.goal in GOAL_RULES:
        goal_message = GOAL_RULES[data.goal].get(risk_label, GOAL_RULES[data.goal]["Medium"])
    return {
        "verdict": verdict, "factor_analysis": factor_analysis,
        "detected_biases": detected_biases, "action_plan": actions,
        "quick_wins": quick_wins, "reversibility_note": rev_note,
        "reversibility_label": rev_label, "goal_message": goal_message
    }

@app.get("/")
def root():
    return {"message":"DecisionIQ API v2.1","endpoints":["/predict","/simulate","/archetypes","/history"]}

@app.post("/predict")
def predict(data: DecisionInput):
    X = np.array([[getattr(data,f) for f in FEATURES]])
    pred_class = MODEL.predict(X)[0]
    pred_proba = MODEL.predict_proba(X)[0]
    risk_label = LABEL_MAP[pred_class]
    success_prob = round(float(pred_proba[0]),3)
    risk_score = compute_risk_score(data)
    raw = {}
    for feat in FEATURES:
        val = getattr(data,feat)
        w = GLOBAL_IMPORTANCE[feat]
        if feat in ["confidence_level","experience_level"]: raw[feat]=(10-val)/10*w
        elif feat=="info_completeness": raw[feat]=(100-val)/100*w
        elif feat=="num_alternatives": raw[feat]=(8-val)/7*w
        else: raw[feat]=val/10*w
        raw[feat]=max(raw[feat],0)
    total=sum(raw.values()) or 1
    fc={k:round(v/total,4) for k,v in raw.items()}
    explanation=generate_explanation(data,risk_label,risk_score,fc)
    arch_comp={}
    for key,arch in ARCHETYPES.items():
        arch_comp[key]={
            "label":arch["label"],"emoji":arch["emoji"],
            "description":arch.get("description",""),
            "risk_score":arch["risk_score"],"risk_label":arch["risk_label"],
            "delta":round(risk_score-arch["risk_score"],2)
        }
    SESSION_HISTORY.append({
        "inputs":{f:getattr(data,f) for f in FEATURES},
        "risk_label":risk_label,"risk_score":risk_score,
        "success_probability":success_prob,"context":data.decision_context
    })
    return {
        "risk_label":risk_label,"risk_score":risk_score,
        "success_probability":success_prob,"feature_contributions":fc,
        "global_feature_importance":GLOBAL_IMPORTANCE,
        "explanation":explanation,"archetype_comparison":arch_comp
    }

@app.post("/simulate")
def simulate(data: DecisionInput):
    X = np.array([[getattr(data,f) for f in FEATURES]])
    pred_class=MODEL.predict(X)[0]
    pred_proba=MODEL.predict_proba(X)[0]
    return {
        "risk_label":LABEL_MAP[pred_class],
        "risk_score":compute_risk_score(data),
        "success_probability":round(float(pred_proba[0]),3)
    }

@app.get("/archetypes")
def get_archetypes(): return ARCHETYPES

@app.get("/history")
def get_history():
    return {
        "count":len(SESSION_HISTORY),"history":SESSION_HISTORY,
        "avg_risk_score":round(sum(h["risk_score"] for h in SESSION_HISTORY)/len(SESSION_HISTORY),2) if SESSION_HISTORY else None
    }