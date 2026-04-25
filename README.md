DecisionIQ — AI Decision Risk Predictor

DecisionIQ is a web-based system that evaluates the **risk of a decision before it is made** by analyzing decision-making conditions such as stress, confidence, and available information.

Features

- Risk Score (0–10)
- Risk Level (Low / Medium / High)
- Success Probability
- Key Risk Factor Analysis
- Cognitive Bias Detection
- Archetype Comparison (You vs Expert)
- Actionable Suggestions

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** FastAPI (Python)  
- **ML Model:** Random Forest (scikit-learn)  
- **Data:** Synthetic dataset (behavior-based)

---

## Project Structure
DecisionIQ/
├── backend/
│ └── main.py
├── frontend/
│ └── index.html
├── ml/
│ ├── generate_dataset.py
│ ├── train_model.py
│ ├── model.pkl
│ ├── decision_dataset.csv
│ ├── feature_importance.json
│ └── archetypes.json
└── README.md



##  Setup Instructions

1️⃣ Install Dependencies
pip install fastapi uvicorn scikit-learn pandas numpy joblib
2️⃣ Generate Dataset
python ml/generate_dataset.py
3️⃣ Train Model
python ml/train_model.py
4️⃣ Run Backend
uvicorn backend.main:app --reload
5️⃣ Run Frontend
Open:
frontend/index.html
