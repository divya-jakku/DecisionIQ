"""
DecisionIQ - Synthetic Dataset Generator
Generates 1200 rows with realistic noise, correlations, and edge cases.
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1200

# --- Core features (with realistic correlations) ---
# High stress often goes with high time pressure in real life
time_pressure = np.random.randint(1, 11, n)
stress_level  = np.clip(
    time_pressure + np.random.randint(-3, 4, n), 1, 10
).astype(int)

# Experienced people tend to be slightly more confident
experience_level  = np.random.randint(1, 11, n)
confidence_level  = np.clip(
    experience_level + np.random.randint(-4, 5, n), 1, 10
).astype(int)

info_completeness = np.random.randint(10, 101, n)
num_alternatives  = np.random.randint(1, 8, n)

# --- Derived feature: decision_type (0=personal, 1=academic, 2=professional) ---
decision_type = np.random.choice([0, 1, 2], n, p=[0.35, 0.30, 0.35])

# --- Reversals (real-world anomalies that make dataset "dirty" but realistic) ---
# Some high-confidence low-experience people (overconfidence bias)
overconfident_mask = (experience_level <= 3) & (confidence_level >= 8)
# Some calm people under high time pressure (experienced professionals)
calm_expert_mask   = (time_pressure >= 8) & (stress_level <= 3) & (experience_level >= 8)

# --- Risk score formula ---
# Core risk drivers
risk_score = (
    time_pressure    * 0.22 +
    stress_level     * 0.20 +
    (10 - confidence_level) * 0.18 +
    (10 - experience_level) * 0.18 +
    (100 - info_completeness) / 10 * 0.14 +
    (8 - num_alternatives) / 7 * 0.08   # fewer alternatives = more risk
)

# Overconfidence penalty: overconfident but inexperienced → higher actual risk
risk_score[overconfident_mask] += np.random.uniform(0.5, 1.5, overconfident_mask.sum())

# Professional decisions slightly higher stakes
risk_score[decision_type == 2] += np.random.uniform(0, 0.4, (decision_type == 2).sum())

# Realistic noise (people are unpredictable)
risk_score += np.random.normal(0, 0.45, n)
risk_score  = np.clip(risk_score, 0, 10).round(2)

# Success probability (inverse of risk, with noise)
success_prob = np.clip(
    1.0 - (risk_score / 10) + np.random.normal(0, 0.05, n),
    0.05, 0.98
).round(3)

# --- Labels ---
# 0=Low, 1=Medium, 2=High — with slightly fuzzy boundaries for realism
def classify(score, noise):
    boundary_low  = 3.5 + noise
    boundary_high = 6.5 + noise
    if score < boundary_low:  return 0
    elif score < boundary_high: return 1
    else: return 2

boundary_noise = np.random.normal(0, 0.2, n)
labels = np.array([classify(s, boundary_noise[i]) for i, s in enumerate(risk_score)])

decision_type_names = {0: "personal", 1: "academic", 2: "professional"}

df = pd.DataFrame({
    "time_pressure":     time_pressure,
    "stress_level":      stress_level,
    "confidence_level":  confidence_level,
    "experience_level":  experience_level,
    "info_completeness": info_completeness,
    "num_alternatives":  num_alternatives,
    "decision_type":     decision_type,
    "risk_score":        risk_score,
    "success_prob":      success_prob,
    "risk_label":        labels
})

df.to_csv("ml/decision_dataset.csv", index=False)
print(f"Dataset saved! Shape: {df.shape}")
print("\nRisk label distribution:")
print(df["risk_label"].map({0:"Low", 1:"Medium", 2:"High"}).value_counts())
print(f"\nOverconfidence bias rows: {overconfident_mask.sum()}")
print(f"Calm expert rows: {calm_expert_mask.sum()}")
print(f"\nRisk score stats:\n{df['risk_score'].describe().round(2)}")
