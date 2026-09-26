import numpy as np
import xgboost as xgb

np.random.seed(42)
N = 5000

dist_km = np.random.uniform(0.5, 30.0, N)
eta_mins = dist_km * np.random.uniform(3.8, 5.2, N)
beds_free = np.random.randint(0, 25, N)
icu_free = np.random.randint(0, 10, N)
specialty_match = np.random.choice([0, 1], size=N, p=[0.35, 0.65])

# Survival/Success probability target
logits = (
    (specialty_match * 3.5)
    + (beds_free * 0.15)
    + (icu_free * 0.30)
    - (eta_mins * 0.12)
    - (dist_km * 0.10)
    - 1.0
)
probs = 1 / (1 + np.exp(-logits))
labels = (probs > 0.5).astype(int)

X = np.column_stack([dist_km, eta_mins, beds_free, icu_free, specialty_match])
dtrain = xgb.DMatrix(
    X,
    label=labels,
    feature_names=["dist_km", "eta_mins", "beds_free", "icu_free", "specialty_match"]
)

params = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "eta": 0.1,
    "eval_metric": "logloss"
}

bst = xgb.train(params, dtrain, num_boost_round=100)
bst.save_model("er_pulse_xgb.json")
print("✅ Saved er_pulse_xgb.json successfully!")