"""Train logistic regression for home win; save model + features list."""
import os, json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss

IN = "src/api_basketball/data/processed/game_features.csv"
OUT_DIR = "src/api_basketball/models/game_outcome"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN)
features = [c for c in df.columns if "rolling" in c]
X = df[features].values
y = df["HOME_WIN"].values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300))]).fit(Xtr, ytr)

auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:,1])
ll  = log_loss(yte, pipe.predict_proba(Xte))
print({"auc": round(auc,3), "log_loss": round(ll,3)})

joblib.dump(pipe, f"{OUT_DIR}/model.pkl")
with open(f"{OUT_DIR}/features.json","w") as f: json.dump({"features": features}, f)
with open(f"{OUT_DIR}/metrics.json","w") as f: json.dump({"auc": float(auc), "log_loss": float(ll)}, f)
print(f"Saved artifacts to {OUT_DIR}")
