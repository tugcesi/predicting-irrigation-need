"""
Modeli eğitip kaydetmek için bu scripti çalıştır:
    python train_model.py

Çıktılar:
    irrigation_model.joblib
    scaler.joblib           (üzerine yazar)
    feature_names.joblib    (üzerine yazar)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─── Veri Yükleme ─────────────────────────────────────────────────────────────
TRAIN_PATH = "train.csv"
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"'{TRAIN_PATH}' bulunamadı. Kaggle'dan indirip bu klasöre koy."
    )

train = pd.read_csv(TRAIN_PATH)
print(f"Veri boyutu: {train.shape}")
print(f"Hedef dağılımı:\n{train['Irrigation_Need'].value_counts()}\n")

# ─── Özellik Mühendisliği ─────────────────────────────────────────────────────
FEATURE_COLS = [
    "Soil_Type", "Soil_pH", "Soil_Moisture", "Organic_Carbon",
    "Electrical_Conductivity", "Temperature_C", "Humidity",
    "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
    "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type",
    "Water_Source", "Field_Area_hectare", "Mulching_Used",
    "Previous_Irrigation_mm", "Region"
]

X = train[FEATURE_COLS]
y = train["Irrigation_Need"]

# One-hot encoding
X_encoded = pd.get_dummies(X)
feature_names = X_encoded.columns.tolist()

# ─── Train / Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Scaler ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── Model Karşılaştırması ────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

best_name, best_model, best_acc = None, None, 0.0

for name, m in models.items():
    m.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, m.predict(X_test_sc))
    print(f"  {name:25s} → Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc, best_name, best_model = acc, name, m

print(f"\n✅ En iyi model: {best_name}  (Accuracy: {best_acc:.4f})")
print(classification_report(y_test, best_model.predict(X_test_sc)))

# ─── Kaydet ───────────────────────────────────────────────────────────────────
joblib.dump(best_model,   "irrigation_model.joblib")
joblib.dump(scaler,       "scaler.joblib")
joblib.dump(feature_names, "feature_names.joblib")

print("💾 Kaydedilen dosyalar:")
print("   irrigation_model.joblib")
print("   scaler.joblib")
print("   feature_names.joblib")