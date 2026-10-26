
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "dataset" / "deam_song_level_features.csv"
MODEL_PATH = PROJECT_DIR / "ml" / "emotion_model.joblib"

# Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Select prediction targets
target_columns = ["valence_mean", "arousal_mean"]
y = df[target_columns].apply(pd.to_numeric, errors="coerce")

# Use acoustic features only; exclude IDs and emotion labels
excluded_columns = [
    "song_id",
    "valence_mean",
    "valence_std",
    "arousal_mean",
    "arousal_std",
]

X = df.drop(
    columns=[c for c in excluded_columns if c in df.columns]
)

# Keep numeric acoustic features
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove unusable columns and rows
X = X.dropna(axis=1, how="all")
valid_rows = y.notna().all(axis=1)
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Split before fitting preprocessing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fill missing feature values using training data only
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train model
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
)

print("Training emotion prediction model...")
model.fit(X_train, y_train)

# Evaluate on held-out test data
predictions = model.predict(X_test)

for i, target in enumerate(target_columns):
    mae = mean_absolute_error(y_test.iloc[:, i], predictions[:, i])
    r2 = r2_score(y_test.iloc[:, i], predictions[:, i])
    print(f"{target}: MAE = {mae:.3f}, R² = {r2:.3f}")

# Save everything needed to reproduce predictions
joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "feature_columns": X.columns.tolist(),
        "target_columns": target_columns,
        
    },
    MODEL_PATH,
)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Training songs: {len(X_train)}")
print(f"Testing songs: {len(X_test)}")
print(f"Acoustic features used: {X.shape[1]}")
