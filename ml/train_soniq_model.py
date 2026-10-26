import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "dataset" / "deam_song_level_features.csv"
OUTPUT = BASE / "ml" / "soniq_emotion_model.joblib"

df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()

targets = ["valence_mean", "arousal_mean"]
y = df[targets].apply(pd.to_numeric, errors="coerce")

# Use only audio features SONIQ can calculate reliably.
feature_columns = [
    "bpm",
    "duration",
    "rms_energy",
    "spectral_centroid_hz",
    "zero_crossing_rate",
]

# The DEAM file contains openSMILE features, not these five
# calculated features. Do not train on unrelated substitute columns.
print("The DEAM file does not contain SONIQ's calculated features.")
print("We need per-song audio files to calculate matching training inputs.")
print("No model was trained or overwritten.")
