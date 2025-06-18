import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# 🔧 CONFIGURATION
FEATURE_VECTOR_FILE = "datasets/NARROWBAND_ZARR/feature_vectors.json"
SCALER_FILE = "datasets/NARROWBAND_ZARR/metadata_scaler.pkl"
SCALED_OUTPUT_FILE = "datasets/NARROWBAND_ZARR/X_scaled.npy"  # optional: save for direct dataset loading

# 🔧 1️⃣ Load your extracted feature vectors
with open(FEATURE_VECTOR_FILE, "r") as f:
    feature_vectors = json.load(f)

# 🔧 2️⃣ Extract the 3 features into NumPy array
X = np.array([
    [fv["center_freq"], fv["bandwidth"], fv["duration"]] 
    for fv in feature_vectors
], dtype=np.float32)

print(f"Loaded {X.shape[0]} feature vectors with shape {X.shape}")

# 🔧 3️⃣ Fit StandardScaler
scaler = StandardScaler()
scaler.fit(X)

# 🔧 4️⃣ Transform features
X_scaled = scaler.transform(X)

# 🔧 5️⃣ Save StandardScaler for deployment/inference
joblib.dump(scaler, SCALER_FILE)
print(f"Scaler saved to: {SCALER_FILE}")

# 🔧 6️⃣ (Optional) Save scaled features to file for training
np.save(SCALED_OUTPUT_FILE, X_scaled)
print(f"Scaled features saved to: {SCALED_OUTPUT_FILE}")

print("✅ StandardScaler fitting complete.")
