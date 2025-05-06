 train_and_save_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("Credit Card Transactions.csv")  # Make sure this file is in the same folder
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply SMOTE
X_train_balanced, y_train_balanced = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_balanced, y_train_balanced)

# Save model
joblib.dump(model, "xgboost_model.pkl")
print("✅ Model saved as xgboost_model.pkl")
