#  python -m pip install pandas xgboost matplotlib scikit-learn joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import joblib

# === CONFIGURATION ===
CSV_PATH = "training_data.csv"
TARGET_COLUMN = "label_3"
RANDOM_SEED = 42
SAVE_METRICS_TO = "run_metrics.json"
SHAP_SAMPLE_SIZE = 1000
EXPERIMENT_NAME = "hint_lover_xgb"

print("🔍 Loading and preprocessing data...")
# === LOAD AND CLEAN DATA ===
df = pd.read_csv(CSV_PATH)
df.columns = [col.lower() for col in df.columns]
df.fillna(0, inplace=True)

# Split features and labels
NON_FEATURE_COLUMNS = ["event_id", "user_id", "derived_tstamp", "has_managed_hint_next_1_days", "has_managed_hint_next_3_days", "has_managed_hint_next_7_days", "label_1", "label_3", "label_7"]
feature_columns = [col for col in df.columns if col not in NON_FEATURE_COLUMNS and df[col].dtype != "object"]

X = df[feature_columns]
y = df[TARGET_COLUMN]

print(f"✅ Loaded {len(df)} rows with {len(feature_columns)} features")

# === SPLIT INTO TRAIN/VAL/TEST ===
print("📊 Splitting data into train/validation/test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp)

print(f"📁 Training set: {len(X_train)} rows")
print(f"📁 Validation set: {len(X_val)} rows")
print(f"📁 Test set: {len(X_test)} rows")


# === TRAIN MODEL ===
print("🚀 Training XGBoost model...")
model = XGBClassifier(
  n_estimators=200,
  max_depth=4,
  learning_rate=0.1,
  subsample=0.8,
  colsample_bytree=0.8,
  random_state=RANDOM_SEED,
  eval_metric="logloss",
  use_label_encoder=False,
  objective="binary:logistic"
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)
print("✅ Model training complete.")

print("📈 Evaluating model on validation set...")
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val, y_prob)
cm = confusion_matrix(y_val, y_pred).tolist()
cr = classification_report(y_val, y_pred, output_dict=True)

# Log metrics
print("roc_auc", roc_auc)
print("confusion_matrix", cm)
print('classification_report', cr)
print("f1_class_0", cr["0"]["f1-score"])
print("f1_class_1", cr["1"]["f1-score"])
print("accuracy", cr["accuracy"])

# Save confusion matrix as artifact
plt.figure()
plt.imshow(confusion_matrix(y_val, y_pred), cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# Save and log model
model_path = "hint_lover_model.pkl"
joblib.dump(model, model_path)

input_example = X_val.sample(1, random_state=RANDOM_SEED)
signature = infer_signature(X_val, model.predict_proba(X_val))


# Log feature importances
fig, ax = plt.subplots()
ax.hist(model.feature_importances_, bins=50)
ax.axvline(0.002, color='green', linestyle='--', label="0.002")
ax.axvline(0.005, color='red', linestyle='--', label="0.005")
ax.set_title("Feature Importance Distribution")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature Count")
ax.legend()
fig.tight_layout()
fig.savefig("feature_importance.png")

# Save ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.grid()
plt.tight_layout()
plt.savefig("roc_curve.png")

# Save PR Curve
precision, recall, _ = precision_recall_curve(y_val, y_prob)
ap_score = average_precision_score(y_val, y_prob)
plt.figure()
plt.plot(recall, precision, label=f"AP = {ap_score:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.tight_layout()
plt.savefig("pr_curve.png")

