from preprocess import *

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. FINAL MODEL
# ============================================================

model = IsolationForest(
    n_estimators=500,
    contamination=0.03,
    max_samples=2048,
    max_features=1.0,
    random_state=42
)


# ============================================================
# 2. TRAIN ONLY ON NORMAL TRAINING DATA
# ============================================================

model.fit(X_train_scaled)


# ============================================================
# 3. PREDICT TEST
# ============================================================

test_pred_raw = model.predict(X_test_scaled)

# -1 = anomaly
#  1 = normal

y_pred = (test_pred_raw == -1).astype(int)


# ============================================================
# 4. TEST METRICS
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# 5. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL ISOLATION FOREST TEST RESULTS")
print("=" * 60)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Failure"],
        zero_division=0
    )
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 8. PREDICTION COUNTS
# ============================================================

print("\n")
print("=" * 60)
print("PREDICTION COUNTS")
print("=" * 60)

print("Normal predictions:", (y_pred == 0).sum())
print("Anomaly predictions:", (y_pred == 1).sum())