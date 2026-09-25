import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("sensor_readings.csv")


# ============================================================
# 2. SENSOR FEATURES
# ============================================================

feature = [
    "vibration_mm_s",
    "bearing_temp_c",
    "motor_current_a",
    "discharge_pressure_bar",
    "rotor_speed_rpm"
]


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

df[feature] = df[feature].fillna(
    df[feature].median()
)


# ============================================================
# 4. TRAIN / CV / TEST SPLIT
# ============================================================

df_train, df_temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

df_cv, df_test = train_test_split(
    df_temp,
    test_size=0.50,
    random_state=42
)


# ============================================================
# 5. TRAINING DATA = NORMAL ONLY
# ============================================================

df_train_normal = df_train[
    df_train["failure_next_24h"] == 0
].copy()


# ============================================================
# 6. CREATE FEATURES
# ============================================================

X_train = df_train_normal[feature]

X_cv = df_cv[feature]

X_test = df_test[feature]


# ============================================================
# 7. SCALE
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_cv_scaled = scaler.transform(X_cv)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 8. LABELS
# ============================================================

y_train = df_train_normal["failure_next_24h"].astype(int)

y_cv = df_cv["failure_next_24h"].astype(int)

y_test = df_test["failure_next_24h"].astype(int)


# ============================================================
# 9. CHECK DATA
# ============================================================

print("========================================")
print("DATASET INFORMATION")
print("========================================")

print("Original train:", df_train.shape)
print("Normal train:", df_train_normal.shape)
print("CV:", df_cv.shape)
print("Test:", df_test.shape)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nCV labels:")
print(y_cv.value_counts())

print("\nTest labels:")
print(y_test.value_counts())