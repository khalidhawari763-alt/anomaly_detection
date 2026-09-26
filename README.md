# Machine Sensor Anomaly Detection using Isolation Forest

An unsupervised machine learning project that uses **Isolation Forest** to detect abnormal machine sensor behavior and investigate its relationship with upcoming machine failures.

## 🎯 Objective

The goal of this project is to determine whether unusual patterns in machine sensor readings can be detected without using failure labels during model training.

The project uses **normal operating data only for training** and evaluates the model against real failure labels.

---

## 📊 Dataset

The dataset contains **52,560 machine sensor records**.

### Sensor Features

The Isolation Forest model uses five sensor measurements:

| Feature | Description |
|---|---|
| `vibration_mm_s` | Machine vibration |
| `bearing_temp_c` | Bearing temperature |
| `motor_current_a` | Motor current |
| `discharge_pressure_bar` | Discharge pressure |
| `rotor_speed_rpm` | Rotor speed |

The dataset also contains information such as:

- Asset ID
- Timestamp
- Operating mode
- Degradation stage
- Remaining Useful Life (RUL)
- Failure within the next 24 hours
- Failure mode

The `failure_next_24h` label is used **only for evaluation** and is not provided to the model during training.

---

## 🔧 Data Preprocessing

### Missing Values

Missing sensor readings were handled using median imputation.

Initial missing values:

```text
vibration_mm_s             896
bearing_temp_c             810
motor_current_a            720
discharge_pressure_bar     570
rotor_speed_rpm            745
```

After preprocessing:

```text
Missing values: 0
```

### Duplicate Sensor Rows

There were **326 duplicated rows across the selected sensor features**.

These rows were retained because repeated sensor readings can represent legitimate repeated machine operating conditions.

### Feature Scaling

The five sensor features were standardized using `StandardScaler`.

---

## 🧪 Train / Validation / Test Strategy

Because Isolation Forest is being used for anomaly detection, the model is trained using **normal machine behavior only**.

```text
Training → Normal data only

Validation → Normal + Failure

Test → Normal + Failure
```

The training data does not use `failure_next_24h`.

This setup simulates a realistic anomaly detection scenario where the model learns normal machine behavior and then identifies unusual observations.

---

# 🌲 Isolation Forest

Isolation Forest detects anomalies by randomly partitioning the feature space.

The key idea is:

> Anomalous observations are usually easier to isolate than normal observations.

The model assigns an anomaly score to each observation and classifies observations as either:

```text
Normal
   or
Anomaly
```

---

## ⚙️ Hyperparameter Tuning

Several Isolation Forest parameters were evaluated using the validation set.

### Contamination

Tested:

```text
0.01
0.02
0.03
0.05
0.08
0.10
0.15
0.20
```

Best validation F1:

```text
contamination = 0.03
F1 = 0.2129
```

### Maximum Samples

Tested:

```text
256
512
1024
2048
4096
```

Best:

```text
max_samples = 2048
```

### Number of Trees

Tested:

```text
50
100
200
300
500
```

Best:

```text
n_estimators = 500
```

### Maximum Features

Tested:

```text
0.5
0.7
1.0
```

Best:

```text
max_features = 1.0
```

---

## 🏆 Final Model

The final Isolation Forest configuration was:

```python
IsolationForest(
    n_estimators=500,
    contamination=0.03,
    max_samples=2048,
    max_features=1.0,
    random_state=42
)
```

---

# 📈 Validation Results

The best configuration achieved:

| Metric | Score |
|---|---:|
| Precision | **16.00%** |
| Recall | **34.92%** |
| F1 Score | **21.95%** |

These results were obtained on the validation set and were used to select the final model configuration.

---

# 🧪 Final Test Results

After selecting the model using the validation set, the final model was evaluated once on the locked test set.

| Metric | Score |
|---|---:|
| Precision | **14.17%** |
| Recall | **24.31%** |
| F1 Score | **17.90%** |
| Accuracy | **96%** |

### Classification Report

```text
              precision    recall  f1-score   support

      Normal       0.99      0.97      0.98      7740
     Failure       0.14      0.24      0.18       144

    accuracy                           0.96      7884
```

### Confusion Matrix

```text
[[7528  212]
 [ 109   35]]
```

The model detected:

**35 out of 144 actual upcoming failures.**

---

# 🔍 Results Interpretation

The model achieved high overall accuracy, but accuracy is not the main metric for this problem because the failure class is highly imbalanced.

The more important metrics are:

- **Precision:** 14.17%
- **Recall:** 24.31%
- **F1:** 17.90%

The model successfully identified some upcoming failures, but it also produced false alarms and missed a significant number of failure cases.

This highlights an important characteristic of anomaly detection:

> An unusual sensor pattern does not necessarily mean that the machine will fail within the next 24 hours.

Isolation Forest detects **unusual behavior**, while `failure_next_24h` represents a specific future event.

---

# 🔎 Sensor Analysis

Analysis of detected anomalous observations showed several notable differences compared with normal observations.

The strongest differences included:

- Higher vibration
- Lower rotor speed
- Lower discharge pressure
- Lower motor current

This suggests that abnormal behavior can involve **combinations of multiple sensor readings**, rather than a single sensor being responsible for an anomaly.

---

# ⚠️ Limitations

### 1. Random Data Split

The current experiment uses a random train/validation/test split.

Because machine sensor data is time-dependent, a future version could use a **chronological split** to better simulate real-world deployment.

### 2. Limited Features

Only five sensor measurements were used.

Additional information such as operating mode, historical trends, and time-based features could potentially improve anomaly detection.

### 3. Anomaly vs Failure

An anomaly does not necessarily represent a failure.

The model is designed to identify unusual machine behavior, while the evaluation target represents failure within the next 24 hours.

---

# 🚀 Future Improvements

Possible future improvements include:

- Chronological train/test splitting
- Time-window features
- Sensor trend analysis
- Additional machine features
- More advanced anomaly detection methods
- Deep Learning Autoencoders
- LSTM-based time-series models
- Real-time anomaly monitoring

---

# 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- PyCharm
- Git & GitHub

---

# 📚 Key Learning Outcomes

Through this project, I learned how to:

- Prepare real-world sensor data
- Handle missing values
- Detect duplicate observations
- Scale numerical features
- Train an unsupervised anomaly detection model
- Train using normal data only
- Tune Isolation Forest hyperparameters
- Evaluate highly imbalanced data
- Interpret precision, recall, and F1
- Analyze detected anomalies
- Understand the difference between anomaly detection and failure prediction

---

## 👤 Author

**Khalid Hawari**

Robotics Student

Focused on:

**Machine Learning • Deep Learning • Robotics • Computer Vision • Autonomous Systems**