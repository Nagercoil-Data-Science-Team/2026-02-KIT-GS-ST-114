# =====================================================
# IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, confusion_matrix,
                             ConfusionMatrixDisplay,
                             roc_curve,
                             precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator   # ⭐ IMPORTANT FIX

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv('PE_teaching_dataset_categorical.csv')

target_column = "teaching_effectiveness_category"

y = df[target_column]
X = df.drop(columns=[target_column])

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

# =====================================================
# PREPROCESSING
# =====================================================

for col in numerical_cols:
    X[col] = X[col].fillna(X[col].mean())

for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].fillna(X[col].mode()[0]))

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

timestamp_cols = [c for c in X.columns if 'time' in c.lower()]
for col in timestamp_cols:
    X[col] = pd.to_datetime(X[col], errors='coerce')
    X[col] = X[col].astype('int64') // 10**9

X_scaled = StandardScaler().fit_transform(X)

# =====================================================
# SMOTE
# =====================================================

X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_scaled, y)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    stratify=y_resampled,
    random_state=42
)

# =====================================================
# DEEP LEARNING MODEL
# =====================================================

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dense(len(np.unique(y_resampled)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
    verbose=1
)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# =====================================================
# BASIC PLOTS
# =====================================================

plt.figure(figsize=(8,6))
plt.bar(['Accuracy','Precision','Recall'], [accuracy, precision, recall])
plt.title("Performance Metrics")
plt.show()

plt.figure(figsize=(8,6))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.legend(['Train','Validation'])
plt.show()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.legend(['Train','Validation'])
plt.show()

# =====================================================
# ROC
# =====================================================

classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

plt.figure(figsize=(8,6))
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f'Class {classes[i]}')
plt.legend()
plt.show()

# =====================================================
# PRECISION RECALL
# =====================================================

plt.figure(figsize=(8,6))
for i in range(len(classes)):
    p, r, _ = precision_recall_curve(y_test_bin[:, i], y_pred_prob[:, i])
    plt.plot(r, p, label=f'Class {classes[i]}')
plt.legend()
plt.show()

# =====================================================
# CALIBRATION
# =====================================================

prob_true, prob_pred = calibration_curve(
    y_test_bin.ravel(),
    y_pred_prob.ravel(),
    n_bins=10
)

plt.figure(figsize=(8,6))
plt.plot(prob_pred, prob_true, marker='o')
plt.title("Calibration Curve")
plt.show()

# =====================================================
# FEATURE IMPORTANCE (FINAL FIX)
# =====================================================

print("\nCalculating Feature Importance...")

# ⭐ sklearn compatible wrapper
class KerasClassifierWrapper(BaseEstimator):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)

wrapped_model = KerasClassifierWrapper(model)

perm_importance = permutation_importance(
    wrapped_model,
    X_test,
    y_test,
    scoring='accuracy',
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

cat_importance = importance_df[importance_df['Feature'].isin(categorical_cols)]
num_importance = importance_df[importance_df['Feature'].isin(numerical_cols)]

plt.figure(figsize=(10,6))
plt.barh(cat_importance['Feature'], cat_importance['Importance'])
plt.title("Categorical Feature Importance")
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(10,6))
plt.barh(num_importance['Feature'], num_importance['Importance'])
plt.title("Numerical Feature Importance")
plt.gca().invert_yaxis()
plt.show()
