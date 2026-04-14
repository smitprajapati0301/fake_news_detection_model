# ===============================
# Model Evaluation
# ===============================

import tensorflow as tf
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ===============================
# 1. Load Dataset
# ===============================

data = pd.read_csv("dataset/preprocessed_news_dataset.csv")

X = data["content"]
y = data["label"]


# ===============================
# 2. Load Tokenizer
# ===============================

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# ===============================
# 3. Convert Text → Sequences
# ===============================

sequences = tokenizer.texts_to_sequences(X)


# ===============================
# 4. Padding
# ===============================

max_length = 200

X = pad_sequences(sequences, maxlen=max_length)


# ===============================
# 5. Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ===============================
# 6. Load Trained Model
# ===============================

model = tf.keras.models.load_model("model/fake_news_model.keras")


# ===============================
# 7. Evaluate Model
# ===============================

loss, accuracy, auc = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)
print("Test AUC:", auc)


# ===============================
# 8. Predictions
# ===============================

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype("int32").flatten()


# ===============================
# 9. Confusion Matrix
# ===============================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ===============================
# 10. Classification Report
# ===============================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ===============================
# 11. Dataset Balance
# ===============================

print("\nLabel Distribution:")
print(data["label"].value_counts())