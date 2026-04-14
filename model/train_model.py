import tensorflow as tf
import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ===============================
# 1. Load Dataset
# ===============================

data = pd.read_csv("dataset/preprocessed_news_dataset.csv")

X = data["content"]
y = data["label"]

print("Dataset size:", len(data))


# ===============================
# 2. Tokenization
# ===============================

tokenizer = Tokenizer(
    num_words=15000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)

pickle.dump(tokenizer, open("model/tokenizer.pkl", "wb"))

vocab_size = 15000


# ===============================
# 3. Padding
# ===============================

max_length = 200

X = pad_sequences(sequences, maxlen=max_length)


# ===============================
# 4. Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ===============================
# 5. Class Weights
# ===============================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))


# ===============================
# 6. Build Fast Model
# ===============================

embedding_dim = 128

model = Sequential()

model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))

model.add(GlobalAveragePooling1D())

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(1, activation="sigmoid"))


# ===============================
# 7. Compile Model
# ===============================

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()


# ===============================
# 8. Early Stopping
# ===============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ===============================
# 9. Train Model
# ===============================

history = model.fit(
    X_train,
    y_train,
    epochs=6,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weights,
    shuffle=True
)


# ===============================
# 10. Save Model
# ===============================

model.save("model/fake_news_model.keras")

print("Model training completed and saved successfully.")