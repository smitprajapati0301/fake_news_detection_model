import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==============================
# 1. Load Dataset
# ==============================

data = pd.read_csv("dataset/preprocessed_news_dataset.csv")

X = data["content"]
y = data["label"]


# ==============================
# 2. Load Tokenizer
# ==============================

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# ==============================
# 3. Convert Text → Sequences
# ==============================

sequences = tokenizer.texts_to_sequences(X)


# ==============================
# 4. Apply Padding
# ==============================

max_length = 300

X = pad_sequences(sequences, maxlen=max_length)


# ==============================
# 5. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# 6. Check Vocabulary
# ==============================

print("Vocabulary size:", len(tokenizer.word_index))


# Example word-token mapping
print("Example tokens:")
print(list(tokenizer.word_index.items())[:10])