import pandas as pd
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==============================
# 1. Load Dataset
# ==============================

data = pd.read_csv("dataset/preprocessed_news_dataset.csv")

print("Dataset shape:", data.shape)


# ==============================
# 2. Initialize Tokenizer
# ==============================

tokenizer = Tokenizer(
    num_words=20000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(data["content"])


# ==============================
# 3. Convert Text to Sequences
# ==============================

sequences = tokenizer.texts_to_sequences(data["content"])


# ==============================
# 4. Apply Padding
# ==============================

max_length = 300

X = pad_sequences(sequences, maxlen=max_length)


# ==============================
# 5. Target Variable
# ==============================

y = data["label"]


# ==============================
# 6. Print Debug Information
# ==============================

print("Vocabulary size:", len(tokenizer.word_index))
print("Input shape:", X.shape)


# ==============================
# 7. Save Tokenizer
# ==============================

pickle.dump(tokenizer, open("model/tokenizer.pkl", "wb"))

print("Tokenizer saved successfully!")