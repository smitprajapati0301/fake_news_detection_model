import tensorflow as tf
import pickle
import re
import nltk

from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ===============================
# Download NLTK resources
# ===============================

nltk.download("stopwords")
nltk.download("wordnet")


# ===============================
# Load Model & Tokenizer
# ===============================

model = tf.keras.models.load_model("model/fake_news_model.keras")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# ===============================
# NLP Tools
# ===============================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ===============================
# Preprocessing Function
# ===============================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# ===============================
# Prediction Function
# ===============================

max_length = 200


def predict_news(text):

    processed_text = preprocess_text(text)

    seq = tokenizer.texts_to_sequences([processed_text])
    
    print("Sequence:", seq)

    padded = pad_sequences(seq, maxlen=max_length)

    prediction = model.predict(padded)[0][0]

    fake_prob = float(prediction)
    real_prob = 1 - fake_prob

    if fake_prob > 0.5:
        label = "Fake News"
    else:
        label = "Real News"

    return label, fake_prob, real_prob


# ===============================
# Test Prediction
# ===============================

news = input("Enter news text: ")

label, fake_prob, real_prob = predict_news(news)

print("\nPrediction:", label)
print("Fake Probability:", round(fake_prob, 3))
print("Real Probability:", round(real_prob, 3))