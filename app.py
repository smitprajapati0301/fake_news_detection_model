import streamlit as st
import tensorflow as tf
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences


model = tf.keras.models.load_model("model/fake_news_model.keras")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
def preprocess_text(text):

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    return text

max_length = 250

def predict_news(text):

    processed_text = preprocess_text(text)

    seq = tokenizer.texts_to_sequences([processed_text])

    padded = pad_sequences(seq, maxlen=max_length)

    prediction = model.predict(padded)[0][0]

    fake_prob = float(prediction)
    real_prob = 1 - fake_prob

    if fake_prob > 0.5:
        label = "Fake News"
    else:
        label = "Real News"

    return label, fake_prob, real_prob

st.title("Fake News Detection System")

st.write("Enter a news article below to check whether it is Fake or Real.")

news_text = st.text_area("News Article")

if st.button("Predict"):

    label, fake_prob, real_prob = predict_news(news_text)

    st.subheader("Prediction")

    st.write("Result:", label)

    st.write("Fake Probability:", round(fake_prob,3))

    st.write("Real Probability:", round(real_prob,3))    