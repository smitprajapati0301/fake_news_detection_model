import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional


vocab_size = 20000
max_length = 300
embedding_dim = 128


model = Sequential()

# Convert tokens into semantic vectors
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))

# Capture context from both directions
model.add(Bidirectional(LSTM(64)))

# Reduce overfitting
model.add(Dropout(0.3))

# Final binary classification
model.add(Dense(1, activation="sigmoid"))


model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


model.summary()