import pandas as pd
import nltk
import re

# NLP tools
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==============================
# 1. Download NLTK resources
# ==============================

nltk.download("stopwords")
nltk.download("wordnet")


# ==============================
# 2. Load Dataset
# ==============================

data = pd.read_csv("dataset/clean_news_dataset.csv")

print("Dataset shape:", data.shape)
print("Columns:", data.columns)


# ==============================
# 3. Handle Missing Values
# ==============================

data["content"] = data["content"].fillna("")


# ==============================
# 4. Convert Text to Lowercase
# ==============================

data["content"] = data["content"].str.lower()


# ==============================
# 5. Remove Punctuation
# ==============================

data["content"] = data["content"].apply(lambda x: re.sub(r"[^\w\s]", "", x))


# ==============================
# 6. Remove Stopwords
# ==============================

stop_words = set(stopwords.words("english"))

data["content"] = data["content"].apply(
    lambda x: " ".join(word for word in x.split() if word not in stop_words)
)


# ==============================
# 7. Lemmatization
# ==============================

lemmatizer = WordNetLemmatizer()

data["content"] = data["content"].apply(
    lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split())
)


# ==============================
# 8. Check Example Output
# ==============================

print("\nSample processed text:")
print(data["content"].iloc[0])


# ==============================
# 9. Save Preprocessed Dataset
# ==============================

data.to_csv("dataset/preprocessed_news_dataset.csv", index=False)

print("\nPreprocessed dataset saved successfully!")