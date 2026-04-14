import pandas as pd

# ==============================
# 1. Load Dataset
# ==============================

fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

print("Fake News:", fake.shape)
print("Real News:", real.shape)


# ==============================
# 2. Inspect Columns
# ==============================

print("\nFake dataset columns:")
print(fake.columns)

print("\nReal dataset columns:")
print(real.columns)


# ==============================
# 3. Check Missing Values
# ==============================

print("\nMissing values in Fake dataset:")
print(fake.isnull().sum())

print("\nMissing values in Real dataset:")
print(real.isnull().sum())


# ==============================
# 4. Add Labels
# ==============================

fake["label"] = 1   # Fake News
real["label"] = 0   # Real News


# ==============================
# 5. Combine Datasets
# ==============================

data = pd.concat([fake, real], axis=0)

print("\nCombined dataset size:", data.shape)


# ==============================
# 6. Keep Important Columns
# ==============================

data = data[["title", "text", "label"]]


# ==============================
# 7. Handle Missing Values
# ==============================

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")


# ==============================
# 8. Combine Title + Text
# ==============================

data["content"] = data["title"] + " " + data["text"]


# ==============================
# 9. Shuffle Dataset
# ==============================

data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# ==============================
# 10. Save Clean Dataset
# ==============================

data.to_csv("dataset/clean_news_dataset.csv", index=False)

print("\nClean dataset saved successfully!")