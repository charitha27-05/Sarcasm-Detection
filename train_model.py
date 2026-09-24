import json
import re
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Standard English stopwords
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
    're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

print("Loading dataset...")
records = []
with open("dataset.json", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

df = pd.DataFrame(records)
print(f"Total rows loaded: {len(df)}")

df["clean_text"] = df["headline"].apply(clean_text)

X = df["clean_text"]
y = df["is_sarcastic"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Fitting TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training Logistic Regression Model...")
model = LogisticRegression(max_iter=1000, C=1.5, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["Not Sarcastic", "Sarcastic"]))

# Save artifacts
print("Saving model and vectorizer...")
joblib.dump(model, "model/sarcasm_model.joblib", compress=3)
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib", compress=3)

print("Artifacts saved in model/ successfully!")

# Quick sanity tests
test_samples = [
    "Oh great, another rainy day when I forgot my umbrella!",
    "Scientists discover a new species of deep sea fish.",
    "I absolutely love sitting in traffic for three hours.",
    "The library is open from 9 AM to 5 PM today."
]

print("\n--- Test Predictions ---")
for s in test_samples:
    cleaned = clean_text(s)
    vec = vectorizer.transform([cleaned])
    prob = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    label = "😏 Sarcastic" if pred == 1 else "🙂 Not Sarcastic"
    confidence = max(prob) * 100
    print(f"Text: '{s}'")
    print(f"  Result: {label} (Confidence: {confidence:.1f}%)\n")
