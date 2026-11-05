import joblib
import os
from datasets import load_dataset
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset
ds = load_dataset(
    "haoxianc/haldonmez_spam-or-ham-a-dataset-for-email-classification"
)

# Step 2: Convert into DataFrame
df = ds["train"].to_pandas()

# Step 3: Clean the data
df = df.drop_duplicates(subset=["text"])
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]

# Step 4: Select input and output
X = df["text"]
y = df["label_num"]

# Step 5: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))

# Step 6: Convert text into numerical features
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 7: Create and train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 8: Predict test emails
y_pred = model.predict(X_test_tfidf)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(accuracy * 100, "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham (Normal)", "Spam"]
))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------------
# MACHINE LEARNING MODEL TRAINING
# -----------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Select input and output
X = df["text"]
y = df["label_num"]

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining emails:", len(X_train))
print("Testing emails:", len(X_test))

# Step 3: Convert email text into numbers
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Create machine learning model
model = MultinomialNB()

# Step 5: Train the model
model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")

# Step 6: Predict test data
y_pred = model.predict(X_test_tfidf)

# Step 7: Evaluate model
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred) * 100, "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham (Normal)", "Spam"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Save trained model and vectorizer

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")