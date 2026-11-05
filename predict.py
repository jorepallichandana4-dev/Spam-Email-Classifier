
import joblib

# Load the saved model
model = joblib.load("models/spam_model.pkl")

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("===== SPAM EMAIL CLASSIFIER =====")

# Get email message from user
email = input("\nEnter your email message: ")

# Convert email text into numerical features
email_tfidf = vectorizer.transform([email])

# Predict email category
prediction = model.predict(email_tfidf)

# Display result
if prediction[0] == 1:
    print("\nResult: SPAM EMAIL")
else:
    print("\nResult: NORMAL EMAIL")
