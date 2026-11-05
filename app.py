
import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# Application heading
st.title("📧 Spam Email Classifier")

st.write(
    "Enter an email message below to check whether "
    "it is Spam or Normal."
)

# Email input
email = st.text_area(
    "Enter your email message:",
    height=180,
    placeholder="Type or paste your email here..."
)

# Check button
if st.button("Check Email"):

    if email.strip() == "":
        st.warning("Please enter an email message.")

    else:
        # Convert email into numerical features
        email_tfidf = vectorizer.transform([email])

        # Predict category
        prediction = model.predict(email_tfidf)

        # Get prediction probabilities
        probabilities = model.predict_proba(email_tfidf)[0]

        confidence = max(probabilities) * 100

        # Display result
        if prediction[0] == 1:
            st.error("🚨 This email is SPAM!")
        else:
            st.success("✅ This email is NORMAL!")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

st.divider()

st.caption("Developed using Python, Scikit-learn and Streamlit.")
