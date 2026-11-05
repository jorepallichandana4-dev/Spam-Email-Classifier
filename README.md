# Spam Email Classifier

## Project Overview
The Spam Email Classifier is a machine learning project that identifies whether an email message is Spam or Ham (Normal).

It uses Natural Language Processing (NLP), TF-IDF Vectorization, and the Naive Bayes algorithm to classify email messages.

## Features
- Classifies emails as Spam or Normal.
- Uses a machine learning model trained on email records.
- Displays prediction confidence.
- Provides an interactive web interface using Streamlit.
- Evaluates model performance using accuracy, precision, recall, and F1-score.

## Technologies Used
- Python
- Pandas
- Hugging Face Datasets
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Joblib
- Streamlit

## Dataset
Dataset: [Spam or Ham Email Classification Dataset](https://huggingface.co/datasets/haoxianc/haldonmez_spam-or-ham-a-dataset-for-email-classification)

The project uses email messages labelled as Spam or Ham.

## Machine Learning Workflow
1. Load the email dataset.
2. Remove duplicate and empty email messages.
3. Separate email text and labels.
4. Split the data into training and testing sets.
5. Convert email text into numerical features using TF-IDF.
6. Train the Multinomial Naive Bayes classifier.
7. Evaluate the model.
8. Save the trained model and vectorizer using Joblib.
9. Use the model in a Streamlit application.

## Model Performance
- Algorithm: Multinomial Naive Bayes
- Feature extraction: TF-IDF
- Training records: 24,127
- Testing records: 6,032
- Test accuracy: 98.34%

## Project Structure
```text
Spam_Email_Classifier/
│
├── dataset/
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd Spam_Email_Classifier
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Project

To train the model:

```bash
python train_model.py
```

To predict an email using the terminal:

```bash
python predict.py
```

To launch the web application:

```bash
streamlit run app.py
```

## Application Output
The application accepts an email message and displays whether it is classified as Spam or Normal, along with the model's prediction confidence.

## Future Improvements
- Improve text preprocessing.
- Compare different machine learning algorithms.
- Add email prediction history.
- Improve the web interface.
- Test the model on additional real-world email datasets.

## Author
Developed as a Machine Learning project using Python and Scikit-learn.