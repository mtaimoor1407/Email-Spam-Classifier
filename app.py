import streamlit as st
from src.preprocess import clean_text
from joblib import load
import os

MODEL_PATH = 'models'

def load_models():
    nb_model = load(os.path.join(MODEL_PATH, 'nb_model.pkl'))
    svm_model = load(os.path.join(MODEL_PATH, 'svm_model.pkl'))
    vectorizer = load(os.path.join(MODEL_PATH, 'tfidf_vectorizer.pkl'))
    return nb_model, svm_model, vectorizer

try:
    nb_model, svm_model, vectorizer = load_models()
except FileNotFoundError:
    st.error("Models not found. Please run classifier.py first.")
    st.stop()

# Streamlit UI
st.title("📧 AI Email Classifier")
st.markdown("Classify emails as **Spam** or **Ham** using Machine Learning")

# Sidebar for model choice
model_choice = st.sidebar.selectbox("Choose Classification Model", ("Naive Bayes", "SVM"))

# Input area
input_text = st.text_area("Enter your email text here", height=200)

if st.button("Classify"):
    if input_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Preprocess and vectorize
        cleaned = clean_text(input_text)
        vectorized = vectorizer.transform([cleaned])

        # Predict
        if model_choice == "Naive Bayes":
            pred = nb_model.predict(vectorized)[0] 
        else:
            pred = svm_model.predict(vectorized)[0]


        # Show prediction result
        st.success(f"**Prediction:** {'📨 Not Spam' if pred == 0 else '🚫 Spam'}")

