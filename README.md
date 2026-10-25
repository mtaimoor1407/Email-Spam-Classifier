# Email Spam Classifier

Classify message text as **spam** or **ham (not spam)** using TF-IDF features and either **Multinomial Naive Bayes** or a **linear-kernel SVM**. A Streamlit interface lets you enter text and choose the classifier.

## Pipeline

1. Lowercase text and remove punctuation and non-alphabetic characters.
2. Remove English stopwords and lemmatize words using NLTK.
3. Split messages into 80% training and 20% test data with `random_state=42`.
4. Fit the TF-IDF vectorizer on training messages and transform the test messages.
5. Apply random undersampling to the training features only.
6. Train both classifiers, print their test metrics, and export the model artifacts.

The application operates on message text; it does not analyze email headers or attachments.

## Run locally

```bash
git clone https://github.com/mtaimoor1407/Email-Spam-Classifier.git
cd Email-Spam-Classifier
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux.

```bash
pip install -r requirements.txt
```

The preprocessing module downloads the NLTK `stopwords`, `wordnet`, and `omw-1.4` resources. Internet access is required if they are not already installed.

### Train and evaluate

The training script expects `data/email.csv`, read with Latin-1 encoding, with these columns:

| Column | Expected value |
| --- | --- |
| `Category` | `ham` or `spam` |
| `Message` | Message text |

Run from the repository root so imports and relative paths resolve correctly:

```bash
python -m src.classifier
```

This prints accuracy and a classification report for both models, then creates:

- `models/nb_model.pkl`
- `models/svm_model.pkl`
- `models/tfidf_vectorizer.pkl`

### Start the interface

```bash
streamlit run app.py
```

Choose **Naive Bayes** or **SVM** in the sidebar, enter a message, and select **Classify**. If the saved models are missing, run the training command first.

## Evaluation

The training script reports held-out accuracy, precision, recall, and F1 through scikit-learn's classification report. Numerical results are not added here without a preserved run output.

For a stronger comparison, preserve the dataset version and class counts, check duplicates across splits, use a stratified split, and report spam-class precision/recall/F1 alongside a confusion matrix. A false positive marks a legitimate message as spam; a false negative lets spam through.

## Repository guide

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and model selection |
| `src/preprocess.py` | Text cleaning, stopword removal, lemmatization |
| `src/classifier.py` | Training, evaluation, undersampling, model export |
| `data/` | Input and cleaned CSV files |
| `models/` | Saved classifiers and TF-IDF vectorizer |
| `Email Spam Classifier Report.docx` | Project report |

## Limitations

English stopwords and the alphabetic-only cleaning rule limit the system's usefulness for multilingual text, code, URLs, and other meaningful symbols. Performance can change when message sources differ from the training data. This is a learning project rather than a complete email-filtering service.

**Tools:** Python, pandas, scikit-learn, imbalanced-learn, NLTK, joblib, Streamlit.
