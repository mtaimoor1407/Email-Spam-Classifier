import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler 

from src.preprocess import clean_text

# Load dataset
DATA_PATH = os.path.join('data', 'email.csv')
df = pd.read_csv(DATA_PATH, encoding='latin-1')

# Drop Null Values
df.dropna(subset=['Category', 'Message'], inplace=True)

# Converting category to numerical values and cleaning message
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
df['clean_message'] = df['Message'].apply(clean_text)

# Save cleaned data
df.to_csv(os.path.join('data', 'cleaned_email.csv'), index=False)

X = df['clean_message']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Random Undersampling 
rus = RandomUnderSampler(random_state=42)
X_resampled , y_resampled = rus.fit_resample(X_train_vec, y_train)

# Train models
nb_model = MultinomialNB()
nb_model.fit(X_resampled, y_resampled)

svm_model = SVC(kernel='linear')
svm_model.fit(X_resampled, y_resampled)

# Save models and vectorizer
os.makedirs('models', exist_ok=True)

joblib.dump(nb_model, 'models/nb_model.pkl')
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

