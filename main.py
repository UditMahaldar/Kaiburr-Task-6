import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Explanatory Data Analysis and Feature Engineering

# Load data
data = pd.read_csv('/content/drive/MyDrive/consumercomplaints.csv')

# Display the first 5 rows of the dataframe
print(data.head())

# Display basic info about the dataset
print(data.info())

# Step 2: Text Pre-Processing

# Text preprocessing function
def preprocess_text(text):
    # Remove punctuations
    text = re.sub('['+string.punctuation+']', '', text)
    # Remove numbers
    text = re.sub('['+string.digits+']', '', text)
    # Lowercase
    text = text.lower()
    return text

# Apply preprocessing to complaints text
data['complaints'] = data['complaints'].apply(preprocess_text)

# Step 3: Selection of Multi Classification model

# Vectorizing the text
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['complaints'])

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, data['product'], test_size=0.2)

# Applying the Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Step 4: Comparison of model performance

# Predictions
y_pred = nb.predict(X_test)

# Confusion matrix
print(confusion_matrix(y_test, y_pred))

# Classification report
print(classification_report(y_test, y_pred))

# Step 5: Model Evolution

print("Model Accuracy:", nb.score(X_test, y_test))

# Step 6: Prediction

# Example prediction
print('complaint: ', data['complaints'][0])
print('Label: ', data['product'][0])
print('Predicted: ', nb.predict(vectorizer.transform([data['complaints'][0]]))[0])
