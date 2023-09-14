# Kaiburr-Task-6
# Consumer Complaints Text Classification

This project uses Python and machine learning (Multinomial Naive Bayes) to classify consumer complaint texts into the following categories:

- 0: Credit reporting, repair, or other
- 1: Debt collection
- 2: Consumer Loan
- 3: Mortgage

## Steps

1. **Explanatory Data Analysis and Feature Engineering** - Analyzing the dataset and creating useful features.
2. **Text Pre-processing** - Cleaning up the text data by removing punctuation, numbers, and converting text to lowercase.
3. **Multiclass Classification Model Selection** - Using the MultinomialNB model from scikit-learn for text classification.
4. **Model Performance Comparison** - Comparing the performance of different models.
5. **Model Evaluation** - Evaluating the model's performance.
6. **Predictions** - Making predictions with the trained model.

## How to Run

- Ensure you have Python 3.x installed.
- Install necessary libraries with pip (i.e., pandas, nltk, re, string, scikit-learn).
- Download or clone this repo.
- Replace file paths in the Python script (`complaint_text_classifier.py`) to match where your dataset is stored.
- From your terminal, navigate to the directory containing `complaint_text_classifier.py` and run the script with `python complaint_text_classifier.py`.

## Contributing
Pull requests are welcome. Please open an issue first to discuss what you would like to change.

## Important
The given dataset for the task was not working, so i tested the code with a sample dataset created by me.
