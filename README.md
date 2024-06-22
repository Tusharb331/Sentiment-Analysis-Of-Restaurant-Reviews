# Sentiment Analysis of Restaurant Reviews


## Overview

This project focuses on sentiment analysis using machine learning techniques. The goal is to classify the sentiment of textual data into positive, negative, or neutral categories.

## Technologies Used

- Python
- NLTK (Natural Language Toolkit)
- Scikit-learn
- Jupyter Notebook

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

3. Download NLTK resources:

   ```bash
   import nltk
   nltk.download('stopwords')

4. Run the Jupyter notebook Restaurant_Review_Sentiment_Analysis.ipynb to see the step-by-step analysis, preprocessing of data, model training, and evaluation.

## Description

### Data Cleaning
The reviews are cleaned by:
- Removing special characters
- Converting to lowercase
- Removing stopwords
- Stemming

### Feature Extraction
The Bag of Words model (using CountVectorizer) is used to convert text data into numerical feature vectors.

### Model Training
Three models are trained:
- **Multinomial Naive Bayes**
- **Bernoulli Naive Bayes**
- **Logistic Regression**

### Model Evaluation
Each model is evaluated on:
- Accuracy
- Precision
- Recall

### Results
- **Multinomial Naive Bayes**: Accuracy - 76.5%, Precision - 0.78, Recall - 0.78
- **Bernoulli Naive Bayes**: Accuracy - 76.5%, Precision - 0.79, Recall - 0.76
- **Logistic Regression**: Accuracy - 75.0%, Precision - 0.82, Recall - 0.68

## Prediction

You can predict the sentiment (positive/negative) of your review messages using the trained models.

### Example

```python
from sentiment_analysis import predict_review

msg = 'The food is really good here.'
prediction = predict_review(msg)
print(f"Sentiment: {'Positive' if prediction else 'Negative'} Review")
