Overview
This project focuses on sentiment analysis of restaurant reviews using machine learning techniques. It aims to classify reviews as either positive or negative based on the text content.

Dependencies
numpy and pandas for data manipulation
nltk for natural language processing tasks
sklearn for machine learning models and evaluation metrics
Dataset
The dataset (Restaurant_Reviews.tsv) contains 996 reviews labeled as liked (1) or not liked (0). After cleaning, duplicates were removed, resulting in 996 unique reviews.

Preprocessing
Cleaned the reviews by removing special characters and converting text to lowercase.
Removed stopwords and performed stemming using the Porter Stemmer from NLTK.
Model Building
Utilized the Bag of Words model with CountVectorizer to convert text data into numerical features.
Split the data into training and test sets for model evaluation.
Model Evaluation
Evaluated three classifiers:
Multinomial Naive Bayes
Logistic Regression
Bernoulli Naive Bayes
Results
Achieved an accuracy of approximately 76.5% with both Multinomial Naive Bayes and Bernoulli Naive Bayes models on the test set.
Precision and recall scores were also calculated to assess model performance.
Prediction Functionality
Implemented a function (predict_review) to predict sentiment (positive or negative) of new restaurant reviews based on trained models.
Example Predictions
Reviewed several sample messages to demonstrate the prediction capability:
"The food is really good here." - Predicted as Positive Review
"Food was pretty bad and the service was very slow." - Predicted as Negative Review
"I liked the food, it was very good." - Predicted as Positive Review
"The food was burnt, it was smelling bad." - Predicted as Negative Review
Conclusion
This project showcases the application of machine learning for sentiment analysis in the context of restaurant reviews, providing insights into the sentiment of customer feedback.
