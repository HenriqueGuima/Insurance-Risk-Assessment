Insurance Risk Assessment using Machine Learning
Overview
This project utilizes machine learning techniques to classify insurance customers into high-risk and low-risk categories. By analyzing customer data, the model predicts which customers are more likely to make claims or commit fraud, helping insurance companies make informed decisions.

Features
The dataset contains various features, including:

Demographics: Age, Gender, Marital Status, etc.
Financial Information: Annual Income, Credit Score, etc.
Health Metrics: Health Score, Smoking Status, Exercise Frequency.
Insurance Information: Previous Claims, Policy Type, Insurance Duration.
Additional Information: Vehicle Age, Premium Amount, Property Type.
The target variable (Risk) is derived based on business rules:

High Risk (1): Customers with more than 2 previous claims, credit scores below 400, or health scores below 20.
Low Risk (0): All other customers.
Project Workflow
Exploratory Data Analysis (EDA):

Visualized data distributions, correlations, and feature importance.
Handled missing values, normalized data, and encoded categorical features.
Feature Engineering:

Scaled numerical features.
Transformed categorical features into numeric representations.
Model Training:

Implemented machine learning models, including Logistic Regression and Random Forest.
Evaluated models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
Results:

The Random Forest model achieved an accuracy of 83.8%, with high precision for high-risk customers and near-perfect recall for low-risk customers.
Technologies Used
Python:
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
Machine Learning:
Random Forest, Logistic Regression.
Data Visualization:
Correlation heatmaps, distribution plots, and feature importance charts.

Key Results
Random Forest Performance:
Accuracy: 83.8%
Precision (High Risk): 99.92%
Recall (High Risk): 67.97%
Confusion Matrix:
True Positives: 9572
False Positives: 8
True Negatives: 13807
False Negatives: 4509
