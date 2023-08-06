# Credit_Card_Fraud_Detection


## Overview

This project aims to build a credit card fraud detection model using machine learning techniques. The dataset used for this project is obtained from Kaggle and contains transaction data with a class label indicating whether the transaction is fraudulent or not. The goal is to train a model that can accurately identify fraudulent transactions based on the features available in the dataset.

## Dataset

The dataset used in this project is named `creditcard.csv`, which contains the following columns:

1. `Time`: Time elapsed between the current transaction and the first transaction in the dataset.
2. `V1-V28`: Anonymized features resulting from a PCA transformation to protect user identities.
3. `Amount`: Transaction amount.
4. `Class`: The class label (0 for legitimate transactions, 1 for fraudulent transactions).

## Preprocessing

1. Displayed the top 5 rows of the dataset using Pandas' `head()` method.
2. Checked for any missing values in the dataset using `isnull().sum()`.
3. Performed feature scaling on the `Amount` column using Scikit-learn's `StandardScaler`.

## Handling Imbalanced Dataset

The dataset is highly imbalanced, with a significantly higher number of legitimate transactions compared to fraudulent ones. To handle this imbalance, two techniques were applied:

1. Undersampling: A random sample of legitimate transactions was taken to match the number of fraudulent transactions, resulting in a balanced dataset.
2. Oversampling: The Synthetic Minority Over-sampling Technique (SMOTE) was used to create synthetic samples of fraudulent transactions to match the number of legitimate transactions, resulting in a balanced dataset.

## Model Building and Evaluation

Three machine learning models were trained and evaluated on both the undersampled and oversampled datasets:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The models were evaluated based on the accuracy, precision, recall, and F1-score on the test set. The final accuracy scores of each model on the test set are shown below:

### Undersampling:

- Logistic Regression: 94.21%
- Decision Tree Classifier: 93.68%
- Random Forest Classifier: 92.11%

### Oversampling:

- Logistic Regression: 94.58%
- Decision Tree Classifier: 99.81%
- Random Forest Classifier: 99.99%

## Model Saving

The Random Forest Classifier trained on the oversampled dataset was saved using the `joblib` library for future use.

Please note that this is a summary of the Credit Card Fraud Detection project's contents. The full code, along with detailed explanations, can be found in the Python script provided in this repository.
