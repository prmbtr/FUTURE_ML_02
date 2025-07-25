# FUTURE_ML_02
Churn Prediction Model – Logistic Regression

This script builds a machine learning model to predict customer churn using Logistic Regression. 
It includes data preprocessing, encoding, scaling, model training, evaluation, and cross-validation.

Dataset:
- File: Churn_Modelling.csv
- Target: 'Exited' (1 = customer left, 0 = customer stayed)
- Features: Customer demographics, account information, and behavior metrics

Steps Covered in the Script:

1. Import Libraries
   - pandas, numpy, matplotlib, seaborn
   - sklearn modules: preprocessing, model_selection, linear_model, metrics, pipeline

2. Load Dataset
   - Reads CSV file into a DataFrame using pandas

3. Data Cleaning and Preprocessing
   - Drops irrelevant columns: 'RowNumber', 'CustomerId', 'Surname'
   - Label encodes 'Gender'
   - One-hot encodes other categorical variables (e.g., Geography)

4. Feature Preparation
   - Combines encoded and numeric features into a clean DataFrame
   - Separates features (X) and label (y)

5. Train-Test Split
   - Splits dataset into training and test sets (80-20)

6. Standardization
   - Applies StandardScaler to ensure features are on the same scale

7. Model Training
   - Trains a Logistic Regression model using the scaled data

8. Evaluation
   - Predicts outcomes on the test set
   - Calculates accuracy score
   - Applies 5-fold cross-validation and reports mean accuracy and standard deviation

Output:
- Test set accuracy
- Cross-validation scores for 5 folds
- Mean accuracy and standard deviation across folds

Dependencies:
- Install required packages before running:
    pip install pandas numpy matplotlib seaborn scikit-learn
