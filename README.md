Stroke Prediction with Machine Learning
This project demonstrates the end-to-end deployment of a machine learning pipeline for predicting the likelihood of a stroke. The pipeline processes raw input data, builds and evaluates models, and serves predictions through a deployment-ready interface.

Steps Involved
Data Preparation
Load and preprocess the dataset:
Categorical data encoded using OneHotEncoder.
Numerical data normalized using MinMaxScaler.
Perform exploratory data analysis (EDA):
Visualized correlations, feature distributions, and relationships between features using Seaborn and Matplotlib.
Model Building
Trained a Logistic Regression model, chosen for its simplicity, interpretability, and robustness.
Hyperparameter tuning:
Optimized the model using GridSearchCV to improve performance.
Evaluated different regularization strengths (C) and solvers (e.g., 'liblinear').
Model Evaluation
Metrics Used:
Confusion Matrix
Accuracy
Precision
Recall
F1-Score
ROC-AUC
Compared model performance before and after hyperparameter tuning.
Model Deployment
Integrated preprocessing and model prediction into a single Pipeline using scikit-learn.
Saved the pipeline using joblib for reuse in deployment.
Developed APIs using Flask to serve predictions.
Future-ready for integration with a Streamlit interface for a user-friendly dashboard.
Technologies Used
Python
Jupyter Notebook
scikit-learn
Flask
Joblib
Seaborn and Matplotlib
Files Included
data.csv: Dataset used for training and testing the model.
model_building.ipynb: Jupyter Notebook containing the complete workflow for data preparation, model building, and evaluation.
app.py: Flask app for serving the model and making predictions.
logistic_regression_pipeline.pkl: Saved pipeline including preprocessing and the trained model.
