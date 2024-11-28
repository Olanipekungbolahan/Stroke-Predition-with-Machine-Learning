
# **Stroke Prediction with Machine Learning**

This project demonstrates how to build, evaluate, and deploy a machine learning model for predicting stroke risks based on patient data.

## **Project Overview**

This project includes the following key steps:

- **Data Preparation**: Loading and preprocessing the dataset, including encoding and normalization.
- **Model Building**: Training and fine-tuning a logistic regression model.
- **Model Evaluation**: Evaluating the model using common metrics (e.g., accuracy, precision, recall).
- **Model Deployment**: Deploying the model using Flask for real-time predictions.

---

## **Technologies Used**

- **Python**
- **Pandas & NumPy**
- **scikit-learn**
- **Flask**
- **Seaborn & Matplotlib**
- **Joblib**

---

## **Files Included**

- **`data.csv`**: The dataset used for training and testing the model.
- **`model_building.ipynb`**: Jupyter notebook for data preprocessing, model training, and evaluation.
- **`app.py`**: Flask API app for deploying the trained model.
- **`logistic_regression_pipeline.pkl`**: Saved model and preprocessing pipeline.
- **`requirements.txt`**: List of dependencies for the project.

---

## **Steps Involved**

### 1. **Data Preparation**
   - **Data Cleaning**: Handling missing values, encoding categorical features, and scaling numerical data.
   - **Exploratory Data Analysis (EDA)**: Visualizing distributions and correlations among features.

### 2. **Model Building**
   - **Logistic Regression**: Trained to predict the likelihood of a stroke.
   - **Hyperparameter Tuning**: Used `GridSearchCV` to optimize the model.

### 3. **Model Evaluation**
   - **Metrics**: Accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
   - **Model Comparison**: Compared results before and after hyperparameter tuning.

### 4. **Model Deployment**
   - **Flask API**: Created a web API to serve the model and make predictions from new data.
   - **Model Serialization**: Saved the pipeline with `joblib` for easy reuse.

---

## **How to Run the Project**

### Prerequisites:
- Python 3.8 or higher.
- Install required dependencies:
  ```bash
  pip install -r requirements.txt

Steps:
1 Clone the repository:

git clone https://github.com/Olanipekungbolahan/Stroke-Prediction-with-Machine-Learning.git
cd Stroke-Prediction-with-Machine-Learning

2 Train the Model: Run the Jupyter notebook model_building.ipynb to preprocess the data, train the model, and evaluate its performance.
python app.py

3 Start the Flask App:
python app.py

4 Test the API: Send a POST request to http://127.0.0.1:5000/predict with a JSON payload (patient data) to get predictions.

Example:
curl -X POST -H "Content-Type: application/json" \
-d '{"ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban", "smoking_status": "never smoked", "age": 45, "hypertension": 1, "heart_disease": 0}' \
http://127.0.0.1:5000/predict

Model Evaluation Metrics
The model was evaluated using the following metrics:

Metric	Value
Accuracy	85%
Precision	87%
Recall	82%
F1-Score	84%
ROC-AUC	90%
Future Work
Improve model performance with more advanced algorithms (e.g., XGBoost, Random Forest).
Add feature engineering to improve predictive power.
Deploy the model to a cloud platform like AWS, Heroku, or Azure for scalability.
Build a Streamlit interface for easier user interaction.
Contributing
Feel free to fork the repository and submit pull requests. Contributions are welcome!

Model Evaluation Metrics
The model was evaluated using the following metrics:

Metric	Value
Accuracy	85%
Precision	87%
Recall	82%
F1-Score	84%
ROC-AUC	90%
Future Work
Improve model performance with more advanced algorithms (e.g., XGBoost, Random Forest).
Add feature engineering to improve predictive power.
Deploy the model to a cloud platform like AWS, Heroku, or Azure for scalability.
Build a Streamlit interface for easier user interaction.
Contributing
Feel free to fork the repository and submit pull requests. Contributions are welcome!

