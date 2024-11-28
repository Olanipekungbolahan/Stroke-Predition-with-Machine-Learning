Stroke Prediction Using Logistic Regression
This project implements a machine learning pipeline to predict the likelihood of a stroke based on patient demographic and health data. The pipeline integrates data preprocessing, feature engineering, and a fine-tuned logistic regression model for robust predictions. The pipeline is designed to handle raw input directly, making it deployment-ready.

Table of Contents
Overview
Features
Technologies Used
Dataset Description
Workflow
How to Run the Project
Model Deployment
Results and Evaluation
Future Work
Contributing
License
Overview
The objective of this project is to predict whether a patient is at risk of a stroke based on their health and lifestyle factors. A logistic regression model was chosen as the preferred approach due to its simplicity, interpretability, and high performance for this problem.

Key Highlights:
Preprocessing steps for both categorical and numerical features.
A fully integrated pipeline for training, testing, and deployment.
API support for real-time predictions using Flask.
Features
Data Preprocessing:

OneHotEncoding for categorical data.
MinMaxScaler for numerical data normalization.
Integration of preprocessing into a single Pipeline for efficiency.
Modeling:

Logistic Regression with hyperparameter tuning using GridSearchCV.
Evaluation metrics: Confusion Matrix, Accuracy, F1-Score, Precision, Recall, and ROC-AUC.
Deployment:

Flask API for real-time predictions.
Saved pipeline model for scalability and reuse.
Technologies Used
Python: Core programming language.
Scikit-learn: For model training, preprocessing, and pipeline integration.
Pandas & NumPy: Data manipulation and preprocessing.
Flask: Backend for deployment.
Joblib: Model saving and loading.
Seaborn & Matplotlib: Data visualization.
Dataset Description
The dataset contains information about patients, including health metrics and demographics. Key columns include:

Column Name	Description	Data Type
ever_married	Patient's marital status	Categorical
work_type	Type of employment	Categorical
Residence_type	Urban or rural residence	Categorical
smoking_status	Smoking habits	Categorical
age	Patient's age	Numerical
hypertension	Whether the patient has hypertension	Numerical (0/1)
heart_disease	Whether the patient has heart disease	Numerical (0/1)
stroke	Target variable (0 = No Stroke, 1 = Stroke)	Numerical (0/1)
Workflow
Data Preprocessing:

Categorical data encoded using OneHotEncoder.
Numerical data normalized using MinMaxScaler.
Model Development:

Split dataset into training (80%) and testing (20%) subsets.
Hyperparameter tuning for Logistic Regression using GridSearchCV.
Model Evaluation:

Metrics computed: Confusion Matrix, Accuracy, F1-Score, Precision, Recall, ROC-AUC.
Deployment:

Trained pipeline saved using joblib.
Flask API for handling input and making predictions.
How to Run the Project
Prerequisites
Python 3.8 or higher.
Install required dependencies:
bash
Copy code
pip install -r requirements.txt
Steps to Run Locally
Clone the repository:

bash
Copy code
git clone https://github.com/<your-username>/stroke-prediction.git
cd stroke-prediction
Train the Model:

bash
Copy code
python train.py
Start the Flask Application:

bash
Copy code
python app.py
Send a POST request to the API: Example using curl:

bash
Copy code
curl -X POST -H "Content-Type: application/json" \
-d '{"ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban", "smoking_status": "never smoked", "age": 45, "hypertension": 1, "heart_disease": 0}' \
http://127.0.0.1:5000/predict
Example Response:

json
Copy code
{
  "prediction": 1,
  "probability": [0.25, 0.75]
}
Model Deployment
The model and preprocessing steps are packaged in a Pipeline object for seamless deployment. The pipeline:

Encodes categorical data.
Normalizes numerical data.
Makes predictions using the Logistic Regression model.
Deployment Steps:
Save the Pipeline:

python
Copy code
joblib.dump(pipeline, 'logistic_regression_pipeline.pkl')
Load for Inference:

python
Copy code
pipeline = joblib.load('logistic_regression_pipeline.pkl')
predictions = pipeline.predict(new_data)
Results and Evaluation
Metrics on Test Data:
Metric	Score
Accuracy	0.85
Precision	0.87
Recall	0.82
F1-Score	0.84
ROC-AUC	0.90
Future Work
Enhance Data Preprocessing:

Add handling for missing or noisy data.
Include feature selection or engineering techniques.
Model Improvement:

Experiment with other models like Random Forest, XGBoost, or Neural Networks.
Scalable Deployment:

Deploy the API on cloud platforms like AWS, Heroku, or Azure.
User Interface:

Build a frontend for easier interaction with the model.
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a feature branch: git checkout -b feature-name.
Commit your changes: git commit -m 'Add some feature'.
Push to the branch: git push origin feature-name.
Open a Pull Request.
License
This project is licensed under the MIT License. See the LICENSE file for details.
