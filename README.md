AI Risk Scorer — Customer Return Risk Prediction
📌 Overview

AI Risk Scorer is a machine learning system designed to predict whether an order is likely to be returned.

The project follows a reproducible ML pipeline using DVC, tracks model experiments using MLflow, performs hyperparameter tuning using GridSearchCV, and provides predictions through a Streamlit application.

The final selected model is Logistic Regression, tuned using GridSearchCV.

🎯 Problem Statement

Customer returns can create additional operational and financial costs for businesses.

The objective of this project is to predict the probability of an order being returned based on customer, order, payment, discount, and delivery-related information.

The prediction can be used to identify orders with a higher return risk.

🏗️ Project Architecture
                    Raw Orders Data
                          │
                          ▼
                 Feature Engineering
                          │
                          ▼
                  Train/Test Split
                          │
                          ▼
                  Feature Selection
                          │
                          ▼
                    Preprocessing
                          │
                          ▼
              ┌───────────────────────┐
              │   Model Comparison    │
              │       MLflow          │
              └───────────┬───────────┘
                          │
                          ▼
                Logistic Regression
                          │
                          ▼
                    GridSearchCV
                          │
                          ▼
                     Best C = 2
                          │
                          ▼
                  Final LR Model
                          │
                          ▼
                    Streamlit App
                          │
                          ▼
                  Return Risk Prediction
🔄 ML Pipeline

The complete pipeline is managed using DVC.

Raw Data
   ↓
Feature Engineering
   ↓
Make Dataset
   ↓
Feature Selection
   ↓
Preprocessing
   ↓
Model Experiments
   ↓
Hyperparameter Tuning
   ↓
Final Model

Run the complete pipeline using:

dvc repro
🧩 Feature Engineering

The project creates additional features from the raw order information.

Examples include:

order_dayofweek
is_weekend
order_month
time_since_last_order
discount_amount
net_order_value
delivery_delay_ratio

Feature engineering is implemented in:

src/features/build_features.py
🎯 Feature Selection

Feature selection is performed before preprocessing.

The selected dataset contains order, customer, discount, delivery, and behavioral features.

The target variable is:

returned

where:

0 → Order not returned
1 → Order returned
⚙️ Preprocessing

The preprocessing pipeline handles different feature types separately.

Numerical Features
Median imputation
RobustScaler
Categorical Features
Most-frequent imputation
OneHotEncoder
drop="first"
handle_unknown="ignore"
Binary Features
Most-frequent imputation

The fitted preprocessing transformer is saved as:

models/preprocessor.pkl
🤖 Model Experiments

Multiple classification algorithms were evaluated and tracked using MLflow.

Models tested:

Logistic Regression
Random Forest
XGBoost
LightGBM
Extra Trees
Hist Gradient Boosting
SVM
KNN

The experiments were tracked using MLflow rather than manually comparing models.

📊 Model Comparison

The best-performing model for this project was Logistic Regression.

The comparison was based primarily on F1-score and recall for the positive class (returned = 1), along with PR-AUC.

The initial experiment produced:

Model	F1	Recall	PR-AUC
Logistic Regression	0.5036	0.6921	0.4779
Random Forest	0.4843	0.6048	0.4623
SVM	0.4814	0.6485	0.4100
XGBoost	0.3599	0.2904	0.3851
LightGBM	0.3398	0.2664	0.3766
Hist Gradient Boosting	0.3189	0.2336	0.4187
Extra Trees	0.2853	0.1965	0.4193
KNN	0.2767	0.2009	0.3351
🔍 Hyperparameter Tuning

After model comparison, Logistic Regression was selected for further optimization.

GridSearchCV was used to tune the regularization parameter:

C

Search space:

tuning:
  C:
    - 0.01
    - 0.1
    - 0.5
    - 1
    - 2
    - 5
    - 10

5-fold cross-validation was used with F1-score as the optimization metric.

Best Parameter
Best C = 2
Best CV F1 = 0.4871
🏆 Final Model

The final model is:

Logistic Regression
C = 2
max_iter = 1000
class_weight = balanced

The final model was evaluated on the held-out test set.

Final Test Performance
Metric	Score
Precision — Class 1	0.3962
Recall — Class 1	0.6921
F1 — Class 1	0.5040
ROC-AUC	0.7423
PR-AUC	0.4779

The final model artifact is stored as:

models/return_risk_model.pkl
📈 Experiment Tracking — MLflow

MLflow is used to track model experiments and final model results.

The project tracks:

Model type
Hyperparameters
F1-score
Recall
PR-AUC
ROC-AUC
Model artifacts

Start the MLflow UI with:

mlflow ui

Then open:

http://127.0.0.1:5000

Two important experiments are used:

return_risk_model_comparison

and

logistic_regression_tuning

The final model is tracked under:

return_risk_final_model
🗂️ DVC

DVC is used to make the ML workflow reproducible.

The pipeline is defined in:

dvc.yaml

Pipeline state is stored in:

dvc.lock

Parameters are maintained in:

params.yaml

Run:

dvc repro

to reproduce the pipeline.

Check pipeline status:

dvc status
🖥️ Streamlit Application

A Streamlit application is provided for model prediction.

The application loads:

models/preprocessor.pkl
models/return_risk_model.pkl

and performs prediction using the trained model.

Run the application:

streamlit run app.py

The application provides an interactive interface for making return-risk predictions.

📁 Project Structure
AI-risk-scorer/
│
├── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── preprocessor.pkl
│   └── return_risk_model.pkl
│
├── notebooks/
│
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── feature_selection.py
│   │
│   ├── preprocessing/
│   │   └── preprocessing.py
│   │
│   └── models/
│       ├── train_model.py
│       ├── tune_model.py
│       └── train_final_model.py
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── .gitignore
🚀 How to Run
1. Clone the repository
git clone <your-repository-url>
cd AI-risk-scorer
2. Create environment
conda create -n project python=3.11
conda activate project
3. Install dependencies
pip install -r requirements.txt
4. Reproduce the ML pipeline
dvc repro
5. Start MLflow
mlflow ui
6. Start Streamlit
streamlit run app.py
🛠️ Tech Stack
Python
Pandas
Scikit-learn
XGBoost
LightGBM
MLflow
DVC
Streamlit
Joblib
PyYAML
💡 Key Highlights
Reproducible ML pipeline using DVC
Automated feature engineering
Feature selection before modeling
Multiple ML algorithms evaluated
MLflow-based experiment tracking
Hyperparameter optimization using GridSearchCV
Final Logistic Regression model with optimized C
Saved preprocessing and model artifacts
Interactive Streamlit prediction application
👨‍💻 Project Status

Status: Completed for Buildathon Demo 🚀

The project currently supports the complete workflow from data processing and model experimentation to final model prediction through Streamlit.
