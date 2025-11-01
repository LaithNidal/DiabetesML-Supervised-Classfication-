## 🩺 Diabetes Prediction: Supervised Binary ML Classification Model (AdaBoost Classifier)

**🚀 Project Overview**

This project aims to predict the likelihood of diabetes in patients based on health and physiological data.
The model leverages an AdaBoost Classifier to provide accurate predictions using a supervised machine learning classification approach.

The project covers the full data science workflow — from data preprocessing and model training to evaluation and result interpretation — demonstrating practical applications of ML in healthcare analytics.

**📊 Workflow**

1. Data Loading & Cleaning – Imported the diabetes dataset and handled missing or inconsistent values.
2. Feature Engineering – Processed relevant medical features such as glucose level, BMI, insulin, and blood pressure.
3. Feature Scaling – Standardized numeric variables for improved classifier performance.
4. Model Training – Trained a RandomForestClassifier on the processed dataset.
5. Model Evaluation – Evaluated performance using key classification metrics including accuracy.

**🧠 Model**

The following models were tested and fitted on training data with accuracy averaged out for 50 trials with the following results:

1. Random Forest  76.1%
2. SVC	75.3%
3. XGB	73.6%
4. Extra Trees Classifier	75.7%
5. Ada Boost Classifier	75.1%



**Accuracy: 85% for AdaBoost after hyperparameter tuning utilizng `Optuna` library.**



**💻 Application**

The model was hosted on a Streamlit file found on the following URL:[Diabetes Classifier](https://doyouhavediabetes.streamlit.app/)



**Inputs:**

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

**Output:**
Predicted Class: Diabetic or Non-Diabetic

**🧩 Tech Stack**

* Python
* Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

**📁 Repository Contents**

* `diabetes_prediction.ipynb` – Jupyter notebook containing data preprocessing, model training, and evaluation

* `diabetes.csv` – Dataset containing patient medical attributes both raw and cleaned file.

* `scaler.joblib` – Saved trained model
  
* `streamlit.app.py` - Stremalit application hositng model.joblib file.
