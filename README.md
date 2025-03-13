# Heart Disease Prediction System

## Overview
The **Heart Disease Prediction System** is an end-to-end machine learning project designed to predict the likelihood of heart disease based on various patient attributes. This project leverages **PyCaret** for model training and evaluation, ensuring an efficient and industrial-level implementation.

## Dataset
The dataset used for this project contains multiple features related to patient health statistics, such as:
- Age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol levels (chol), fasting blood sugar (fbs), resting electrocardiographic results (restecg), maximum heart rate achieved (thalach), exercise-induced angina (exang), oldpeak, slope, number of major vessels (ca), and thal.
- **Target variable**: `target` (0 = No heart disease, 1 = Heart disease)

## Data Preprocessing
- Handled missing values by imputing **median** for continuous variables and **mode** for categorical variables.
- Removed invalid values for **ca (>=4)** and **thal (<=0)**.
- Saved the cleaned dataset for further analysis.

## Model Training & Evaluation
- **Automated Model Selection** using PyCaret.
- Compared multiple classification models and selected the best-performing one.
- **Best Model:** `Extra Trees Classifier`

### Model Performance:
| Metric        | Value  |
|--------------|--------|
| Accuracy     | 0.9847 |
| AUC          | 0.9995 |
| Recall       | 0.981  |
| Precision    | 0.9898 |
| F1 Score     | 0.9847 |
| Kappa        | 0.9694 |
| MCC          | 0.9707 |
| Train Time (Sec) | 0.088 |

## Dependencies
- Python 3.8+
- Pandas, NumPy
- PyCaret
- Scikit-learn

## License
This project is licensed under the MIT License.

