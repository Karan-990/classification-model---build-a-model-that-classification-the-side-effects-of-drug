# Drug Side Effects Classification Model

📌 Project Overview
This project builds a **classification model** to predict the side effects of a drug based on patient attributes. Several machine learning models were trained and evaluated, with **Random Forest** achieving the highest accuracy of **98.33%**.

## 📂 Dataset
- The dataset (`drug200.csv`) contains **patient attributes** such as:
  - Age
  - Sex
  - Blood Pressure (BP)
  - Cholesterol Level
  - Sodium-to-Potassium Ratio (Na_to_K)
  - Drug Type (Target Variable)

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Libraries:**
  - NumPy, Pandas (Data Manipulation)
  - Matplotlib, Seaborn (Data Visualization)
  - Scikit-learn (Machine Learning Models & Evaluation)

## 🏆 Model Performance
| Model                 | Accuracy  | Precision | Recall  | F1-Score |
|----------------------|----------|----------|---------|----------|
| Logistic Regression  | ~94.16%  | 94.50%   | 94.16%  | 94.18%   |
| Decision Tree       | ~98.33%  | 98.40%   | 98.33%  | 98.31%   |
| **Random Forest**    | **98.33%** | **98.40%** | **98.33%** | **98.32%** |
| KNN                 | ~92.5%   | 92.8%    | 92.5%   | 92.4%    |
| SVM                 | ~95.0%   | 95.2%    | 95.0%   | 94.9%    |
| Naïve Bayes         | ~91.66%  | 91.8%    | 91.66%  | 91.5%    |

## 🎯 Predictions for a New Patient
A new patient's data is fed into the trained models:
```python
new_patient_data = pd.DataFrame({'Age':[30], 'Sex':['F'], 'BP':['NORMAL'], 'Cholesterol':['HIGH'], 'Na_to_K':[15.0]})
```
### Prediction Output:
- **Random Forest Prediction:** Drug Y ✅ (Final Decision)
- Decision Tree Prediction: Drug Y
- Logistic Regression Prediction: Drug X
- KNN Prediction: Drug Y
- SVM Prediction: Drug X
- Naïve Bayes Prediction: Drug Y


