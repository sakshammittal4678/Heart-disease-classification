# 🫀 Heart Disease Classification using Machine Learning

This project aims to predict whether a patient has heart disease based on medical features using various classification algorithms. The goal is to build a reliable model to assist in early diagnosis using machine learning techniques.

---

## 📁 Dataset

- Source: [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Format: CSV
- Records: 303 rows, 13 features + 1 target
- Target Variable: `target`  
  - 0 = No Heart Disease  
  - 1 = Presence of Heart Disease

---

## 📌 Features Used

- `age` – Age of the patient
- `sex` – Gender (1 = male; 0 = female)
- `cp` – Chest pain type (0-3)
- `trestbps` – Resting blood pressure
- `chol` – Serum cholesterol (mg/dl)
- `fbs` – Fasting blood sugar > 120 mg/dl
- `restecg` – Resting electrocardiographic results
- `thalach` – Maximum heart rate achieved
- `exang` – Exercise-induced angina
- `oldpeak` – ST depression induced by exercise
- `slope` – Slope of the peak exercise ST segment
- `ca` – Number of major vessels colored by fluoroscopy
- `thal` – Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)

---

## 🧠 Algorithms Used

- Logistic Regression ✅ (Final model)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

Hyperparameter tuning was done using **GridSearchCV** to get the best model performance.

---

## 🔍 Evaluation Metrics

- **Accuracy**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-score)**
- **Cross-validation Score**
- **ROC AUC Score**

---

## ✅ Final Model Results

- **Best Model:** Logistic Regression (with GridSearchCV)
- **Accuracy:** ~86%
- **Confusion Matrix:**

| Actual\Predicted |  0 |  1 |
|------------------|----|----|
| 0                | 20 |  8 |
| 1                | 0  | 33 |

- **Insights:**
  - High true positive rate (early detection potential)
  - Very few false negatives — model is cautious about predicting "no disease"

---

## 📊 Visualizations

- Correlation Heatmap
- Feature Importances (Random Forest)
- ROC Curve
- Confusion Matrix (Seaborn heatmap)

---

## 💾 Model Saving

The final logistic regression model was saved using `pickle`:

```python
import pickle
with open('heart_disease_classification_lr.pkl', 'wb') as f:
    pickle.dump(lr_gs_model, f)
