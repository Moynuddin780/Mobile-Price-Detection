# 📊 Mobile Price Range Prediction using ML

This project explores machine learning approaches to classify mobile phones into price categories based on various features. The dataset contains numerical attributes such as battery power, RAM, internal memory, etc., and the target is to predict the price range (0 = low cost to 3 = very high cost).

---

## 🔧 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
```

---

## 📁 Dataset Overview

- **Rows**: 2000
- **Target**: `price_range` (0 to 3)
- **Features**: All numerical (e.g., battery_power, ram, px_height)

---

## 📊 Exploratory Data Analysis (EDA)

### 🔹 Class Distribution

Balanced distribution across all 4 price categories.

![Output 1](data:image/png;base64,{{IMAGE1}})

### 🔹 Heatmap of Feature Correlation

Shows strong positive and negative correlations among features.

![Output 2](data:image/png;base64,{{IMAGE2}})

### 🔹 Top Features by Correlation with Price

- `ram`, `battery_power`, `px_height` are highly correlated with price.

---

## ⚙️ Feature Scaling & Selection

- **StandardScaler** used to scale features.
- **PCA** applied to reduce dimensionality (optional step, shown in notebook).

---

## 🧠 Model Used: Random Forest Classifier

### ✅ Justification

- Handles large feature spaces well.
- Robust against overfitting.
- Can output feature importances.

---

## 🎯 Model Evaluation

- **Accuracy**: ~88%
- **Classification Report**:

```plaintext
              precision    recall  f1-score   support

           0       0.88      0.93      0.90       124
           1       0.84      0.86      0.85       139
           2       0.87      0.85      0.86       117
           3       0.93      0.89      0.91       120

    accuracy                           0.88       500
   macro avg       0.88      0.88      0.88       500
weighted avg       0.88      0.88      0.88       500
```

### 🔹 Confusion Matrix

![Output 3](data:image/png;base64,{{IMAGE3}})

---

## 🌟 Feature Importance

Top contributing features:

1. RAM
2. Battery Power
3. Mobile Weight
4. Internal Memory

![Output 4](data:image/png;base64,{{IMAGE4}})

---

## ✅ Conclusion

- The model performs with high accuracy (~88%) using only basic numeric features.
- Random Forest proved to be a powerful baseline classifier.
- Feature importance and visual analysis helped justify the model's decisions.

---

## 📌 Future Work

- Try more classifiers: XGBoost, SVM, Deep Learning
- Hyperparameter tuning
- UI integration for prediction

---

> 📁 Notebook: `8eb8c391-1a83-4ab7-a7f8-1c60ad35640a.ipynb`  
> 🧠 Model: `RandomForestClassifier`
