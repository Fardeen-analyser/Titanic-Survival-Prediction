# 🛳️ Titanic Dataset Project

This project explores the **Titanic dataset**, one of the most famous datasets used for learning data analysis, machine learning, and predictive modeling. The dataset provides information about the passengers aboard the RMS Titanic, which sank on April 15, 1912, after colliding with an iceberg.

---

## 📘 Overview

The goal of this project is to analyze the data and build predictive models to determine whether a passenger survived the disaster based on features such as:
- Age  
- Gender  
- Class (1st, 2nd, or 3rd)  
- Fare  
- Family relations aboard (Siblings/Spouses, Parents/Children)  
- Embarkation port  

---

## 📂 Dataset Description

The dataset typically includes the following files:

| File | Description |
|------|--------------|
| `train.csv` | Training dataset containing passenger information and survival labels. |
| `test.csv` | Test dataset without survival labels (used for prediction). |
| `gender_submission.csv` | Sample submission file for prediction results. |

### Columns

| Column | Description |
|--------|-------------|
| `PassengerId` | Unique ID for each passenger |
| `Survived` | Survival (0 = No, 1 = Yes) |
| `Pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | Passenger’s full name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number (if available) |
| `Embarked` | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## ⚙️ Project Steps

1. **Data Loading & Cleaning**  
   - Handle missing values (Age, Cabin, Embarked)
   - Convert categorical variables (Sex, Embarked) into numeric form

2. **Exploratory Data Analysis (EDA)**  
   - Visualize survival rates by gender, age, and class  
   - Identify correlations between features and survival  

3. **Feature Engineering**  
   - Create new features such as `FamilySize` and `IsAlone`  
   - Encode categorical variables  

4. **Model Building**  
   - Train models (e.g., Logistic Regression, Random Forest, XGBoost)  
   - Tune hyperparameters for optimal performance  

5. **Model Evaluation**  
   - Evaluate accuracy, precision, recall, F1-score  
   - Generate ROC and confusion matrix  

6. **Prediction & Submission**  
   - Predict survival on the test set  
   - Save results to `submission.csv`  

---

## 🧰 Requirements

Make sure you have the following installed:

```bash
python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn