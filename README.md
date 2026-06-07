# Telecom Customer Churn Prediction

Predicting which telecom customers are likely to churn using 
machine learning and deep learning models.

---

## Why This Project?

Customer churn is one of the biggest problems in the telecom industry. 
Retaining an existing customer is far cheaper than acquiring a new one. 
This project builds a framework to identify high-risk customers 
before they leave, so retention teams can act early.

---

## Dataset

- Source: IBM Telco Customer Churn Dataset (Kaggle)
- 7,043 customers, 21 features
- Churn rate: ~26.5% — imbalanced dataset

---

## Models Used

| Model | Type |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Ensemble |
| ANN (TensorFlow) | Deep Learning |

---

## Key Results

| Model | Recall (Churn) | AUC |
|---|---|---|
| Logistic Regression | ~76% | ~0.83 |
| Random Forest | ~78% | ~0.85 |
| ANN (TensorFlow) | ~81% | ~0.85 |

> ANN with threshold 0.35 was selected as final model  
> Recall prioritized over Accuracy — missing a churner costs more than a false alarm

---

## Main Findings

- Month-to-month contract customers churn at 42% vs 3% for 2-year contracts
- New customers (low tenure) have highest churn risk
- High monthly charges increase churn probability
- Top predictors: tenure, monthly charges, contract type

---

## Business Impact

Assuming a telecom base of 500,000 customers:
- Model identifies ~106,000 potential churners per year
- Retaining just 20% of those saves ~$1.3M per month

---

## Project Structure



---

## How to Run

1. Clone the repo
2. Open churn_prediction.ipynb in Jupyter or Google Colab
3. Run all cells top to bottom
4. Dataset loads automatically from URL — no setup needed

---

## Libraries Used

- Python, Pandas, NumPy
- Scikit-Learn
- TensorFlow / Keras
- Matplotlib, Seaborn

---

## Author

Isuri Dharmasena  
[LinkedIn](https://www.linkedin.com/in/isuri-dharmasena-7137791ba/) | 
[GitHub](https://github.com/isuridharmasena10-gif)
