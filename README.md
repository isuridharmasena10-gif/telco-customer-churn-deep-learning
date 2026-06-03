# 📊 Telecom Customer Churn Prediction using Machine Learning & Deep Learning

This repository showcases an end-to-end data science pipeline utilizing **Classical Machine Learning** and **Deep Learning (Artificial Neural Networks)** in TensorFlow/Keras to predict customer churn for a telecommunications provider. 

The project emphasizes benchmarking simpler models against complex architectures, solving severe class imbalances, and mathematically justifying model selection based on business-critical metrics.

---

## 🎯 Objective
The primary goal is to build a predictive framework that identifies subscription customers at high risk of churning (leaving the company). By intercepting these individuals early, the business can proactively deploy targeted retention strategies to preserve recurring revenue.

---

## 🛠️ Technologies Used
- **Data Manipulation:** Python, Pandas, NumPy
- **Machine Learning & Evaluation:** Scikit-learn
- **Deep Learning Framework:** TensorFlow / Keras
- **Data Visualization:** Matplotlib, Seaborn

---

## 📋 Project Workflow
1. **Data Cleaning & Type Rectification:** Fixed the `TotalCharges` object-type trap by coercing empty spaces into NaNs and dropping missing rows.
2. **Feature Engineering & Encoding:** Mapped binary categorical variables and performed One-Hot Encoding on nominal multi-class columns.
3. **Data Leakage Prevention:** Performed Stratified Train-Test Split (80/20) *before* applying `StandardScaler` to prevent feature information from leaking into the test partition.
4. **Cost-Sensitive Learning:** Applied inverse class weighting (`class_weight='balanced'`) across all models to handle the severe 73.5% vs 26.5% target class imbalance.
5. **Model Benchmarking:** Trained Baseline ML (Logistic Regression) and Classical ML (Random Forest) models.
6. **Deep Learning Framework:** Engineered a multi-layer Sequential ANN with Dropout and Early Stopping callbacks to optimize and eliminate overfitting.
7. **Comparative Evaluation:** Cross-examined models based on Accuracy vs. Recall trade-offs.

---

## 📊 Model Comparison & Results

Since customer churn is an imbalanced classification problem, global accuracy is misleading. In telecom retention, **Recall (capturing actual churners)** is the most critical business metric. 

The table below demonstrates the final performance across all implemented models on the identical test partition:

| Model Architecture | Overall Accuracy | Recall (True Positive Churn Rate) |
| :--- | :---: | :---: |
| 1. Logistic Regression (Baseline ML) | 72.64% | 79.68% |
| 2. Random Forest (Classical ML) | 78.89% | 48.66% |
| 3. **Artificial Neural Network (TensorFlow DL)** | **71.62%** | **80.75%** |

### 💡 Core Engineering & Business Justification
* **The Random Forest Failure:** Although Random Forest yields the highest global accuracy (**78.89%**), it severely suffers from majority-class bias, failing to capture the minority class with a critical Recall drop to **48.66%** (missing over half of the churning customers).
* **The Deep Learning Advantage:** By deploying an optimized **Artificial Neural Network (ANN)** combined with dynamic cost-sensitive class weights, we achieved the highest financial and operational utility with a **Recall of 80.75%**. Out of every 10 customers planning to leave the network, this model successfully intercepts 8 of them, mathematically and operationally justifying the shift from classical machine learning to deep learning.

---

## 🧠 Technical Model Architectures

### 1. Classical Machine Learning Models
- **Logistic Regression:** Regularized linear baseline with inverse class frequency weights (`max_iter=1000`).
- **Random Forest:** Ensemble of 100 Decision Trees with balanced bootstrapping.

### 2. Deep Learning Architecture (TensorFlow/Keras)
- **Input Layer:** 30 Scaled Continuous & Encoded Features
- **Hidden Layer 1:** Dense Layer (16 Neurons, ReLU Activation) + `Dropout(0.2)`
- **Hidden Layer 2:** Dense Layer (8 Neurons, ReLU Activation) + `Dropout(0.2)`
- **Output Layer:** Dense Layer (1 Neuron, Sigmoid Activation)
- **Optimization:** Adam Optimizer operating against a Binary Cross-Entropy loss function.
- **Regularization:** Monitored via an `EarlyStopping` callback (patience=5) that terminated training at Epoch 16 and restored the absolute best mathematical weights from Epoch 11.

---

## 📂 Dataset Information
- **Source:** IBM Telco Customer Churn Dataset
- **Sample Size:** 7,032 records (post-cleaning)
- **Target Variable:** `Churn` (Yes / No)

---

## 🚀 Future Improvements
- Hyperparameter Optimization using KerasTuner or GridSearchCV.
- Integration of gradient boosting benchmarks (XGBoost / LightGBM).
- Creating an interactive deployment interface using Streamlit.
- Containerizing the final network into a production-ready Model API.

---

## 👩‍💻 Author

**Isuri Dharmasena**
- **BSc (Hons) in Applied Statistics** | Data Science | AI & ML
- LinkedIn: [Isuri Dharmasena](https://www.linkedin.com/in/isuri-dharmasena-7137791ba/)
- GitHub: [isuridharmasena10-gif](https://github.com/isuridharmasena10-gif)

⭐ *If you found this project benchmarking insightful, feel free to star the repository!*
