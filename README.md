# Heart Disease Prediction Model

This project implements machine learning models to predict heart disease using the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. The implemented models include Decision Tree, Random Forest, Logistic Regression, Support Vector Machine (SVM), and a hybrid model (HRFLM) combining Random Forest and Logistic Regression.

---

## **Dataset Overview**
The dataset used is the Cleveland Heart Disease dataset:
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Features**: 13 clinical and demographic attributes such as age, sex, chest pain type, cholesterol level, maximum heart rate, etc.
- **Target Variable**: Binary classification:
  - `0`: No heart disease
  - `1`: Presence of heart disease
- **Missing Data**: Rows with missing values are removed during preprocessing.

---

## **Models Implemented**
### 1. **Decision Tree (DT)**
- Constructs interpretable rules to classify data.
- May overfit for small datasets.

### 2. **Random Forest (RF)**
- An ensemble of decision trees that improves accuracy and reduces overfitting.
- Handles both linear and non-linear relationships.

### 3. **Logistic Regression (LR)**
- A simple and interpretable model suitable for linear relationships.

### 4. **Support Vector Machine (SVM)**
- Uses kernels to classify data with both linear and non-linear boundaries.
- Computationally intensive on larger datasets.

### 5. **Hybrid Random Forest and Logistic Regression (HRFLM)**
- Combines the probabilities from RF and LR models to improve prediction accuracy.
- Balances strengths of ensemble methods (RF) and linear models (LR).

---

## **Metrics Used**
The models are evaluated using the following metrics:
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Proportion of positive predictions that are correct.
- **Recall (Sensitivity)**: Proportion of actual positives correctly identified.
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC-AUC Score**: Measures the ability of the model to distinguish between classes.
- **Confusion Matrix**: Provides detailed insight into true positives, false positives, true negatives, and false negatives.

---

## **How to Run the Code**
1. **Prerequisites**:
   - Python 3.8+
   - Required libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost` (optional for extended model).
2. **Steps**:
   - Clone this repository or download the files.
   - Install dependencies using:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the script:
     ```bash
     python heart_disease_model.py
     ```
3. **Output**:
   - The script prints evaluation metrics for all implemented models, including the hybrid model.

---

## **Future Improvements**
- **Advanced Feature Selection**: Use genetic algorithms or reinforcement learning for feature optimization.
- **Deep Learning Models**: Implement CNNs or RNNs to analyze time-series or ECG data.
- **Explainability**: Add SHAP or LIME for interpretable AI.
- **Real-time Prediction**: Integrate the model into a clinical decision support system or wearable device.

---

## **Contact**
For questions or contributions, feel free to reach out:

- **LinkedIn**: https://www.linkedin.com/in/shubhampawade/

---

## **Acknowledgments**
- UCI Machine Learning Repository for providing the dataset.
- Scikit-learn for machine learning tools.
- XGBoost for advanced gradient boosting techniques.
