# Alzheimer's Diagnosis Prediction — ML Exploratory Project

An exploratory machine learning project predicting Alzheimer's diagnosis from a public clinical dataset using Random Forest and XGBoost classifiers. The initial model achieved ~95% accuracy, but a follow-up investigation revealed this was driven almost entirely by cognitive/functional assessment features (MMSE, FunctionalAssessment, MemoryComplaints, BehavioralProblems, ADL) — when these were removed, performance dropped to ~0.48 ROC-AUC (essentially random), suggesting these features are closely tied to the diagnostic criteria itself rather than independent risk factors. This reframes the project's key contribution from "a predictive model" to a demonstration of how important it is to check for data leakage in clinical ML.

## Key Findings

- **Baseline performance**: XGBoost achieved ~95% accuracy and ~0.94 ROC-AUC using the full feature set.
- **Leakage discovery**: Removing cognitive/functional assessment features dropped ROC-AUC to ~0.48 (random guessing), indicating these features likely overlap with the diagnostic criteria rather than acting as independent predictors.
- **Overfitting check**: Training accuracy of 1.0 vs. validation accuracy of ~0.95 indicated overfitting; hyperparameter tuning (RandomizedSearchCV) narrowed this gap without meaningfully changing validation performance.
- **Interpretability**: SHAP values were used to understand feature contributions to individual predictions.

## Dataset

This project uses the [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) by Rabie El Kharoua, available on Kaggle. The dataset is not included in this repository — download it from Kaggle and place `alzheimers_disease_data.csv` in the same directory as the notebook (or update the file path in the notebook to match where you saved it).

## How to Run

1. Clone this repository:
   ```bash
   git https://github.com/edwarddong1927-glitch/alzheimers-diagnosis-ml.git
   cd alzheimers-diagnosis-ml
   ```
2. Install the required packages:
   ```bash
   pip install numpy pandas matplotlib scikit-learn xgboost shap
   ```
3. Download the dataset from Kaggle (link above) and place it in the project directory.
4. Update the file path in the notebook's data-loading cell to point to your local CSV file.
5. Open the notebook and run all cells:
   ```bash
   jupyter notebook alzheimers_diagnosis_prediction.ipynb
   ```

## Project Structure

- `alzheimers_diagnosis_prediction.ipynb` — the full analysis: EDA, preprocessing, baseline (Random Forest) and main (XGBoost) models, evaluation, hyperparameter tuning, the leakage investigation, SHAP interpretability, and a discussion of limitations.

## Limitations

This project is exploratory and not intended as a clinically validated diagnostic tool. It uses tabular clinical/lifestyle data rather than imaging, genetic, or biomarker data, and has not been validated on external datasets. See the notebook's Limitations section for full details, including the data leakage discussion.
