# Alzheimer's Diagnosis Prediction — Clinical ML Leakage Investigation

An exploratory machine learning project using a public Alzheimer’s disease dataset to classify whether a patient has an Alzheimer’s diagnosis. This project uses a full machine learning workflow: exploratory data analysis, preprocessing, baseline modeling, XGBoost classification, hyperparameter tuning, feature importance analysis, leakage investigation, and SHAP interpretability.

The initial models achieved strong validation performance, with XGBoost reaching approximately **95% validation accuracy** and **0.947 ROC-AUC**. However, feature importance and SHAP analysis showed that the model relied heavily on cognitive, functional, and behavioral assessment variables, including `MemoryComplaints`, `FunctionalAssessment`, `BehavioralProblems`, `ADL`, and `MMSE`.

After removing these assessment-based features, performance dropped sharply to approximately **58.6% accuracy** and **0.475 ROC-AUC**, suggesting that the original model was largely recovering diagnosis from variables closely tied to the diagnostic process rather than predicting Alzheimer’s disease from independent risk factors.

This project highlights an important clinical machine learning lesson: strong accuracy alone is not enough. Model performance must be interpreted alongside feature meaning, leakage risk, and clinical context.

---
## Key Findings

- **Random Forest baseline:** Achieved approximately **94.2% accuracy** and **0.946 ROC-AUC**.
- **XGBoost model:** Achieved approximately **94.9% accuracy** and **0.947 ROC-AUC** on the validation set.
- **Overfitting check:** The initial XGBoost model reached **1.0 training accuracy** and approximately **94.9% validation accuracy**, suggesting mild overfitting.
- **Hyperparameter tuning:** `RandomizedSearchCV` reduced the training-validation gap, with tuned XGBoost reaching approximately **96.2% training accuracy** and **95.1% validation accuracy**.
- **Potential leakage discovered:** The strongest predictors were cognitive, functional, and behavioral assessment features closely related to Alzheimer’s diagnosis.
- **Leakage check:** Removing suspected diagnostic-overlap features dropped ROC-AUC to approximately **0.475**, showing little independent predictive signal from the remaining demographic, lifestyle, and medical-history variables.
- **Interpretability:** Feature importance and SHAP were used to understand which variables drove model predictions.

---

## Model Results Summary

| Model / Feature Set | Accuracy | ROC-AUC | Interpretation |
|---|---:|---:|---|
| Random Forest baseline | 94.2% | 0.946 | Strong baseline performance |
| XGBoost, all features | 94.9% | 0.947 | Strong validation performance, but likely driven by assessment-based features |
| Tuned XGBoost | 95.1% | 0.945 | Reduced overfitting gap, but did not remove the leakage concern |
| XGBoost without suspected leakage features | 58.6% | 0.475 | Little independent signal from remaining features |

---

## Dataset

This project uses the [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) by Rabie El Kharoua, available on Kaggle.

The dataset contains demographic, lifestyle, medical-history, cognitive, functional, and behavioral variables related to Alzheimer’s disease diagnosis.

The dataset is not included in this repository. To run the notebook, download the dataset from Kaggle and place `alzheimers_disease_data.csv` in the project directory, or update the file path in the notebook.

---
## Methods

This notebook follows a structured clinical machine learning workflow:

1. Loaded and inspected the Alzheimer’s disease dataset
2. Reviewed dataset shape, variables, diagnosis distribution, and missing values
3. Performed exploratory data analysis, including MMSE score comparison by diagnosis
4. Built preprocessing pipelines for numerical and categorical features
5. Trained a Random Forest baseline model
6. Trained and evaluated an XGBoost classifier
7. Compared training and validation accuracy to check for overfitting
8. Tuned XGBoost hyperparameters using `RandomizedSearchCV`
9. Examined feature importance to identify possible diagnostic-overlap features
10. Removed suspected leakage features and retrained the model
11. Used SHAP to interpret feature contributions
12. Discussed limitations and clinical implications
---

## Potential Leakage Investigation

The most important part of this project is the leakage investigation.

The original XGBoost model performed well, but the top features were:

- `MemoryComplaints`
- `FunctionalAssessment`
- `BehavioralProblems`
- `ADL`
- `MMSE`

These variables are clinically meaningful, but they are also closely related to how Alzheimer’s disease may be diagnosed. For example, MMSE measures cognitive impairment, while ADL and FunctionalAssessment measure daily functioning.

To test whether the model was learning independent risk patterns or mainly recovering the diagnosis from assessment-based variables, these features were removed and the model was retrained. After removal, ROC-AUC dropped from approximately **0.947** to **0.475**.

This suggests that the original high performance should be interpreted cautiously.

---
## Main Takeaway
The strongest result of this project is not just that the model achieved high accuracy. The stronger finding is that the high accuracy was likely driven by features closely tied to the diagnosis label.

This reframes the project from simply building a predictive model to evaluating whether a clinical ML model is learning meaningful independent patterns or relying on diagnostic-overlap features.

---


## How to Run

Clone this repository:

```bash
git clone https://github.com/edwarddong1927-glitch/alzheimers-diagnosis-ml.git
cd alzheimers-diagnosis-ml

Install the required packages:

pip install -r requirements.txt

Download the dataset from Kaggle and place the CSV file in the project directory.

Open the notebook:

jupyter notebook alzheimers-diagnosis-prediction.ipynb

Then run all cells.
```
---

## Project 
alzheimers-diagnosis-ml/
│
├── alzheimers-diagnosis-prediction.ipynb   # Full analysis notebook
├── requirements.txt                        # Python dependencies
└── README.md                               # Project overview
---

## Limitations

This project is exploratory and should not be interpreted as a clinically validated diagnostic tool. The model was trained on a public tabular dataset and was not externally validated.

The dataset does not include neuroimaging, genetic, biomarker, or longitudinal clinical data, which are often important in Alzheimer’s disease research. In addition, several of the strongest predictors appear closely related to the diagnosis process itself, creating a potential leakage or diagnostic-overlap concern.

Future work should test the model on independent datasets, evaluate calibration, include richer clinical or biomarker features, and further investigate whether predictions remain useful when diagnostic-assessment variables are removed.

---
## What I Learned
This project showed that high machine learning performance does not automatically mean a model is clinically useful. In clinical ML, it is important to ask whether the model is learning independent predictors or simply relying on variables that overlap with the target label.

The most valuable part of this project was identifying why the model performed well, testing for leakage, and reframing the results honestly based on the evidence.