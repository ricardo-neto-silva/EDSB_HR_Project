**Enterprise Data Science Bootcamp**

**EDSB25_26**

Rita Martins 20240821

Joana Coelho 2024080

Pedro Fernandes 20240823

Ricardo Silva 20240824

The content of this repository corresponds to a project developed by Group 26 in the Enterprise Data Science Bootcamp course, as part of the Enterprise Data Science and Analytics Postgraduate Program at Nova IMS in 2025.

This project aims to understand the factors influencing employee turnover in an organization, through a detailed exploratory analysis and the identification of key variables that contribute most to this effect. Based on this, a machine learning predictive model was developed to anticipate the probability of each employee leaving. The generated insights serve as a basis for strategic recommendations and targeted actions, with the goal of increasing talent retention and strengthening the company’s competitiveness.

# 📉 Employee Attrition Risk Prediction

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20App-blue)](https://huggingface.co/spaces/ricardo-neto-silva/hr-attrition-predictor)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This project leverages Machine Learning to predict employee attrition. By analyzing demographic, job-related, and behavioral data, the model identifies employees at risk of leaving the organization.

The goal is to provide Human Resources with actionable insights to improve retention strategies. The project compares a baseline **XGBoost** model against a tuned **Support Vector Classifier (SVC)**. The SVC was ultimately selected for its robust performance and stability during Cross-Validation.

**✨ Live Demo:** [Click here to try the Web App on Hugging Face](https://huggingface.co/spaces/ricardo-neto-silva/hr-attrition-predictor)

---

## 📂 Repository Structure

HR_PROJECT/
├── data/
│   └── raw/
│       └── HR_Attrition_Dataset.csv      # Original dataset
├── src/
│   ├── app.py                            # Gradio application for deployment
│   ├── hr_attrition.py                   # Main modeling scripts
│   ├── hr_attrition.ipynb                # Analysis and Training notebook
│   ├── config.json                       # Config for engineered features (medians)
│   ├── role_medians.json                 # Role-specific income statistics
│   └── svc_attrition_pipeline.joblib     # Serialized final model
├── requirements.txt                      # Dependencies for reproduction
├── pipeline_viz.png                      # Pipeline visualization image
├── .gitignore                            # Git configuration
├── EDSB25_26.csv                         # Identification of the group
└── README.md                             # Project documentation

---

### Part 2: Methodology & Metrics

## ⚙️ The Methodology

### 1. Advanced Feature Engineering
To maximize predictive power, I engineered several custom features based on domain knowledge:
* **TenureIndex:** A composite score weighting years at the company vs. years in the current role.
* **PromotionGap:** Measures the stagnation period since the last promotion.
* **Income_Rate_Ratio:** Ratio of monthly income to daily rate to detect compensation disparities.
* **EngagementIndex:** Aggregation of Job Satisfaction, Environment Satisfaction, and Job Involvement.

### 2. The Pipeline
The project implements a robust `sklearn.pipeline.Pipeline` designed to prevent data leakage.

![Pipeline Visualization](pipeline_viz.png)

1.  **Custom Preprocessing (`PreprocessToDF`):** Calculates engineered features using training-set statistics (saved in `config.json`) to ensure new data is treated consistently.
2.  **Encoding (`ColumnTransformer`):**
    * `OrdinalEncoder` for ranked variables (e.g., *BusinessTravel*, *Education*).
    * `OneHotEncoder` for nominal variables (e.g., *Department*, *JobRole*).
3.  **Feature Selection:** A strict subset of predictive features was selected using multi-method analysis.
4.  **Scaling:** `StandardScaler` normalizes inputs (crucial for the SVC model).
5.  **Inference:** Probability estimation for attrition risk.

---

## 📊 Model Performance & Selection

I evaluated two primary candidates: a baseline XGBoost model and a tuned Support Vector Classifier (SVC).

### 🥇 Final Model: SVC (Tuned + Threshold Optimized)
* **Algorithm:** Support Vector Classifier (RBF Kernel)
* **Feature Set:** Strict (Selected Features)
* **Hyperparameters:** `C=8.53`, `gamma=0.0117`
* **Threshold:** **0.33** (Tuned)

| Metric | Cross-Validation Score | Test Set Score |
| :--- | :--- | :--- |
| **Accuracy** | **87.4%** | **85.4%** |
| **Precision** | **64.4%** | **55.0%** |
| **Recall** | **54.2%** | 46.8% |
| **F1-Score** | **58.0%** | **50.6%** |

### 🥈 Baseline Model: XGBoost
* **Algorithm:** Gradient Boosted Trees
* **Feature Set:** Moderate
* **Threshold:** Default (0.50)

| Metric | Cross-Validation Score | Test Set Score |
| :--- | :--- | :--- |
| **Accuracy** | 85.6% | 84.0% |
| **Precision** | 56.6% | 50.0% |
| **Recall** | 48.9% | **48.9%** |
| **F1-Score** | 52.4% | 49.5% |

**Conclusion:** The SVC model was selected because it outperformed the baseline XGBoost across all Cross-Validation metrics, demonstrating better generalization. On the Test Set, it achieved higher Accuracy (+1.4%) and F1-Score (+1.1%), validating the decision to use a stricter feature set and tuned threshold.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* pip

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ricardo-neto-silva/HR_PROJECT.git](https://github.com/ricardo-neto-silva/HR_PROJECT.git)
    cd HR_PROJECT
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App Locally:**
    The project includes a Gradio interface for easy interaction.
    ```bash
    cd src
    python app.py
    ```
    Open your browser to the URL shown in the terminal (usually `http://127.0.0.1:7860`).

---

## ⚠️ Disclaimer
This tool is intended for informational purposes only. The predictions are based on historical data patterns and should be interpreted by HR professionals in conjunction with qualitative factors. It should not be used as the sole basis for hiring or firing decisions.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📜 License
[MIT](https://choosealicense.com/licenses/mit/)
