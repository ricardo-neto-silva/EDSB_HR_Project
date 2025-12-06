import gradio as gr
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessToDF(BaseEstimator, TransformerMixin):
    def __init__(self, ct):
        self.ct = ct

    def fit(self, X, y=None):
        self.ct.fit(X, y)
        self.feature_names_ = self.ct.get_feature_names_out()
        return self

    def transform(self, X):
        X_arr = self.ct.transform(X)
        index = getattr(X, "index", None)
        return pd.DataFrame(X_arr, columns=self.feature_names_, index=index)
    
class ColumnSelectorByName(BaseEstimator, TransformerMixin):
    def __init__(self, names=None):
        # IMPORTANT: do NOT modify 'names' here; just store it
        self.names = names

    def fit(self, X, y=None):
        # If names is None, select nothing
        if self.names is None:
            self.active_names_ = []
        else:
            # keep only those names that actually exist in this fold
            self.active_names_ = [n for n in self.names if n in X.columns]
        return self

    def transform(self, X):
        if not hasattr(self, "active_names_"):
            raise RuntimeError("ColumnSelectorByName is not fitted yet.")
        return X[self.active_names_]

model = joblib.load("svc_attrition_pipeline.joblib")

with open("config.json") as f:
    cfg = json.load(f)
THRESHOLD = cfg["threshold"]

with open("role_medians.json") as f:
    role_medians = json.load(f)   # dict: {jobrole: median_income}

# 2. Define the prediction function
# The arguments MUST match the order of the 'inputs' list defined below
def predict_attrition(
    TenureIndex, YearsInCurrentRole, Education, MonthlyIncome, YearsWithCurrManager,
    Department, JobLevel, PerformanceRating, NumCompaniesWorked, MaritalStatus,
    Gender, DistanceFromHome, JobRole, OverTime, PromotionGap,
    Income_Rate_Ratio, YearsSinceLastPromotion, HourlyRate, RelationshipSatisfaction,
    IncomeVsRoleMedian, TotalWorkingYears, BusinessTravel, EnvironmentSatisfaction,
    JobSatisfaction, DailyRate, EducationField, YearsAtCompany, JobInvolvement,
    PercentSalaryHike, TrainingTimesLastYear, StockOptionLevel, EngagementIndex,
    Age, WorkLifeBalance, MonthlyRate
):
    
    # Build a DataFrame matching the EXACT column names from X_full
    row = {
        "TenureIndex": TenureIndex,
        "YearsInCurrentRole": YearsInCurrentRole,
        "Education": Education,
        "MonthlyIncome": MonthlyIncome,
        "YearsWithCurrManager": YearsWithCurrManager,
        "Department": Department,
        "JobLevel": JobLevel,
        "PerformanceRating": PerformanceRating,
        "NumCompaniesWorked": NumCompaniesWorked,
        "MaritalStatus": MaritalStatus,
        "Gender": Gender, # Expecting 0 or 1
        "DistanceFromHome": DistanceFromHome,
        "JobRole": JobRole,
        "OverTime": OverTime, # Expecting 0 or 1
        "PromotionGap": PromotionGap,
        "Income_Rate_Ratio": Income_Rate_Ratio,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "HourlyRate": HourlyRate,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "IncomeVsRoleMedian": IncomeVsRoleMedian,
        "TotalWorkingYears": TotalWorkingYears,
        "BusinessTravel": BusinessTravel,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "JobSatisfaction": JobSatisfaction,
        "DailyRate": DailyRate,
        "EducationField": EducationField,
        "YearsAtCompany": YearsAtCompany,
        "JobInvolvement": JobInvolvement,
        "PercentSalaryHike": PercentSalaryHike,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "StockOptionLevel": StockOptionLevel,
        "EngagementIndex": EngagementIndex,
        "Age": Age,
        "WorkLifeBalance": WorkLifeBalance,
        "MonthlyRate": MonthlyRate
    }

    # Create DataFrame (1 row)
    df = pd.DataFrame([row])

    # ---- ENGINEERED FEATURES (same logic as training!) ----

    # TenureIndex
    df["TenureIndex"] = (
        df["YearsAtCompany"]
        + df["YearsInCurrentRole"]
        + df["YearsWithCurrManager"]
    ) / 3.0

    # PromotionGap (no negatives)
    df["PromotionGap"] = (df["YearsAtCompany"] - df["YearsSinceLastPromotion"]).clip(lower=0)

    # EngagementIndex
    engagement_cols = [
        "JobInvolvement",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
    ]
    df["EngagementIndex"] = df[engagement_cols].mean(axis=1)

    # IncomeVsRoleMedian (using TRAIN medians loaded from JSON)
    df["IncomeVsRoleMedian"] = df["MonthlyIncome"] / df["JobRole"].map(role_medians)

    # -------------------------------------------------------
    # Now df has ALL the columns the pipeline expects
    # -------------------------------------------------------

    # Get probability
    # Note: pipeline handles scaling/encoding internally based on this raw data
    proba = model.predict_proba(df)[:, 1][0]
    
    label = "Likely to leave" if proba >= THRESHOLD else "Likely to stay"

    return {
        "Predicted label": label,
        "Probability of attrition": round(float(proba), 3)
    }

# 3. Build Gradio Inputs based on your Feature Inspector results
inputs = [
    # --- Group 1: Engineered & Calculated Indices ---
    gr.Slider(0.0, 20.0, value=5.12, label="TenureIndex"),
    gr.Slider(0.03, 9.18, value=0.67, label="Income_Rate_Ratio"),
    gr.Slider(0.35, 3.63, value=1.11, label="IncomeVsRoleMedian"),
    gr.Slider(1.25, 4.0, value=2.72, label="EngagementIndex"),
    gr.Slider(0, 36, value=4.82, label="PromotionGap"),

    # --- Group 2: Demographics & Personal ---
    gr.Slider(18, 60, step=1, value=37, label="Age"),
    gr.Radio([0, 1], label="Gender (0=Female, 1=Male)"), # Adjusted to numeric based on stats
    gr.Dropdown(["Single", "Married", "Divorced"], label="MaritalStatus"),
    gr.Slider(1, 5, step=1, value=3, label="Education"),
    gr.Dropdown(['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'], label="EducationField"),
    gr.Slider(1, 29, step=1, value=9, label="DistanceFromHome"),

    # --- Group 3: Job Details ---
    gr.Dropdown(['Sales', 'Research & Development', 'Human Resources'], label="Department"),
    gr.Dropdown(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'], label="JobRole"),
    gr.Slider(1, 5, step=1, value=2, label="JobLevel"),
    gr.Dropdown(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], label="BusinessTravel"),
    gr.Radio([0, 1], label="OverTime (0=No, 1=Yes)"), # Adjusted to numeric based on stats

    # --- Group 4: Compensation ---
    gr.Slider(1009, 19999, value=6500, label="MonthlyIncome"),
    gr.Slider(2094, 26999, value=14313, label="MonthlyRate"),
    gr.Slider(102, 1499, value=802, label="DailyRate"),
    gr.Slider(30, 100, value=65, label="HourlyRate"),
    gr.Slider(11, 25, value=15, label="PercentSalaryHike"),
    gr.Slider(0, 3, step=1, value=1, label="StockOptionLevel"),

    # --- Group 5: Experience & Tenure ---
    gr.Slider(0, 40, step=1, value=11, label="TotalWorkingYears"),
    gr.Slider(0, 40, step=1, value=7, label="YearsAtCompany"),
    gr.Slider(0, 18, step=1, value=4, label="YearsInCurrentRole"),
    gr.Slider(0, 17, step=1, value=4, label="YearsWithCurrManager"),
    gr.Slider(0, 15, step=1, value=2, label="YearsSinceLastPromotion"),
    gr.Slider(0, 9, step=1, value=3, label="NumCompaniesWorked"),
    gr.Slider(0, 6, step=1, value=3, label="TrainingTimesLastYear"),

    # --- Group 6: Satisfaction & Performance ---
    gr.Slider(1, 4, step=1, value=3, label="JobSatisfaction"),
    gr.Slider(1, 4, step=1, value=3, label="EnvironmentSatisfaction"),
    gr.Slider(1, 4, step=1, value=3, label="RelationshipSatisfaction"),
    gr.Slider(1, 4, step=1, value=3, label="JobInvolvement"),
    gr.Slider(1, 4, step=1, value=3, label="WorkLifeBalance"),
    gr.Slider(3, 4, step=1, value=3, label="PerformanceRating"),
]

# Note: The 'inputs' list MUST match the function arguments order EXACTLY.
# I have re-ordered the function arguments in 'predict_attrition' above to match the random order
# produced by the inspector, but mapped specifically to the logical groups here. 

# RE-MAPPING: Because the list above is grouped logically for the User Interface,
# we need to be careful. The safest way is to wrap the function so arguments map by name,
# or simply match the list order to the function definition order.

# Let's align the list below EXACTLY to the function signature I wrote in step 2 to prevent errors.
inputs_aligned = [
    gr.Slider(0.0, 20.0, value=5.12, label="TenureIndex"),
    gr.Slider(0, 18, step=1, value=4.23, label="YearsInCurrentRole"),
    gr.Slider(1, 5, step=1, value=2.91, label="Education"),
    gr.Slider(1009, 19999, value=6502, label="MonthlyIncome"),
    gr.Slider(0, 17, step=1, value=4.12, label="YearsWithCurrManager"),
    gr.Dropdown(['Sales', 'Research & Development', 'Human Resources'], label="Department"),
    gr.Slider(1, 5, step=1, value=2.06, label="JobLevel"),
    gr.Slider(3, 4, step=1, value=3.15, label="PerformanceRating"),
    gr.Slider(0, 9, step=1, value=2.69, label="NumCompaniesWorked"),
    gr.Dropdown(['Single', 'Married', 'Divorced'], label="MaritalStatus"),
    gr.Radio([0, 1], label="Gender (0=Female, 1=Male)"),
    gr.Slider(1, 29, step=1, value=9.19, label="DistanceFromHome"),
    gr.Dropdown(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'], label="JobRole"),
    gr.Radio([0, 1], label="OverTime (0=No, 1=Yes)"),
    gr.Slider(0, 36, value=4.82, label="PromotionGap"),
    gr.Slider(0.037, 9.17, value=0.67, label="Income_Rate_Ratio"),
    gr.Slider(0, 15, step=1, value=2.19, label="YearsSinceLastPromotion"),
    gr.Slider(30, 100, value=65.89, label="HourlyRate"),
    gr.Slider(1, 4, step=1, value=2.71, label="RelationshipSatisfaction"),
    gr.Slider(0.35, 3.63, value=1.11, label="IncomeVsRoleMedian"),
    gr.Slider(0, 40, step=1, value=11.28, label="TotalWorkingYears"),
    gr.Dropdown(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], label="BusinessTravel"),
    gr.Slider(1, 4, step=1, value=2.72, label="EnvironmentSatisfaction"),
    gr.Slider(1, 4, step=1, value=2.73, label="JobSatisfaction"),
    gr.Slider(102, 1499, value=802.49, label="DailyRate"),
    gr.Dropdown(['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'], label="EducationField"),
    gr.Slider(0, 40, step=1, value=7.01, label="YearsAtCompany"),
    gr.Slider(1, 4, step=1, value=2.73, label="JobInvolvement"),
    gr.Slider(11, 25, value=15.21, label="PercentSalaryHike"),
    gr.Slider(0, 6, step=1, value=2.80, label="TrainingTimesLastYear"),
    gr.Slider(0, 3, step=1, value=0.79, label="StockOptionLevel"),
    gr.Slider(1.25, 4.0, value=2.72, label="EngagementIndex"),
    gr.Slider(18, 60, step=1, value=36.92, label="Age"),
    gr.Slider(1, 4, step=1, value=2.76, label="WorkLifeBalance"),
    gr.Slider(2094, 26999, value=14313, label="MonthlyRate"),
]

outputs = gr.JSON(label="Prediction")

demo = gr.Interface(
    fn=predict_attrition,
    inputs=inputs_aligned,
    outputs=outputs,
    title="Employee Attrition Risk (SVC Model)",
    description="Enter employee information (35 Features) to estimate the probability of attrition."
)

if __name__ == "__main__":
    demo.launch()