# %% [markdown]
# ## **_Enterprise Data Science and Analytics - Enterprise Data Science Bootcamp_**
# 
# ### **HR Attrition Project - EDSB25_26**
# 
# - Ana Rita Martins 20240821
# 
# - Joana Coelho 2024080
# 
# - Pedro Fernandes 20240823
# 
# - Ricardo Silva 20240824

# %% [markdown]
# Data Science and Analytics are reshaping how organizations solve problems across diverse industries. Through systematic data analysis and predictive modeling, evidence-based solutions can be developed, enabling more reliable decision-making and greater efficiency.
# 
# In Human Resources, predictive analytics supports critical functions such as employee retention, workforce planning, and automated CV screening.
# 
# This project focuses on developing predictive models to assess the likelihood of employee resignation. By analyzing factors ranging from demographics to job satisfaction, the models aim to provide interpretable insights that highlight key drivers of attrition. These insights will help HR leaders take proactive steps to reduce turnover and retain talent.

# %% [markdown]
# ## 1. Importing Packages

# %%
import numpy as np
import pandas as pd
from summarytools import dfSummary
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from itertools import product
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.base import clone
import shap
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json
import joblib


# %% [markdown]
# ## 2. Importing Data and Initial Exploration

# %%
data = pd.read_csv('../data/raw/HR_Attrition_Dataset.csv')
print(data.head())


# %%
data.info()

# %%
pd.set_option('display.max_columns', None) 
data.describe() 

# %%
data.describe(include='object')

# %% [markdown]
# From this initial inspection what immediately stands out is that we have 3 constant features: "EmployeeCount", "StandardHours", and "Over18". We can remove those straight away. Additionally, the employee number (ID) feature, does not seem to contain any relevant info, and  we'll drop it too.

# %%
data.drop(columns=['EmployeeCount','Over18','StandardHours','EmployeeNumber'],inplace=True)

# %%
cat_cols = data.select_dtypes(include=["object"]).columns

for col in cat_cols: 
    print(f"Value counts for column '{col}':")
    print(data[col].value_counts())
    print("\n") 

# %%
dfSummary(data)

# %% [markdown]
# From the summary above, we verified that the data set doesn't contain duplicates, and we also gathered information about the data's distribution and main statistics.
# 
# What we can note is that, beasides our target, we have a couple of other binary features. Let's encode those.

# %%
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})

data.head()

# %% [markdown]
# Let's now have a look at how the distribution of the target variable.

# %%
ax = sns.countplot(x=data['Attrition'], hue=data['Attrition'], legend=False)
for container in ax.containers:
    ax.bar_label(container)

plt.title('Distribution of the Target Variable (Attrition)')
plt.show()


# %% [markdown]
# We can observe that our target cariable is quite imbalanced. This will require extra attention in later steps, namely when splitting the dataset into train, validation and test sets, as well as during the modelling stage.

# %% [markdown]
# # **3. Exploratory Data Analysis**

# %% [markdown]
# We'll start by plotting histograms to visually assess the distribution of the numeric features; this will allows us to spot any relevant patterns or trends in the data.

# %%
data.hist(figsize=(20, 15))
plt.show()


# %% [markdown]
# The histograms reveal some important patterns in the dataset.
# 
# - Once again we can observe that the **target variable** is highly skewed toward staying in the company.
# 
# - Concerning demographics, **age** follows an approximately bell-shaped distribution, centered around 30-40; **Gender** is skewed with more males than females.
# 
# - Features that are related to **work characteristics** (YearsAtCompany, TotalWorkingYears, YearsInCurrentRole, Overtime) are right-skewed, indicating many relatively new employees and fewer with long careers; working overtime is not common.
# 
# - **Income**: Salaries and rates are right-skewed, with few very high earners.
# 
# - **Satisfaction-related** variables are discrete and somewhat skewed toward higher ratings, while PerformanceRating shows very little variation (nearly all at level 3), suggesting limited predictive value.
# 
# Overall, the data displays strong imbalance and skewness patterns that will require careful consideration during modeling, suggesting it could benefit from stratified splits, and algorithms robust to class imbalance.

# %%
# Selecting  numerical columns (binaries excluded)
binary_cols = ['Attrition', 'Gender', 'OverTime']
num_cols = [col for col in data.select_dtypes(include=['int64', 'float64']).columns if col not in binary_cols]

# Boxplots for each numerical feature
n_cols = 5
n_rows = -(-len(num_cols) // n_cols)  

plt.figure(figsize=(20, 4*n_rows))

for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(y=data[col])
    plt.title(col)

plt.tight_layout()
plt.show()


# %% [markdown]
# The boxplots highlight the extent of skewness and make the outliers stand out clearly, which complements the histogram analysis above.
# 
# - Outliers are especially relevant in income and employment duration related-variables, which may need special handling. We'll decide how to handle them further down.
# 
# - For demographic/job characteristics (Age, DistanceFromHome, JobLevel, Education) featured the distributions are fairly compact with few outliers, aligning with the unimodal/bell-like shapes seen in histograms.
# 
# 
# - Ordinal satisfaction and variables show limited spread, consistent with their discrete scale, with some level of skew toward higher values. Their limited range may reduce their explanatory power.
# 
# - PerformanceRating shows very little variation (nearly all values at level 3) confirming its limited usefulness as a predictive feature.

# %% [markdown]
# Subsequent steps may differ based on the category of each feature. Therefore, we’ll create lists that group feature names by their respective types.

# %%
# Explicitly define groups that cannot be inferred reliably
feature_groups = {
    "binary": ['Gender', 'OverTime'],
    "ordinal": [
        'Education','EnvironmentSatisfaction','JobInvolvement',
        'JobLevel','JobSatisfaction','PerformanceRating',
        'RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance'
    ]
}

# Infer remaining types
all_features = data.columns.drop('Attrition')

# Categorical = object dtype except those explicitly listed
explicit_non_continuous = feature_groups["binary"] + feature_groups["ordinal"]
categorical = (
    data.select_dtypes(include='object')
        .columns.difference(explicit_non_continuous)
        .tolist()
)

# Continuous = numeric except explicit lists
continuous = (
    all_features
        .difference(categorical + explicit_non_continuous)
        .tolist()
)

feature_groups["categorical"] = categorical
feature_groups["continuous"] = continuous
feature_groups['non-continuous'] = feature_groups['binary'] + feature_groups['ordinal'] + categorical

feature_groups



# %% [markdown]
# Let's now look at the distribution of our non-continuous features.

# %%
for feature in feature_groups['non-continuous']:

    ax = sns.countplot(y=data[feature],order=data[feature].value_counts(ascending=False).index)
    ax.set_xlabel('Number of Employees')

    # Get data label values and concatenate them
    abs_values = data[feature].value_counts(ascending=False).values
    rel_values = data[feature].value_counts(ascending=False, normalize=True).values * 100
    data_labels = [f'{label[0]} ({label[1]:.1f}%)' for label in zip(abs_values, rel_values)]

    ax.bar_label(container=ax.containers[0], labels=data_labels)
    ax.margins(x=0.25)
    
 
    plt.show()


# %% [markdown]
# From the variables that, a priori, we'd think could be related with attrition, we find that:
# 
# - roughly 30% of employees work overtime
# 
# - roughly 40% have low to medium levels of satisfaction with the work environment
# 
# - roughly 30% report low to medium levels of job involvement
# 
# - nearly 40% report low to medium job satisfaction
# 
# - another nearly 40% have low to medium levels of satisfaction with relationships at work
# 
# - and about 5% report bad work-life balance

# %% [markdown]
# To better understand what might be contributing to employees’ decisions to quit, we'll next plot the non-continuous features against the target variable. We’ll also measure the attrition rate within each category. This will show us whether some groups are more prone to leaving than others, irrespective of their overall frequency.

# %%
for feature in feature_groups['non-continuous']:

    # Get within category proportions
    proportions = data.groupby(feature)['Attrition'].value_counts(normalize=True)

    # Plot
    ax = sns.countplot(y=data[feature], hue=data['Attrition'], order=data[feature].value_counts().sort_index().index)
    ax.set_xlabel('Number of Employees')

    # Insert proportions as data labels
    for i, container in enumerate(ax.containers):
        labels = [f'{proportions.loc[d,i]:.1%}' for d in sorted(data[feature].unique())]
        ax.bar_label(container, labels)

    ax.margins(x=0.15)

    plt.show()


# %% [markdown]
# From the plots above we find the following trends:
# 
# Department-level & Job roles
# 
# - Sales and Human Resources show a higher proportion of employees quitting compared to R&D.
# 
# - Within job roles, HR professionals tend to leave more often, but so do Lab Technicians, even though they are part of the R&D department.
# 
# - Sales Representatives have the highest attrition rate across all job roles, whereas higher-level roles—such as managers and directors—show very low attrition.
# 
# Personal characteristics
# 
# - Single employees appear more likely to quit.
# 
# Work conditions and workload
# 
# - Employees who work overtime, travel frequently, or have poor work–life balance are more likely to leave.
# 
# - Low satisfaction with the work environment, job involvement, overall job satisfaction, and relationships at work is also strongly associated with higher attrition.
# 
# Job level and hierarchy
# 
# - Employees in lower hierarchical levels tend to leave more often. However, attrition proportions do not strictly follow the hierarchical ranking order.
# 
# Stock ownership
# 
# - Employees with no stock options (stock option level 0) are more prone to quitting. This is not surprising, as offering stock is a common strategy to increase engagement.

# %% [markdown]
# Let's now run an equivalent analysis with our continuous features.
# 
# We'll plot both their probability density function and violin plots and assess how their distribution relates to the target.

# %%
# Ensure Attrition is binary for plotting aesthetics
df_plot = data.copy()
df_plot["Attrition"] = df_plot["Attrition"].astype(str)

#attrition_num = df_plot["Attrition"]

continuous_vars = feature_groups["continuous"]

def plot_kde_violin(df, col):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # KDE plot
    sns.kdeplot(
        data=df, x=col, hue="Attrition",
        common_norm=False, fill=True, alpha=0.4, ax=axes[0]
    )
    axes[0].set_title(f"KDE of {col} by Attrition")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Density")
    
    # Violin plot
    sns.violinplot(
        data=df, hue="Attrition", y=col,
        inner="box", ax=axes[1]
    )
    axes[1].set_title(f"Violin Plot of {col} by Attrition")
    axes[1].set_xlabel("Attrition")
    axes[1].set_ylabel(col)
    
    plt.tight_layout()
    plt.show()

# Generate combined plots for all continuous variables
for col in continuous_vars:
    plot_kde_violin(df_plot, col)


# %% [markdown]
# Some features show noticeable differences in their distributions depending on whether the employee quit or stayed.
# 
# Age and career stage
# 
# - Employees who quit tend to be younger.
# 
# - This aligns with lower values observed in Total Working Years, Years at Company, Years in Current Role, and Years with Current Manager.
# 
# Early-career employees may be more inclined to change jobs or roles, contributing to these lower tenure metrics.
# 
# Compensation
# 
# - Monthly income appears influential: employees with lower income are more likely to leave, which is expected. The same applies to daily rate.
# 
# Distance from home
# 
# - The larger the distance from home to work, the more likely the employees are to leave.
# 
# Other features
# 
# - The remaining continuous features either show similar distributions across attrition groups or differences too small to be clearly meaningful.

# %% [markdown]
# We’ll now take look at the correlations among the features, including the target variable. This will help us identify potential collinearity, as well as highlight which features are associated with attrition. Since several features are not strictly numeric or continuous, we’ll use Spearman’s correlation, which measures monotonic relationships by correlating feature ranks rather than their raw values.

# %% [markdown]
# We'll exclude strictly nominal categorical variables (like Gender, Department, JobRole) because Spearman is rank-based, not meant for unordered categories.

# %%
# Selecting valid variables for Spearman

df_corr = data.copy()

ordinal_features = feature_groups["ordinal"]
continuous_features = feature_groups["continuous"]

spearman_vars = continuous_features + ordinal_features + ["Attrition"]
df_spearman = df_corr[spearman_vars]


# %%
# Computing Spearman correlation matrix

spearman_matrix = df_spearman.corr(method="spearman")


# %%
# Extracting sorted correlations with Attrition

attrition_corr = spearman_matrix["Attrition"].drop("Attrition")
attrition_corr_sorted = attrition_corr.sort_values(ascending=False)

print(attrition_corr_sorted)


# %%
# Visualizing Spearman correlation matrix

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(spearman_matrix, dtype=bool))

sns.heatmap(spearman_matrix, mask=mask, cmap="coolwarm", center=0, annot = True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title("Spearman Correlation Matrix")
plt.show()


# %% [markdown]
#  From the analyses and visualization above we observe that:
# 
#  - YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager, TotalWorkingYears, JobLevel, MonthlyIncome, StockOptionLevel and Age are the strongest monotonic predictors of Attrition.
# 
# These are indicators that relate to tenure, seniority, and stability and they're in agreement with HR domain knowledge: attrition is highest among newer, younger, lower-level employees.
# 
#  - JobSatisfaction, JobInvolvement, EnvironmentSatisfaction Tshow mild but potentially meaningful associations.
# 
# Employees with lower satisfaction or lower involvement show slightly higher attrition.

# %%
top_features = attrition_corr_sorted.abs().sort_values(ascending=False).head(12).index

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(spearman_matrix.loc[top_features, top_features], dtype=bool))
sns.heatmap(spearman_matrix.loc[top_features, top_features], mask=mask,
            cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Top Spearman Correlated Variables")
plt.show()


# %% [markdown]
# The heatmap shows that several of the variables most strongly correlated with attrition are also highly collinear with each other. In particular, the following groups demonstrate very strong monotonic relationships (ρ > 0.70):
# 
#  - JobLevel — MonthlyIncome (ρ ≈ 0.92)
# 
#  - YearsInCurrentRole — YearsWithCurrManager (ρ ≈ 0.85)
# 
#  - TotalWorkingYears — MonthlyIncome (ρ ≈ 0.71)
# 
# These features are all measures of: Tenure, Seniority, Career progression, Employee stability, which explains why they are tightly correlated with each other and with lower attrition.

# %% [markdown]
#  While colinearity doesn't harm tree-based models, it does affect linear models like linear regression. Besides, it It also leads to unnecessary redundancy in the feature set. Keeping all of them increases the demand for computational powerr and increases the risk of overfitting. By the end of our feature selection process, we should aim to keep at most 2 or 3 representative variables of this set. And for regression models, we'll explicitly remove correlated pairs.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
#  Another way to circumvent colinearity is to combine several colinear raw variables into a single engineered feature. Let's do that below.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# %% [markdown]
# ## Feature Engineering

# %%
# TenureIndex (Average of three tenure-related variables)

data["TenureIndex"] = (
    data["YearsAtCompany"] +
    data["YearsInCurrentRole"] +
    data["YearsWithCurrManager"]
) / 3


# PromotionGap (Time in company since last promotion: a proxy for stagnation)

data["PromotionGap"] = data["YearsAtCompany"] - data["YearsSinceLastPromotion"]

# avoid negative values if any weird records exist
data["PromotionGap"] = data["PromotionGap"].clip(lower=0)

# EngagementIndex (Composite of satisfaction / involvement metrics)

engagement_cols = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction"
]

data["EngagementIndex"] = data[engagement_cols].mean(axis=1)

data['Income_Rate_Ratio'] = data['MonthlyIncome'] / data['MonthlyRate']

# %%
engineered = ["TenureIndex", "PromotionGap", "EngagementIndex", "Income_Rate_Ratio"]

spearman_corrs = (
    data[engineered + ["Attrition"]]
    .corr(method="spearman")["Attrition"]
    .drop("Attrition")
)

print(spearman_corrs)

# %% [markdown]
# # Preprocessing Steps

# %% [markdown]
# ## Train-Test Split

# %% [markdown]
# Before any encoding and feature selection steps we'll start by defining x and y, and defining the train–test split. Doing this at this stage is critical to avoid data leakage.

# %%
# Separate features and target
X = data.drop('Attrition', axis=1).copy()
y = data['Attrition'].copy()

# Train–test split (20% test, stratified by target)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,      # keep same attrition proportion in train/test; very important given class imbalance
    shuffle=True     
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# %%
# IncomeVsRoleMedian (MonthlyIncome relative to median for JobRole)

# Compute medians by JobRole on TRAIN ONLY
role_medians = X_train.groupby("JobRole")["MonthlyIncome"].median()

# Map to train and test
X_train["IncomeVsRoleMedian"] = (
    X_train["MonthlyIncome"] / X_train["JobRole"].map(role_medians)
)

X_test["IncomeVsRoleMedian"] = (
    X_test["MonthlyIncome"] / X_test["JobRole"].map(role_medians)
)

# %%
# Save for deployment
role_medians_dict = role_medians.to_dict()

import json
with open("role_medians.json", "w") as f:
    json.dump(role_medians_dict, f)

# %% [markdown]
# ## Rebuilding feature groups on X_train

# %%
# Categorical features inferred from dtype 'object'
categorical_features = list(
    X_train.select_dtypes(include='object').columns.drop(['BusinessTravel'])
)

# Binary features (defined them manually above)
binary_features = ['Gender', 'OverTime']

# Ordinal features (predefined list above)
ordinal_features = [
    'BusinessTravel','Education','EnvironmentSatisfaction','JobInvolvement',
    'JobLevel','JobSatisfaction','PerformanceRating',
    'RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance'
]

# Non-continuous = categorical + binary + ordinal
non_continuous_features = categorical_features + binary_features + ordinal_features

# Continuous = everything else except the target (including engineered features)
continuous_features = list(
    X_train.columns.difference(non_continuous_features)
)

# %%
print("Categorical nominal:", categorical_features)
print("Binary:", binary_features)
print("Ordinal:", ordinal_features)
print("Continuous (incl. engineered):", continuous_features)

# %% [markdown]
# ## Defining the preprocessing (encoders + passthrough)

# %%
# Split ordinal features: BusinessTravel vs the rest 
ordinal_bt = ["BusinessTravel"]
ordinal_other = [f for f in ordinal_features if f != "BusinessTravel"]

# Ordinal encoder for BusinessTravel with meaningful order
bt_categories = [["Non-Travel", "Travel_Rarely", "Travel_Frequently"]]

bt_ordinal_transformer = OrdinalEncoder(
    categories=bt_categories,
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

# Ordinal encoder for all other ordinal features (numeric scales 1–4/5 etc.)
other_ordinal_transformer = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

# One-hot encoder for nominal + binary
onehot_transformer = OneHotEncoder(
    drop=None,
    handle_unknown="ignore",
    sparse_output=False,
)

# ColumnTransformer tying everything together
preprocess = ColumnTransformer(
    transformers=[
        # BusinessTravel with explicit order: Non-Travel < Rarely < Frequently
        ("ord_bt", bt_ordinal_transformer, ordinal_bt),

        # Other ordinal features (Education, JobSatisfaction, etc.)
        ("ord", other_ordinal_transformer, ordinal_other),

        # Nominal + binary features
        ("nom", onehot_transformer, categorical_features + binary_features),

        # Continuous (including engineered ones like TenureIndex, IncomeVsRoleMedian)
        ("num", "passthrough", continuous_features),
    ]
)



# %% [markdown]
# When preprocess.fit(X_train) is called, it learns: category mappings for ordinal features; dummy columns for nominal + binary. Then preprocess.transform(...) will apply this same mapping to train & test.

# %% [markdown]
# ## Combining preprocessing + scaling into a Pipeline

# %%
pipeline_preprocess = Pipeline([
    ('preprocess', preprocess),
    ('scale', StandardScaler())
])


# %% [markdown]
# ## Fitting preprocessing only on training data and transform both sets

# %%
# Fit on training data only (no leakage)
pipeline_preprocess.fit(X_train)

# Transform train and test with the same fitted pipeline
X_train_processed = pipeline_preprocess.transform(X_train)
X_test_processed  = pipeline_preprocess.transform(X_test)



# %%
print("X_train_processed shape:", X_train_processed.shape)
print("X_test_processed shape:",  X_test_processed.shape)

#print("Number of feature names:", len(feature_names))

# %%
# Grab the ColumnTransformer from the pipeline
ct = pipeline_preprocess.named_steps['preprocess']

# Ask it for the output feature names
feature_names = ct.get_feature_names_out()

# Now rebuild the DataFrames
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
X_test_df  = pd.DataFrame(X_test_processed,  columns=feature_names, index=X_test.index)


# %%
X_train_df.shape


# %%
X_test_df.shape


# %% [markdown]
# # Feature Selection

# %% [markdown]
# ## Chi-square

# %% [markdown]
# The chi-square test is a statistical method used to assess whether there is a significant association between categorical features and a categorical target variable. In the context of feature selection, it allows us to quantify how strongly each feature is related to the outcome by comparing observed frequencies with those expected under independence. Features that show a higher chi-square statistic are considered more informative because they deviate more from what would be expected if there were no relationship with the target.

# %%
def chi_square_for_feature(X_col, y):
    """Return chi2 and p-value for one categorical feature vs target."""
    table_observed = pd.crosstab(y, X_col)
    chi2, pvalue, dof, expected = stats.chi2_contingency(table_observed.values)
    return chi2, pvalue


def chi_square_for_features(X_train, y_train, alpha=0.05):
    """Run chi-square for each column in X_train and return a summary DataFrame."""
    results = []

    for var in X_train.columns:
        chi2, pvalue = chi_square_for_feature(X_train[var], y_train)
        results.append({
            "feature": var,
            "chi2": chi2,
            "p_value": pvalue,
            "significant": pvalue < alpha
        })

    results_df = pd.DataFrame(results)
    # Sort by p-value (smallest p-value = strongest evidence of association)
    results_df = results_df.sort_values("p_value")

    return results_df


# %%
chi2_results = chi_square_for_features(
    X_train[non_continuous_features],
    y_train,
    alpha=0.05
)

chi2_results


# %% [markdown]
# ## Mutual Information

# %% [markdown]
# Mutual information is a measure that quantifies how much knowing the value of one variable reduces uncertainty about another. Unlike traditional correlation measures, it can capture both linear and nonlinear relationships, making it particularly useful in feature selection. When applied to a feature and a target variable, a higher mutual information score indicates that the feature provides more predictive signal about the target. This allows mutual information to identify informative features even when their relationships with the outcome are complex or non-monotonic, offering a flexible and powerful method for selecting variables in a machine-learning pipeline.

# %%
def compute_mutual_information_from_ct(ct, X_train, y_train):
    """
    Compute mutual information between each encoded feature and the target,
    using a fitted ColumnTransformer 'ct' (without scaling).
    Returns a sorted DataFrame with MI scores.
    """

    # 1. Transform X_train with the fitted ColumnTransformer (no scaling)
    X_train_enc = ct.transform(X_train)
    feature_names = ct.get_feature_names_out()

    X_train_enc_df = pd.DataFrame(
        X_train_enc,
        columns=feature_names,
        index=X_train.index
    )

    # 2. Build a mask of which features are discrete
    col_series = X_train_enc_df.columns.to_series()
    discrete_mask = col_series.str.startswith(('ord__', 'nom__'))

    # 3. Compute mutual information
    mi_scores = mutual_info_classif(
        X_train_enc_df,
        y_train,
        discrete_features=discrete_mask.values,
        random_state=42
    )

    # 4. Build results DataFrame
    mi_df = pd.DataFrame({
        'Feature': X_train_enc_df.columns,
        'MI': mi_scores,
        'Discrete': discrete_mask.values
    })

    mi_df.sort_values('MI', ascending=False, inplace=True)
    mi_df.reset_index(drop=True, inplace=True)

    return mi_df




# %%
# Get the fitted ColumnTransformer from the pipeline
ct = pipeline_preprocess.named_steps['preprocess']

mi_results = compute_mutual_information_from_ct(ct, X_train, y_train)
display(mi_results.head(20))


# %% [markdown]
# ## L1 Logistic Regression (LASSO)

# %% [markdown]
# L1-regularized Logistic Regression, commonly known as LASSO, is a linear classification model that incorporates an L1 penalty to encourage sparsity in its coefficients. During training, this penalty pushes the weights of less informative features toward zero, effectively removing them from the model. This makes LASSO a powerful embedded feature-selection method.

# %%
def select_with_l1_logistic(X_train_df, y_train, C=1.0):
    """
    Run L1-penalized Logistic Regression to select features.
    Returns a DataFrame with coefficients and selection mask.
    """
    # L1 logistic regression with class balancing (important for attrition)
    l1_model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        class_weight='balanced',
        C=C,
        random_state=42
    )

    selector = SelectFromModel(l1_model, prefit=False)
    selector.fit(X_train_df, y_train)

    # Selected mask
    mask = selector.get_support()

    # Coefficients
    coefs = selector.estimator_.coef_[0]

    # Build results table
    results = pd.DataFrame({
        "Feature": X_train_df.columns,
        "Coefficient": coefs,
        "Selected": mask
    })

    # Absolute magnitude for sorting
    results["AbsCoef"] = results["Coefficient"].abs()
    results = results.sort_values("AbsCoef", ascending=False)

    return results, mask

l1_results, l1_mask = select_with_l1_logistic(X_train_df, y_train)
display(l1_results.head(20))


# %% [markdown]
#  Features with Selected = True are part of the sparse LASSO-selected subset. Larger coefficients (in magnitude) reflect stronger linear effect.

# %% [markdown]
# ## Random Forest Classifier

# %% [markdown]
# For feature selection, Random Forest provides a natural measure of importance by evaluating how much each feature contributes to reducing impurity across the ensemble’s trees. Features that consistently yield larger decreases in impurity are considered more influential, while weaker predictors receive lower importance scores. Because Random Forest handles interactions, nonlinear effects, and mixed data types with ease, it serves as a powerful and versatile tool for identifying the most informative features in a dataset.

# %%
def select_with_random_forest(X_train_df, y_train, n_estimators=500):
    """
    Train a Random Forest and return a DataFrame with feature importances.
    """

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight='balanced',
        max_depth=None,
        n_jobs=-1
    )

    rf.fit(X_train_df, y_train)

    importances = rf.feature_importances_

    results = pd.DataFrame({
        "Feature": X_train_df.columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    return results, rf

rf_results, rf_model = select_with_random_forest(X_train_df, y_train)
display(rf_results.head(20))


# %% [markdown]
# ## XGBoost Feature Importance

# %% [markdown]
# XGBoost provides built-in feature importance metrics that quantify how much each variable contributes to the model’s predictive performance. Features with higher importance values are those that consistently help the model make better decisions. Because XGBoost captures complex interactions and nonlinear patterns, its feature importance offers a powerful, model-driven way to identify the most influential predictors in the dataset. XGBoost is often very strong at discovering: threshold effects, feature interactions, nonlinear jump patterns, sparse informative features.

# %%
def select_with_xgboost(X_train_df, y_train):
    """
    Train XGBoost and return feature importances.
    """

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    xgb.fit(X_train_df, y_train)

    importances = xgb.feature_importances_

    results = pd.DataFrame({
        "Feature": X_train_df.columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    return results, xgb

xgb_results, xgb_model = select_with_xgboost(X_train_df, y_train)
display(xgb_results.head(20))



# %% [markdown]
# ## Table Combining Feature Selection Results

# %%
#S tarting from all encoded features

# Base table: one row per encoded feature
unified_fs = pd.DataFrame({
    "Feature": X_train_df.columns
})


# Merging MI results

# Keep only needed columns from MI
mi_short = mi_results[["Feature", "MI"]]

unified_fs = unified_fs.merge(
    mi_short,
    on="Feature",
    how="left")

#Merging L1 Logistic Regression results

l1_short = l1_results[["Feature", "Coefficient", "Selected"]]

unified_fs = unified_fs.merge(
    l1_short,
    on="Feature",
    how="left")

# Merging Random Forest results

rf_short = rf_results.rename(columns={"Importance": "RF_importance"})[
    ["Feature", "RF_importance"]
]

unified_fs = unified_fs.merge(
    rf_short,
    on="Feature",
    how="left"
)

# Merging XGBoost results

xgb_short = xgb_results.rename(columns={"Importance": "XGB_importance"})[
    ["Feature", "XGB_importance"]
]

unified_fs = unified_fs.merge(
    xgb_short,
    on="Feature",
    how="left"
)


# %%
# Mapping encoded features back to their raw feature name

def get_raw_feature_name(encoded_feature):
    """
    Map an encoded feature name (ord__/nom__/num__) back to the original column name.

    Examples:
      'ord__JobLevel'                      -> 'JobLevel'
      'nom__BusinessTravel_Travel_Rarely'  -> 'BusinessTravel'
      'nom__MaritalStatus_Single'          -> 'MaritalStatus'
      'num__Age'                           -> 'Age'
    """
    if encoded_feature.startswith("num__") or encoded_feature.startswith("ord__"):
        # pattern: 'num__Age' or 'ord__JobLevel'
        return encoded_feature.split("__", 1)[1]

    if encoded_feature.startswith("nom__"):
        # pattern: 'nom__Column_Category_With_Underscores'
        tmp = encoded_feature.split("__", 1)[1]  # 'Column_Category...'
        raw_col = tmp.split("_", 1)[0]           # take part before first '_'
        return raw_col

    # For any unexpected feature, return None
    return None

unified_fs["raw_feature"] = unified_fs["Feature"].apply(get_raw_feature_name)


# %%
# Merging Chi-square results by raw feature

chi2_short = chi2_results.rename(columns={
    "feature": "raw_feature",
    "chi2": "chi2_stat",
    "p_value": "chi2_pvalue",
    "significant": "chi2_significant"
})[["raw_feature", "chi2_stat", "chi2_pvalue", "chi2_significant"]]

unified_fs = unified_fs.merge(
    chi2_short,
    on="raw_feature",
    how="left"
)


# %%
# Flagging whether feature is discrete (ordinal or one-hot)

unified_fs["is_discrete"] = unified_fs["Feature"].str.startswith(("ord__", "nom__"))

# Possible sorting: by Random Forest importance (descending)
unified_fs_sorted = unified_fs.sort_values(
    by=["RF_importance", "XGB_importance", "MI"],
    ascending=False
)

unified_fs_sorted.head(30)


# %% [markdown]
# ## Finding which variables are consistently selected by the different feature selection methods

# %%
df = unified_fs.copy()  # keeping the original safe

# Fill NaNs with 0 where it makes sense 
df["MI"] = df["MI"].fillna(0)
df["RF_importance"] = df["RF_importance"].fillna(0)
df["XGB_importance"] = df["XGB_importance"].fillna(0)

# chi2_significant may be NaN for numeric features; treat those as False
df["chi2_significant"] = df["chi2_significant"].fillna(False)

# L1 Selected may be NaN for some features; treat as False
df["Selected"] = df["Selected"].fillna(False)


# %% [markdown]
# Establishing dynamic thresholds (quantile-based)

# %%
# Helper to get a quantile threshold, but avoid NaNs / all-zeros issues
def safe_quantile(series, q, default=0.0):
    vals = series.dropna()
    if (vals > 0).sum() == 0:
        return default
    return vals.quantile(q)

# Top 30% for MI, RF, XGB
mi_thresh  = safe_quantile(df["MI"],             0.70, default=0.0)
rf_thresh  = safe_quantile(df["RF_importance"],  0.70, default=0.0)
xgb_thresh = safe_quantile(df["XGB_importance"], 0.70, default=0.0)

print("MI threshold:", mi_thresh)
print("RF threshold:", rf_thresh)
print("XGB threshold:", xgb_thresh)


# %% [markdown]
# Defining binary flags

# %%
# Chi-square (categoricals only) – already a boolean
df["chi2_good"] = df["chi2_significant"].astype(bool)

# Mutual Information – above quantile threshold
df["mi_good"] = (df["MI"] >= mi_thresh)

# L1 Logistic Regression – already boolean
df["l1_good"] = df["Selected"].astype(bool)

# Random Forest – above quantile threshold
df["rf_good"] = (df["RF_importance"] >= rf_thresh)

# XGBoost – above quantile threshold
df["xgb_good"] = (df["XGB_importance"] >= xgb_thresh)


# %% [markdown]
#  Building the consensus score

# %%
method_flags = ["chi2_good", "mi_good", "l1_good", "rf_good", "xgb_good"]

# Converting to int and sum
df["consensus_score"] = df[method_flags].astype(int).sum(axis=1)

# Checking distribution
print(df["consensus_score"].value_counts().sort_index())


# %% [markdown]
# Defining Feature Set A (strict) and Feature Set B (moderate)

# %%
# Strict: features that get at least 3 "votes"
strict_mask = df["consensus_score"] >= 3

# Moderate: at least 2 votes
moderate_mask = df["consensus_score"] >= 2

# Feature names (encoded, ready for modelling)
features_strict   = df.loc[strict_mask,   "Feature"].tolist()
features_moderate = df.loc[moderate_mask, "Feature"].tolist()

print("Number of features in strict set:", len(features_strict))
print("Number of features in moderate set:", len(features_moderate))


# %%
df_consensus_view = df[[
    "Feature",
    "raw_feature",
    "MI",
    "Coefficient",
    "RF_importance",
    "XGB_importance",
    "chi2_stat",
    "chi2_pvalue",
    "chi2_good",
    "mi_good",
    "l1_good",
    "rf_good",
    "xgb_good",
    "consensus_score"
]].sort_values("consensus_score", ascending=False)

df_consensus_view.head(30)


# %%
# Strict set of features

for i, f in enumerate(features_strict, 1):
    print(f"{i}. {f}")


# %% [markdown]
# This strict set leans heavily toward a few thematic clusters:
# 
# - Career progression & seniority (Age, TotalWorkingYears, YearsAtCompany, YearsWithCurrManager, JobLevel)
# 
# - Income & compensation (MonthlyIncome, Income_Rate_Ratio, StockOptionLevel)
# 
# - Job role (4 JobRole dummies)
# 
# - Marital status (Single / Divorced)
# 
# - OverTime
# 
# - EngagementIndex
# 
# - BusinessTravel
# 
# These are well known drivers of attrition.

# %%
# Modearate set of features

for i, f in enumerate(features_moderate, 1):
    print(f"{i}. {f}")


# %% [markdown]
# The moderate feature set as expected is more comprehensive and could be particularly useful for tree-based models. Of note, all engineered features are included in this set.

# %% [markdown]
#  # Modelling

# %% [markdown]
# The following code defines two custom scikit-learn transformers to improve workflow when working with feature pipelines. PreprocessToDF converts the output of a ColumnTransformer into a pandas DataFrame with properly named columns, preserving indices and making downstream analysis much easier. ColumnSelectorByName then allows selecting a subset of these encoded features by their column names, ensuring that only features present in the current fold or dataset are retained. Together, these components enable clean, name-aware preprocessing and flexible feature selection within scikit-learn pipelines. 

# %%
from sklearn.base import BaseEstimator, TransformerMixin, clone

# Turn a ColumnTransformer into a DataFrame with named columns
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


# Select a subset of encoded features by name
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




# %% [markdown]
# With the following function we perform a leak-free evaluation of a model using a fully integrated scikit-learn pipeline. It first builds a pipeline that applies the provided preprocessing (pipeline_preprocess’s ColumnTransformer), converts the result to a DataFrame, selects only the desired encoded features (feature_list), scales them, and then fits the specified model. Using StratifiedKFold, it runs cross-validation directly on the raw training data, ensuring that all preprocessing and feature selection are done inside each fold to avoid data leakage. After computing average CV metrics (accuracy, precision, recall, F1), it refits the pipeline on the full training set and evaluates it once on the untouched test set. The function returns both the CV and test metrics, along with the fully fitted pipeline for later use.

# %%
def evaluate_model_preprocessed(
    model,
    feature_list,
    X_train, y_train,
    X_test, y_test,
    pipeline_preprocess,
    n_splits=5
):
    """
    Leak-free evaluation:

    - Builds Pipeline: preprocess → DataFrame → select(encoded cols) → scale → model
    - Runs StratifiedKFold CV on RAW X_train (preprocessing inside CV)
    - Fits final model on full X_train
    - Tests once on untouched X_test
    """

    # Clone unfitted ColumnTransformer from your original pipeline
    ct = pipeline_preprocess.named_steps["preprocess"]
    ct = clone(ct)

    # Build full modeling pipeline
    base_pipe = Pipeline([
        ("preprocess", PreprocessToDF(ct)),
        ("select",    ColumnSelectorByName(feature_list)),
        ("scale",     StandardScaler()),
        ("model",     clone(model)),
    ])

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_acc, cv_prec, cv_rec, cv_f1 = [], [], [], []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        pipe_fold = clone(base_pipe)
        pipe_fold.fit(X_tr, y_tr)
        preds = pipe_fold.predict(X_val)

        cv_acc.append(accuracy_score(y_val, preds))
        cv_prec.append(precision_score(y_val, preds))
        cv_rec.append(recall_score(y_val, preds))
        cv_f1.append(f1_score(y_val, preds))

    # Final fit on full training set
    final_pipe = clone(base_pipe)
    final_pipe.fit(X_train, y_train)

    # Test evaluation (still untouched)
    test_preds = final_pipe.predict(X_test)

    results = {
        "cv_accuracy":    np.mean(cv_acc),
        "cv_precision":   np.mean(cv_prec),
        "cv_recall":      np.mean(cv_rec),
        "cv_f1":          np.mean(cv_f1),
        "test_accuracy":  accuracy_score(y_test, test_preds),
        "test_precision": precision_score(y_test, test_preds),
        "test_recall":    recall_score(y_test, test_preds),
        "test_f1":        f1_score(y_test, test_preds),
        "fitted_pipeline": final_pipe,
    }

    return results



# %% [markdown]
# Model factory functions: each call returns a FRESH instance 

# %%
def make_lr():
    return LogisticRegression(
        class_weight='balanced',
        max_iter=500,
        solver='lbfgs',
        random_state=42
    )

def make_rf():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

def make_xgb():
    # handle imbalance with scale_pos_weight
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / pos
    
    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )

def make_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(32,),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=42
    )

def make_svc():
    return SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,         # in case later want predict_proba
        class_weight='balanced',
        random_state=42
    )

def make_lgbm():
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

def make_cat():
    return CatBoostClassifier(
        iterations=400,
        depth=4,
        learning_rate=0.1,
        loss_function='Logloss',
        eval_metric='F1',
        auto_class_weights='Balanced',
        random_state=42,
        verbose=False
    )


# %%
model_factories = {
    "LR":   make_lr,
    "RF":   make_rf,
    "XGB":  make_xgb,
    "MLP":  make_mlp,
    "SVC":  make_svc,
    "LGBM": make_lgbm,
    "CAT":  make_cat
}

feature_sets = {
    "strict":   features_strict,
    "moderate": features_moderate
}



# %% [markdown]
# Using the code below we run a systematic baseline evaluation across multiple models and feature sets. For each model–feature-set combination, we create a fresh model instance, evaluate it using the leak-free evaluate_model_preprocessed pipeline, and store both cross-validation and test-set metrics. All results are assembled into a DataFrame making it easy to compare model performance across different preprocessing and feature-selection configurations.

# %%
baseline_results = []

for model_name, factory in model_factories.items():
    for fs_name, fs_list in feature_sets.items():
        print(f"Running {model_name} with {fs_name} features...")
        
        model = factory()  # FRESH instance each time
        
        res = evaluate_model_preprocessed(
            model=model,
            feature_list=fs_list,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            pipeline_preprocess=pipeline_preprocess
        )
        
        # add identifiers
        res["model"] = model_name
        res["feature_set"] = fs_name
        
        baseline_results.append(res)

# Convert to DataFrame for nice viewing
results_df = pd.DataFrame(baseline_results)

# Put columns in a convenient order
cols_order = [
    "model", "feature_set",
    "cv_accuracy", "cv_precision", "cv_recall", "cv_f1",
    "test_accuracy", "test_precision", "test_recall", "test_f1"
]
results_df = results_df[cols_order]

results_df


# %% [markdown]
# Based on the baseline results, XGBoost paired with the moderate feature set provides one of the most favourable trade-offs across all evaluation metrics. This makes it our best candidate for further optimisation, so we will focus our hyperparameter tuning (using GridSearch and Optuna) on this configuration.

# %% [markdown]
# ## XGBoost Hyperparameter Tuning Using Grid Search

# %% [markdown]
# We'll reuse our evaluate_model_preprocessed while looping over a small grid to assess:
# 
# - which max_depth generally works best
# 
# - whether smaller/larger learning_rate helps
# 
# - whether more trees improve things
# 
# - whether subsampling is beneficial

# %%
# helper to compute imbalance weight

def compute_spw(y):
    pos = y.sum()
    neg = len(y) - pos
    return neg / pos

spw = compute_spw(y_train)

param_grid = {
    "max_depth":      [3, 4, 5],
    "learning_rate":  [0.05, 0.1],
    "n_estimators":   [200, 400],
    "subsample":      [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

grid_results = []

for max_depth, lr, n_est, subs, colsub in product(
    param_grid["max_depth"],
    param_grid["learning_rate"],
    param_grid["n_estimators"],
    param_grid["subsample"],
    param_grid["colsample_bytree"],
):
    print(f"Testing: depth={max_depth}, lr={lr}, n_est={n_est}, subs={subs}, col={colsub}")

    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=lr,
        n_estimators=n_est,
        subsample=subs,
        colsample_bytree=colsub,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",  # faster if available
    )

    res = evaluate_model_preprocessed(
        model=model,
        feature_list=features_moderate,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        pipeline_preprocess=pipeline_preprocess
    )

    res["max_depth"] = max_depth
    res["learning_rate"] = lr
    res["n_estimators"] = n_est
    res["subsample"] = subs
    res["colsample_bytree"] = colsub

    grid_results.append(res)

grid_df = pd.DataFrame(grid_results)

# sort by CV F1 (primary) then test F1 (secondary)
grid_df_sorted = grid_df.sort_values(
    by=["cv_f1", "test_f1"], ascending=False
).reset_index(drop=True)

grid_df_sorted.head(10)

# %%
grid_best_results = grid_df_sorted.iloc[0]


# %% [markdown]
# By performing grid search we obtained modest but consistent improvements over the baseline XGBoost model in cross-validation performance, particularly in F1. These results justify trying Optuna in order to explore the hyperparameter space more efficiently.

# %% [markdown]
# ## XGBoost Hyperparameter Tuning Using Optuna

# %% [markdown]
# Optuna is a powerful framework for automated hyperparameter optimization. It uses intelligent search strategies to efficiently explore the hyperparameter space and identify high-performing configurations with far fewer trials than traditional grid or random search. Optuna’s flexibility allows it to handle complex search spaces, conditional parameters, and custom objective functions, making it especially useful for tuning models like XGBoost. 

# %% [markdown]
# We define our Optuna search space based on the information we obtained from the (coarse) grid search above

# %%
# Best row from grid search (already sorted by cv_f1 desc)

best_row = grid_df_sorted.iloc[0]

best_md   = int(best_row["max_depth"])
best_lr   = float(best_row["learning_rate"])
best_ne   = int(best_row["n_estimators"])
best_sub  = float(best_row["subsample"])
best_col  = float(best_row["colsample_bytree"])

print("Best grid params (CV):")
print("max_depth:", best_md)
print("learning_rate:", best_lr)
print("n_estimators:", best_ne)
print("subsample:", best_sub)
print("colsample_bytree:", best_col)

# Helper to keep ranges reasonable
def clip_int(low, high, min_val, max_val):
    return max(min_val, low), min(max_val, high)

def clip_float(low, high, min_val, max_val):
    return max(min_val, low), min(max_val, high)

# Ranges centred around grid best (can tweak)

md_low, md_high     = clip_int(best_md - 1, best_md + 1, 2, 8)
ne_low, ne_high     = clip_int(best_ne - 100, best_ne + 200, 100, 800)
sub_low, sub_high   = clip_float(best_sub - 0.2, best_sub + 0.2, 0.5, 1.0)
col_low, col_high   = clip_float(best_col - 0.2, best_col + 0.2, 0.5, 1.0)

lr_low  = max(0.01, best_lr / 3)
lr_high = min(0.3,  best_lr * 3)

print("\nOptuna search ranges:")
print("max_depth:", (md_low, md_high))
print("learning_rate:", (lr_low, lr_high))
print("n_estimators:", (ne_low, ne_high))
print("subsample:", (sub_low, sub_high))
print("colsample_bytree:", (col_low, col_high))



# %% [markdown]
# This code defines an Optuna objective function that tunes key XGBoost hyperparametersOptuna objective + study (optimising CV F1)

# %%
def xgb_objective(trial):

    max_depth = trial.suggest_int("max_depth", md_low, md_high)
    learning_rate = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)
    n_estimators = trial.suggest_int("n_estimators", ne_low, ne_high)
    subsample = trial.suggest_float("subsample", sub_low, sub_high)
    colsample_bytree = trial.suggest_float("colsample_bytree", col_low, col_high)

    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=spw,          # your imbalance weight
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )

    res = evaluate_model_preprocessed(
        model=model,
        feature_list=features_moderate,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pipeline_preprocess=pipeline_preprocess,
        n_splits=5
    )

    # We optimise CV F1 (NOT test F1)
    return res["cv_f1"]


study = optuna.create_study(
    direction="maximize",
    study_name="xgb_moderate_optuna",
    sampler=TPESampler(seed=42),
)

study.optimize(xgb_objective, n_trials=40, show_progress_bar=True)

print("Number of finished trials:", len(study.trials))
print("Best trial CV F1:", study.best_value)
print("Best trial params:", study.best_params)



# %% [markdown]
#  Train + evaluate final model with best Optuna params

# %% [markdown]
#  Now we build a fresh XGB model with the best Optuna params, run it through the same evaluation function, and inspect both CV + test metrics.

# %%
best_params = study.best_params
print("\nBest Optuna params:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

best_xgb = XGBClassifier(
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    n_estimators=best_params["n_estimators"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    scale_pos_weight=spw,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist",
)

optuna_results = evaluate_model_preprocessed(
    model=best_xgb,
    feature_list=features_moderate,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    pipeline_preprocess=pipeline_preprocess,
    n_splits=5
)

print("\nFinal XGBoost (Optuna) performance:")
for k, v in optuna_results.items():
    if k != "fitted_pipeline":   # don't spam with the whole pipeline object
        print(f"{k}: {v}")



# %% [markdown]
#  Model Comparison Table

# %%
def summarize_model(name, feature_set, row_or_results):
    """row_or_results can be a pandas row (baseline) or the optuna_results dict."""
    if isinstance(row_or_results, dict):
        # optuna_results style
        return {
            "model": name,
            "feature_set": feature_set,
            "cv_accuracy":  row_or_results["cv_accuracy"],
            "cv_precision": row_or_results["cv_precision"],
            "cv_recall":    row_or_results["cv_recall"],
            "cv_f1":        row_or_results["cv_f1"],
            "test_accuracy":  row_or_results["test_accuracy"],
            "test_precision": row_or_results["test_precision"],
            "test_recall":    row_or_results["test_recall"],
            "test_f1":        row_or_results["test_f1"],
        }
    else:
        # pandas Series from results_df
        r = row_or_results
        return {
            "model": name,
            "feature_set": feature_set,
            "cv_accuracy":  r["cv_accuracy"],
            "cv_precision": r["cv_precision"],
            "cv_recall":    r["cv_recall"],
            "cv_f1":        r["cv_f1"],
            "test_accuracy":  r["test_accuracy"],
            "test_precision": r["test_precision"],
            "test_recall":    r["test_recall"],
            "test_f1":        r["test_f1"],
        }



# %%
# Baseline XGB with moderate features
xgb_base_mod = results_df.query("model == 'XGB' and feature_set == 'moderate'").iloc[0]


# %%
rows = []

rows.append(summarize_model("XGB_baseline", "moderate", xgb_base_mod))
rows.append(summarize_model("XGB_GridBest", "moderate", grid_best_results))
rows.append(summarize_model("XGB_Optuna", "moderate", optuna_results))

comparison_df = pd.DataFrame(rows)
comparison_df



# %% [markdown]
# Tuning with Optuna didn’t beat the baseline - test_f1 Optuna 0.44 vs test_f1 base model 0.49
# 
# After tuning XGBoost with 2 different approaches, we conclude that the baseline configuration was already near the performance ceiling for this dataset; additional tuning brought only marginal changes in test F1.

# %% [markdown]
# ## XGBoost Model Evaluation

# %% [markdown]
# For model evaluation we'll use our XGB base model.

# %%
# Fit BASELINE model using the leak-free evaluator

baseline_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,   # imbalance weight computed earlier
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist",
)

baseline_results = evaluate_model_preprocessed(
    model=baseline_xgb,
    feature_list=features_moderate,   
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    pipeline_preprocess=pipeline_preprocess,
    n_splits=5
)

print("Baseline XGB (moderate) CV/Test metrics:")
for k, v in baseline_results.items():
    if k != "fitted_pipeline":
        print(f"{k}: {v}")

# grab the fitted full pipeline

baseline_pipe = baseline_results["fitted_pipeline"]


# Get predicted probabilities on the TEST set

y_test_proba = baseline_pipe.predict_proba(X_test)[:, 1]



# %% [markdown]
# ### ROC Curve

# %%
# ROC curve + AUC

fpr, tpr, roc_thresh = roc_curve(y_test, y_test_proba)
roc_auc = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.7)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Baseline XGBoost (moderate)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %% [markdown]
# The XGBoost model achieves a ROC-AUC of 0.761, indicating good ability to discriminate between employees who leave and those who stay. The curve rises steeply at low false-positive rates, showing that the model correctly identifies many true quitters before confusing them with non-quitters.
# 
#  However, ROC-AUC evaluates performance across all possible thresholds and is insensitive to class imbalance. In an attrition scenario where only ~15% of employees leave, a model can obtain a relatively high ROC-AUC even when the practical precision and recall trade-off is challenging.
# 
# Therefore, while the ROC curve confirms that the model produces a meaningful ranking of risk scores, it must be interpreted alongside the Precision–Recall curve and F1 score to fully understand real-world classification performance under imbalance.

# %% [markdown]
# ### Precision-Recall Curve

# %%
# Precision–Recall curve + AP

prec, rec, pr_thresh = precision_recall_curve(y_test, y_test_proba)
ap = average_precision_score(y_test, y_test_proba)

plt.figure(figsize=(6, 5))
plt.plot(rec, prec, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Baseline XGBoost (moderate)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %% [markdown]
# The Precision–Recall curve provides a more realistic evaluation of model performance under class imbalance than ROC-AUC. The model achieves an Average Precision (AP) of 0.5, substantially higher than the baseline positive rate of ~0.15, confirming that it captures meaningful signal related to employee attrition. Precision is extremely high at low recall (0.85–1.00), indicating that the model is very confident in identifying the top-risk employees. As recall increases, precision decreases, reflecting a typical precision–recall tradeoff in imbalanced datasets, where achieving high recall requires accepting more false positives. These results align with the observed F1 scores (0.45–0.49), confirming that the model captures meaningful signal but cannot achieve high precision and recall simultaneously.
# Although the model ranks employees well (as shown by the ROC curve), achieving strong precision and recall simultaneously is challenging due to the underlying class imbalance. The PR curve therefore offers a realistic assessment of actionable performance and complements the ROC curve by showing where the model can be most effectively used.

# %% [markdown]
# ### Threshold vs F1 on the Test Set

# %%
# Threshold vs F1 on the TEST set (diagnostic)

thresholds = np.linspace(0.05, 0.95, 181)  # step = 0.005
f1_scores = []

for t in thresholds:
    preds_t = (y_test_proba >= t).astype(int)
    f1_scores.append(f1_score(y_test, preds_t))

f1_scores = np.array(f1_scores)
best_idx = np.argmax(f1_scores)
best_t = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

plt.figure(figsize=(7, 5))
plt.plot(thresholds, f1_scores, label="F1 score")
plt.axvline(0.5, color="red", linestyle="--", label="Default 0.5")
plt.axvline(best_t, color="green", linestyle="--",
            label=f"Best on test: t={best_t:.3f}, F1={best_f1:.3f}")
plt.xlabel("Threshold")
plt.ylabel("F1 score")
plt.title("Threshold vs F1 – Baseline XGBoost (moderate)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.show()





# %% [markdown]
# The Threshold–F1 curve illustrates how the model’s F1 score changes as the decision threshold varies between 0 and 1. The model achieves its highest F1 values—peaking just above 0.50, which is when the threshold is in the 0.53 to 0.54 range. This means that, on the test set, increasing the threshold just a notch above the default 0.50 improves the balance between precision and recall.
# 
# However, this peak reflects test-set behavior and should not be used for model selection, as tuning a threshold on test data introduces data leakage.
# 
# Overall, the curve highlights how sensitive F1 performance is to threshold choice, especially in imbalanced datasets, and reinforces the importance of selecting thresholds based on training-only validation procedures rather than test-set optimization.

# %% [markdown]
# ## XGBoost Threshold Optimization

# %% [markdown]
# The Threshold–F1 curve above shows that the model’s performance varies substantially across different probability cutoffs. Although the default threshold of 0.50 is commonly used, it is not the point that maximizes the F1 score on this dataset. This variation suggests that adjusting the classification threshold could meaningfully improve the balance between precision and recall.
# 
# Therefore, in the next section, we perform a dedicated threshold optimization procedure using using validation performed only on the training data (avoiding leakage) in an attempt identify a more effective decision threshold for this model.

# %%
# Helper function to find best threshold on one set

def find_best_threshold(y_true, y_proba, metric=f1_score, thresholds=None):
    """
    Given true labels and predicted probabilities, find the threshold
    that maximizes the chosen metric (default: F1).
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)  # step = 0.005

    scores = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        scores.append(metric(y_true, preds))

    scores = np.array(scores)
    best_idx = np.argmax(scores)
    return float(thresholds[best_idx]), float(scores[best_idx])



# %%
# Main function to evaluate model with CV threshold tuning

def evaluate_model_with_threshold_cv(
    model,
    feature_list,
    X_train, y_train,
    X_test, y_test,
    pipeline_preprocess,
    n_splits=5,
    metric=f1_score
):
    """
    Leak-free evaluation with threshold optimization via cross-validation.

    - Pipeline: preprocess -> DataFrame -> select(encoded cols) -> scale -> model
    - For each CV fold:
        * fit pipeline on fold's train
        * get predict_proba on fold's val
        * find best threshold on that fold (for chosen metric, default F1)
    - Global threshold T* = mean of per-fold best thresholds
    - Final pipeline fit on full X_train
    - Test metrics computed using T* on predict_proba(X_test)
    """

    # Clone unfitted ColumnTransformer from your pipeline_preprocess
    ct = pipeline_preprocess.named_steps["preprocess"]
    ct = clone(ct)

    # Base pipeline (unfitted)
    base_pipe = Pipeline([
        ("preprocess", PreprocessToDF(ct)),
        ("select",    ColumnSelectorByName(feature_list)),
        ("scale",     StandardScaler()),
        ("model",     clone(model)),
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_thresholds = []
    cv_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    # CV loop with threshold tuning 
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        pipe_fold = clone(base_pipe)
        pipe_fold.fit(X_tr, y_tr)

        # probabilities on validation fold
        y_val_proba = pipe_fold.predict_proba(X_val)[:, 1]

        # best threshold on this fold
        t_fold, _ = find_best_threshold(y_val, y_val_proba, metric=metric)
        fold_thresholds.append(t_fold)

        # apply this fold's threshold to compute metrics
        y_val_pred = (y_val_proba >= t_fold).astype(int)

        cv_metrics["accuracy"].append(accuracy_score(y_val, y_val_pred))
        cv_metrics["precision"].append(precision_score(y_val, y_val_pred))
        cv_metrics["recall"].append(recall_score(y_val, y_val_pred))
        cv_metrics["f1"].append(f1_score(y_val, y_val_pred))

    # Global threshold = mean of best thresholds
    T_star = float(np.mean(fold_thresholds))

    # Final fit on full training data 
    final_pipe = clone(base_pipe)
    final_pipe.fit(X_train, y_train)

    # Test evaluation using T_star 
    y_test_proba = final_pipe.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= T_star).astype(int)

    test_results = {
        "test_accuracy":  accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "test_recall":    recall_score(y_test, y_test_pred),
        "test_f1":        f1_score(y_test, y_test_pred),
    }

    results = {
        "cv_accuracy":   float(np.mean(cv_metrics["accuracy"])),
        "cv_precision":  float(np.mean(cv_metrics["precision"])),
        "cv_recall":     float(np.mean(cv_metrics["recall"])),
        "cv_f1":         float(np.mean(cv_metrics["f1"])),
        "cv_thresholds": fold_thresholds,
        "best_threshold": T_star,
        "test_accuracy":  test_results["test_accuracy"],
        "test_precision": test_results["test_precision"],
        "test_recall":    test_results["test_recall"],
        "test_f1":        test_results["test_f1"],
        "fitted_pipeline": final_pipe,
    }

    return results



# %% [markdown]
#  Applying to our baseline XGBoost (moderate features)

# %%
baseline_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist",
)

xgb_threshold_results = evaluate_model_with_threshold_cv(
    model=baseline_xgb,
    feature_list=features_moderate,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    pipeline_preprocess=pipeline_preprocess,
    n_splits=5,
    metric=f1_score   # you can swap to recall_score if you prefer
)

print("Cross-validated threshold tuning results (XGB, moderate):")
for k, v in xgb_threshold_results.items():
    if k not in ("fitted_pipeline", "cv_thresholds"):
        print(f"{k}: {v}")

print("\nPer-fold thresholds:", xgb_threshold_results["cv_thresholds"])
print("Global T* (mean threshold):", xgb_threshold_results["best_threshold"])



# %% [markdown]
# Using stratified K-fold cross-validation, the threshold-optimization procedure identified a threshold of 0.49. When this cross-validated threshold was applied to the test set, the model achieved performance metrics very similar to those obtained with the 0.5 threshold, which tells us that, for this dataset, optimizing the decision threshold does not result in an improvement over the baseline setting. This is not surprising considering how close the optimized and the base threshold are.

# %%
# Use fitted baseline pipeline & get test probabilities from earlier baseline evaluation: baseline_results = evaluate_model_preprocessed(...)

final_pipeline = baseline_results["fitted_pipeline"]

# Probabilities for the test set

y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]


# Final threshold

final_threshold = 0.50   # default or your chosen business threshold

# Final predictions
y_pred = (y_test_proba >= final_threshold).astype(int)


# Compute metrics

test_accuracy  = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall    = recall_score(y_test, y_pred)
test_f1        = f1_score(y_test, y_pred)

metrics_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1"],
    "Score": [test_accuracy, test_precision, test_recall, test_f1]
})

print(f"Final Classification Metrics (threshold = {final_threshold:.2f})")
display(metrics_table)


# Confusion Matrix (Heatmap)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted No Attrition", "Predicted Attrition"],
    yticklabels=["Actual No Attrition", "Actual Attrition"]
)
plt.title(f"Confusion Matrix (threshold = {final_threshold:.2f})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()



# %% [markdown]
# ## XGBoost Model Explainability

# %% [markdown]
# ### SHAP Analysis

# %% [markdown]
#  To better understand how our XGBoost model makes predictions, we now turn to SHAP (SHapley Additive exPlanations). SHAP sheds light on the "black box" behind predictions by showing how each feature influences the model’s output, both at the dataset level and for individual employees.

# %% [markdown]
#    Compute SHAP values (TreeExplainer)

# %%
shap.initjs()

# Pull pieces out of your fitted baseline pipeline

final_pipeline = baseline_results["fitted_pipeline"]

preprocess_df = final_pipeline.named_steps["preprocess"]   # PreprocessToDF
selector      = final_pipeline.named_steps["select"]       # ColumnSelectorByName
model         = final_pipeline.named_steps["model"]        # XGBClassifier

# Encode + select TRAIN features (same as model sees)
X_train_encoded = preprocess_df.transform(X_train)         # DataFrame with encoded cols
encoded_feature_names = selector.active_names_             # selected encoded feature names
X_train_sel = X_train_encoded[encoded_feature_names]

# Optional: smaller background for speed
background = X_train_sel.sample(n=min(200, len(X_train_sel)), random_state=42)


# TreeExplainer on encoded features

explainer = shap.TreeExplainer(model, data=background)
shap_values = explainer.shap_values(X_train_sel)

# For binary XGBoost, TreeExplainer returns list [class0, class1]
shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values

# %% [markdown]
# #### "Raw" SHAP Plots with Encoded Features

# %% [markdown]
# Bar plot: feature importance ranking

# %%
# Bar plot (global importance)

shap.summary_plot(
    shap_values_pos,
    X_train_sel,
    feature_names=encoded_feature_names,
    plot_type="bar",
    show=True
)

# %% [markdown]
# This SHAP bar plot shows the features with the greatest overall influence on the model’s attrition predictions. Income Rate Ratio stands out as the most impactful factor, suggesting that employees whose monthly income is low relative to their hourly rate are more likely to leave. Stock Option Level, Training Times Last Year, and OverTime also play major roles, indicating that compensation structure, development opportunities, and workload meaningfully affect retention. Additional influential factors include NumCompaniesWorked, Income vs. Role Median, PromotionGap, and time with the current manager, all reflecting aspects of career progression and employment history.

# %% [markdown]
# Beeswarm: direction + magnitude

# %%
# Beeswarm plot (direction + distribution)

shap.summary_plot(
    shap_values_pos,
    X_train_sel,
    feature_names=encoded_feature_names,
    show=True
)

# %% [markdown]
# This SHAP beeswarm plot illustrates how individual feature values influence the model’s attrition predictions. Features are ordered by importance, with each point representing an employee and colored by whether the feature value is high (pink) or low (blue). 
# 
# The most influential features, such as Income Rate Ratio, Stock Option Level, Training Times Last Year, and OverTime—display clear directional effects: for example, low Stock Option Level and high OverTime push predictions toward quitting, whereas higher training frequency and higher relative pay (Income Rate Ratio) generally reduce predicted attrition risk. Other features like NumCompaniesWorked, Income vs. Role Median, PromotionGap, and time with the current manager also show meaningful but more nuanced patterns. Overall, the plot highlights dominant financial, developmental, and workload-related factors.

# %% [markdown]
# #### SHAP Plots for General Audience

# %%
# Helper: map encoded name back to raw feature name

def get_raw_name(encoded_name):
    """
    'nom__BusinessTravel_Travel_Frequently' -> 'BusinessTravel'
    'ord__JobLevel'                         -> 'JobLevel'
    'num__Age'                              -> 'Age'
    """
    name = encoded_name.split("__", 1)[1]   # strip num__/ord__/nom__
    if "_" in name:                         # one-hot: raw_feature_category
        return name.split("_", 1)[0]
    return name

raw_names = [get_raw_name(f) for f in encoded_feature_names]
unique_raw = sorted(set(raw_names))




# %%
# Group SHAP values by raw feature

# Group SHAP values: sum impacts of all one-hot columns per raw feature
grouped_shap_df = (
    pd.DataFrame(shap_values_pos, columns=encoded_feature_names)
      .groupby(raw_names, axis=1)
      .sum()
)


# Build grouped feature values (for beeswarm colors)

grouped_features_dict = {}

for raw in unique_raw:
    # encoded columns belonging to this raw feature
    cols = [c for c, r in zip(encoded_feature_names, raw_names) if r == raw]

    if len(cols) == 1:
        # ordinal or numeric → keep as-is
        grouped_features_dict[raw] = X_train_sel[cols[0]].values
    else:
        # multi-category one-hot (e.g., BusinessTravel, JobRole)
        # represent category as argmax index across dummies
        encoded = X_train_sel[cols].values
        grouped_features_dict[raw] = encoded.argmax(axis=1)

# Overwrite OverTime with the original binary column so colors make sense
if "OverTime" in X_train.columns:
    grouped_features_dict["OverTime"] = X_train["OverTime"].values

# Build DataFrame with same index as X_train
grouped_features = pd.DataFrame(grouped_features_dict, index=X_train.index)

# Align column order with grouped_shap_df just in case
grouped_features = grouped_features[grouped_shap_df.columns]


# %% [markdown]
# Bar plot: feature importance ranking

# %%
# Bar plot (global importance)
shap.summary_plot(
    grouped_shap_df.values,
    grouped_features,
    feature_names=grouped_shap_df.columns,
    plot_type="bar",
    show=True
)



# %% [markdown]
# Beeswarm: direction + magnitude

# %%
# Beeswarm plot (direction + distribution)

shap.summary_plot(
    grouped_shap_df.values,
    grouped_features,
    feature_names=grouped_shap_df.columns,
    show=True
)

# %% [markdown]
# ## SVC hyperparameter tuning using randomized search

# %% [markdown]
#   From the base models we trained, our second best was a Support Vector Machine model using the restricted set of features. Let's try to optimize it and see how it compares with the XGBoost model above.

# %% [markdown]
#   We start by tuning it with RandomizedSearchCV. The function earches over a logarithmic range of C and gamma values (for an RBF kernel) using stratified K-fold cross-validation and F1 score as the optimization metric. After fitting on the raw training data (with all transforms performed inside each CV fold to avoid leakage), it reports the best parameters and CV F1, and returns the best full pipeline along with the fitted search object.

# %%
RANDOM_STATE = 42

def make_svc():
    """Fresh SVC instance."""
    return SVC(
        kernel="rbf",
        probability=True,         # needed for ROC, PR, threshold tuning
        class_weight="balanced",
        random_state=RANDOM_STATE
    )


def tune_svc_random_search(
    feature_list,
    X_train, y_train,
    pipeline_preprocess,
    n_splits=5,
    n_iter=40
):
    """
    Leak-free SVC tuner:
    - Uses same pattern as evaluate_model_preprocessed:
      preprocess (ColumnTransformer) -> DF -> select -> scale -> model
    - Wraps everything in a Pipeline and tunes only SVC hyperparameters.
    """

    # 1) Clone the ColumnTransformer from your preprocessing pipeline
    ct = pipeline_preprocess.named_steps["preprocess"]
    ct = clone(ct)

    # 2) Build full modeling pipeline (same structure as eval function)
    base_pipe = Pipeline([
        ("preprocess", PreprocessToDF(ct)),           # returns DataFrame with feature names
        ("select",    ColumnSelectorByName(feature_list)),
        ("scale",     StandardScaler()),
        ("model",     make_svc()),
    ])

    # 3) Define search space on the *model* step
    param_dist = {
        "model__C":     np.logspace(-2, 3, 30),   # 0.01 → 1000
        "model__gamma": np.logspace(-4, 1, 30),   # 1e-4 → 10
        "model__kernel": ["rbf"],                # could add "linear" if desired
    }

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True
    )

    # 4) Fit search on RAW training data (pipeline handles transforms inside CV)
    search.fit(X_train, y_train)

    print("Best SVC parameters:", search.best_params_)
    print("Best CV F1 (RandomizedSearchCV):", search.best_score_)

    # best_estimator_ is the full pipeline
    return search.best_estimator_, search



# %% [markdown]
# Now let's tune and evaluate the SVC model using the strict feature set.

# %%
svc_features = features_strict   

best_svc_pipeline, svc_search = tune_svc_random_search(
    feature_list=svc_features,
    X_train=X_train,
    y_train=y_train,
    pipeline_preprocess=pipeline_preprocess,
    n_splits=5,
    n_iter=40
)

# Evaluate using the same leak-free evaluator
svc_tuned_results = evaluate_model_preprocessed(
    model=best_svc_pipeline.named_steps["model"],  # tuned SVC
    feature_list=svc_features,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    pipeline_preprocess=pipeline_preprocess
)

svc_tuned_results["model"] = "SVC_tuned"
svc_tuned_results["feature_set"] = "strict"

pd.DataFrame([svc_tuned_results])



# %% [markdown]
# ## SVC hyperparameter tuning using grid search

# %% [markdown]
#   We used the best parameters identified in the randomized search to guide the selection of values explored in the grid search.

# %%
RANDOM_STATE = 42

def tune_svc_grid_search_around_best(
    svc_search,
    feature_list,
    X_train, y_train,
    pipeline_preprocess,
    n_splits=5
):
    
    # Extract best params
    
    best_params = svc_search.best_params_
    best_C = best_params["model__C"]
    best_gamma = best_params["model__gamma"]
    print("Random search best C:", best_C)
    print("Random search best gamma:", best_gamma)

    
    # Build local grids around them (log-spaced neighborhood)
    
    def around_log(value, span=1.0, num=5, vmin=1e-4, vmax=1e4):
        """Return log-spaced values around 'value' within +/- span in log10."""
        center = np.log10(value)
        low = max(center - span, np.log10(vmin))
        high = min(center + span, np.log10(vmax))
        return np.logspace(low, high, num)

    C_grid = around_log(best_C, span=0.7, num=5, vmin=1e-2, vmax=1e3)
    gamma_grid = around_log(best_gamma, span=0.7, num=5, vmin=1e-4, vmax=1e1)

    print("C grid:", C_grid)
    print("gamma grid:", gamma_grid)

    
    # 3. Rebuild SVC pipeline (same as in random search)
     
    ct = pipeline_preprocess.named_steps["preprocess"]
    ct = clone(ct)

    base_pipe = Pipeline([
        ("preprocess", PreprocessToDF(ct)),
        ("select",    ColumnSelectorByName(feature_list)),
        ("scale",     StandardScaler()),
        ("model",     make_svc()),   # fresh SVC, params set via grid
    ])

    param_grid = {
        "model__C": C_grid,
        "model__gamma": gamma_grid,
        "model__kernel": ["rbf"],
    }

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

  
    # Fit grid search on RAW X_train
   
    grid.fit(X_train, y_train)

    print("Best params (GridSearchCV):", grid.best_params_)
    print("Best CV F1 (GridSearchCV):", grid.best_score_)

    best_svc_pipeline = grid.best_estimator_
    return best_svc_pipeline, grid



# %%
best_svc_grid_pipeline, svc_grid = tune_svc_grid_search_around_best(
    svc_search=svc_search,
    feature_list=svc_features,
    X_train=X_train,
    y_train=y_train,
    pipeline_preprocess=pipeline_preprocess,
    n_splits=5
)

svc_grid_results = evaluate_model_preprocessed(
    model=best_svc_grid_pipeline.named_steps["model"],
    feature_list=svc_features,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    pipeline_preprocess=pipeline_preprocess
)

svc_grid_results["model"] = "SVC_tuned_grid"
svc_grid_results["feature_set"] = "strict"

pd.DataFrame([svc_grid_results])



# %% [markdown]
#  The grid search mostly confirmed the random-search best point.

# %% [markdown]
# ## SVC Model Evaluation

# %%
# Use the best model from the grid search as final SVC pipeline
best_svc_pipeline = svc_grid.best_estimator_

# Alternatively, the random-search one preferred swap to:
# best_svc_pipeline = svc_search.best_estimator_

# %%
# This pipeline already includes preprocess + select + scale + SVC

y_test_proba = best_svc_pipeline.predict_proba(X_test)[:, 1]


# %% [markdown]
# ### ROC Curve

# %%
# ROC curve + AUC

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
auc_score = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Tuned SVC (strict features)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()



# %% [markdown]
# This ROC curve shows how well the tuned SVC model (using the strict feature set) distinguishes between employees who stay and those who leave across all possible classification thresholds. The curve rises well above the diagonal baseline, indicating that the model performs substantially better than random guessing.

# %% [markdown]
# ### Precision-Recall Curve

# %%
# Precision–Recall curve + AP

prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
ap = average_precision_score(y_test, y_test_proba)

plt.figure(figsize=(7, 6))
plt.plot(rec, prec, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Tuned SVC (strict features)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()



# %% [markdown]
# This Precision–Recall curve evaluates how well the tuned SVC model identifies employees who are likely to quit, focusing specifically on performance for the positive (attrition) class. Precision starts high when recall is low—meaning that when the model is very selective, the employees it flags as high-risk are mostly correct. As recall increases, precision gradually declines, which is expected in imbalanced datasets where true quitters are relatively few. The Average Precision (AP) of 0.536 indicates that the model maintains a reasonably strong balance between precision and recall across thresholds, and performs well above a random baseline. Overall, the tuned SVC demonstrates solid ability to detect likely quitters while controlling false alarms.

# %% [markdown]
# ### Threshold vs F1 on the Test Set

# %%
# Threshold vs F1 on TEST set

thresholds = np.linspace(0, 1, 201)
f1_scores = []
precisions = []
recalls = []

for t in thresholds:
    y_pred_t = (y_test_proba >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_t))
    precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_t))

f1_scores = np.array(f1_scores)
precisions = np.array(precisions)
recalls = np.array(recalls)

best_idx = np.argmax(f1_scores)
best_t = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

# F1 vs threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores, label="F1", linewidth=2)
plt.axvline(0.5, color="red", linestyle="--", label="Threshold 0.5")
plt.axvline(best_t, color="green", linestyle="--",
            label=f"Best on test: t={best_t:.3f}, F1={best_f1:.3f}")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 vs Threshold – Tuned SVC (strict features)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()



# %% [markdown]
# This plot shows how the tuned SVC model’s F1 score changes as we vary the classification threshold. While the default threshold of 0.5 (red dashed line) yields a moderate F1, the curve reveals that the model performs significantly better at lower thresholds. The optimal point on the test set occurs around t = 0.22 (green dashed line), where the model reaches its highest F1 score of 0.518. This indicates that, for attrition prediction, lowering the threshold improves the balance between precision and recall—capturing more true quitters while keeping false positives at a manageable level. 
# 
# Overall, the plot highlights that threshold tuning is essential for extracting the model’s best real-world performance, which we will perform below.

# %% [markdown]
#    # SVC Threshold Optimization

# %% [markdown]
# The function below performs cross-validated threshold optimisation for SVC wrapped in a pipeline. It uses Stratified K-fold CV on the training data; in each fold, it fits the pipeline, scores predicted probabilities on the validation split, and searches over a grid of thresholds to find the one that maximises the F1 score for that fold. It records per-fold metrics at the fold-specific best threshold, and then averages the optimal thresholds to obtain a single global threshold. 
# 
# The pipeline is then refit on the full training set, applied to the test set using this global threshold, and the resulting test metrics are computed. The function returns the cross-validated metrics, the per-fold and global thresholds, and the final test performance, giving a leak-free way to tune and evaluate the decision threshold.

# %%
def optimize_threshold_cv(
    pipeline,
    X_train, y_train,
    X_test, y_test,
    n_splits=5,
    thresholds=np.linspace(0, 1, 101)
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_thresholds = []

    cv_acc = []
    cv_prec = []
    cv_rec = []
    cv_f1 = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Fit pipeline for this fold
        pipeline.fit(X_tr, y_tr)

        # Predict probabilities
        val_proba = pipeline.predict_proba(X_val)[:, 1]

        # Search best threshold
        best_f1 = -1
        best_t = 0.5

        for t in thresholds:
            y_pred_t = (val_proba >= t).astype(int)
            f1 = f1_score(y_val, y_pred_t)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        fold_thresholds.append(best_t)

        # CV metrics at that threshold
        y_pred_best = (val_proba >= best_t).astype(int)
        cv_acc.append(accuracy_score(y_val, y_pred_best))
        cv_prec.append(precision_score(y_val, y_pred_best, zero_division=0))
        cv_rec.append(recall_score(y_val, y_pred_best))
        cv_f1.append(f1_score(y_val, y_pred_best))

        print(f"Fold {fold}: best_t = {best_t:.3f}, F1 = {best_f1:.3f}")

    # Global threshold = mean of fold thresholds
    global_t = np.mean(fold_thresholds)

    # Refit pipeline on full training set
    pipeline.fit(X_train, y_train)

    # Predict on test
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (test_proba >= global_t).astype(int)

    results = {
        "cv_accuracy":  np.mean(cv_acc),
        "cv_precision": np.mean(cv_prec),
        "cv_recall":    np.mean(cv_rec),
        "cv_f1":        np.mean(cv_f1),

        "best_threshold": global_t,

        "test_accuracy":  accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "test_recall":    recall_score(y_test, y_test_pred),
        "test_f1":        f1_score(y_test, y_test_pred),

        "per_fold_thresholds": fold_thresholds,
        "global_threshold": global_t
    }

    return results



# %%
svc_threshold_results = optimize_threshold_cv(
    pipeline=best_svc_pipeline,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_splits=5
)

svc_threshold_results

# %% [markdown]
#  The previous SVC tuned model had test F1 = 0.47; after threshold tuning it improved to 0.51. The tuned SVC now generalizes as well as XGBoost on test, while being simpler (uses less features).

# %% [markdown]
# ### SVC Confusion Matrix

# %%
# Get optimized threshold

t_opt = float(svc_threshold_results["global_threshold"])
print("Using optimized threshold:", t_opt)


# Predict on test set

test_proba_svc = best_svc_pipeline.predict_proba(X_test)[:, 1]
y_pred_opt = (test_proba_svc >= t_opt).astype(int)


# Confusion matrix

cm = confusion_matrix(y_test, y_pred_opt)


# Heatmap

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Predicted: No Attrition", "Predicted: Attrition"],
    yticklabels=["Actual: No Attrition", "Actual: Attrition"]
)
plt.title(f"Tuned SVC Confusion Matrix (Threshold = {t_opt:.3f})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


# Classification report

print("\nClassification Report (Optimized Threshold):\n")
print(classification_report(y_test, y_pred_opt, digits=4))



# %%
# Use optimized threshold

t_opt = float(svc_threshold_results["global_threshold"])
print("Using optimized threshold:", t_opt)

test_proba_svc = best_svc_pipeline.predict_proba(X_test)[:, 1]
y_pred_opt = (test_proba_svc >= t_opt).astype(int)


# Confusion matrix (raw + normalized)

cm = confusion_matrix(y_test, y_pred_opt)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

tn, fp, fn, tp = cm.ravel()

# Additional metrics
precision = precision_score(y_test, y_pred_opt, zero_division=0)
recall = recall_score(y_test, y_pred_opt)
f1 = f1_score(y_test, y_pred_opt)
specificity = tn / (tn + fp)


# Plotting

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

titles = ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized %)"]
mats = [cm, cm_norm]
fmts = ["d", ".2f"]

for ax, mat, title, fmt in zip(axes, mats, titles, fmts):

    sns.heatmap(
        mat,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="gray",
        xticklabels=["Predicted: No Attrition", "Predicted: Attrition"],
        yticklabels=["Actual: No Attrition", "Actual: Attrition"],
        ax=ax,
        annot_kws={"size": 14}
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)

plt.tight_layout()


# Below-plot metrics annotation

plt.figtext(
    0.5, -0.05,
    f"Threshold = {t_opt:.3f} | "
    f"Precision = {precision:.3f} | "
    f"Recall = {recall:.3f} | "
    f"Specificity = {specificity:.3f} | "
    f"F1 Score = {f1:.3f}",
    wrap=True,
    horizontalalignment='center',
    fontsize=14
)

plt.show()


# 5. Detailed classification report in text

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_opt, digits=4))



# %% [markdown]
# ### Model Comparison: XGBoost (Baseline) vs SVC (Tuned)

# %%
# 1) Grab baseline XGB row from your results_df
xgb_base_mod = results_df.query("model == 'XGB' and feature_set == 'moderate'").iloc[0]

# 2) Build comparison table (test metrics only)
rows = [
    {
        "model": "XGB_baseline (moderate)",
        "Accuracy":  xgb_base_mod["test_accuracy"],
        "Precision": xgb_base_mod["test_precision"],
        "Recall":    xgb_base_mod["test_recall"],
        "F1":        xgb_base_mod["test_f1"],
    },
    {
        "model": "SVC_tuned + thresh (strict)",
        "Accuracy":  float(svc_threshold_results["test_accuracy"]),
        "Precision": float(svc_threshold_results["test_precision"]),
        "Recall":    float(svc_threshold_results["test_recall"]),
        "F1":        float(svc_threshold_results["test_f1"]),
    }
]

df_compare = pd.DataFrame(rows)
display(df_compare)

# 3) Melt to long format for plotting
df_long = df_compare.melt(id_vars="model", var_name="Metric", value_name="Score")
metric_order = ["Accuracy", "Precision", "Recall", "F1"]
df_long["Metric"] = pd.Categorical(df_long["Metric"], categories=metric_order, ordered=True)

# 4) Side-by-side bar plot
plt.figure(figsize=(8, 6))
sns.barplot(
    data=df_long,
    x="Metric",
    y="Score",
    hue="model"
)

plt.ylim(0, 1.0)
plt.title("Model Comparison – XGB Baseline vs Tuned SVC", fontsize=14, fontweight="bold")
plt.ylabel("Score")
plt.xlabel("")
plt.legend(title="Model")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()



# %% [markdown]
# # Model Cards

# %%
# Extract SVC params
svc_model = best_svc_pipeline.named_steps["model"]
svc_params = {
    "C": svc_model.C,
    "gamma": svc_model.gamma,
    "kernel": svc_model.kernel,
    "class_weight": svc_model.class_weight,
    "probability": svc_model.probability,
}

xgb_card = {
    "model_name": "XGB_baseline",
    "feature_set": "moderate",
    "algorithm": "XGBoost (tree-based gradient boosting)",
    "threshold": 0.50,  # or xgb_threshold_results['global_threshold'] if you optimized
    "cv_accuracy":  xgb_base_mod["cv_accuracy"],
    "cv_precision": xgb_base_mod["cv_precision"],
    "cv_recall":    xgb_base_mod["cv_recall"],
    "cv_f1":        xgb_base_mod["cv_f1"],
    "test_accuracy":  xgb_base_mod["test_accuracy"],
    "test_precision": xgb_base_mod["test_precision"],
    "test_recall":    xgb_base_mod["test_recall"],
    "test_f1":        xgb_base_mod["test_f1"],
}

svc_card = {
    "model_name": "SVC_tuned_thresholded",
    "feature_set": "strict",
    "algorithm": "Support Vector Classifier (RBF kernel)",
    "threshold": float(svc_threshold_results["global_threshold"]),
    "svc_params": svc_params,
    "cv_accuracy":  float(svc_threshold_results["cv_accuracy"]),
    "cv_precision": float(svc_threshold_results["cv_precision"]),
    "cv_recall":    float(svc_threshold_results["cv_recall"]),
    "cv_f1":        float(svc_threshold_results["cv_f1"]),
    "test_accuracy":  float(svc_threshold_results["test_accuracy"]),
    "test_precision": float(svc_threshold_results["test_precision"]),
    "test_recall":    float(svc_threshold_results["test_recall"]),
    "test_f1":        float(svc_threshold_results["test_f1"]),
}

import pandas as pd
model_cards_df = pd.DataFrame([xgb_card, svc_card])
model_cards_df



# %% [markdown]
# ## SVC Model Explainability

# %% [markdown]
# ## SHAP Analysis

# %% [markdown]
# The code below prepares SHAP visualizations by fdefining a helper function to map encoded feature names back to their original raw column names. The mapping logic handles numeric (num__), ordinal (ord__), and one-hot–encoded nominal (nom__) features, stripping prefixes and categories so each encoded feature can be associated with its original source variable. 
# 
# Using this function, the code converts the strict SVC feature set from encoded names to their corresponding raw feature names, producing a clean, deduplicated list of the original HR variables used by the model.

# %%
shap.initjs()

# Mapping encoded strict features back to raw feature name (reuse the same logic we used before)

def get_raw_feature_name(encoded_feature):
    """
    'ord__JobLevel'                         -> 'JobLevel'
    'num__Age'                              -> 'Age'
    'nom__OverTime_1'                       -> 'OverTime'
    'nom__BusinessTravel_Travel_Rarely'     -> 'BusinessTravel'
    """
    if encoded_feature.startswith(("num__", "ord__")):
        return encoded_feature.split("__", 1)[1]
    if encoded_feature.startswith("nom__"):
        tmp = encoded_feature.split("__", 1)[1]  # 'Column_Category...'
        raw_col = tmp.split("_", 1)[0]
        return raw_col
    return encoded_feature

# strict encoded feature list used for SVC
svc_features_encoded = features_strict

strict_raw_features = sorted(
    {get_raw_feature_name(f) for f in svc_features_encoded}
)

print("Raw features used by SVC (strict set):")
print(strict_raw_features)


# %% [markdown]
# Below we define a prediction wrapper that allows SHAP to send raw feature values directly into the tuned SVC pipeline. It first stores the original training column order, then creates f_pred, a function that converts any incoming NumPy array into a DataFrame with the correct column names and order. The function then feeds this DataFrame through the full preprocessing-and-model pipeline (best_svc_pipeline) and returns the predicted probability of attrition. This setup ensures that SHAP can evaluate the model end-to-end.

# %%
# Define prediction function over *raw* X (full X, pipeline handles preprocessing)

# We'll pass full X (all columns) to the pipeline.
all_feature_names = list(X_train.columns)

def f_pred(X_array):
    """
    X_array: numpy array with columns in the same order as X_train.columns
    Returns: P(Attrition=1)
    """
    X_df = pd.DataFrame(X_array, columns=all_feature_names)
    return best_svc_pipeline.predict_proba(X_df)[:, 1]


# %% [markdown]
# Below we prepare the data needed for SHAP’s KernelExplainer. We start by selecting a small, representative background dataset from the training set—up to 100 samples—which SHAP uses to estimate the model’s expected output and serve as the reference distribution. Then, we select a subset of up to 200 rows from the test set as the instances to explain.

# %%
# Background data for KernelExplainer

background_size = 100
background_df = X_train.sample(
    n=min(background_size, len(X_train)),
    random_state=42
)
background = background_df.values


# Data to explain (subset of test set)

n_explain = 200
X_test_explain_df = X_test.sample(
    n=min(n_explain, len(X_test)),
    random_state=123
)
X_test_explain = X_test_explain_df.values



# %% [markdown]
# This code runs SHAP’s KernelExplainer to compute feature attributions for the tuned SVC model. These SHAP values quantify how each feature contributes to the model’s predicted probability of attrition for every explained sample.

# %%
# Run KernelExplainer

explainer = shap.KernelExplainer(f_pred, background)
shap_values_full = explainer.shap_values(X_test_explain, nsamples="auto")

# shap_values_full can be list or array depending on SHAP version
if isinstance(shap_values_full, list):
    shap_values_full = shap_values_full[0]  # for scalar output

shap_values_full = np.array(shap_values_full)  # (n_samples, n_all_features)


# %% [markdown]
# This block filters the SHAP values and the corresponding feature data to include only the strict raw feature set used by the SVC model, ensuring that all subsequent SHAP visualizations or analyses focus only on the variables that the SVC model actually uses.

# %%
# Restrict SHAP & X to the strict raw features indices of strict_raw_features in the full DataFrame

idx_strict = [all_feature_names.index(col) for col in strict_raw_features]

shap_values_strict = shap_values_full[:, idx_strict]
X_explain_strict = X_test_explain_df[strict_raw_features]

print("SHAP array shape (strict):", shap_values_strict.shape)


# %%
# 7. SHAP plots (lay-audience friendly feature names)
# -------------------------------------------------

# Bar plot: global importance
shap.summary_plot(
    shap_values_strict,
    X_explain_strict,
    feature_names=strict_raw_features,
    plot_type="bar",
    max_display=20
)


# %% [markdown]
# This SHAP bar plot shows the relative importance of each raw feature used by the tuned SVC model in predicting employee attrition. OverTime is by far the strongest driver of the model’s decisions, with substantial average impact on the predicted risk of quitting. Other influential factors include EngagementIndex, JobRole, TotalWorkingYears, and NumCompaniesWorked, suggesting that employee engagement, role type, career stage, and previous job mobility all play meaningful roles in the model’s assessment. Demographic and compensation-related features, such as MaritalStatus, Age, StockOptionLevel, and Income_Rate_Ratio—have moderate influence, while variables like MonthlyIncome, IncomeVsRoleMedian, and YearsAtCompany contribute to a lesser extent. 
# 
# Overall, the plot indicates that workload intensity, engagement, and career history are the most impactful predictors within the strict feature set selected for the SVC.

# %%
# Beeswarm: direction + distribution
shap.summary_plot(
    shap_values_strict,
    X_explain_strict,
    feature_names=strict_raw_features,
    max_display=20
)


# %% [markdown]
# This SHAP beeswarm plot shows how each strict raw feature influences the tuned SVC model’s prediction of attrition. Features at the top—such as OverTime and EngagementIndex—have the strongest overall impact. High (pink) values of OverTime consistently push predictions toward higher attrition risk, while low (blue) values reduce it. EngagementIndex shows the opposite pattern: higher engagement tends to decrease predicted risk, while lower engagement increases it. Career-related variables such as TotalWorkingYears, NumCompaniesWorked, and YearsWithCurrManager show more nuanced effects, with both high and low values influencing predictions depending on the context. Compensation and job-level features—such as StockOptionLevel, JobLevel, and Income_Rate_Ratio—also contribute but with smaller SHAP magnitudes. 
# 
# Overall, the plot highlights that workload, engagement, and career history are the primary drivers in how the SVC model determines the likelihood of an employee leaving.

# %%
# SHAP values -> DataFrame for convenience
df_shap = pd.DataFrame(shap_values_strict, columns=strict_raw_features)
df_feats = X_explain_strict[strict_raw_features].copy()

# Simple dtype-based split
numeric_features = [c for c in strict_raw_features if df_feats[c].dtype != "object"]
categorical_features = [c for c in strict_raw_features if df_feats[c].dtype == "object"]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# %%
force_cats = ["OverTime", "JobRole", "MaritalStatus", "BusinessTravel"]
for c in force_cats:
    if c in strict_raw_features and c not in categorical_features:
        categorical_features.append(c)
        if c in numeric_features:
            numeric_features.remove(c)



# %%
if numeric_features:
    shap.summary_plot(
        df_shap[numeric_features].values,
        df_feats[numeric_features],
        feature_names=numeric_features,
        max_display=20
    )

# %%
# Optional: nicer labels for OverTime, etc.
overtime_labels = {0: "No overtime", 1: "Overtime"}

for feat in categorical_features:
    shap_vals = df_shap[feat]
    vals = df_feats[feat].copy()

    # Apply human-friendly mapping where useful
    if feat == "OverTime":
        vals = vals.map(overtime_labels).fillna(vals)

    # Build plotting DataFrame
    plot_df = pd.DataFrame({
        "SHAP value": shap_vals,
        feat: vals
    })

    # Create a palette with one colour per category
    categories = plot_df[feat].astype(str).unique()
    palette = sns.color_palette("Set2", len(categories))

    plt.figure(figsize=(8, max(3, 0.5 * len(categories))))
    sns.stripplot(
        data=plot_df,
        x="SHAP value",
        y=feat,
        hue=feat,
        palette=palette,
        dodge=False,
        alpha=0.7,
        orient="h",
        jitter=0.25,
        linewidth=0.5,
        edgecolor="k"
    )

    plt.axvline(0, color="gray", linestyle="--", alpha=0.7)
    plt.title(f"SHAP values for {feat}", fontsize=14, fontweight="bold")
    plt.xlabel("Impact on attrition prediction (SHAP value)")
    plt.ylabel(feat)
    plt.legend(title=feat, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# # Train Final Model Using ALL Available Data

# %% [markdown]
# Adding 'IncomeVsRoleMedian' to data

# %%
import json

# Load the TRAIN medians (computed earlier)
with open("role_medians.json") as f:
    role_medians = json.load(f)

# Add IncomeVsRoleMedian to the FULL dataset (data)

data["IncomeVsRoleMedian"] = data["MonthlyIncome"] / data["JobRole"].map(role_medians)


# %%
print(data.columns)

# %%
X_full = data.drop("Attrition", axis=1).copy()
y_full = data["Attrition"].copy()

print(X_full.shape, y_full.shape)


# %% [markdown]
# ## Re-fitting best SVC pipeline on all data

# %%
print(best_svc_pipeline.named_steps['model'].n_features_in_)

# %%
print(strict_raw_features)

# %%
print(len(strict_raw_features))

# %%
# Create a new dataframe with only the selected features
X_full_selected = X_full[strict_raw_features].copy()

# Optional: Verify the result
print(f"Original shape: {X_full.shape}")
print(f"New shape: {X_full_selected.shape}")

# %%
# Refit best SVC pipeline on *all* data
best_svc_pipeline.fit(X_full, y_full)


# %% [markdown]
# ## Saving Optimized Threshold

# %%
t_opt = svc_threshold_results["global_threshold"]  
t_opt

# %% [markdown]
# ## Prediction Helper for Deployment

# %%
FINAL_THRESHOLD = float(svc_threshold_results["global_threshold"])

def predict_attrition_proba(df):
    """
    df: pandas DataFrame with the same columns as X_full
    Returns: numpy array of P(attrition=1)
    """
    return best_svc_pipeline.predict_proba(df)[:, 1]

def predict_attrition_label(df, threshold=FINAL_THRESHOLD):
    """
    Returns 0/1 predictions using the chosen decision threshold.
    """
    proba = predict_attrition_proba(df)
    return (proba >= threshold).astype(int)


# %% [markdown]
# ## Saving Model Locally

# %%
# Save model
joblib.dump(best_svc_pipeline, "svc_attrition_pipeline.joblib")

# Save threshold
FINAL_THRESHOLD = float(svc_threshold_results["global_threshold"])
with open("config.json", "w") as f:
    json.dump({"threshold": FINAL_THRESHOLD}, f)


# %%
print(best_svc_pipeline.steps)

# %% [markdown]
# ## Retrieving Cols Necesary to Build Gradio App

# %%
# 1. Access the custom preprocessing step (named 'preprocess')
preprocess_step = best_svc_pipeline.named_steps['preprocess']

# 2. Access the internal ColumnTransformer stored inside it
# Note: Assuming your class stores the 'ct' argument as an attribute named 'ct'
column_transformer = preprocess_step.ct 

# 3. Now extract the features just like before
required_columns = []

for name, transformer, columns in column_transformer.transformers_:
    if name != 'remainder':  # Skip the 'remainder' config
        # 'columns' is the list of raw feature names for this transformer
        required_columns.extend(columns)

# Remove duplicates
required_columns = list(set(required_columns))

print(f"Total inputs needed for Gradio: {len(required_columns)}")
print(required_columns)

# %%
import pandas as pd
import numpy as np

# 1. Create a subset dataframe with only the features the app needs
# (Make sure 'required_columns' is defined from the previous step)
app_data = X_full[required_columns]

print(f"--- ANALYZING {len(required_columns)} FEATURES FOR GRADIO ---\n")

for col in app_data.columns:
    print(f"Feature: '{col}'")
    
    # CHECK: Is it numerical?
    if pd.api.types.is_numeric_dtype(app_data[col]):
        min_val = app_data[col].min()
        max_val = app_data[col].max()
        mean_val = app_data[col].mean()
        
        # Suggest Gradio code for numerical inputs
        print(f"   Type: Numerical")
        print(f"   Stats: Min={min_val}, Max={max_val}, Mean={mean_val:.2f}")
        print(f"   Recommendation: gr.Slider(minimum={min_val}, maximum={max_val}, value={mean_val:.2f}, label='{col}')")
        
    # CHECK: Is it categorical (object/category)?
    else:
        # Get unique values and convert to a clean list
        unique_vals = app_data[col].unique().tolist()
        
        # Suggest Gradio code for categorical inputs
        print(f"   Type: Categorical")
        print(f"   Choices ({len(unique_vals)}): {unique_vals}")
        print(f"   Recommendation: gr.Dropdown(choices={unique_vals}, label='{col}')")
    
    print("-" * 30)


