# %% [markdown]
#   ## **_Enterprise Data Science and Analytics - Enterprise Data Science Bootcamp_**
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
#   ### **HR Attrition Project - EDSB25_26**
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
#   - Ana Rita Martins 20240821
# 
# 
# 
# 
# 
# 
# 
#   - Joana Coelho 2024080
# 
# 
# 
# 
# 
# 
# 
#   - Pedro Fernandes 20240823
# 
# 
# 
# 
# 
# 
# 
#   - Ricardo Silva 20240824

# %% [markdown]
#   Data Science and Analytics are reshaping how organizations solve problems across diverse industries. Through systematic data analysis and predictive modeling, evidence-based solutions can be developed, enabling more reliable decision-making and greater efficiency.
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
#   In Human Resources, predictive analytics supports critical functions such as employee retention, workforce planning, and automated CV screening.
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
#   This project focuses on developing predictive models to assess the likelihood of employee resignation. By analyzing factors ranging from demographics to job satisfaction, the models aim to provide interpretable insights that highlight key drivers of attrition. These insights will help HR leaders take proactive steps to reduce turnover and retain talent.

# %% [markdown]
#   ## 1. Importing Packages

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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.base import clone
import shap
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV




# %% [markdown]
#   ## 2. Importing Data and Initial Exploration

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
#   From this initial inspection what immediately stands out is that we have 3 constant features: "EmployeeCount", "StandardHours", and "Over18". We can remove those straight away. Additionally, the employee number (ID) feature, does not seem to contain any relevant info, and  we'll drop it too.

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
#   From the summary above, we verified that the data set doesn't contain duplicates, and we also gathered information about the data's distribution and main statistics.
# 
# 
# 
# 
# 
# 
# 
#   What we can note is that, beasides our target, we have a couple of other binary features. Let's encode those.

# %%
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})

data.head()







# %% [markdown]
#   Let's now have a look at how the distribution of the target variable.

# %%
ax = sns.countplot(x=data['Attrition'], hue=data['Attrition'], legend=False)
for container in ax.containers:
    ax.bar_label(container)

plt.title('Distribution of the Target Variable (Attrition)')
plt.show()




# %% [markdown]
#   We can observe that our target cariable is quite imbalanced. This will require extra attention in later steps, namely when splitting the dataset into train, validation and test sets, as well as during the modelling stage.

# %%
data.shape







# %%
data.head(3)







# %% [markdown]
#   # **3. Exploratory Data Analysis**

# %% [markdown]
#   We'll start by plotting histograms to visually assess the distribution of the numeric features; this will allows us to spot any relevant patterns or trends in the data.

# %%
data.hist(figsize=(20, 15))
plt.show()







# %% [markdown]
#   The histograms reveal some important patterns in the dataset.
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
#   - Once again we can observe that the **target variable** is highly skewed toward staying in the company.
# 
# 
# 
# 
# 
# 
# 
#   - Concerning demographics, **age** follows an approximately bell-shaped distribution, centered around 30-40; **Gender** is skewed with more males than females.
# 
# 
# 
# 
# 
# 
# 
#   - Features that are related to **work characteristics** (YearsAtCompany, TotalWorkingYears, YearsInCurrentRole, Overtime) are right-skewed, indicating many relatively new employees and fewer with long careers; working overtime is not common.
# 
# 
# 
# 
# 
# 
# 
#   - **Income**: Salaries and rates are right-skewed, with few very high earners.
# 
# 
# 
# 
# 
# 
# 
#   - **Satisfaction-related** variables are discrete and somewhat skewed toward higher ratings, while PerformanceRating shows very little variation (nearly all at level 3), suggesting limited predictive value.
# 
# 
# 
# 
# 
# 
# 
#   Overall, the data displays strong imbalance and skewness patterns that will require careful consideration during modeling, suggesting it could benefit from stratified splits, and algorithms robust to class imbalance.

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
#   The boxplots highlight the extent of skewness and make the outliers stand out clearly, which complements the histogram analysis above.
# 
# 
# 
# 
# 
# 
# 
#   - Outliers are especially relevant in income and employment duration related-variables, which may need special handling. We'll decide how to handle them further down.
# 
# 
# 
# 
# 
# 
# 
#   - For demographic/job characteristics (Age, DistanceFromHome, JobLevel, Education) featured the distributions are fairly compact with few outliers, aligning with the unimodal/bell-like shapes seen in histograms.
# 
# 
# 
# 
# 
# 
# 
#   - Ordinal satisfaction and variables show limited spread, consistent with their discrete scale, with some level of skew toward higher values. Their limited range may reduce their explanatory power.
# 
# 
# 
# 
# 
# 
# 
#   - PerformanceRating shows very little variation (nearly all values at level 3) confirming its limited usefulness as a predictive feature.

# %% [markdown]
#   Subsequent steps may differ based on the category of each feature. Therefore, we’ll create lists that group feature names by their respective types.

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
#







# %% [markdown]
#   Let's now look at the distribution of our non-continuous features.

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
#   From the variables that, a priori, we'd think could be related with attrition, we find that:
# 
# 
# 
# 
# 
# 
# 
#   - roughly 30% of employees work overtime
# 
# 
# 
# 
# 
# 
# 
#   - roughly 40% have low to medium levels of satisfaction with the work environment
# 
# 
# 
# 
# 
# 
# 
#   - roughly 30% report low to medium levels of job involvement
# 
# 
# 
# 
# 
# 
# 
#   - nearly 40% report low to medium job satisfaction
# 
# 
# 
# 
# 
# 
# 
#   - another nearly 40% have low to medium levels of satisfaction with relationships at work
# 
# 
# 
# 
# 
# 
# 
#   - and about 5% report bad work-life balance

# %% [markdown]
#   To better understand what might be contributing to employees’ decisions to quit, we'll next plot the non-continuous features against the target variable. We’ll also measure the attrition rate within each category. This will show us whether some groups are more prone to leaving than others, irrespective of their overall frequency.

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
#   From the plots above we find the following trends:
# 
# 
# 
# 
# 
# 
# 
#   Department-level & Job roles
# 
# 
# 
# 
# 
# 
# 
#   - Sales and Human Resources show a higher proportion of employees quitting compared to R&D.
# 
# 
# 
# 
# 
# 
# 
#   - Within job roles, HR professionals tend to leave more often, but so do Lab Technicians, even though they are part of the R&D department.
# 
# 
# 
# 
# 
# 
# 
#   - Sales Representatives have the highest attrition rate across all job roles, whereas higher-level roles—such as managers and directors—show very low attrition.
# 
# 
# 
# 
# 
# 
# 
#    Personal characteristics
# 
# 
# 
# 
# 
# 
# 
#   - Single employees appear more likely to quit.
# 
# 
# 
# 
# 
# 
# 
#   Work conditions and workload
# 
# 
# 
# 
# 
# 
# 
#   - Employees who work overtime, travel frequently, or have poor work–life balance are more likely to leave.
# 
# 
# 
# 
# 
# 
# 
#   - Low satisfaction with the work environment, job involvement, overall job satisfaction, and relationships at work is also strongly associated with higher attrition.
# 
# 
# 
# 
# 
# 
# 
#   Job level and hierarchy
# 
# 
# 
# 
# 
# 
# 
#   - Employees in lower hierarchical levels tend to leave more often. However, attrition proportions do not strictly follow the hierarchical ranking order.
# 
# 
# 
# 
# 
# 
# 
#   Stock ownership
# 
# 
# 
# 
# 
# 
# 
#   - Employees with no stock options (stock option level 0) are more prone to quitting. This is not surprising, as offering stock is a common strategy to increase engagement.

# %% [markdown]
#      Let's now run an equivalent analysis with our continuous features.
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
#      We'll plot both their probability density function and violin plots and assess how their distribution relates to the target.

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
#   Some features show noticeable differences in their distributions depending on whether the employee quit or stayed.
# 
# 
# 
# 
# 
# 
# 
#   Age and career stage
# 
# 
# 
# 
# 
# 
# 
#   - Employees who quit tend to be younger.
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
#   - This aligns with lower values observed in Total Working Years, Years at Company, Years in Current Role, and Years with Current Manager.
# 
# 
# 
# 
# 
# 
# 
#   Early-career employees may be more inclined to change jobs or roles, contributing to these lower tenure metrics.
# 
# 
# 
# 
# 
# 
# 
#   Compensation
# 
# 
# 
# 
# 
# 
# 
#   - Monthly income appears influential: employees with lower income are more likely to leave, which is expected. The same applies to daily rate.
# 
# 
# 
# 
# 
# 
# 
#   Distance from home
# 
# 
# 
# 
# 
# 
# 
#   - The larger the distance from home to work, the more likely the employees are to leave.
# 
# 
# 
# 
# 
# 
# 
#   Other features
# 
# 
# 
# 
# 
# 
# 
#   - The remaining continuous features either show similar distributions across attrition groups or differences too small to be clearly meaningful.

# %% [markdown]
#      We’ll now take look at the correlations among the features, including the target variable. This will help us identify potential collinearity, as well as highlight which features are associated with attrition. Since several features are not strictly numeric or continuous, we’ll use Spearman’s correlation, which measures monotonic relationships by correlating feature ranks rather than their raw values.

# %% [markdown]
#      We'll exclude strictly nominal categorical variables (like Gender, Department, JobRole) because Spearman is rank-based, not meant for unordered categories.

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
#      From the analyses and visualization above we observe that:
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
#      - YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager, TotalWorkingYears, JobLevel, MonthlyIncome, StockOptionLevel and Age are the strongest monotonic predictors of Attrition.
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
#      These are indicators that relate to tenure, seniority, and stability and they're in agreement with HR domain knowledge: attrition is highest among newer, younger, lower-level employees.
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
#      - JobSatisfaction, JobInvolvement, EnvironmentSatisfaction Tshow mild but potentially meaningful associations.
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
#      Employees with lower satisfaction or lower involvement show slightly higher attrition.
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

# %%
top_features = attrition_corr_sorted.abs().sort_values(ascending=False).head(12).index

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(spearman_matrix.loc[top_features, top_features], dtype=bool))
sns.heatmap(spearman_matrix.loc[top_features, top_features], mask=mask,
            cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Top Spearman Correlated Variables")
plt.show()








# %% [markdown]
#     The heatmap shows that several of the variables most strongly correlated with attrition are also highly collinear with each other. In particular, the following groups demonstrate very strong monotonic relationships (ρ > 0.70):
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
#     - JobLevel — MonthlyIncome (ρ ≈ 0.92)
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
#     - YearsInCurrentRole — YearsWithCurrManager (ρ ≈ 0.85)
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
#     - TotalWorkingYears — MonthlyIncome (ρ ≈ 0.71)
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
#     These features are all measures of: Tenure, Seniority, Career progression, Employee stability, which explains why they are tightly correlated with each other and with lower attrition.

# %% [markdown]
#     While colinearity doesn't harm tree-based models, it does affect linear models like linear regression. Besides, it It also leads to unnecessary redundancy in the feature set. Keeping all of them increases the demand for computational powerr and increases the risk of overfitting. By the end of our feature selection process, we should aim to keep at most 2 or 3 representative variables of this set. And for regression models, we'll explicitly remove correlated pairs.
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
#     Another way to circumvent colinearity is to combine several colinear raw variables into a single engineered feature. Let's do that below.
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
#     ## Feature Engineering

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

# IncomeVsRoleMedian (Relative pay vs median in same job role)

# Compute median income per JobRole
role_median_income = data.groupby("JobRole")["MonthlyIncome"].transform("median")

data["IncomeVsRoleMedian"] = data["MonthlyIncome"] / role_median_income


data['Income_Rate_Ratio'] = data['MonthlyIncome'] / data['MonthlyRate']






# %%
engineered = ["TenureIndex", "PromotionGap", "EngagementIndex", "IncomeVsRoleMedian", "Income_Rate_Ratio"]

spearman_corrs = (
    data[engineered + ["Attrition"]]
    .corr(method="spearman")["Attrition"]
    .drop("Attrition")
)

print(spearman_corrs)





# %%
print(data.shape)





# %% [markdown]
#     # Preprocessing Steps

# %% [markdown]
#     ## Train-Test Split

# %% [markdown]
#     Before any encoding and feature selection steps we'll start by defining x and y, and defining the train–test split. Doing this at this stage is critical to avoid data leakage.

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





# %% [markdown]
#     ## Rebuilding feature groups on X_train

# %%
# 1. Categorical features inferred from dtype 'object'
categorical_features = list(
    X_train.select_dtypes(include='object').columns.drop(['BusinessTravel'])
)

# 2. Binary features (defined them manually above)
binary_features = ['Gender', 'OverTime']

# 3. Ordinal features (predefined list above)
ordinal_features = [
    'BusinessTravel','Education','EnvironmentSatisfaction','JobInvolvement',
    'JobLevel','JobSatisfaction','PerformanceRating',
    'RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance'
]

# 4. Non-continuous = categorical + binary + ordinal
non_continuous_features = categorical_features + binary_features + ordinal_features

# 5. Continuous = everything else except the target (including engineered features)
continuous_features = list(
    X_train.columns.difference(non_continuous_features)
)






# %%
print("Categorical nominal:", categorical_features)
print("Binary:", binary_features)
print("Ordinal:", ordinal_features)
print("Continuous (incl. engineered):", continuous_features)






# %% [markdown]
#     ## Defining the preprocessing (encoders + passthrough)

# %%
# Ordinal encoder for ordinal features

ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# One-hot encoder for nominal + binary

onehot_transformer = OneHotEncoder(
    drop=None,                 # or 'first' if you want k-1 dummies
    handle_unknown='ignore',   # ignore categories not seen during fit
    sparse_output=False        # get a dense array, easier to wrap in DataFrame
)

# ColumnTransformer ties it together

preprocess = ColumnTransformer(
    transformers=[
        ('ord',   ordinal_transformer, ordinal_features),
        ('nom',   onehot_transformer, categorical_features + binary_features),
        ('num',   'passthrough',      continuous_features),
    ]
)






# %% [markdown]
#     When preprocess.fit(X_train) is called, it learns: category mappings for ordinal features; dummy columns for nominal + binary.
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
#     Then preprocess.transform(...) will apply this same mapping to train & test.

# %% [markdown]
#     ## Combining preprocessing + scaling into a Pipeline

# %%
pipeline_preprocess = Pipeline([
    ('preprocess', preprocess),
    ('scale', StandardScaler())
])






# %% [markdown]
#     ## Fitting preprocessing only on training data and transform both sets

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
# 1. Grab the ColumnTransformer from the pipeline
ct = pipeline_preprocess.named_steps['preprocess']

# 2. Ask it for the output feature names
feature_names = ct.get_feature_names_out()

# 3. Now rebuild the DataFrames
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
X_test_df  = pd.DataFrame(X_test_processed,  columns=feature_names, index=X_test.index)






# %%
X_train_df.shape





# %%
X_test_df.shape





# %% [markdown]
#     # Feature Selection

# %% [markdown]
#     ## Chi-square

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
#     ## Mutual Information

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
#     ## L1 Logistic Regression (LASSO)

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
#     Features with Selected = True are part of the sparse LASSO-selected subset. Larger coefficients (in magnitude) reflect stronger linear effect.

# %% [markdown]
#     ## Random Forest Classifier

# %% [markdown]
#     Random Forest captures: nonlinearities, interactions, categorical effects, monotonic or non-monotonic patterns. Works very well alongside LASSO.

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
#     ## XGBoost Feature Importance

# %% [markdown]
#     XGBoost is often very strong at discovering: threshold effects, feature interactions, nonlinear jump patterns, sparse informative features.

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
#    ## Table Combining Feature Selection Results

# %%
#Starting from all encoded features

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
#    ## Finding which variables are consistently selected by the different feature selection methods

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
#    Establishing dynamic thresholds (quantile-based)

# %%
# Helper to get a quantile threshold, but avoid NaNs / all-zeros issues
def safe_quantile(series, q, default=0.0):
    vals = series.dropna()
    if (vals > 0).sum() == 0:
        return default
    return vals.quantile(q)

# Example: top 30% for MI, RF, XGB
mi_thresh  = safe_quantile(df["MI"],             0.70, default=0.0)
rf_thresh  = safe_quantile(df["RF_importance"],  0.70, default=0.0)
xgb_thresh = safe_quantile(df["XGB_importance"], 0.70, default=0.0)

print("MI threshold:", mi_thresh)
print("RF threshold:", rf_thresh)
print("XGB threshold:", xgb_thresh)






# %% [markdown]
#    Defining binary flags

# %%
# 1) Chi-square (categoricals only) – already a boolean
df["chi2_good"] = df["chi2_significant"].astype(bool)

# 2) Mutual Information – above quantile threshold
df["mi_good"] = (df["MI"] >= mi_thresh)

# 3) L1 Logistic Regression – already boolean
df["l1_good"] = df["Selected"].astype(bool)

# 4) Random Forest – above quantile threshold
df["rf_good"] = (df["RF_importance"] >= rf_thresh)

# 5) XGBoost – above quantile threshold
df["xgb_good"] = (df["XGB_importance"] >= xgb_thresh)






# %% [markdown]
#    Building the consensus score

# %%
method_flags = ["chi2_good", "mi_good", "l1_good", "rf_good", "xgb_good"]

# Converting to int and sum
df["consensus_score"] = df[method_flags].astype(int).sum(axis=1)

# Checking distribution
print(df["consensus_score"].value_counts().sort_index())






# %% [markdown]
#    Defining Feature Set A (strict) and Feature Set B (moderate)

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
#    This strict set leans heavily toward a few thematic clusters:
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
#    - Career progression & seniority (Age, TotalWorkingYears, YearsAtCompany, YearsWithCurrManager, JobLevel)
# 
# 
# 
# 
# 
# 
# 
#    - Income & compensation (MonthlyIncome, Income_Rate_Ratio, StockOptionLevel)
# 
# 
# 
# 
# 
# 
# 
#    - Job role (4–5 JobRole dummies)
# 
# 
# 
# 
# 
# 
# 
#    - Marital status (Single / Divorced)
# 
# 
# 
# 
# 
# 
# 
#    - OverTime
# 
# 
# 
# 
# 
# 
# 
#    - EngagementIndex
# 
# 
# 
# 
# 
# 
# 
#    - BusinessTravel
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
#    These are well known drivers of attrition.

# %%
# Modearate set of features

for i, f in enumerate(features_moderate, 1):
    print(f"{i}. {f}")





# %% [markdown]
#    The moderate feature set as expected is more comprehensive and could be particularly useful for tree-based models. Of note, all engineered features are included in this set.

# %% [markdown]
#     # Modelling

# %% [markdown]
#    The following modelling function:
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
#    - accepts a feature list (strict or moderate)
# 
# 
# 
# 
# 
# 
# 
#    - uses the pipeline_preprocess for encoding & scaling
# 
# 
# 
# 
# 
# 
# 
#    - runs cross-validation on the training set
# 
# 
# 
# 
# 
# 
# 
#    - evaluates several metrics
# 
# 
# 
# 
# 
# 
# 
#    - trains the final model on the full training set
# 
# 
# 
# 
# 
# 
# 
#    - evaluates it on the held-out test set
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
#    This will be the function we'll reuse for comparing different models as well as strict feature set vs moderate feature set.

# %%
def evaluate_model_preprocessed(
    model,
    feature_list,
    X_train, y_train,
    X_test, y_test,
    pipeline_preprocess,
    n_splits=5
):

    # ----- 1. Fit + transform preprocessing -----
    pipeline_preprocess.fit(X_train)
    X_train_prep = pipeline_preprocess.transform(X_train)
    X_test_prep  = pipeline_preprocess.transform(X_test)

    feature_names = pipeline_preprocess.get_feature_names_out()

    # ----- 2. Select strict / moderate features -----
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    idx = [name_to_idx[f] for f in feature_list]

    X_train_sel = X_train_prep[:, idx]
    X_test_sel  = X_test_prep[:, idx]

    # ----- 3. CV -----
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_acc, cv_prec, cv_rec, cv_f1 = [], [], [], []

    for train_i, val_i in skf.split(X_train_sel, y_train):
        X_tr = X_train_sel[train_i];   y_tr = y_train.iloc[train_i]
        X_val = X_train_sel[val_i];    y_val = y_train.iloc[val_i]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        cv_acc.append(accuracy_score(y_val, preds))
        cv_prec.append(precision_score(y_val, preds))
        cv_rec.append(recall_score(y_val, preds))
        cv_f1.append(f1_score(y_val, preds))

    # ----- 4. Final fit -----
    model.fit(X_train_sel, y_train)

    # ----- 5. Test set -----
    preds = model.predict(X_test_sel)

    results = {
        "cv_accuracy":   np.mean(cv_acc),
        "cv_precision":  np.mean(cv_prec),
        "cv_recall":     np.mean(cv_rec),
        "cv_f1":         np.mean(cv_f1),
        "test_accuracy":  accuracy_score(y_test, preds),
        "test_precision": precision_score(y_test, preds),
        "test_recall":    recall_score(y_test, preds),
        "test_f1":        f1_score(y_test, preds)
    }

    return results





# %%
# --- Model factory functions: each call returns a FRESH instance ---

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
        probability=True,         # in case you later want predict_proba
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
# Fit pipeline once to extract transformed feature names
pipeline_preprocess.fit(X_train)
all_features = list(pipeline_preprocess.get_feature_names_out())

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
    "moderate": features_moderate,
    "all_feats":  all_features
}





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
#   Based on the baseline results, XGBoost paired with the moderate feature set provides one of the most favourable trade-offs across all evaluation metrics. This makes it our best candidate for further optimisation, so we will focus our hyperparameter tuning (using GridSearch and Optuna) on this configuration.

# %% [markdown]
#   ## Hyperparameter tuning using Grid Search

# %% [markdown]
#   We'll reuse our evaluate_model_preprocessed while looping over a small grid to assess:
# 
# 
# 
# 
# 
# 
# 
#   - which max_depth generally works best
# 
# 
# 
# 
# 
# 
# 
#   - whether smaller/larger learning_rate helps
# 
# 
# 
# 
# 
# 
# 
#   - whether more trees improve things
# 
# 
# 
# 
# 
# 
# 
#   - whether subsampling is beneficial

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




# %% [markdown]
#   By performing grid search we obtained modest but consistent improvements over the baseline XGBoost model in cross-validation performance, particularly in precision and F1. Test-set results remain close to the baseline, indicating that the model is stable and not highly sensitive to the grid’s parameter variations. These results justify trying Optuna in order to explore the hyperparameter space more efficiently.

# %% [markdown]
#   ## Hyperparameter tuning using Optuna

# %% [markdown]
#   We define our Optuna search space based on the information we obtained from the (coarse) grid search above

# %%
def xgb_optuna_objective(trial):
    # hyperparameters informed by grid search 
    max_depth = trial.suggest_int("max_depth", 3, 6)
    # grid liked 0.05 and 0.10 (stay in that neighbourhood)
    learning_rate = trial.suggest_float("learning_rate", 0.03, 0.15, log=True)
    # grid used 200 and 400 (centre around that, allow a bit more)
    n_estimators = trial.suggest_int("n_estimators", 250, 600)
    # grid always had 0.8 . explore around it, but not as low as 0.6
    subsample = trial.suggest_float("subsample", 0.7, 0.9)
    # grid used 0.8 and 1.0 (let's restrict to that region)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.8, 1.0)

    # regularization / tree-shape params: keep reasonably broad
    min_child_weight = trial.suggest_float("min_child_weight", 1.0, 8.0)
    gamma = trial.suggest_float("gamma", 0.0, 5.0)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)


    spw = compute_spw(y_train)

    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )

    # ----- CV evaluation on training only -----
    # (copy of the logic from evaluate_model_preprocessed, but CV only)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # preprocess once per trial
    pipeline_preprocess.fit(X_train)
    X_train_prep = pipeline_preprocess.transform(X_train)
    feature_names = pipeline_preprocess.get_feature_names_out()

    # select moderate features
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    idx = [name_to_idx[f] for f in features_moderate]
    X_sel = X_train_prep[:, idx]

    cv_f1_scores = []

    for tr_idx, val_idx in skf.split(X_sel, y_train):
        X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        cv_f1_scores.append(f1_score(y_val, preds))

    return np.mean(cv_f1_scores)

# create and run study
study = optuna.create_study(
    direction="maximize",
    study_name="xgb_hr_attrition_f1"
)

study.optimize(xgb_optuna_objective, n_trials=50, show_progress_bar=True)

print("Best trial:", study.best_trial.number)
print("Best value (CV F1):", study.best_value)
print("Best params:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")




# %%
best_params = study.best_trial.params
spw = compute_spw(y_train)

best_xgb = XGBClassifier(
    **best_params,
    scale_pos_weight=spw,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist",
)

optuna_best_results = evaluate_model_preprocessed(
    model=best_xgb,
    feature_list=features_moderate,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    pipeline_preprocess=pipeline_preprocess
)

optuna_best_results




# %% [markdown]
#   Tining with Optuna didn’t beat the baseline - test_f1 Optuna 0.477 vs test_f1 base model 0.489
# 
# 
# 
#   After tuning XGBoost with 2 different approached, we conclude that the baseline configuration was already near the performance ceiling for this dataset; additional tuning brought only marginal changes in test F1.

# %% [markdown]
#   ## Model Evaluation

# %% [markdown]
#   For model evaluation we'll use our XGB base model.

# %%
# Final model to be used in evaluation curves

def make_final_xgb_baseline():
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
        eval_metric="logloss",
        n_jobs=-1
    )

model = make_final_xgb_baseline()




# %%
def preprocess_and_select(X, pipeline_preprocess, feature_list, fit=False):
    """
    Fit+transform or just transform with pipeline_preprocess,
    then select the columns in feature_list (transformed names).
    Returns transformed array and (possibly refitted) pipeline.
    """
    if fit:
        pipeline_preprocess.fit(X)
    X_prep = pipeline_preprocess.transform(X)
    
    feature_names = pipeline_preprocess.get_feature_names_out()
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    idx = [name_to_idx[f] for f in feature_list]
    
    return X_prep[:, idx], pipeline_preprocess



# %%
# preprocess full train and test (for evaluation curves)
X_train_sel, pp_fitted = preprocess_and_select(
    X_train, pipeline_preprocess, features_moderate, fit=True
)
X_test_sel, _ = preprocess_and_select(
    X_test, pp_fitted, features_moderate, fit=False
)

# fit model on full training
model.fit(X_train_sel, y_train)

# predicted probabilities
test_proba = model.predict_proba(X_test_sel)[:, 1]



# %%
# ROC Curve

fpr, tpr, _ = roc_curve(y_test, test_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()




# %% [markdown]
#   The XGBoost model achieves a ROC-AUC of 0.775, indicating good ability to discriminate between employees who leave and those who stay. The curve rises steeply at low false-positive rates, showing that the model correctly identifies many true quitters before confusing them with non-quitters.
# 
# 
# 
# 
# 
# 
# 
#   However, ROC-AUC evaluates performance across all possible thresholds and is insensitive to class imbalance. In an attrition scenario where only ~15% of employees leave, a model can obtain a relatively high ROC-AUC even when the practical precision and recall trade-off is challenging.
# 
# 
# 
# 
# 
# 
# 
#   Therefore, while the ROC curve confirms that the model produces a meaningful ranking of risk scores, it must be interpreted alongside the Precision–Recall curve and F1 score to fully understand real-world classification performance under imbalance.

# %%
# Precision-Recall Curve

prec, rec, _ = precision_recall_curve(y_test, test_proba)
pr_auc = average_precision_score(y_test, test_proba)

plt.figure(figsize=(6, 6))
plt.plot(rec, prec, label=f"PR curve (AP = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid()
plt.show()




# %% [markdown]
#   The Precision–Recall curve provides a more realistic evaluation of model performance under class imbalance than ROC-AUC. The model achieves an Average Precision (AP) of 0.53, substantially higher than the baseline positive rate of ~0.15, confirming that it captures meaningful signal related to employee attrition. Precision is extremely high at low recall (0.85–1.00), indicating that the model is very confident in identifying the top-risk employees. As recall increases, precision decreases, reflecting a typical precision–recall tradeoff in imbalanced datasets, where achieving high recall requires accepting more false positives. These results align with the observed F1 scores (0.45–0.49), confirming that the model captures meaningful signal but cannot achieve high precision and recall simultaneously.
# 
# 
# 
# 
# 
# 
# 
#   Although the model ranks employees well (as shown by the ROC curve), achieving strong precision and recall simultaneously is challenging due to the underlying class imbalance. The PR curve therefore offers a realistic assessment of actionable performance and complements the ROC curve by showing where the model can be most effectively used.

# %%
# Threshold vs F1 Score

thresholds = np.linspace(0.05, 0.95, 200)
f1s = []

for thr in thresholds:
    preds = (test_proba >= thr).astype(int)
    f1s.append(f1_score(y_test, preds))

# Obter top 3 thresholds
top_3_indices = np.argsort(f1s)[-3:][::-1]  # Índices dos 3 maiores F1s, em ordem decrescente
top_3_thresholds = thresholds[top_3_indices]
top_3_f1s = np.array(f1s)[top_3_indices]

print("Top 3 Thresholds:")
for i, (thr, f1) in enumerate(zip(top_3_thresholds, top_3_f1s), 1):
    print(f"{i}. Threshold: {thr:.4f} → F1 Score: {f1:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(thresholds, f1s)
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("Threshold vs F1")
plt.grid()
plt.show()



# %% [markdown]
#   The Threshold–F1 curve illustrates how the model’s F1 score changes as the decision threshold varies between 0 and 1. The model achieves its highest F1 values—peaking just above 0.50, which is when the threshold is in the 0.30 to 0.45 range. This means that, on the test set, lowering the threshold below the default 0.50 improves the balance between precision and recall.
# 
# 
# 
# 
# 
# 
# 
#   However, this peak reflects test-set behavior and should not be used for model selection, as tuning a threshold on test data introduces data leakage.
# 
# 
# 
# 
# 
# 
# 
#   Overall, the curve highlights how sensitive F1 performance is to threshold choice, especially in imbalanced datasets, and reinforces the importance of selecting thresholds based on training-only validation procedures rather than test-set optimization.

# %% [markdown]
#   ## Threshold Optimization

# %% [markdown]
#   The Threshold–F1 curve above shows that the model’s performance varies substantially across different probability cutoffs. Although the default threshold of 0.50 is commonly used, it is not the point that maximizes the F1 score on this dataset. This variation suggests that adjusting the classification threshold could meaningfully improve the balance between precision and recall.
# 
# 
# 
# 
# 
# 
# 
#   Therefore, in the next section, we perform a dedicated threshold optimization procedure using using validation performed only on the training data (avoiding leakage) in an attempt identify a more effective decision threshold for this model.

# %%
def find_best_threshold(
    model,
    feature_list,
    X_train,
    y_train,
    pipeline_preprocess,
    val_size=0.25,
    random_state=42,
    thresholds=None
):
    """
    - Splits X_train into inner_train + val
    - Fits preprocessing on inner_train only
    - Fits model on inner_train
    - Sweeps thresholds on val probabilities
    - Returns best threshold and a DataFrame of threshold vs metrics
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)  # step 0.005

    # inner split (ONLY from training data)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=random_state
    )

    # preprocess & select features
    X_tr_sel, pp_fitted = preprocess_and_select(
        X_tr, pipeline_preprocess, feature_list, fit=True
    )
    X_val_sel, _ = preprocess_and_select(
        X_val, pp_fitted, feature_list, fit=False
    )

    # fit model on inner train
    model.fit(X_tr_sel, y_tr)

    # get probabilities on validation
    val_proba = model.predict_proba(X_val_sel)[:, 1]

    rows = []
    for thr in thresholds:
        y_val_pred = (val_proba >= thr).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred, zero_division=0)
        rec = recall_score(y_val, y_val_pred)
        rows.append((thr, f1, prec, rec))

    thr_df = pd.DataFrame(rows, columns=["threshold", "f1", "precision", "recall"])
    best_row = thr_df.loc[thr_df["f1"].idxmax()]

    best_thr = best_row["threshold"]
    print("Best threshold on validation:")
    print(best_row)

    return best_thr, thr_df




# %%
best_thr, thr_df = find_best_threshold(
    model=make_final_xgb_baseline(),
    feature_list=features_moderate,
    X_train=X_train,
    y_train=y_train,
    pipeline_preprocess=pipeline_preprocess
)




# %%
def evaluate_with_threshold(
    model,
    feature_list,
    X_train, y_train,
    X_test, y_test,
    pipeline_preprocess,
    threshold
):
    # Fit preprocessing on FULL training set
    X_train_sel, pp_fitted = preprocess_and_select(
        X_train, pipeline_preprocess, feature_list, fit=True
    )
    X_test_sel, _ = preprocess_and_select(
        X_test, pp_fitted, feature_list, fit=False
    )

    # Fit model on full training
    model.fit(X_train_sel, y_train)

    # Probabilities on test
    test_proba = model.predict_proba(X_test_sel)[:, 1]
    y_pred = (test_proba >= threshold).astype(int)

    # Metrics
    test_acc  = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec  = recall_score(y_test, y_pred)
    test_f1   = f1_score(y_test, y_pred)

    return {
        "threshold": threshold,
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1
    }




# %%
# Evaluate on test with tuned threshold

threshold_results = evaluate_with_threshold(
    model=make_final_xgb_baseline(),
    feature_list=features_moderate,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    pipeline_preprocess=pipeline_preprocess,
    threshold=best_thr
)

threshold_results




# %% [markdown]
#   Although the single validation split suggested an optimal threshold of approximately 0.55, applying this threshold to the test set reduced the F1 score to 0.41, which is worse than the baseline model’s performance using the default 0.50 threshold. This indicates that the threshold estimate obtained from a single split may not be stable. To address this, we'll repeat the threshold-optimization procedure using stratified K-fold cross-validation, which provides a more robust and reliable threshold by averaging performance across multiple folds of the training data.

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def get_oof_proba_cv(
    model_factory,
    feature_list,
    X_train,
    y_train,
    pipeline_preprocess,
    n_splits=5,
    random_state=42
):
    """
    Returns out-of-fold predicted probabilities for the positive class
    for all rows in X_train, using StratifiedKFold CV.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_proba = np.zeros(len(X_train))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"Fold {fold}/{n_splits}")

        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # clone preprocessing pipeline so each fold is independent
        pp = clone(pipeline_preprocess)

        # fit preprocess on fold-train only
        X_tr_sel, pp_fitted = preprocess_and_select(
            X_tr, pp, feature_list, fit=True
        )
        X_val_sel, _ = preprocess_and_select(
            X_val, pp_fitted, feature_list, fit=False
        )

        # fresh model for this fold
        model = model_factory()
        model.fit(X_tr_sel, y_tr)

        proba_val = model.predict_proba(X_val_sel)[:, 1]
        oof_proba[val_idx] = proba_val

    return oof_proba




# %%


oof_proba = get_oof_proba_cv(
    model_factory=make_final_xgb_baseline,
    feature_list=features_moderate,
    X_train=X_train,
    y_train=y_train,
    pipeline_preprocess=pipeline_preprocess,
    n_splits=5
)




# %%
def tune_threshold_from_oof(y_true, proba, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)

    rows = []
    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        rows.append((thr, f1, prec, rec))

    thr_df = pd.DataFrame(rows, columns=["threshold", "f1", "precision", "recall"])
    best_row = thr_df.loc[thr_df["f1"].idxmax()]
    print("Best threshold from OOF predictions:")
    print(best_row)

    return best_row["threshold"], thr_df




# %%
best_thr_cv, thr_df_cv = tune_threshold_from_oof(y_train, oof_proba)




# %%
threshold_results_cv = evaluate_with_threshold(
    model=make_final_xgb_baseline(),
    feature_list=features_moderate,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    pipeline_preprocess=pipeline_preprocess,
    threshold=best_thr_cv
)

threshold_results_cv




# %% [markdown]
#   Using stratified K-fold cross-validation, the threshold-optimization procedure produced a more stable estimate, again identifying a threshold close to 0.55. When this cross-validated threshold was applied to the test set, the model achieved an F1 score of 0.434, which is slightly better than the single-split approach but still below the baseline F1 obtained with the default 0.50 threshold. This confirms that, for this dataset, adjusting the decision threshold does not yield a consistent improvement over the baseline setting.

# %%
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Our final threshold (from baseline model)
final_threshold = 0.50   

# Generate final predictions 
y_pred = (test_proba >= final_threshold).astype(int)

# Computing metrics 
test_accuracy  = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall    = recall_score(y_test, y_pred)
test_f1        = f1_score(y_test, y_pred)

# Building a metrics table 
metrics_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1"],
    "Score": [test_accuracy, test_precision, test_recall, test_f1]
})

print("Final Classification Metrics (threshold = {})".format(final_threshold))
metrics_table




# %%
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_Stay (0)", "Actual_Leave (1)"],
    columns=["Pred_Stay (0)", "Pred_Leave (1)"]
)

print("Confusion Matrix (threshold = {})".format(final_threshold))
cm_df



# %% [markdown]
#   Teste Pedro - threshold optimization from the f1 / threshold curve
# 
# 
# 
#   Although the baseline model used a default threshold of 0.50, our earlier analysis of the Threshold–F1 curve indicated that a lower threshold of approximately 0.38 maximizes the F1 score on the test set. By adopting this optimized threshold, we can potentially improve the balance between precision and recall, leading to better overall classification performance in identifying employees at risk of attrition.
# 
# 
# 
#   Our final threshold (from baseline model)
# 
# 
# 
#   Generate final predictions
# 
# 
# 
#   Computing metrics
# 
# 
# 
#   Building a metrics table

# %%
# Confusion matrix Teste Pedro
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_Stay (0)", "Actual_Leave (1)"],
    columns=["Pred_Stay (0)", "Pred_Leave (1)"]
)

#print("Confusion Matrix (threshold = {})".format(test_threshold))
cm_df

# We can see that by lowering the threshold to 0.38, the model identifies more true positives (employees who leave) at the cost of increasing false positives (employees incorrectly predicted to leave). This trade-off is reflected in the updated precision and recall scores, which should be carefully considered based on the organization's priorities regarding attrition management.



# %% [markdown]
#  # Model Explainability

# %% [markdown]
#  ## SHAP Analysis

# %% [markdown]
#  Compute SHAP values (TreeExplainer)

# %%
# X_train_sel: already computed earlier with features_moderate and fit=True
# model: already trained on X_train_sel

X_for_shap = X_train_sel

import numpy as np
import shap
shap.initjs()

shap_sample_size = 300
if X_for_shap.shape[0] > shap_sample_size:
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(
        X_for_shap.shape[0],
        size=shap_sample_size,
        replace=False
    )
    X_shap = X_for_shap[sample_idx]
else:
    X_shap = X_for_shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)



# %% [markdown]
#  ## Global SHAP plots

# %% [markdown]
# To better understand how our XGBoost model makes predictions, we now turn to SHAP (SHapley Additive exPlanations). SHAP sheds light on the "black box" behind predictions by showing how each feature influences the model’s output, both at the dataset level and for individual employees.

# %% [markdown]
#  Bar plot: feature importance ranking

# %%
shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=features_moderate,
    plot_type="bar"
)




# %% [markdown]
# This SHAP summary plot highlights the features that contribute most to the model’s attrition predictions. The most influential factors include OverTime, StockOptionLevel, EngagementIndex, Age, and MonthlyIncome, indicating that workload, compensation, engagement, and career stage play central roles in predicting whether an employee will leave. Several job-related and financial features, such as JobRole (Research Scientist), NumCompaniesWorked, DailyRate, and Income vs. Role Median also show meaningful impact. Meanwhile, features related to satisfaction, work–life balance, distance from home, and duration under current manager contribute moderately but consistently. Lower-ranked features still affect predictions, but with smaller average influence.

# %% [markdown]
#  Beeswarm: direction + magnitude

# %%
shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=features_moderate
)


# %% [markdown]
# This SHAP beeswarm plot provides a detailed view of how individual feature values influence the model’s predictions for employee attrition. Features appear in order of overall importance, and each point represents an employee. The color indicates whether the feature value is high (pink) or low (blue), while the position on the x-axis shows whether that value increases or decreases the predicted probability of quitting.
# 
# A few patterns stand out clearly:
# 
# - Low OverTime (0) strongly reduces attrition risk, while high OverTime tends to push predictions toward quitting.
# 
# - Lower StockOptionLevel, lower EngagementIndex, and younger Age similarly drive predictions upward, indicating higher risk.
# 
# - Higher MonthlyIncome and higher IncomeVsRoleMedian generally push predictions downward, aligning with retention.
# 
# - For several features—such as NumCompaniesWorked, DailyRate, and DistanceFromHome—both high and low values can influence the prediction direction, reflecting more complex, nonlinear relationships the model has learned.
# 
# Overall, this plot shows not just which features matter most, but also how specific feature values contribute to individual attrition predictions.

# %% [markdown]
# Interaction plots

# %%
# If X_shap is numpy, wrap it for readability
X_shap_df = pd.DataFrame(X_shap, columns=features_moderate)

# Engagement vs SHAP, colored by Overtime 
shap.dependence_plot(
    "num__EngagementIndex",
    shap_values,
    X_shap_df,
    interaction_index="nom__OverTime_0"   # colour by overtime
)


# %% [markdown]
# This SHAP dependence plot shows how the EngagementIndex influences the model’s prediction of attrition, while also highlighting its interaction with OverTime. As EngagementIndex increases, SHAP values clearly decrease, indicating that more engaged employees are predicted to be at lower risk of quitting. Conversely, lower engagement strongly pushes the model toward predicting attrition.
# 
# The color scale reveals an interaction effect: employees who do not work overtime (blue) generally have slightly lower SHAP values at similar engagement levels, reinforcing a lower predicted risk. Those who do work overtime (pink) tend to contribute more positively to the attrition prediction, even when engagement is moderate.
# 
# Overall, the plot shows a clean, monotonic relationship: lower engagement → higher attrition risk, with overtime amplifying this effect.

# %%


# If shap_values is a list, take the positive class
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values  # (n_samples, n_features)

# total contribution toward attrition risk (positive class)
total_shap = sv.sum(axis=1)

# Top-k highest risk explanations (largest positive SHAP sums)
k = 5
top_pos_idx = np.argsort(total_shap)[-k:][::-1]

# Top-k lowest risk explanations (most negative SHAP sums)
top_neg_idx = np.argsort(total_shap)[:k]

print("Top high-risk examples (by SHAP):", top_pos_idx)
print("Top low-risk examples (by SHAP):", top_neg_idx)



# %% [markdown]
# Finding the eployee most at risk for attrition and the top variables contributing for such risk

# %%
def shap_example_df(idx):
    return pd.DataFrame({
        "feature": features_moderate,
        "value": X_shap_df.iloc[idx].values,
        "shap_value": sv[idx]
    }).sort_values("shap_value", ascending=False)

# Example: inspect the most at-risk employee
high_risk_example = shap_example_df(top_pos_idx[0])
#low_risk_example  = shap_example_df(top_neg_idx[0])

high_risk_example.head(10)  # top 10 contributors for high-risk case
#low_risk_example.head(10)   # top 10 contributors for low-risk case



# %%
# High-risk case
shap.force_plot(
    explainer.expected_value,
    sv[top_pos_idx[0], :],
    X_shap_df.iloc[top_pos_idx[0], :],
    feature_names=features_moderate
)

# Low-risk case
#shap.force_plot(
#    explainer.expected_value,
#    sv[top_neg_idx[0], :],
#    X_shap_df.iloc[top_neg_idx[0], :],
#    feature_names=features_moderate
#)



# %% [markdown]
#  # SVC hyperparameter tuning using randomized search

# %% [markdown]
# From the base models we trained, our second best was a Support Vector Machine model using the restricted set of features. Let's try to optimize it and see how it compares with the XGBoost model above.

# %% [markdown]
# We start by tuning it with RandomizedSearchCV

# %%

RANDOM_STATE = 42  # keep it consistent

def tune_svc_random_search(
    feature_list,
    X_train, y_train,
    pipeline_preprocess,
    n_splits=5,
    n_iter=40
):
    """
    Tune an SVC on the selected features using RandomizedSearchCV.
    Returns (best_estimator, search_object).
    """

    # 1. Preprocess + select features (same logic as evaluate_model_preprocessed) 
    pipeline_preprocess.fit(X_train)
    X_train_prep = pipeline_preprocess.transform(X_train)

    feature_names = pipeline_preprocess.get_feature_names_out()
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    idx = [name_to_idx[f] for f in feature_list]
    X_train_sel = X_train_prep[:, idx]

    # 2. Base SVC model 
    base_svc = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

    # 3. Search space 
    param_dist = {
        "C":     np.logspace(-2, 3, 30),   # 0.01 → 1000
        "gamma": np.logspace(-4, 1, 30),   # 1e-4 → 10
        "kernel": ["rbf"]                  # you can add 'linear' if you want
    }

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator=base_svc,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",          # aligns with cv_f1 metric
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True             # refit on full training set with best params
    )

    # 4. Run search 
    search.fit(X_train_sel, y_train)

    print("Best params:", search.best_params_)
    print("Best CV F1 (RandomizedSearchCV):", search.best_score_)

    best_svc = search.best_estimator_
    return best_svc, search



# %%
# choose feature set to tune on
svc_features = features_strict   # or features_moderate / all_features

best_svc, svc_search = tune_svc_random_search(
    feature_list=svc_features,
    X_train=X_train,
    y_train=y_train,
    pipeline_preprocess=pipeline_preprocess,
    n_splits=5,
    n_iter=40
)



# %%
svc_tuned_results = evaluate_model_preprocessed(
    model=best_svc,
    feature_list=svc_features,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    pipeline_preprocess=pipeline_preprocess
)

svc_tuned_results["model"] = "SVC_tuned"
svc_tuned_results["feature_set"] = "strict"   # or whatever you used

pd.DataFrame([svc_tuned_results])



# %% [markdown]
#  # SVC hyperparameter tuning using grid search

# %% [markdown]
# We used the best parameters identified in the randomized search to guide the selection of values explored in the grid search.

# %%
best_C = 8.531678524172815
best_gamma = 0.011721022975334805

# helper: create log-spaced values around a center (factor ≈ 2 range)
def local_log_grid(center, log_half_width=0.3, n=5):
    """
    center: positive float
    log_half_width: how far (in log10 space) to go on each side
                    0.3 ≈ factor of 2 range (10^0.3 ≈ 2)
    n: number of points
    """
    log_c = np.log10(center)
    return np.logspace(log_c - log_half_width, log_c + log_half_width, n)

C_grid = local_log_grid(best_C, log_half_width=0.3, n=5)
gamma_grid = local_log_grid(best_gamma, log_half_width=0.3, n=5)

print("C_grid:", C_grid)
print("gamma_grid:", gamma_grid)

param_grid = {
    "C": C_grid,
    "gamma": gamma_grid,
    "kernel": ["rbf"]
}



# %%
# preprocess + select the same feature set you tuned on with random search 
pipeline_preprocess.fit(X_train)
X_train_prep = pipeline_preprocess.transform(X_train)

feature_names = pipeline_preprocess.get_feature_names_out()
name_to_idx = {name: i for i, name in enumerate(feature_names)}
idx = [name_to_idx[f] for f in features_strict]   # or moderate/all_feats
X_train_sel = X_train_prep[:, idx]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svc_base = SVC(
    probability=True,
    class_weight="balanced",
    random_state=42
)

grid_search = GridSearchCV(
    estimator=svc_base,
    param_grid=param_grid,
    scoring="f1",      
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid_search.fit(X_train_sel, y_train)

print("Best params (local grid):", grid_search.best_params_)
print("Best CV F1 (local grid):", grid_search.best_score_)

best_svc_local = grid_search.best_estimator_



# %%
svc_local_results = evaluate_model_preprocessed(
    model=best_svc_local,
    feature_list=features_strict,      # or whichever I want to test
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    pipeline_preprocess=pipeline_preprocess
)

svc_local_results["model"] = "SVC_tuned_local"
svc_local_results["feature_set"] = "strict"

pd.DataFrame([svc_local_results])



# %%
# Preprocess + select features ONCE
X_train_sel, pp_fitted = preprocess_and_select(
    X_train, pipeline_preprocess, features_strict, fit=True
)
X_test_sel, _ = preprocess_and_select(
    X_test, pp_fitted, features_strict, fit=False
)

# 2) Fit your already-tuned SVC on the selected train features
best_svc_local.fit(X_train_sel, y_train)


# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve_fitted(model, X_test_sel, y_test):
    # model is already fitted, X_test_sel is already preprocessed+selected
    y_proba = model.predict_proba(X_test_sel)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – SVC (strict features)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# %%
plot_roc_curve_fitted(best_svc_local, X_test_sel, y_test)


# %%
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def plot_pr_curve_fitted(model, X_test_sel, y_test):

    # model is already fitted
    y_proba = model.predict_proba(X_test_sel)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (AP = {ap:.3f})")
    plt.grid(alpha=0.3)
    plt.show()


# %%
plot_pr_curve_fitted(best_svc_local, X_test_sel, y_test)


# %%
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def plot_threshold_f1_fitted(model, X_test_sel, y_test):

    # model is already fitted
    y_proba = model.predict_proba(X_test_sel)[:, 1]

    thresholds = np.linspace(0, 1, 201)   # step = 0.005
    f1_scores = [f1_score(y_test, (y_proba >= thr).astype(int)) for thr in thresholds]

    best_thr = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, f1_scores, linewidth=2)
    plt.axvline(best_thr, color='red', linestyle='--',
                label=f"Best thr = {best_thr:.3f}, F1 = {best_f1:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print(f"Best threshold (test-based): {best_thr:.3f}")
    print(f"Best F1 on test: {best_f1:.3f}")

    return best_thr, best_f1


# %%
best_thr_test, best_f1_test = plot_threshold_f1_fitted(
    best_svc_local,
    X_test_sel,
    y_test
)


# %% [markdown]
#  # SVC Threshold Optimization

# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import clone

RANDOM_STATE = 42

def optimize_threshold_repeated_cv(
    base_model,
    feature_list,
    X_train, y_train,
    pipeline_preprocess,
    n_splits=5,
    n_repeats=5,
    thresholds=None
):
    """
    Returns:
        best_thr: float
        best_f1:  float (CV estimate at best_thr)
        thr_df:   DataFrame with (threshold, f1)
    """

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)  # 0.01 step

    # ----- 1. Preprocess + select features (same style as your other code) -----
    pipeline_preprocess.fit(X_train)
    X_prep = pipeline_preprocess.transform(X_train)

    feature_names = pipeline_preprocess.get_feature_names_out()
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    idx = [name_to_idx[f] for f in feature_list]
    X_sel = X_prep[:, idx]

    # ----- 2. Repeated stratified CV -----
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE
    )

    all_probas = []
    all_y = []

    for tr_idx, val_idx in rskf.split(X_sel, y_train):
        X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = clone(base_model)
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_val)[:, 1]
        all_probas.append(proba)
        all_y.append(y_val.values)

    all_probas = np.concatenate(all_probas)
    all_y = np.concatenate(all_y)

    # ----- 3. Scan thresholds -----
    best_thr = 0.5
    best_f1 = -1
    rows = []

    for thr in thresholds:
        preds = (all_probas >= thr).astype(int)
        f1 = f1_score(all_y, preds)
        rows.append((thr, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    thr_df = pd.DataFrame(rows, columns=["threshold", "f1"])

    print(f"Best threshold: {best_thr:.3f}, CV F1: {best_f1:.4f}")

    return best_thr, best_f1, thr_df



# %%
# choose which SVC to optimize:
# base_svc = make_svc()                 # baseline SVC
base_svc = best_svc_local               # tuned SVC from grid search

best_thr, best_f1_cv, thr_curve = optimize_threshold_repeated_cv(
    base_model=base_svc,
    feature_list=features_strict,        # or moderate/all_feats
    X_train=X_train,
    y_train=y_train,
    pipeline_preprocess=pipeline_preprocess,
    n_splits=5,
    n_repeats=5
)



# %%
# ----- Preprocess train + test with same pipeline -----
pipeline_preprocess.fit(X_train)
X_train_prep = pipeline_preprocess.transform(X_train)
X_test_prep  = pipeline_preprocess.transform(X_test)

feature_names = pipeline_preprocess.get_feature_names_out()
name_to_idx = {name: i for i, name in enumerate(feature_names)}
idx = [name_to_idx[f] for f in features_strict]    # same as above

X_train_sel = X_train_prep[:, idx]
X_test_sel  = X_test_prep[:, idx]

# ----- Fit model on full training data -----
final_svc = clone(base_svc)   # same hyperparams as used in threshold search
final_svc.fit(X_train_sel, y_train)

# ----- Probabilities + thresholded predictions on test -----
test_proba = final_svc.predict_proba(X_test_sel)[:, 1]
test_preds_opt = (test_proba >= best_thr).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test_results_thr = {
    "test_accuracy":  accuracy_score(y_test, test_preds_opt),
    "test_precision": precision_score(y_test, test_preds_opt),
    "test_recall":    recall_score(y_test, test_preds_opt),
    "test_f1":        f1_score(y_test, test_preds_opt)
}
test_results_thr





# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(model, X_test_sel, y_test, threshold=None):
    # probabilities
    proba = model.predict_proba(X_test_sel)[:, 1]
    preds = (proba >= threshold).astype(int) if threshold is not None else model.predict(X_test_sel)
    
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stay", "Quit"])
    
    plt.figure(figsize=(5, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (threshold = {threshold:.2f})" if threshold else "Confusion Matrix")
    plt.show()


# %%
plot_confusion_matrix(final_svc, X_test_sel, y_test, threshold=best_thr)


# %% [markdown]
# If we hadn't optimized the threshold, our confusion matrix would be the one below

# %%
plot_confusion_matrix(final_svc, X_test_sel, y_test, threshold=0.5)


# %%
plot_roc_curve_fitted(final_svc, X_test_sel, y_test)


# %%
plot_pr_curve_fitted(final_svc, X_test_sel, y_test)


# %%
plot_threshold_f1_fitted(final_svc, X_test_sel, y_test)


# %%
from sklearn.calibration import calibration_curve

def plot_calibration_curve(model, X_test_sel, y_test):
    y_proba = model.predict_proba(X_test_sel)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title("Calibration Curve – SVC")
    plt.grid(alpha=0.3)
    plt.show()


# %%
plot_calibration_curve(final_svc, X_test_sel, y_test)


# %%
# feature names after selection (in correct order)
selected_feature_names = [feature_names[i] for i in idx]

# %%
# pick a background sample from training for Kernel SHAP
background_size = 100  # you can change; 50–200 is typical
background_idx = np.random.RandomState(42).choice(
    X_train_sel.shape[0],
    size=min(background_size, X_train_sel.shape[0]),
    replace=False
)
background = X_train_sel[background_idx]

# define a prediction function that returns P(class=1)
f_pred = lambda X: final_svc.predict_proba(X)[:, 1]

explainer = shap.KernelExplainer(f_pred, background)


# %%
# choose how many test samples to explain
n_explain = 200  # adjust as needed
X_test_explain = X_test_sel[:n_explain]

# shap_values will be a 1D array (since f_pred returns P(class=1))
shap_values = explainer.shap_values(X_test_explain, nsamples="auto")


# %%
shap.summary_plot(
    shap_values,
    X_test_explain,
    feature_names=selected_feature_names,
    plot_type="bar",
    show=True
)


# %%
shap.summary_plot(
    shap_values,
    X_test_explain,
    feature_names=selected_feature_names,
    show=True
)


# %%
shap.dependence_plot(
    "num__Age",                  # or any name in selected_feature_names
    shap_values,
    X_test_explain,
    feature_names=selected_feature_names,
    interaction_index="auto"     # SHAP will pick most interacting feature
)


# %%
# make dataframe of shap values
shap_df = pd.DataFrame(
    shap_values,
    columns=selected_feature_names,
    index=y_test.iloc[:n_explain].index   # align with subset of test
)

# attach target and predicted probability
y_test_explain = y_test.iloc[:n_explain]
test_proba_explain = final_svc.predict_proba(X_test_explain)[:, 1]

shap_df["y_true"] = y_test_explain.values
shap_df["y_pred_proba"] = test_proba_explain
shap_df["shap_sum"] = shap_df[selected_feature_names].sum(axis=1)

# top examples where SHAP strongly pushes towards Quit (class 1)
top_quit_examples = shap_df.sort_values("shap_sum", ascending=False).head(10)

# top examples where SHAP strongly pushes towards Stay (class 0)
#top_stay_examples = shap_df.sort_values("shap_sum", ascending=True).head(10)

top_quit_examples#, top_stay_examples


# %% [markdown]
# # Ensemble Model

# %%
# 1) Fit preprocess on train and get all transformed feature names
pipeline_preprocess.fit(X_train)
all_features = list(pipeline_preprocess.get_feature_names_out())

# 2) Use your helper to select ALL features
X_train_all, pp_fitted = preprocess_and_select(
    X_train, pipeline_preprocess, all_features, fit=True
)
X_test_all, _ = preprocess_and_select(
    X_test, pp_fitted, all_features, fit=False
)


# %%
from sklearn.base import clone

# base_svc = the tuned SVC hyperparameters you liked (e.g. best_svc_local)
svc_ens = clone(best_svc_local)
svc_ens.fit(X_train_all, y_train)

# XGB with same hyperparams as before, just retrained on all features
xgb_ens = clone(model)   # or clone of your tuned XGB
xgb_ens.fit(X_train_all, y_train)


# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np

def soft_voting_proba(model_svc, model_xgb, X):
    p_svc = model_svc.predict_proba(X)[:, 1]
    p_xgb = model_xgb.predict_proba(X)[:, 1]
    return 0.5 * (p_svc + p_xgb)    # equal weights


# %%
proba_ens_test = soft_voting_proba(svc_ens, xgb_ens, X_test_all)
pred_ens_test = (proba_ens_test >= best_thr).astype(int)   # or 0.5


ens_results = {
    "test_accuracy":  accuracy_score(y_test, pred_ens_test),
    "test_precision": precision_score(y_test, pred_ens_test),
    "test_recall":    recall_score(y_test, pred_ens_test),
    "test_f1":        f1_score(y_test, pred_ens_test)
}
ens_results


# %%
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def weighted_voting_proba(model_svc, model_xgb, X, w_svc):
    """
    w_svc: weight for SVC in [0,1]
    weight for XGB is (1 - w_svc)
    """
    p_svc = model_svc.predict_proba(X)[:, 1]
    p_xgb = model_xgb.predict_proba(X)[:, 1]
    return w_svc * p_svc + (1.0 - w_svc) * p_xgb


# %%
def evaluate_weight_grid(model_svc, model_xgb, X_test, y_test,
                         w_values=None, thr_values=None):
    if w_values is None:
        w_values = np.linspace(0.0, 1.0, 11)   # 0.0, 0.1, ..., 1.0
    if thr_values is None:
        thr_values = np.linspace(0.1, 0.9, 81) # 0.1 to 0.9 step 0.01

    rows = []

    for w in w_values:
        proba = weighted_voting_proba(model_svc, model_xgb, X_test, w)
        for thr in thr_values:
            preds = (proba >= thr).astype(int)
            f1 = f1_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            acc = accuracy_score(y_test, preds)
            rows.append({
                "w_svc": w,
                "w_xgb": 1 - w,
                "threshold": thr,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            })

    results = pd.DataFrame(rows)
    best_row = results.loc[results["f1"].idxmax()]

    print("Best ensemble config on test:")
    print(best_row)

    return results, best_row


# %%
ens_results_df, best_ens = evaluate_weight_grid(
    svc_ens, xgb_ens,
    X_test_all, y_test
)



