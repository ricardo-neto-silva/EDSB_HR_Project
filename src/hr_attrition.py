# %% [markdown]
#   ## **_Enterprise Data Science and Analytics - Enterprise Data Science Bootcamp_**
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
#   - Ana Rita Martins 20240821
# 
# 
# 
#   - Joana Coelho 20240801
# 
# 
# 
#   - Pedro Fernandes 20240823
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
#   In Human Resources, predictive analytics supports critical functions such as employee retention, workforce planning, and automated CV screening.
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
#   From the summary above, we verified that the data set does't contain duplicates, and we also gathered information about the data's distribution and main statistics.
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
#   - Once again we can observe that the **target variable** is highly skewed toward staying in the company.
# 
# 
# 
#   - Concerning demographics, **age** follows an approximately bell-shaped distribution, centered around 30-40; **Gender** is skewed with more males than females.
# 
# 
# 
#   - Features that are related to **work characteristics** (YearsAtCompany, TotalWorkingYears, YearsInCurrentRole, Overtime) are right-skewed, indicating many relatively new employees and fewer with long careers; working overtime is not common.
# 
# 
# 
#   - **Income**: Salaries and rates are right-skewed, with few very high earners.
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
#   - Outliers are especially relevant in income and emplyment duration related-variables, which may need special handling. We'll decide how to handle them further down.
# 
# 
# 
#   - For demographic/job characteristics (Age, DistanceFromHome, JobLevel, Education) featured the distributions are fairly compact with few outliers, aligning with the unimodal/bell-like shapes seen in histograms.
# 
# 
# 
#   - Ordinal satisfaction and variables show limited spread, consistent with their discrete scale, with some level of skew toward higher values. Their limited range may reduce their explanatory power.
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
#   From the variables that, a priori, we'd think could be related with attrition we find that:
# 
# 
# 
#    - roughly 30% of employees work overtime
# 
# 
# 
#    - roughly 40% have low to medium levels of satisfaction with the work environment
# 
# 
# 
#    - roughly 30% report low to medium levels of job involvement
# 
# 
# 
#    - nearly 40% report low to medium job satisfaction
# 
# 
# 
#    - another nearly 40% have low to medium levels of satisfaction with relationships at work
# 
# 
# 
#    - and about 5% report bad work-life balance

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
#   Department-level & Job roles:
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
#   Personal characteristics
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
#  Let's now run an equivalent analysis with our continuous features.
# 
# 
# 
#  We'll plot both their probability density function and violin plots and assess how their distribution relates to the target.

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
# 
# 
# 
# 
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
#  We’ll now take look at the correlations among the features, including the target variable. This will help us identify potential collinearity, as well as highlight which features are associated with attrition. Since several features are not strictly numeric or continuous, we’ll use Spearman’s correlation, which measures monotonic relationships by correlating feature ranks rather than their raw values.

# %% [markdown]
#  We'll exclude strictly nominal categorical variables (like Gender, Department, JobRole) because Spearman is rank-based, not meant for unordered categories.

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
# 
# 
#  - YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager, TotalWorkingYears, JobLevel, MonthlyIncome, StockOptionLevel and Age are the strongest monotonic predictors of Attrition.
# 
# 
# 
#  These are indicators that relate to tenure, seniority, and stability and they're in agreement with HR domain knowledge: attrition is highest among newer, younger, lower-level employees.
# 
# 
# 
# 
# 
#  - JobSatisfaction, JobInvolvement, EnvironmentSatisfaction Tshow mild but potentially meaningful associations.
# 
# 
# 
#  Employees with lower satisfaction or lower involvement show slightly higher attrition.
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
# The heatmap shows that several of the variables most strongly correlated with attrition are also highly collinear with each other. In particular, the following groups demonstrate very strong monotonic relationships (ρ > 0.70):
# 
# - JobLevel — MonthlyIncome (ρ ≈ 0.92)
# 
# - YearsInCurrentRole — YearsWithCurrManager (ρ ≈ 0.85)
# 
# - TotalWorkingYears — MonthlyIncome (ρ ≈ 0.71)
# 
# These features are all measures of: Tenure, Seniority, Career progression, Employee stability, which explains why they are tightly correlated with each other and with lower attrition.

# %% [markdown]
# While colinearity doesn't harm tree-based models, it does affect linear models like linear regression. Besides, it It also leads to unnecessary redundancy in the feature set. Keeping all of them increases the demand for computational powerr and increases the risk of overfitting. By the end of our feature selection process, we should aim to keep at most 2 or 3 representative variables of this set. And for regression models, we'll explicitly remove correlated pairs.
# Another way to circumvent colinearity is to combine several colinear raw variables into a single engineered feature. Let's do that below.
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
# # Preprocessing Steps

# %% [markdown]
# ## Train-Test Split

# %% [markdown]
# Before any encoding and feature selection steps we'll start by defining X and y, and defining the train–test split. Doing this at this satge is critical to avoid data leakage.

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
# ## Rebuilding feature groups on X_train

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
# ## Defining the preprocessing (encoders + passthrough)

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
# When preprocess.fit(X_train) is called, it learns: category mappings for ordinal features; dummy columns for nominal + binary. 
# Then preprocess.transform(...) will apply this same mapping to train & test.

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

print("Number of feature names:", len(feature_names))


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
# # Feature Selection

# %% [markdown]
# ## Chi-square

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
# Features with Selected = True are part of the sparse LASSO-selected subset. Larger coefficients (in magnitude) reflect stronger linear effect.

# %% [markdown]
# ## Random Forest Classifier

# %% [markdown]
# Random Forest captures: nonlinearities, interactions, categorical effects, monotonic or non-monotonic patterns. Works very well alongside LASSO.

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
# XGBoost is often very strong at discovering: threshold effects, feature interactions, nonlinear jump patterns, sparse informative features.

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


# %%
# Base table: one row per encoded feature
unified_fs = pd.DataFrame({
    "Feature": X_train_df.columns
})


# %%
# Keep only needed columns from MI
mi_short = mi_results[["Feature", "MI"]]

unified_fs = unified_fs.merge(
    mi_short,
    on="Feature",
    how="left"
)


# %%
l1_short = l1_results[["Feature", "Coefficient", "Selected"]]

unified_fs = unified_fs.merge(
    l1_short,
    on="Feature",
    how="left"
)


# %%
rf_short = rf_results.rename(columns={"Importance": "RF_importance"})[
    ["Feature", "RF_importance"]
]

unified_fs = unified_fs.merge(
    rf_short,
    on="Feature",
    how="left"
)


# %%
xgb_short = xgb_results.rename(columns={"Importance": "XGB_importance"})[
    ["Feature", "XGB_importance"]
]

unified_fs = unified_fs.merge(
    xgb_short,
    on="Feature",
    how="left"
)


# %%
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
# Flag whether feature is discrete (ordinal or one-hot)
unified_fs["is_discrete"] = unified_fs["Feature"].str.startswith(("ord__", "nom__"))

# Example sorting: by Random Forest importance (descending)
unified_fs_sorted = unified_fs.sort_values(
    by=["RF_importance", "XGB_importance", "MI"],
    ascending=False
)

unified_fs_sorted.head(30)


# %%
df = unified_fs.copy()  # just to keep the original safe

# --- Fill NaNs with 0 where it makes sense ---
df["MI"] = df["MI"].fillna(0)
df["RF_importance"] = df["RF_importance"].fillna(0)
df["XGB_importance"] = df["XGB_importance"].fillna(0)

# chi2_significant may be NaN for numeric features; treat those as False
df["chi2_significant"] = df["chi2_significant"].fillna(False)

# L1 Selected may be NaN for some features; treat as False
df["Selected"] = df["Selected"].fillna(False)


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


# %%
method_flags = ["chi2_good", "mi_good", "l1_good", "rf_good", "xgb_good"]

# Convert to int and sum
df["consensus_score"] = df[method_flags].astype(int).sum(axis=1)

# Optional: see distribution
print(df["consensus_score"].value_counts().sort_index())


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


# %% [markdown]
# # Modelling

# %%
def evaluate_model(model, feature_list, 
                   X_train, y_train, 
                   X_test, y_test, 
                   pipeline_preprocess, 
                   n_splits=5):
    """
    Train and evaluate a model using selected features.
    
    Steps:
    - Select raw features
    - Apply preprocessing (encode + scale)
    - Run CV inside training
    - Train final model on full training set
    - Evaluate on test set

    Returns:
        results_dict containing CV metrics and test metrics
    """

    # -----------------------------
    # 1. Select only chosen features
    # -----------------------------
    X_train_sel = X_train[feature_list].copy()
    X_test_sel  = X_test[feature_list].copy()

    # -----------------------------
    # 2. Preprocess (fit on train only)
    # -----------------------------
    # Fit the preprocessing on training subset
    pipeline_preprocess.fit(X_train_sel)
    X_train_prep = pipeline_preprocess.transform(X_train_sel)
    X_test_prep  = pipeline_preprocess.transform(X_test_sel)

    # -----------------------------
    # 3. Cross-validation
    # -----------------------------
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    f1_scores, acc_scores, prec_scores, rec_scores = [], [], [], []

    for train_idx, val_idx in skf.split(X_train_prep, y_train):
        X_tr_cv, X_val_cv = X_train_prep[train_idx], X_train_prep[val_idx]
        y_tr_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr_cv, y_tr_cv)
        preds = model.predict(X_val_cv)

        acc_scores.append(accuracy_score(y_val_cv, preds))
        prec_scores.append(precision_score(y_val_cv, preds))
        rec_scores.append(recall_score(y_val_cv, preds))
        f1_scores.append(f1_score(y_val_cv, preds))

    # -----------------------------
    # 4. Fit final model on full training set
    # -----------------------------
    model.fit(X_train_prep, y_train)

    # -----------------------------
    # 5. Evaluate on test set
    # -----------------------------
    test_preds = model.predict(X_test_prep)

    test_acc  = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds)
    test_rec  = recall_score(y_test, test_preds)
    test_f1   = f1_score(y_test, test_preds)

    # -----------------------------
    # 6. Package results
    # -----------------------------
    results = {
        "cv_accuracy":   np.mean(acc_scores),
        "cv_precision":  np.mean(prec_scores),
        "cv_recall":     np.mean(rec_scores),
        "cv_f1":         np.mean(f1_scores),

        "test_accuracy":  test_acc,
        "test_precision": test_prec,
        "test_recall":    test_rec,
        "test_f1":        test_f1
    }

    return results



