# %% [markdown]
#  ## **_Enterprise Data Science and Analytics - Enterprise Data Science Bootcamp_**
# 
# 
# 
#  ### **HR Attrition Project - EDSB25_26**
# 
# 
# 
#  - Ana Rita Martins 20240821
# 
#  - Joana Coelho 20240801
# 
#  - Pedro Fernandes 20240823
# 
#  - Ricardo Silva 20240824

# %% [markdown]
#  Data Science and Analytics are reshaping how organizations solve problems across diverse industries. Through systematic data analysis and predictive modeling, evidence-based solutions can be developed, enabling more reliable decision-making and greater efficiency.
# 
# 
# 
#  In Human Resources, predictive analytics supports critical functions such as employee retention, workforce planning, and automated CV screening.
# 
# 
# 
#  This project focuses on developing predictive models to assess the likelihood of employee resignation. By analyzing factors ranging from demographics to job satisfaction, the models aim to provide interpretable insights that highlight key drivers of attrition. These insights will help HR leaders take proactive steps to reduce turnover and retain talent.

# %% [markdown]
#  ## 1. Importing Packages

# %%
import numpy as np
import pandas as pd
from summarytools import dfSummary
import matplotlib.pyplot as plt
import seaborn as sns
#from pandas.io.formats.style import Styler


# %% [markdown]
#  ## 2. Importing Data and Initial Exploration

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
#  From this initial inspection what immediately stands out is that we have 3 constant features: "EmployeeCount", "StandardHours", and "Over18". We can remove those straight away. Additionally, the employee number (ID) feature, does not seem to contain any relevant info, and  we'll drop it too.

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
#  From the summary above, we verified that the data set does't contain duplicates, and we also gathered information about the data's distribution and main statistics.
# 
# 
# 
#  What we can note is that, beasides our target, we have a couple of other binary features. Let's encode those.

# %%
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})

data.head()


# %% [markdown]
#  Let's now have a look at how the distribution of the target variable.

# %%
ax = sns.countplot(x=data['Attrition'], hue=data['Attrition'], legend=False)
for container in ax.containers:
    ax.bar_label(container)

plt.title('Distribution of the Target Variable (Attrition)')
plt.show()


# %% [markdown]
#  We can observe that our target cariable is quite imbalanced. This will require extra attention in later steps, namely when splitting the dataset into train, validation and test sets, as well as during the modelling stage.

# %%
data.shape


# %%
data.head(3)


# %% [markdown]
#  # **3. Exploratory Data Analysis**

# %% [markdown]
#  We'll start by plotting histograms to visually assess the distribution of the numeric features; this will allows us to spot any relevant patterns or trends in the data.

# %%
data.hist(figsize=(20, 15))
plt.show()


# %% [markdown]
#  The histograms reveal some important patterns in the dataset.
# 
#  - Once again we can observe that the **target variable** is highly skewed toward staying in the company.
# 
#  - Concerning demographics, **age** follows an approximately bell-shaped distribution, centered around 30-40; **Gender** is skewed with more males than females.
# 
#  - Features that are related to **work characteristics** (YearsAtCompany, TotalWorkingYears, YearsInCurrentRole, Overtime) are right-skewed, indicating many relatively new employees and fewer with long careers; working overtime is not common.
# 
#  - **Income**: Salaries and rates are right-skewed, with few very high earners.
# 
#  - **Satisfaction-related** variables are discrete and somewhat skewed toward higher ratings, while PerformanceRating shows very little variation (nearly all at level 3), suggesting limited predictive value.
# 
# 
# 
#  Overall, the data displays strong imbalance and skewness patterns that will require careful consideration during modeling, suggesting it could benefit from stratified splits, and algorithms robust to class imbalance.

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
#  The boxplots highlight the extent of skewness and make the outliers stand out clearly, which complements the histogram analysis above.
# 
#  - Outliers are especially relevant in income and emplyment duration related-variables, which may need special handling. We'll decide how to handle them further down.
# 
#  - For demographic/job characteristics (Age, DistanceFromHome, JobLevel, Education) featured the distributions are fairly compact with few outliers, aligning with the unimodal/bell-like shapes seen in histograms.
# 
#  - Ordinal satisfaction and variables show limited spread, consistent with their discrete scale, with some level of skew toward higher values. Their limited range may reduce their explanatory power.
# 
#  - PerformanceRating shows very little variation (nearly all values at level 3) confirming its limited usefulness as a predictive feature.

# %% [markdown]
#  Subsequent steps may differ based on the category of each feature. Therefore, we’ll create lists that group feature names by their respective types.

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
#  Let's now look at the distribution of our non-continuous features.

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
#  From the variables that, a priori, we'd think could be related with attrition we find that:
# 
#   - roughly 30% of employees work overtime
# 
#   - roughly 40% have low to medium levels of satisfaction with the work environment
# 
#   - roughly 30% report low to medium levels of job involvement
# 
#   - nearly 40% report low to medium job satisfaction
# 
#   - another nearly 40% have low to medium levels of satisfaction with relationships at work
# 
#   - and about 5% report bad work-life balance

# %% [markdown]
#  To better understand what might be contributing to employees’ decisions to quit, we'll next plot the non-continuous features against the target variable. We’ll also measure the attrition rate within each category. This will show us whether some groups are more prone to leaving than others, irrespective of their overall frequency.

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
#  From the plots above we find the following trends:
# 
# 
# 
#  Department-level & Job roles:
# 
# 
# 
#  - Sales and Human Resources show a higher proportion of employees quitting compared to R&D.
# 
# 
# 
#  - Within job roles, HR professionals tend to leave more often, but so do Lab Technicians, even though they are part of the R&D department.
# 
# 
# 
#  - Sales Representatives have the highest attrition rate across all job roles, whereas higher-level roles—such as managers and directors—show very low attrition.
# 
# 
# 
#  Personal characteristics
# 
# 
# 
#  - Single employees appear more likely to quit.
# 
# 
# 
#  Work conditions and workload
# 
# 
# 
#  - Employees who work overtime, travel frequently, or have poor work–life balance are more likely to leave.
# 
# 
# 
#  - Low satisfaction with the work environment, job involvement, overall job satisfaction, and relationships at work is also strongly associated with higher attrition.
# 
# 
# 
#  Job level and hierarchy
# 
# 
# 
#  - Employees in lower hierarchical levels tend to leave more often. However, attrition proportions do not strictly follow the hierarchical ranking order.
# 
# 
# 
#  Stock ownership
# 
# 
# 
#  - Employees with no stock options (stock option level 0) are more prone to quitting. This is not surprising, as offering stock is a common strategy to increase engagement.

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
# 
# 
#  Some features show noticeable differences in their distributions depending on whether the employee quit or stayed.
# 
# 
# 
#  Age and career stage
# 
# 
# 
#  - Employees who quit tend to be younger.
# 
# 
# 
#  - This aligns with lower values observed in Total Working Years, Years at Company, Years in Current Role, and Years with Current Manager.
# 
# 
# 
#  Early-career employees may be more inclined to change jobs or roles, contributing to these lower tenure metrics.
# 
# 
# 
#  Compensation
# 
# 
# 
#  - Monthly income appears influential: employees with lower income are more likely to leave, which is expected. The same applies to daily rate.
# 
# 
# 
#  Distance from home
# 
# 
# 
#  - The larger the distance from home to work, the more likely the employees are to leave.
# 
# 
# 
#  Other features
# 
# 
# 
#  - The remaining continuous features either show similar distributions across attrition groups or differences too small to be clearly meaningful.

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
# From the analyses and visualization above we observe that:
# 
# - YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager, TotalWorkingYears, JobLevel, MonthlyIncome, StockOptionLevel and Age are the strongest monotonic predictors of Attrition.
# 
# These are indicators that relate to tenure, seniority, and stability and they're in agreement with HR domain knowledge: attrition is highest among newer, younger, lower-level employees.
# 
# 
# - JobSatisfaction, JobInvolvement, EnvironmentSatisfaction Tshow mild but potentially meaningful associations.
# 
# Employees with lower satisfaction or lower involvement show slightly higher attrition.
# 

# %%
top_features = attrition_corr_sorted.abs().sort_values(ascending=False).head(12).index

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(spearman_matrix.loc[top_features, top_features], dtype=bool))
sns.heatmap(spearman_matrix.loc[top_features, top_features], mask=mask,
            cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Top Spearman Correlated Variables")
plt.show()



# %%
categorical_features = list(data.select_dtypes(include='object').columns.drop(['BusinessTravel']))

binary_features = ['Gender','OverTime']

ordinal_features = ['BusinessTravel','Education','EnvironmentSatisfaction','JobInvolvement',
                    'JobLevel', 'JobSatisfaction','PerformanceRating',
                    'RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance']

non_continuous_features = categorical_features + binary_features + ordinal_features

continuous_features = list(data.columns.difference(non_continuous_features).drop(['Attrition']))


# %%
# Selecionar apenas colunas do tipo object (qualitativas)
categorical_cols = data.select_dtypes(include='object').columns.tolist()

# Remover a variável Attrition da lista
if 'Attrition' in categorical_cols:
    categorical_cols.remove('Attrition')

# Criar gráficos para cada variável qualitativa
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    
    # Criar tabela de contingência
    contingency = data.groupby([col, 'Attrition']).size().unstack(fill_value=0)
    
    ax = sns.countplot(x=data[col], hue=data['Attrition'])
    
    for container in ax.containers:
        labels = []
        for i, bar in enumerate(container):
            height = bar.get_height()
            if height > 0:
                # Obter a categoria pela posição da barra
                categories = data[col].unique()
                category = categories[i] if i < len(categories) else categories[0]
                category_total = len(data[data[col] == category])
                percentage = (height / category_total) * 100
                labels.append(f'{percentage:.1f}%\n(n={int(height)})')
            else:
                labels.append('')
        ax.bar_label(container, labels=labels)
    
    plt.title(f'Distribution of {col} by Attrition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# %% [markdown]
#  By analyzing these bar charts, we can identify notable differences in the distribution of Attrition across our categorical variables. Key findings include:
# 
# 
# 
#  - **Business Travel**: Higher travel frequency might correlate with increased attrition rates, considering the differences between their percentages. This may suggest that frequent business travel may negatively impact employee retention.
# 
# 
# 
#  - **Education Field and Job Role**: While some correlation between education field and job role is probable (Human Resources, for instance), the Job Role variable reveals a striking pattern: Sales Representatives show nearly 40% attrition rate, significantly higher than other positions.
# 
# 
# 
#  - **Marital Status**: Single employees demonstrate substantially higher attrition rates compared to their married counterparts. The proportion of single employees leaving the company is more than double that of married employees (relative to their respective group sizes), indicating that marital status may be a relevant predictor of attrition.
# 
# 
# 
#  These preliminary observations suggest that work-life balance factors (travel, marital status) and specific job roles warrant further investigation in our analysis.

# %% [markdown]
#  We will also analyze the numerical variables in our dataset. However, before proceeding, it would be interesting to explore a potential relationship between MonthlyIncome and MonthlyRate. While MonthlyIncome represents the actual salary an employee receives each month, MonthlyRate might reflect the hourly rate or the standardized value the company attributes to that employee on a monthly basis.
# 
# 
# 
#  To investigate this relationship, we propose creating a new variable - Income_Rate_Ratio - which will capture the ratio between these two metrics and may provide insights into compensation structures and their potential impact on attrition.

# %%
data['Income_Rate_Ratio'] = data['MonthlyIncome'] / data['MonthlyRate']


# %%
# Dicionário para guardar as percentagens
percentage_dict = {}

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=data['Attrition'], y=data[col], errorbar='sd')

    # Remover a variável Attrition da lista
    if 'Attrition' in numeric_cols:
        numeric_cols.remove('Attrition')

    # Calcular médias
    means = data.groupby('Attrition')[col].mean()
    
    mean_no = means.get('No', means.iloc[0])
    mean_yes = means.get('Yes', means.iloc[1])
    
    # Calcular diferença e percentagem
    difference = mean_yes - mean_no
    percentage = ((mean_yes / mean_no) - 1) * 100 if mean_no != 0 else 0
    
    # Guardar percentagem
    percentage_dict[col] = percentage
    
    # Adicionar valores das médias
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    # Adicionar texto com a relação entre médias
    plt.text(0.5, 0.95, f'Difference: {difference:.2f} | Change: {percentage:.1f}%', 
             transform=ax.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.title(f'Mean {col} by Attrition')
    plt.tight_layout()
    plt.show()

# Mostrar top 10 variáveis com maior percentagem (em valor absoluto)
top_10 = pd.Series(percentage_dict).abs().sort_values(ascending=False).head(10)
print("\nTop 10 variáveis com maior variação percentual:")
print(top_10)


# %% [markdown]
#  By analyzing the variables with the largest percentage differences between attrition outcomes, several key patterns emerge:
# 
# 
# 
#  - **OverTime** demonstrates by far the highest percentage variation (128.6%), more than three times greater than any other variable. This suggests that overtime work is the single most critical factor associated with employee attrition and warrants immediate attention in retention strategies.
# 
# 
# 
#  - **Stock Options** show a substantial 37.6% variation, with employees possessing stock options exhibiting significantly lower attrition rates. This suggests that equity compensation may serve as an effective retention strategy, by strengthening organizational commitment and long-term alignment with the company's success.
# 
# 
# 
#  - **Tenure-related variables** (Years in Current Role: 35.3%, Years with Current Manager: 34.7%, Years at Company: 30.4%, Total Working Years: 30.5%) consistently rank among the top differentiators. Employees with higher tenure across these dimensions show lower attrition rates, indicating that role stability, manager continuity, and organizational tenure contribute positively to retention.
# 
# 
# 
#  - The newly created **Income_Rate_Ratio** (30.7%) exhibits greater percentage variation than either MonthlyIncome (29.9%) or MonthlyRate alone. This suggests that the relationship between these compensation metrics may reveal misalignment between employee remuneration and their perceived organizational value, potentially contributing to attrition decisions.
# 
# 
# 
#  However, these observations are preliminary and reflect observable relationships. Statistical testing and multivariate analysis will be necessary to establish the strength and significance of these relationships while controlling for potential confounding variables and interactions.


