# %% [markdown]
# ## **_Enterprise Data Science and Analytics - Enterprise Data Science Bootcamp_**
# 
# ### **HR Attrition Project - EDSB25_26**
# 
# - Ana Rita Martins 20240821
# - Joana Coelho 20240801
# - Pedro Fernandes 20240823
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
import pandas as pd
from summarytools import dfSummary
import matplotlib.pyplot as plt
import seaborn as sns
#from pandas.io.formats.style import Styler

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
# From the summary above, we verified that the data set does't contain duplicates, and we also gathered information about the data's distribution and main statistics.
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

# %%
data.shape

# %%
data.head(3)

# %% [markdown]
# # **3. Exploratory Data Analysis**

# %% [markdown]
# We'll start by plotting histograms to visually assess the distribution of the numeric features; this will allows us to spot any relevant patterns or trends in the data.

# %%
data.hist(figsize=(20, 15))
plt.show()

# %% [markdown]
# The histograms reveal some important patterns in the dataset. 
# - Once again we can observe that the **target variable** is highly skewed toward staying in the company.
# - Concerning demographics, **age** follows an approximately bell-shaped distribution, centered around 30-40; **Gender** is skewed with more males than females.
# - Features that are related to **work characteristics** (YearsAtCompany, TotalWorkingYears, YearsInCurrentRole, Overtime) are right-skewed, indicating many relatively new employees and fewer with long careers; working overtime is not common.
# - **Income**: Salaries and rates are right-skewed, with few very high earners.
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
# - Outliers are especially relevant in income and emplyment duration related-variables, which may need special handling. We'll decide how to handle them further down.
# - For demographic/job characteristics (Age, DistanceFromHome, JobLevel, Education) featured the distributions are fairly compact with few outliers, aligning with the unimodal/bell-like shapes seen in histograms.
# - Ordinal satisfaction and variables show limited spread, consistent with their discrete scale, with some level of skew toward higher values. Their limited range may reduce their explanatory power.
# - PerformanceRating shows very little variation (nearly all values at level 3) confirming its limited usefulness as a predictive feature.


